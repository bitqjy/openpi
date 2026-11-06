"""Offline, training-only weak phases for Panda pd_joint_pos demonstrations.

The rules use gripper commands, measured opening and TCP motion. They do NOT
establish physical contact or success. Future samples are used only to construct
supervision; none of these features enter the inference router.
"""

import dataclasses
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

PHASE_NAMES = ("approach", "align_grasp", "constrained_operation", "release_retreat")
SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class PhaseRules:
    align_radius_m: float = 0.04
    align_lookback_s: float = 2.0
    closing_settle_s: float = 0.25
    release_window_s: float = 0.6
    # Panda normalized pd_joint_pos gripper command: -1 close, +1 open.
    close_command: float = -0.5
    open_command: float = 0.5


DEFAULT_RULES = PhaseRules()


def read_columns(root: Path):
    """Read only low-dimensional columns; no RGB decoding or dataset mutation."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    keys = ("index", "episode_index", "frame_index", "state", "actions")
    parts = {key: [] for key in keys}
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        ep = episode["episode_index"]
        path = root / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        table = pq.read_table(path, columns=list(keys))
        for key in keys:
            parts[key].append(np.asarray(table[key].to_pylist()))
    data = {key: np.concatenate(value) for key, value in parts.items()}
    order = np.argsort(data["index"])
    data = {key: value[order] for key, value in data.items()}
    if not np.array_equal(data["index"], np.arange(info["total_frames"])):
        raise ValueError("Expected unique, contiguous LeRobot global frame indices.")
    if data["state"].shape != (len(order), 25) or data["actions"].shape != (len(order), 8):
        raise ValueError("Phase rules require Panda state=qpos9+qvel9+tcp_pose7, actions=8 absolute joint targets.")
    for key in ("state", "actions"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite {key} in source dataset.")
    for ep in np.unique(data["episode_index"]):
        frames = data["frame_index"][data["episode_index"] == ep]
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"Episode {ep}: frame indices are not ordered and contiguous.")
    return data, info


def source_digest(data, fps):
    digest = hashlib.sha256(str(float(fps)).encode())
    for key in ("index", "episode_index", "frame_index", "state", "actions"):
        dtype = "<f4" if key in ("state", "actions") else "<i8"
        value = np.ascontiguousarray(data[key], dtype=dtype)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def label_episode(state, actions, fps, rules=DEFAULT_RULES):
    if fps <= 0 or len(state) == 0:
        raise ValueError("Need a nonempty episode and positive FPS.")
    grip = actions[:, 7]
    if np.max(np.abs(grip)) > 1.01:
        raise ValueError("Expected normalized pd_joint_pos gripper commands in [-1, 1].")
    closed = np.zeros(len(state), dtype=bool)
    mode = False
    for i, command in enumerate(grip):
        if command <= rules.close_command:
            mode = True
        elif command >= rules.open_command:
            mode = False
        closed[i] = mode

    phase = np.where(closed, 2, 0).astype(np.int32)
    confidence = np.full(len(state), 0.6, dtype=np.float32)
    tcp = state[:, 18:21]
    quat = state[:, 21:25]
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(quat_norm < 0.5) or np.any(quat_norm > 1.5):
        raise ValueError("Invalid TCP quaternion; check state layout before generating labels.")
    quat = quat / quat_norm
    distance = np.r_[0.0, np.linalg.norm(np.diff(tcp, axis=0), axis=-1)]
    angle = np.r_[0.0, 2 * np.arccos(np.clip(np.abs(np.sum(quat[1:] * quat[:-1], -1)), 0, 1))]
    opening_speed = np.r_[0.0, np.abs(np.diff(state[:, 7:9].sum(-1)))] * fps
    # Closed gripper and motion are only proxies for manipulation, not contact labels.
    confidence[closed & (distance * fps < 0.005) & (angle * fps < 0.05)] = 0.25
    confidence[closed & (opening_speed > 0.02)] = 0.25

    closing = np.flatnonzero(closed & ~np.r_[False, closed[:-1]])
    opening = np.flatnonzero(~closed & np.r_[False, closed[:-1]])
    for start in opening:
        end = min(len(state), start + max(1, round(rules.release_window_s * fps)))
        next_close = closing[closing > start]
        if len(next_close):
            end = min(end, next_close[0])
        phase[start:end] = 3
        confidence[start:end] = 0.6
    for start in closing:
        begin = start
        lower = max(0, start - round(rules.align_lookback_s * fps))
        while (
            begin > lower
            and not closed[begin - 1]
            and np.linalg.norm(tcp[begin - 1] - tcp[start]) <= rules.align_radius_m
        ):
            begin -= 1
        end = min(len(state), start + max(1, round(rules.closing_settle_s * fps)))
        next_open = opening[opening > start]
        if len(next_open):
            end = min(end, next_open[0])
        phase[begin:end] = 1
        confidence[begin:end] = 0.8
    # Ambiguous command ranges provide no phase supervision.
    confidence[(grip > rules.close_command) & (grip < rules.open_command)] = 0
    changes = np.flatnonzero(np.diff(phase)) + 1
    confidence[changes] *= 0.5
    return phase, confidence


def chunk_targets(phase, confidence, horizon):
    if horizon <= 0:
        raise ValueError("Action horizon must be positive.")
    targets = np.zeros((len(phase), 4), dtype=np.float32)
    weights = np.zeros(len(phase), dtype=np.float32)
    for i in range(len(phase)):
        end = min(i + horizon, len(phase))  # never cross episode boundaries or count padding as evidence
        counts = np.bincount(phase[i:end], weights=confidence[i:end], minlength=4)
        if counts.sum() > 0:
            targets[i] = counts / counts.sum()
        weights[i] = np.mean(confidence[i:end])
    return targets, weights


def build_labels(root: Path, output: Path, *, repo_id: str, action_horizon=10, rules=DEFAULT_RULES):
    data, info = read_columns(root)
    targets = np.zeros((len(data["index"]), 4), dtype=np.float32)
    weights = np.zeros(len(targets), dtype=np.float32)
    phases = np.zeros(len(targets), dtype=np.int32)
    episodes = []
    for ep in np.unique(data["episode_index"]):
        idx = np.flatnonzero(data["episode_index"] == ep)
        phase, confidence = label_episode(data["state"][idx], data["actions"][idx], info["fps"], rules)
        targets[idx], weights[idx] = chunk_targets(phase, confidence, action_horizon)
        phases[idx] = phase
        changes = np.r_[0, np.flatnonzero(np.diff(phase)) + 1, len(phase)]
        episodes.append(
            {
                "episode_index": int(ep),
                "frames": len(idx),
                "phase_counts": dict(zip(PHASE_NAMES, np.bincount(phase, minlength=4).tolist(), strict=True)),
                "segments": [
                    {"start": int(a), "end": int(b), "phase": PHASE_NAMES[phase[a]]}
                    for a, b in itertools.pairwise(changes)
                ],
            }
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": repo_id,
        "action_horizon": action_horizon,
        "fps": info["fps"],
        "source_digest": source_digest(data, info["fps"]),
        "phase_names": PHASE_NAMES,
        "rules": dataclasses.asdict(rules),
        "state_layout": "panda_qpos9_qvel9_tcp7",
        "gripper_convention": "-1 close, +1 open",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata=np.asarray(json.dumps(metadata)),
        phase_targets=targets,
        phase_confidence=weights,
        frame_phase=phases,
        episode_index=data["episode_index"],
        frame_index=data["frame_index"],
    )
    report = {
        **metadata,
        "total_frames": len(targets),
        "total_episodes": len(episodes),
        "phase_counts": dict(zip(PHASE_NAMES, np.bincount(phases, minlength=4).tolist(), strict=True)),
        "mean_confidence": float(weights.mean()),
        "episodes": episodes,
        "caveat": "Weak gripper/TCP proxies. Inspect against videos; these are not verified contact or success labels.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class PhaseLabeledDataset:
    def __init__(self, dataset, path, *, repo_id, action_horizon):
        self._dataset = dataset
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Missing phase labels: {path}. Run scripts/build_maniskill_phase_labels.py first.")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            self._targets = saved["phase_targets"]
            self._confidence = saved["phase_confidence"]
            self._episode = saved["episode_index"]
            self._frame = saved["frame_index"]
        if (metadata["schema_version"], metadata["repo_id"], metadata["action_horizon"]) != (
            SCHEMA_VERSION,
            repo_id,
            action_horizon,
        ):
            raise ValueError("Phase label schema, repo ID or action horizon mismatch; regenerate labels.")
        data, info = read_columns(Path(dataset.root))
        if metadata["source_digest"] != source_digest(data, info["fps"]):
            raise ValueError("Phase labels do not match dataset state/actions/indices/FPS; regenerate labels.")
        if self._targets.shape != (len(dataset), 4) or self._confidence.shape != (len(dataset),):
            raise ValueError("Phase label shape mismatch.")
        if not np.array_equal(self._episode, data["episode_index"]) or not np.array_equal(
            self._frame, data["frame_index"]
        ):
            raise ValueError("Phase label frame ordering mismatch.")
        if not np.isfinite(self._targets).all() or not np.isfinite(self._confidence).all():
            raise ValueError("Non-finite phase labels.")
        if np.any(self._targets < 0) or np.any((self._confidence < 0) | (self._confidence > 1)):
            raise ValueError("Invalid phase probabilities or confidence.")
        if not np.allclose(self._targets[self._confidence > 0].sum(-1), 1):
            raise ValueError("Supervised phase targets must sum to one.")

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        idx = int(sample["index"])
        if int(sample["episode_index"]) != self._episode[idx] or int(sample["frame_index"]) != self._frame[idx]:
            raise ValueError("Sample and phase label indices disagree.")
        return {**sample, "phase_targets": self._targets[idx], "phase_confidence": self._confidence[idx]}
