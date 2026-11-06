import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import phase_labels as phases


def episode():
    state = np.zeros((80, 25), dtype=np.float32)
    state[:, 21] = 1  # TCP quaternion wxyz
    state[:, 18] = np.linspace(0.2, 0, 80)
    state[:, 7:9] = 0.04
    state[30:60, 7:9] = 0.01
    actions = np.zeros((80, 8), dtype=np.float32)
    actions[:, 7] = 1
    actions[30:60, 7] = -1
    return state, actions


def write_dataset(root, state, actions):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    n = len(state)
    info = {"fps": 20, "total_frames": n, "chunks_size": 1000, "data_path": "data/episode_{episode_index}.parquet"}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": n}) + "\n")
    pq.write_table(
        pa.table(
            {
                "state": state.tolist(),
                "actions": actions.tolist(),
                "index": np.arange(n),
                "episode_index": np.zeros(n, dtype=int),
                "frame_index": np.arange(n),
            }
        ),
        root / "data/episode_0.parquet",
    )


def test_phase_events_and_motion_proxies():
    state, actions = episode()
    phase, confidence = phases.label_episode(state, actions, fps=20)
    assert phase[0] == 0
    assert phase[30] == 1
    assert phase[45] == 2
    assert phase[60] == 3
    assert phase[-1] == 0
    assert np.all((confidence >= 0) & (confidence <= 1))
    # q and -q describe the same orientation: sign flips must not look like motion.
    signed = state.copy()
    signed[::2, 21:25] *= -1
    np.testing.assert_array_equal(phases.label_episode(signed, actions, 20)[1], confidence)


def test_chunk_boundaries_and_unknown_labels():
    phase = np.array([0, 0, 1, 1, 2])
    targets, confidence = phases.chunk_targets(phase, np.ones(5), horizon=3)
    np.testing.assert_allclose(targets[0], [2 / 3, 1 / 3, 0, 0])
    np.testing.assert_array_equal(targets[-1], [0, 0, 1, 0])
    np.testing.assert_array_equal(confidence, 1)
    targets, confidence = phases.chunk_targets(phase, np.zeros(5), horizon=10)
    assert not targets.any()
    assert not confidence.any()


def test_sidecar_is_bound_to_data_and_horizon(tmp_path):
    state, actions = episode()
    root = tmp_path / "dataset"
    write_dataset(root, state, actions)
    output = tmp_path / "labels.npz"
    report = phases.build_labels(root, output, repo_id="local/test", action_horizon=10)
    assert report["total_frames"] == 80

    class Dataset:
        def __init__(self):
            self.root = root

        def __len__(self):
            return 80

        def __getitem__(self, index):
            return {"index": index, "episode_index": 0, "frame_index": index, "state": state[index]}

    dataset = Dataset()
    labeled = phases.PhaseLabeledDataset(dataset, output, repo_id="local/test", action_horizon=10)
    assert labeled[20]["phase_targets"].shape == (4,)
    np.testing.assert_array_equal(labeled[20]["state"], state[20])
    with pytest.raises(ValueError, match="horizon mismatch"):
        phases.PhaseLabeledDataset(dataset, output, repo_id="local/test", action_horizon=5)
    actions[0, 0] += 0.1
    write_dataset(root, state, actions)
    with pytest.raises(ValueError, match="do not match"):
        phases.PhaseLabeledDataset(dataset, output, repo_id="local/test", action_horizon=10)


def test_supervision_survives_repack_without_becoming_state(tmp_path, monkeypatch):
    from openpi import transforms
    from openpi.models import tokenizer
    from openpi.models.action_moe import ActionMoEConfig
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import LeRobotManiSkillDataConfig

    # Repack/data transforms only: keep the unit test independent of tokenizer downloads.
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda *_, **__: object())
    model_config = Pi0Config(pi05=True, action_moe=ActionMoEConfig())
    config = LeRobotManiSkillDataConfig(phase_labels_path="labels.npz").create(tmp_path, model_config)
    state, actions = episode()
    sample = {
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "wrist_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "state": state[0],
        "actions": actions[:10],
        "prompt": "turn the valve",
        "phase_targets": np.array([0, 0, 1, 0], dtype=np.float32),
        "phase_confidence": np.float32(0.6),
    }
    transform = transforms.compose([*config.repack_transforms.inputs, *config.data_transforms.inputs])
    result = transform(sample)
    np.testing.assert_array_equal(result["state"], sample["state"])
    np.testing.assert_array_equal(result["phase_targets"], sample["phase_targets"])
    assert result["prompt"] == sample["prompt"]
    assert result["phase_confidence"] == sample["phase_confidence"]


def test_bad_layout_fails_before_labeling(tmp_path):
    state, actions = episode()
    write_dataset(tmp_path, state[:, :18], actions)
    with pytest.raises(ValueError, match="state=qpos9"):
        phases.read_columns(tmp_path)
