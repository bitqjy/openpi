"""Generate a sidecar and phase timelines without modifying the LeRobot dataset."""

import argparse
import json
from pathlib import Path

from openpi.training.phase_labels import PhaseRules
from openpi.training.phase_labels import build_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/luna_cleaned_20260908")
    parser.add_argument("--output", type=Path, default=Path("assets/phase_labels/luna_cleaned_20260908_h10.npz"))
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--align-radius-m", type=float, default=0.04)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists; choose a new output or pass --overwrite.")
    if args.align_radius_m <= 0 or args.action_horizon <= 0:
        parser.error("Alignment radius and action horizon must be positive.")
    report = build_labels(
        args.dataset_dir,
        args.output,
        repo_id=args.repo_id,
        action_horizon=args.action_horizon,
        rules=PhaseRules(align_radius_m=args.align_radius_m),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
    print(f"Labels: {args.output}\nTimeline report: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
