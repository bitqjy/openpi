#!/usr/bin/env bash
# Run from any directory. Inherit the already deployed A800 CUDA environment.
set -euo pipefail

OPENPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${MANISKILL_PROJECT_ROOT:-$(cd -- "$OPENPI_ROOT/../.." && pwd)}"
PYTHON="${OPENPI_PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
REPO_ID="${REPO_ID:-local/luna_cleaned_20260908}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/dataset/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/.cache/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

CONFIG=pi05_maniskill_phase_moe_full
BASE_PARAMS="${PI05_BASE_PARAMS:-/mnt/data/base_model/pi05_base/params}"
ASSETS_BASE="${ASSETS_BASE:-$PROJECT_ROOT/assets_openpi}"
# Same state/actions and transforms as the audited dense recipe: reuse its stats.
NORM_ASSETS_DIR="${NORM_ASSETS_DIR:-$ASSETS_BASE/pi05_maniskill}"
PHASE_LABELS="${PHASE_LABELS:-$ASSETS_BASE/phase_labels/${REPO_ID//\//_}_h10.npz}"

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=true
    shift
fi
resume=false
for arg in "$@"; do
    if [[ "$arg" == --resume ]]; then resume=true; fi
done

label_cmd=("$PYTHON" "$OPENPI_ROOT/scripts/build_maniskill_phase_labels.py"
    --dataset-dir "$HF_LEROBOT_HOME/$REPO_ID" --repo-id "$REPO_ID"
    --output "$PHASE_LABELS" --action-horizon 10)
train_cmd=("$PYTHON" "$OPENPI_ROOT/scripts/train.py" "$CONFIG"
    --exp-name "${EXP_NAME:-luna_joint_pi05_phase_moe_full}"
    --data.repo-id "$REPO_ID" --data.phase-labels-path "$PHASE_LABELS"
    --data.assets.assets-dir "$NORM_ASSETS_DIR"
    --assets-base-dir "$ASSETS_BASE"
    --checkpoint-base-dir "${CHECKPOINT_BASE:-$PROJECT_ROOT/checkpoints_openpi}"
    --weight-loader.params-path "$BASE_PARAMS"
    --batch-size "${BATCH_SIZE:-8}" --num-workers "${NUM_WORKERS:-2}"
    --fsdp-devices "${FSDP_DEVICES:-1}" --num-train-steps "${NUM_TRAIN_STEPS:-30000}"
    --no-wandb-enabled "$@")

if "$dry_run"; then
    printf 'Generate labels if missing: '; printf '%q ' "${label_cmd[@]}"; printf '\n'
    printf 'Train: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
    exit 0
fi

[[ -x "$PYTHON" ]] || { printf 'Missing Python environment: %s\n' "$PYTHON" >&2; exit 1; }
[[ -f "$HF_LEROBOT_HOME/$REPO_ID/meta/info.json" ]] || { printf 'Missing LeRobot dataset: %s\n' "$REPO_ID" >&2; exit 1; }
[[ -f "$NORM_ASSETS_DIR/$REPO_ID/norm_stats.json" ]] || {
    printf 'Missing audited normalization stats: %s/%s/norm_stats.json\n' "$NORM_ASSETS_DIR" "$REPO_ID" >&2; exit 1;
}
if ! "$resume" && [[ ! -d "$BASE_PARAMS" ]]; then
    printf 'Set PI05_BASE_PARAMS to the local JAX params directory: %s\n' "$BASE_PARAMS" >&2
    exit 1
fi
cd "$OPENPI_ROOT"
if [[ ! -f "$PHASE_LABELS" ]]; then "${label_cmd[@]}"; fi
exec "${train_cmd[@]}"
