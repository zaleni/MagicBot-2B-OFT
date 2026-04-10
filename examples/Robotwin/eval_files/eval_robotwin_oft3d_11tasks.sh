#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="${SCRIPT_DIR}/robotwin_oft3d_11tasks.txt"
START_EVAL_SCRIPT="${SCRIPT_DIR}/start_eval.sh"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_robotwin_eval.py"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash eval_robotwin_oft3d_11tasks.sh -c <ckpt_path> [options]

Required:
  -c, --ckpt              Checkpoint file path (.pt or .safetensors)

Optional:
  -n, --name              Policy name / run label. Default: ckpt file stem
  -m, --mode              clean | randomized | both. Default: both
  -s, --seed              Eval seed. Default: 0
  -j, --jobs-per-gpu      Concurrent jobs per GPU. Default: 1
  -p, --base-port         First server port. Default: 5694
      --server-timeout    Seconds to wait for policy server. Default: 600
      --install-deps      Forwarded to start_eval.sh
  -h, --help              Show this help

Environment:
  ROBOTWIN_PATH           Third-party RoboTwin checkout
  STARVLA_PYTHON          Python used for the policy server and summary script
  ROBOTWIN_PYTHON         Python used for RoboTwin eval

Examples:
  bash eval_robotwin_oft3d_11tasks.sh -c /path/to/final_model/model.safetensors
  bash eval_robotwin_oft3d_11tasks.sh -c /path/to/steps_10000_pytorch_model.pt -m randomized -j 2
EOF
}

normalize_mode() {
    case "$1" in
        clean|demo_clean) printf 'demo_clean\n' ;;
        randomized|demo_randomized) printf 'demo_randomized\n' ;;
        both) printf 'both\n' ;;
        *)
            echo "Unsupported mode: $1" >&2
            exit 1
            ;;
    esac
}

resolve_summary_python() {
    if [[ -n "${STARVLA_PYTHON:-}" ]]; then
        printf '%s\n' "${STARVLA_PYTHON}"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    command -v python
}

CKPT_PATH=""
POLICY_NAME=""
MODE="both"
SEED="0"
JOBS_PER_GPU="1"
BASE_PORT="5694"
SERVER_TIMEOUT="600"
INSTALL_DEPS=0

while (( $# > 0 )); do
    case "$1" in
        -c|--ckpt) CKPT_PATH="$2"; shift 2 ;;
        -n|--name) POLICY_NAME="$2"; shift 2 ;;
        -m|--mode) MODE="$(normalize_mode "$2")"; shift 2 ;;
        -s|--seed) SEED="$2"; shift 2 ;;
        -j|--jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        -p|--base-port) BASE_PORT="$2"; shift 2 ;;
        --server-timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
        --install-deps) INSTALL_DEPS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${CKPT_PATH}" ]]; then
    echo "Missing required --ckpt" >&2
    usage
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint path does not exist: ${CKPT_PATH}" >&2
    exit 1
fi

if [[ -z "${POLICY_NAME}" ]]; then
    ckpt_name="$(basename "${CKPT_PATH}")"
    POLICY_NAME="${ckpt_name%.*}"
fi

SUMMARY_PYTHON="$(resolve_summary_python)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROBOTWIN_EVAL_ROOT:-$(dirname "${CKPT_PATH}")/robotwin_eval_runs/${POLICY_NAME}_${TIMESTAMP}}"
mkdir -p "${RUN_ROOT}"

MODES=()
if [[ "${MODE}" == "both" ]]; then
    MODES=(demo_clean demo_randomized)
else
    MODES=("${MODE}")
fi

echo "[INFO] checkpoint: ${CKPT_PATH}"
echo "[INFO] policy_name: ${POLICY_NAME}"
echo "[INFO] task_file: ${TASK_FILE}"
echo "[INFO] run_root: ${RUN_ROOT}"
echo "[INFO] modes: ${MODES[*]}"

for eval_mode in "${MODES[@]}"; do
    MODE_LOG_DIR="${RUN_ROOT}/${eval_mode}"
    mkdir -p "${MODE_LOG_DIR}"

    echo "[INFO] Running ${eval_mode} for 11 Robotwin tasks"
    ROBOTWIN_LOG_ROOT="${MODE_LOG_DIR}" \
    bash "${START_EVAL_SCRIPT}" \
        -m "${eval_mode}" \
        -n "${POLICY_NAME}" \
        -c "${CKPT_PATH}" \
        -s "${SEED}" \
        -j "${JOBS_PER_GPU}" \
        -p "${BASE_PORT}" \
        --server-timeout "${SERVER_TIMEOUT}" \
        $([[ "${INSTALL_DEPS}" == "1" ]] && printf '%s' "--install-deps") \
        "${TASK_FILE}"

    "${SUMMARY_PYTHON}" "${SUMMARY_SCRIPT}" \
        --log-dir "${MODE_LOG_DIR}" \
        --task-file "${TASK_FILE}" \
        --mode "${eval_mode}"
done

"${SUMMARY_PYTHON}" - "${RUN_ROOT}" "${TASK_FILE}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
task_file = Path(sys.argv[2])
tasks = [line.strip() for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]

mode_summaries = {}
for mode_dir in sorted(run_root.iterdir()):
    if not mode_dir.is_dir():
        continue
    summary_path = mode_dir / "summary.json"
    if summary_path.exists():
        mode_summaries[mode_dir.name] = json.loads(summary_path.read_text(encoding="utf-8"))

combined = {
    "run_root": str(run_root),
    "modes": mode_summaries,
    "tasks": [],
}

for task_name in tasks:
    row = {"task_name": task_name}
    for mode_name, summary in mode_summaries.items():
        task_map = {item["task_name"]: item for item in summary.get("tasks", [])}
        item = task_map.get(task_name, {})
        row[f"{mode_name}_status"] = item.get("status", "missing")
        row[f"{mode_name}_success_rate"] = item.get("success_rate")
        row[f"{mode_name}_success_count"] = item.get("success_count")
        row[f"{mode_name}_test_num"] = item.get("test_num")
    combined["tasks"].append(row)

(run_root / "summary.json").write_text(
    json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

lines = [f"run_root: {run_root}", ""]
for mode_name, summary in mode_summaries.items():
    lines.append(
        f"{mode_name}: avg_task_success_rate={summary.get('avg_task_success_rate', 0):.2f}% "
        f"| overall_episode_success_rate={summary.get('overall_episode_success_rate', 0):.2f}% "
        f"| completed_tasks={summary.get('completed_tasks', 0)}/{summary.get('expected_tasks', 0)}"
    )

lines.append("")
lines.append("per_task:")
for row in combined["tasks"]:
    parts = [row["task_name"]]
    for mode_name in mode_summaries.keys():
        parts.append(
            f"{mode_name}={row.get(f'{mode_name}_success_rate', 'NA')} "
            f"status={row.get(f'{mode_name}_status', 'missing')}"
        )
    lines.append(" | ".join(parts))

(run_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[INFO] Saved combined summary to {run_root / 'summary.json'}")
PY

echo "[INFO] Finished Robotwin OFT3D 11-task evaluation"
echo "[INFO] Results:"
echo "  ${RUN_ROOT}/summary.txt"
echo "  ${RUN_ROOT}/summary.json"
