#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS_RATE_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9]+(?:\.[0-9]+)?)%")


def load_tasks(task_file: Path) -> list[str]:
    tasks = []
    for raw_line in task_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            tasks.append(line)
    if not tasks:
        raise ValueError(f"No tasks found in {task_file}")
    return tasks


def parse_eval_log(eval_log: Path) -> dict | None:
    text = eval_log.read_text(encoding="utf-8", errors="ignore")
    clean_text = ANSI_ESCAPE_RE.sub("", text)
    matches = SUCCESS_RATE_RE.findall(clean_text)
    if not matches:
        return None

    success_count_str, test_num_str, success_rate_str = matches[-1]
    success_count = int(success_count_str)
    test_num = int(test_num_str)
    success_rate = float(success_rate_str)
    return {
        "success_count": success_count,
        "test_num": test_num,
        "success_rate": round(success_rate, 2),
    }


def find_log(log_dir: Path, task_name: str, suffix: str) -> Path | None:
    matches = sorted(log_dir.glob(f"{task_name}_*_{suffix}.log"))
    if not matches:
        return None
    return matches[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    task_file = args.task_file.resolve()
    tasks = load_tasks(task_file)

    task_rows = []
    completed_rows = []

    for task_name in tasks:
        eval_log = find_log(log_dir, task_name, "eval")
        server_log = find_log(log_dir, task_name, "server")
        row = {
            "task_name": task_name,
            "status": "missing_log",
            "eval_log": str(eval_log) if eval_log else None,
            "server_log": str(server_log) if server_log else None,
        }

        if eval_log is not None:
            parsed = parse_eval_log(eval_log)
            if parsed is None:
                row["status"] = "missing_summary"
            else:
                row.update(parsed)
                row["status"] = "completed"
                completed_rows.append(row)

        task_rows.append(row)

    completed_count = len(completed_rows)
    total_tasks = len(tasks)
    total_success = sum(item["success_count"] for item in completed_rows)
    total_tests = sum(item["test_num"] for item in completed_rows)
    avg_task_success_rate = round(
        sum(item["success_rate"] for item in completed_rows) / completed_count, 2
    ) if completed_count else 0.0
    overall_episode_success_rate = round((total_success / total_tests) * 100, 2) if total_tests else 0.0

    summary = {
        "log_dir": str(log_dir),
        "task_file": str(task_file),
        "mode": args.mode,
        "completed_tasks": completed_count,
        "expected_tasks": total_tasks,
        "avg_task_success_rate": avg_task_success_rate,
        "overall_episode_success_rate": overall_episode_success_rate,
        "total_success": total_success,
        "total_tests": total_tests,
        "tasks": task_rows,
    }

    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tsv_lines = ["task_name\tstatus\tsuccess_count\ttest_num\tsuccess_rate\teval_log\tserver_log"]
    for row in task_rows:
        tsv_lines.append(
            "\t".join(
                [
                    row["task_name"],
                    row["status"],
                    str(row.get("success_count", "")),
                    str(row.get("test_num", "")),
                    str(row.get("success_rate", "")),
                    row.get("eval_log") or "",
                    row.get("server_log") or "",
                ]
            )
        )
    (log_dir / "summary.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    text_lines = [
        f"log_dir: {log_dir}",
        f"mode: {args.mode or 'unknown'}",
        f"completed_tasks: {completed_count}/{total_tasks}",
        f"avg_task_success_rate: {avg_task_success_rate:.2f}%",
        f"overall_episode_success_rate: {overall_episode_success_rate:.2f}%",
        f"total_success: {total_success}",
        f"total_tests: {total_tests}",
        "",
        "per_task:",
    ]
    for row in task_rows:
        if row["status"] == "completed":
            text_lines.append(
                f"{row['task_name']}: {row['success_rate']:.2f}% ({row['success_count']}/{row['test_num']})"
            )
        else:
            text_lines.append(f"{row['task_name']}: {row['status']}")
    (log_dir / "summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print(f"[INFO] Saved Robotwin eval summary to {log_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
