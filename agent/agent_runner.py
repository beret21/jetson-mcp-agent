#!/usr/bin/env python3
"""
Jetson Xavier 자율 Agent Runner.

사용법:
  python3 -m agent.agent_runner submit --task "CNC EDA" --dataset raw/cnc_mill_real.csv
  python3 -m agent.agent_runner status --task-id agent_abc12345
  python3 -m agent.agent_runner result --task-id agent_abc12345
  python3 -m agent.agent_runner list
"""
import argparse
import json
import os
import sys
import subprocess
from datetime import datetime

# 모듈 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.task_store import submit_task, load_task, update_task, list_tasks
from agent.config import AGENT_TASKS_DIR, WORKSPACE_ROOT


def cmd_submit(args):
    """새 EDA 작업 제출 (백그라운드 실행)."""
    dataset = args.dataset
    # 상대경로 → 절대경로
    if not os.path.isabs(dataset):
        dataset = os.path.join(WORKSPACE_ROOT, dataset)

    if not os.path.exists(dataset):
        print(f"Error: 데이터셋을 찾을 수 없습니다: {dataset}")
        sys.exit(1)

    task_data = submit_task(
        task=args.task,
        dataset=dataset,
        max_iterations=args.max_iterations,
    )
    task_id = task_data["id"]

    # 백그라운드에서 EDA 루프 실행
    log_path = os.path.join(AGENT_TASKS_DIR, f"{task_id}.log")
    runner_script = os.path.abspath(__file__)

    cmd = [
        sys.executable, runner_script, "run", "--task-id", task_id
    ]

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # 부모 프로세스 종료 시에도 계속 실행
        )

    update_task(task_id, {"pid": proc.pid})

    print(f"Task submitted: {task_id}")
    print(f"  Task: {args.task}")
    print(f"  Dataset: {dataset}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  PID: {proc.pid}")
    print(f"  Log: {log_path}")
    print(f"\nCheck status: python3 -m agent.agent_runner status --task-id {task_id}")


def cmd_run(args):
    """EDA 루프 실행 (직접 호출용, submit이 백그라운드로 호출)."""
    from agent.eda_agent import run_eda_loop
    run_eda_loop(args.task_id)


def cmd_status(args):
    """작업 상태 확인."""
    task = load_task(args.task_id)
    if not task:
        print(f"Error: 작업을 찾을 수 없습니다: {args.task_id}")
        sys.exit(1)

    print(f"Task: {task['id']}")
    print(f"  Status: {task['status']}")
    print(f"  Task: {task['task']}")
    print(f"  Submitted: {task.get('submitted_at', '')}")

    iterations = task.get("iterations", [])
    if iterations:
        print(f"  Iterations: {len(iterations)}")
        for it in iterations:
            acc = it.get("accuracy", "N/A")
            actions = ", ".join(it.get("actions", []))[:50]
            print(f"    Iter {it['iteration']}: {acc}% — {actions}")

    if task["status"] == "completed":
        print(f"  Best Accuracy: {task.get('best_accuracy')}%")
        print(f"  Finished: {task.get('finished_at', '')}")
    elif task["status"] == "failed":
        print(f"  Error: {task.get('error', '')[:200]}")


def cmd_result(args):
    """완료된 작업의 보고서 출력."""
    task = load_task(args.task_id)
    if not task:
        print(f"Error: 작업을 찾을 수 없습니다: {args.task_id}")
        sys.exit(1)

    if task["status"] != "completed":
        print(f"작업이 아직 완료되지 않았습니다. 현재 상태: {task['status']}")
        sys.exit(1)

    report = task.get("final_report", "")
    if report:
        print(report)
    else:
        print("보고서가 없습니다.")
        print(json.dumps(task.get("iterations", []), indent=2, ensure_ascii=False))


def cmd_list(args):
    """전체 작업 목록."""
    tasks = list_tasks()
    if not tasks:
        print("등록된 작업이 없습니다.")
        return

    print(f"{'ID':<20} {'Status':<12} {'Iters':<6} {'Task'}")
    print("-" * 70)
    for t in tasks:
        print(f"{t['id']:<20} {t['status']:<12} {t['iterations']:<6} {t['task']}")


def main():
    parser = argparse.ArgumentParser(description="Jetson Xavier 자율 Agent Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = subparsers.add_parser("submit", help="새 EDA 작업 제출")
    p_submit.add_argument("--task", required=True, help="작업 설명")
    p_submit.add_argument("--dataset", required=True, help="데이터셋 경로")
    p_submit.add_argument("--max-iterations", type=int, default=5, help="최대 반복 횟수")

    # run (내부용)
    p_run = subparsers.add_parser("run", help="EDA 루프 실행 (내부용)")
    p_run.add_argument("--task-id", required=True)

    # status
    p_status = subparsers.add_parser("status", help="작업 상태 확인")
    p_status.add_argument("--task-id", required=True)

    # result
    p_result = subparsers.add_parser("result", help="작업 결과 조회")
    p_result.add_argument("--task-id", required=True)

    # list
    subparsers.add_parser("list", help="전체 작업 목록")

    args = parser.parse_args()

    commands = {
        "submit": cmd_submit,
        "run": cmd_run,
        "status": cmd_status,
        "result": cmd_result,
        "list": cmd_list,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
