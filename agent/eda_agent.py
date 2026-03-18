"""EDA 자율 Agent — Claude Code CLI를 활용한 반복 EDA 루프."""
import asyncio
import json
import os
import re
import subprocess
from datetime import datetime

from .config import CLAUDE_CLI, MAX_ITERATIONS, TIMEOUT_SECONDS, PROMPTS_DIR, MCP_SERVER_DIR
from .task_store import load_task, update_task


def _load_prompt() -> str:
    """EDA 시스템 프롬프트 로드."""
    path = os.path.join(PROMPTS_DIR, "eda_system.md")
    with open(path, "r") as f:
        return f.read()


def _build_user_prompt(task: str, dataset: str, iteration: int, context: list) -> str:
    """반복별 사용자 프롬프트 생성."""
    if iteration == 0:
        return f"""다음 데이터셋에 대해 EDA를 수행하세요.

데이터셋: {dataset}
작업: {task}

1단계: data(action="stats")로 데이터 구조를 파악하세요.
2단계: xai(action="explain")로 상관관계, 이상치, 분포를 분석하세요.
3단계: Baseline 모델을 학습하세요 (job submit).
4단계: xai(action="trace")로 학습 결과를 해석하세요.

반드시 === ITERATION 0 COMPLETE === 형식으로 결과를 출력하세요."""

    # Iteration 1+: 이전 결과 기반 개선
    prev_summary = "\n".join([
        f"- Iter {c['iteration']}: {c.get('accuracy', 'N/A')}% ({', '.join(c.get('actions', []))})"
        for c in context
    ])

    return f"""이전 반복 결과:
{prev_summary}

현재 반복: {iteration}
데이터셋: {dataset}

1단계: xai(action="diagnose", job_id="{context[-1].get('job_id', '')}", path="{dataset}")로 진단하세요.
2단계: diagnose 추천을 기반으로 피처 엔지니어링을 수행하세요 (execute python).
3단계: 개선된 데이터로 재학습하세요 (job submit).
4단계: xai(action="compare", job_ids="{','.join(c.get('job_id', '') for c in context)},{'{new_job_id}'}")로 비교하세요.

반드시 === ITERATION {iteration} COMPLETE === 형식으로 결과를 출력하세요."""


def _parse_iteration_result(output: str, iteration: int) -> dict:
    """Claude Code 출력에서 반복 결과 파싱."""
    result = {
        "iteration": iteration,
        "accuracy": None,
        "actions": [],
        "job_id": None,
        "features": None,
        "should_stop": False,
        "reason": "",
        "timestamp": datetime.now().isoformat(),
    }

    # 정확도 파싱
    acc_match = re.search(r'Accuracy:\s*([\d.]+)%', output)
    if acc_match:
        result["accuracy"] = float(acc_match.group(1))

    # 액션 파싱
    actions_match = re.search(r'Actions:\s*(.+)', output)
    if actions_match:
        result["actions"] = [a.strip() for a in actions_match.group(1).split(",")]

    # Job ID 파싱
    job_match = re.search(r'Job ID:\s*(\S+)', output)
    if job_match:
        result["job_id"] = job_match.group(1)

    # Features 파싱
    feat_match = re.search(r'Features:\s*(\d+)', output)
    if feat_match:
        result["features"] = int(feat_match.group(1))

    # 중단 판단
    stop_match = re.search(r'Should Stop:\s*(true|false)', output, re.IGNORECASE)
    if stop_match:
        result["should_stop"] = stop_match.group(1).lower() == "true"

    reason_match = re.search(r'Reason:\s*(.+)', output)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()

    return result


def run_iteration(task_id: str, task: str, dataset: str,
                  iteration: int, context: list) -> dict:
    """Claude Code CLI로 단일 EDA 반복 실행."""
    system_prompt = _load_prompt()
    user_prompt = _build_user_prompt(task, dataset, iteration, context)

    # Claude Code CLI 호출
    # CLI 사용법: claude [options] [prompt]  (위치 인자로 프롬프트 전달)
    cmd = [
        CLAUDE_CLI,
        "--print",                       # 결과만 출력
        "--dangerously-skip-permissions", # 자율 모드
        "--output-format", "text",       # 텍스트 출력
        "--max-turns", "30",             # 턴 제한
        "--system-prompt", system_prompt,
        user_prompt,                     # 위치 인자로 프롬프트 전달
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=MCP_SERVER_DIR,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[STDERR]: {result.stderr}"
    except subprocess.TimeoutExpired:
        output = f"[TIMEOUT] {TIMEOUT_SECONDS}s 초과"
    except Exception as e:
        output = f"[ERROR] {str(e)}"

    # 결과 파싱
    iter_result = _parse_iteration_result(output, iteration)
    iter_result["raw_output_length"] = len(output)

    return iter_result


def run_eda_loop(task_id: str):
    """전체 EDA 반복 루프 실행."""
    task_data = load_task(task_id)
    if not task_data:
        return

    update_task(task_id, {"status": "running", "started_at": datetime.now().isoformat()})

    task = task_data["task"]
    dataset = task_data["dataset"]
    max_iter = task_data.get("config", {}).get("max_iterations", MAX_ITERATIONS)
    context = []

    try:
        for iteration in range(max_iter):
            print(f"\n{'='*60}")
            print(f"  EDA Iteration {iteration} / {max_iter}")
            print(f"{'='*60}\n")

            iter_result = run_iteration(task_id, task, dataset, iteration, context)
            context.append(iter_result)

            # 작업 업데이트
            update_task(task_id, {
                "iterations": context,
                "current_iteration": iteration,
            })

            print(f"  Accuracy: {iter_result.get('accuracy')}%")
            print(f"  Actions: {iter_result.get('actions')}")
            print(f"  Should Stop: {iter_result.get('should_stop')}")

            if iter_result.get("should_stop"):
                print(f"  Reason: {iter_result.get('reason')}")
                break

        # 최종 보고서 생성
        best_iter = max(context, key=lambda x: x.get("accuracy") or 0)
        report = _generate_report(task, dataset, context, best_iter)

        update_task(task_id, {
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "final_report": report,
            "best_accuracy": best_iter.get("accuracy"),
            "total_iterations": len(context),
        })
        print(f"\n{'='*60}")
        print(f"  EDA COMPLETE — Best: {best_iter.get('accuracy')}%")
        print(f"{'='*60}\n")

    except Exception as e:
        import traceback
        update_task(task_id, {
            "status": "failed",
            "finished_at": datetime.now().isoformat(),
            "error": traceback.format_exc(),
        })
        print(f"  EDA FAILED: {e}")


def _generate_report(task: str, dataset: str, iterations: list, best: dict) -> str:
    """마크다운 보고서 생성."""
    lines = [
        f"# EDA Report: {os.path.basename(dataset)}",
        f"\n## Task\n{task}",
        f"\n## Summary",
        f"- Total iterations: {len(iterations)}",
        f"- Best accuracy: {best.get('accuracy')}% (Iteration {best.get('iteration')})",
        f"\n## Iteration History\n",
        "| # | Actions | Accuracy | Change |",
        "|---|---------|----------|--------|",
    ]

    prev_acc = None
    for it in iterations:
        acc = it.get("accuracy")
        actions = ", ".join(it.get("actions", []))[:40]
        delta = f"+{acc - prev_acc:.1f}%p" if acc and prev_acc else "-"
        lines.append(f"| {it['iteration']} | {actions} | {acc}% | {delta} |")
        prev_acc = acc

    return "\n".join(lines)
