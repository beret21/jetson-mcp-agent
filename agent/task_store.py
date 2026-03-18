"""Agent 작업 영속화 — JSON 파일 기반."""
import json
import os
import uuid
from datetime import datetime
from .config import AGENT_TASKS_DIR


def _task_path(task_id: str) -> str:
    return os.path.join(AGENT_TASKS_DIR, f"{task_id}.json")


def submit_task(task: str, dataset: str, max_iterations: int = 5) -> dict:
    """새 작업 제출. task_id 반환."""
    task_id = f"agent_{uuid.uuid4().hex[:8]}"
    task_data = {
        "id": task_id,
        "task": task,
        "dataset": dataset,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "config": {"max_iterations": max_iterations},
        "iterations": [],
        "final_report": "",
        "error": None,
    }
    with open(_task_path(task_id), "w") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    return task_data


def load_task(task_id: str) -> dict | None:
    path = _task_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def update_task(task_id: str, updates: dict):
    """작업 데이터 부분 업데이트."""
    task = load_task(task_id)
    if not task:
        return
    task.update(updates)
    with open(_task_path(task_id), "w") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)


def list_tasks(limit: int = 20) -> list:
    """최근 작업 목록."""
    tasks = []
    if not os.path.exists(AGENT_TASKS_DIR):
        return tasks
    for fname in sorted(os.listdir(AGENT_TASKS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(AGENT_TASKS_DIR, fname), "r") as f:
                t = json.load(f)
                tasks.append({
                    "id": t["id"],
                    "task": t["task"][:60],
                    "status": t["status"],
                    "submitted_at": t.get("submitted_at", ""),
                    "iterations": len(t.get("iterations", [])),
                })
            if len(tasks) >= limit:
                break
    return tasks
