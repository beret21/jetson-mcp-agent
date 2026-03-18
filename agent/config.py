"""Agent 설정 및 비용 제어."""
import os

# Claude Code CLI 경로
CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "~/.local/bin/claude")

# 비용/시간 제한
MAX_ITERATIONS = 5          # EDA 반복 최대 횟수
MAX_TURNS_PER_ITER = 30     # 반복당 최대 턴
TIMEOUT_SECONDS = 1800      # 작업당 최대 30분

# 경로
MCP_SERVER_DIR = "/home/YOUR_USERNAME/mcp-server"
WORKSPACE_ROOT = os.path.join(MCP_SERVER_DIR, "data")
AGENT_TASKS_DIR = os.path.join(MCP_SERVER_DIR, "agent_tasks")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# 작업 디렉토리 자동 생성
os.makedirs(AGENT_TASKS_DIR, exist_ok=True)
