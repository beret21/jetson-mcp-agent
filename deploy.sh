#!/bin/bash
# ============================================================
# Jetson Xavier MCP Server 배포 스크립트
#
# ⚠️ 환경 제약:
#   - MCP 서버: Python 3.10 venv (MCP SDK requires-python >=3.10)
#   - PyTorch/CUDA: 시스템 Python 3.8 (JetPack R35.6.1 전용)
#   - 시스템 Python 변경 금지
#
# 사용법: Mac 터미널에서 실행
#   chmod +x deploy.sh
#   ./deploy.sh
# ============================================================

set -e

# ── 설정 (본인 환경에 맞게 수정하세요) ─────────────────────────
JETSON_HOST="${JETSON_HOST:-YOUR_JETSON_IP}"
JETSON_USER="${JETSON_USER:-YOUR_USERNAME}"
JETSON_DIR="/home/${JETSON_USER}/mcp-server"
MCP_PORT=8765
PYTHON310="/usr/local/bin/python3.10"
VENV_DIR="${JETSON_DIR}/venv"

if [ "$JETSON_HOST" = "YOUR_JETSON_IP" ] || [ "$JETSON_USER" = "YOUR_USERNAME" ]; then
    echo "⚠️  먼저 환경 변수를 설정하세요:"
    echo "   export JETSON_HOST=<Jetson IP>"
    echo "   export JETSON_USER=<Jetson 사용자명>"
    echo ""
    echo "   또는 deploy.sh 상단의 기본값을 직접 수정하세요."
    exit 1
fi

echo "📦 Jetson Xavier MCP Server 배포 시작"
echo "   Target: ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}"
echo "   Python: ${PYTHON310} (venv: ${VENV_DIR})"
echo ""

# ── 1. Jetson에 디렉토리 생성 ────────────────────────────────
echo "1️⃣  Jetson에 디렉토리 생성..."
ssh ${JETSON_USER}@${JETSON_HOST} "mkdir -p ${JETSON_DIR}"

# ── 2. 파일 전송 ────────────────────────────────────────────
echo "2️⃣  파일 전송 중..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
scp "${SCRIPT_DIR}/jetson_mcp_server.py" ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/
scp "${SCRIPT_DIR}/requirements.txt" ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/

# Agent 모듈 전송
echo "   Agent 모듈 전송 중..."
ssh ${JETSON_USER}@${JETSON_HOST} "mkdir -p ${JETSON_DIR}/agent/prompts ${JETSON_DIR}/agent_tasks"
scp "${SCRIPT_DIR}/agent/__init__.py" \
    "${SCRIPT_DIR}/agent/config.py" \
    "${SCRIPT_DIR}/agent/task_store.py" \
    "${SCRIPT_DIR}/agent/eda_agent.py" \
    "${SCRIPT_DIR}/agent/agent_runner.py" \
    ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/agent/
scp "${SCRIPT_DIR}/agent/prompts/eda_system.md" \
    ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/agent/prompts/

# ── 3. Python 3.10 venv 생성 및 의존성 설치 ──────────────────
echo "3️⃣  Python 3.10 venv 생성 및 의존성 설치 중..."
ssh ${JETSON_USER}@${JETSON_HOST} "
    # venv가 없으면 생성
    if [ ! -f ${VENV_DIR}/bin/python3 ]; then
        echo '   venv 생성 중...'
        ${PYTHON310} -m venv ${VENV_DIR}
    fi
    # venv 활성화 후 의존성 설치
    ${VENV_DIR}/bin/pip install --upgrade pip
    ${VENV_DIR}/bin/pip install -r ${JETSON_DIR}/requirements.txt
"

# ── 4. systemd 서비스 등록 ────────────────────────────────────
echo "4️⃣  systemd 서비스 등록 중..."
ssh ${JETSON_USER}@${JETSON_HOST} "cat > /tmp/jetson-mcp.service << UNIT
[Unit]
Description=Jetson Xavier MCP Server
After=network.target

[Service]
Type=simple
User=${JETSON_USER}
WorkingDirectory=${JETSON_DIR}
ExecStart=${VENV_DIR}/bin/python3 ${JETSON_DIR}/jetson_mcp_server.py --port ${MCP_PORT}
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/usr/local/cuda/bin:\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64

[Install]
WantedBy=multi-user.target
UNIT
sudo mv /tmp/jetson-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jetson-mcp
sudo systemctl restart jetson-mcp"

echo ""
echo "✅ 배포 완료!"
echo ""
echo "   서버 상태 확인: ssh ${JETSON_USER}@${JETSON_HOST} 'sudo systemctl status jetson-mcp'"
echo "   서버 로그 확인: ssh ${JETSON_USER}@${JETSON_HOST} 'sudo journalctl -u jetson-mcp -f'"
echo "   MCP 엔드포인트: http://${JETSON_HOST}:${MCP_PORT}/mcp"
echo ""
echo "📝 Claude Code에서 연결하려면:"
echo "   claude mcp add jetson-xavier --transport streamable-http http://${JETSON_HOST}:${MCP_PORT}/mcp"
