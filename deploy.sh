#!/bin/bash
# ============================================================
# Jetson Xavier MCP Server 배포 스크립트
#
# ⚠️ 환경 제약:
#   - MCP 서버: Python 3.10 venv (MCP SDK requires-python >=3.10)
#   - PyTorch/CUDA: 시스템 Python 3.8 (JetPack R35.6.1 전용)
#   - 시스템 Python 변경 금지
#
# 사용법:
#   1. 아래 설정 섹션의 JETSON_HOST, JETSON_USER를 본인 환경에 맞게 수정
#   2. chmod +x deploy.sh && ./deploy.sh
# ============================================================

set -e

# ── 설정 (본인 환경에 맞게 수정하세요) ──────────────────────
JETSON_HOST="YOUR_JETSON_IP"       # 예: 192.168.1.100
JETSON_USER="YOUR_USERNAME"        # 예: jetson
JETSON_DIR="/home/${JETSON_USER}/mcp-server"
MCP_PORT=8765
PYTHON310="/usr/local/bin/python3.10"
VENV_DIR="${JETSON_DIR}/venv"

# ── 설정 검증 ────────────────────────────────────────────────
if [ "$JETSON_HOST" = "YOUR_JETSON_IP" ]; then
    echo "❌ deploy.sh의 JETSON_HOST를 Jetson의 IP 주소로 수정하세요."
    exit 1
fi
if [ "$JETSON_USER" = "YOUR_USERNAME" ]; then
    echo "❌ deploy.sh의 JETSON_USER를 Jetson의 사용자명으로 수정하세요."
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
Environment=PATH=/usr/local/cuda/bin:/home/${JETSON_USER}/.local/bin:/usr/local/bin:/usr/bin:/bin
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
