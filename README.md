# Jetson Xavier MCP Server

NVIDIA Jetson Xavier의 CUDA/GPU 자원을 [Claude Code](https://claude.ai/code)에서 원격으로 활용할 수 있는 [MCP(Model Context Protocol)](https://modelcontextprotocol.io/) 서버입니다.

## Architecture

```
┌──────────────┐   Streamable HTTP   ┌──────────────────────────────┐
│  Mac         │                     │  Jetson Xavier               │
│  Claude Code │  ◄──── :8765 ────►  │  MCP Server (Python 3.10)    │
│              │                     │     │                        │
└──────────────┘                     │     ├─ CUDA 11.4 / GPU       │
                                     │     ├─ PyTorch (Python 3.8)  │
                                     │     └─ Shell / File I/O      │
                                     └──────────────────────────────┘
```

## Why Two Python Versions?

JetPack의 의존성 관리가 핵심입니다.

| Runtime | Python | Reason |
|---------|--------|--------|
| **MCP Server** | 3.10 (venv) | MCP SDK requires `python >= 3.10` |
| **PyTorch/CUDA** | 3.8 (system) | NVIDIA JetPack R35.6.1 wheel은 cp38 전용 |

시스템 Python(3.8)을 업그레이드하면 JetPack ↔ CUDA ↔ cuDNN ↔ TensorRT ↔ PyTorch 간 의존성이 깨집니다. **절대 변경하지 마세요.**

MCP 서버는 Python 3.10 venv에서 실행되고, PyTorch가 필요한 도구(`run_python`, `cuda_benchmark`)는 내부적으로 `/usr/bin/python3.8`을 호출합니다.

## Requirements

### Jetson Xavier
- JetPack R35.x (L4T R35)
- CUDA 11.4
- Python 3.8 (system) + Python 3.10 (`/usr/local/bin/python3.10`)
- PyTorch (installed via NVIDIA JetPack wheel for cp38)

### Client (Mac/Linux)
- [Claude Code](https://claude.ai/code) or any MCP client

## Quick Start

### 0. Configuration (필수)

배포 전에 `deploy.sh`를 열어서 **상단 2개 변수를 본인 환경에 맞게 수정**하세요:

```bash
# deploy.sh 상단
JETSON_HOST="YOUR_JETSON_IP"       # ← Jetson의 IP 주소 (예: 192.168.1.100)
JETSON_USER="YOUR_USERNAME"        # ← Jetson의 사용자명 (예: jetson)
```

SSH 키 인증도 미리 설정해야 합니다 (배포 스크립트가 SSH로 접속하므로):

```bash
# Mac에서 실행
ssh-copy-id <user>@<jetson-ip>
```

### 1. Deploy to Jetson

```bash
# From your Mac
chmod +x deploy.sh
./deploy.sh
```

Or manually:

```bash
# Transfer files
scp jetson_mcp_server.py requirements.txt <user>@<jetson-ip>:~/mcp-server/

# On Jetson
cd ~/mcp-server
/usr/local/bin/python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python3 jetson_mcp_server.py --port 8765
```

### 2. Connect from Claude Code

```bash
claude mcp add jetson-xavier --transport streamable-http http://<jetson-ip>:8765/mcp
```

Or add to `.mcp.json`:

```json
{
  "mcpServers": {
    "jetson-xavier": {
      "type": "streamable-http",
      "url": "http://<jetson-ip>:8765/mcp"
    }
  }
}
```

### 3. Use in Claude Code

Just ask naturally:

- *"Jetson GPU 상태 확인해줘"*
- *"Jetson에서 CUDA 벤치마크 돌려줘"*
- *"Jetson에서 PyTorch로 행렬 연산 실행해줘"*

## Available Tools (17)

### System & Connectivity
| Tool | Description |
|------|-------------|
| `ping` | Health check |
| `system_info` | OS, CPU, memory, disk info |
| `gpu_status` | CUDA/GPU status, tegrastats, JetPack version |
| `python_env` | Python version and installed ML packages |
| `list_processes` | List running processes |

### Execution
| Tool | Description |
|------|-------------|
| `execute_command` | Run shell commands (with safety filters) |
| `run_python` | Execute Python code with CUDA acceleration |
| `read_file` | Read files on Jetson |
| `write_file` | Write files on Jetson |
| `cuda_benchmark` | Matrix multiplication benchmark on GPU |

### Package Management
| Tool | Description |
|------|-------------|
| `install_package` | Install Python packages with JetPack compatibility check |
| `list_compatible_packages` | List JetPack R35.6.1 compatible package versions |

### Task Queue (Async Jobs)
| Tool | Description |
|------|-------------|
| `submit_job` | Submit long-running jobs (background execution with auto fan control) |
| `check_job` | Check job status or list all jobs |
| `get_result` | Get completed job results |
| `get_log` | Get real-time job logs (useful for monitoring running jobs) |

### Fan Cooling Control
| Tool | Description |
|------|-------------|
| `set_fan_profile` | View or change fan cooling profile (quiet/cool/aggressive) |

## Fan Cooling Control

Jetson Xavier의 팬 냉각을 3단계로 제어합니다.

| Profile | Description | Use Case |
|---------|-------------|----------|
| `quiet` | 소음 최소. 50°C부터 팬 시작, 유휴 시 정지 | 유휴/저부하 |
| `cool` | 균형 모드. 35°C부터 팬 시작 | 일반 운영 |
| `aggressive` | 팬 항상 동작. 50°C에서 최대 속도 | AI 학습/추론 |

### Manual Control
`set_fan_profile` 도구로 직접 변경하거나 현재 상태를 조회할 수 있습니다.

### Automatic Control
`submit_job`으로 작업을 제출하면:
1. 작업 시작 시 팬이 `aggressive` 모드로 자동 전환 (기본값)
2. 작업 완료 시 이전 프로파일로 자동 복귀
3. `fan_profile` 파라미터로 작업별 프로파일 지정 가능

> ⚠️ `aggressive` 모드에서는 팬 소음이 크게 증가합니다.

## Systemd Service

The deploy script registers a systemd service for auto-start on boot:

```bash
# Status
sudo systemctl status jetson-mcp

# Logs
sudo journalctl -u jetson-mcp -f

# Restart
sudo systemctl restart jetson-mcp
```

## Tested Environment

| Component | Version |
|-----------|---------|
| Jetson Xavier | AGX Xavier |
| JetPack | R35.6.1 |
| CUDA | 11.4 |
| cuDNN | 8.6.0 |
| TensorRT | 8.5.2 |
| PyTorch | 2.1.0a0+nv23.06 (cp38) |
| MCP SDK | 1.26.0 |
| Python (MCP) | 3.10.13 |
| Python (System) | 3.8.10 |

## Roadmap

- [x] Async task queue (submit job, check later)
- [x] JetPack-safe package management
- [x] Fan cooling control (manual + automatic)
- [ ] ESP32 IoT data pipeline integration
- [ ] Claude Agent SDK hybrid architecture (Mac Agent ↔ Jetson Agent)
- [ ] Model inference endpoints (ONNX, TensorRT)
- [ ] Resource monitoring dashboard

## License

MIT
