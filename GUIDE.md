# Jetson Xavier MCP Server 설정 가이드

## ⚠️ Jetson Xavier 환경 제약사항 (반드시 준수)

### Python 버전 정책
- **시스템 Python**: 3.8.10 (`/usr/bin/python3` → `python3.8`)
  - JetPack R35.6.1의 기본 Python. 시스템 패키지가 의존
  - **절대 업그레이드/변경 금지** — 시스템이 망가짐
  - PyTorch 2.1.0 (NVIDIA JetPack 전용 wheel, cp38) 설치됨
- **Python 3.10** (`/usr/local/bin/python3.10`)
  - 소스 빌드로 설치됨. 시스템과 완전 격리
  - MCP SDK가 `requires-python: >=3.10`이므로 MCP 서버 실행에 사용
  - **이 버전도 함부로 업그레이드 금지**

### 이원화 구조
```
MCP 서버 실행:  Python 3.10 venv (~/<deploy-dir>/venv/)
                └─ mcp, uvicorn, httpx

PyTorch/CUDA:   Python 3.8 (시스템)
                └─ torch 2.1.0a0 (NVIDIA JetPack wheel)
                └─ run_python(), cuda_benchmark()에서 /usr/bin/python3.8로 호출
```

### JetPack 의존성
- **JetPack**: R35.6.1 (L4T R35.6.1)
- **CUDA**: 11.4 (`/usr/local/cuda-11.4/`)
- **cuDNN**: 8.6.0
- **TensorRT**: 8.5.2
- **무작정 버전 업그레이드 금지**: JetPack, CUDA, Python, PyTorch 모두 상호 의존성이 있으므로 개별 업그레이드 시 시스템 전체가 깨질 수 있음

### 환경변수 필수 설정
```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 아키텍처

```
[Mac: Claude Code]  ──── Streamable HTTP ────▶  [Jetson Xavier: MCP Server]
                                                        │
                                                  ┌─────┴─────┐
                                                  │ CUDA GPU  │
                                                  │ PyTorch   │
                                                  │ Shell/Py  │
                                                  └───────────┘
```

## 배포 방법

### 방법 1: 배포 스크립트 사용 (추천)

```bash
# deploy.sh 상단의 JETSON_HOST, JETSON_USER를 수정한 후 실행
chmod +x deploy.sh
./deploy.sh
```

### 방법 2: 수동 배포

```bash
# 1. 파일 전송
scp jetson_mcp_server.py requirements.txt <user>@<jetson-ip>:~/mcp-server/

# 2. Jetson에서 설치 및 실행
ssh <user>@<jetson-ip>
cd ~/mcp-server
/usr/local/bin/python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python3 jetson_mcp_server.py --port 8765
```

## Claude Code에서 연결

```bash
# MCP 서버 등록
claude mcp add jetson-xavier --transport streamable-http http://<jetson-ip>:8765/mcp
```

또는 프로젝트 `.mcp.json`에 직접 추가:

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

## 제공 도구 목록

| 도구 | 설명 |
|------|------|
| `ping` | 서버 연결 상태 확인 |
| `system_info` | OS, CPU, 메모리, 디스크 정보 |
| `gpu_status` | CUDA/GPU 상태 및 사용률 |
| `python_env` | Python 버전 및 ML 패키지 목록 |
| `execute_command` | 셸 커맨드 실행 |
| `run_python` | Python 코드 실행 (CUDA 가속) |
| `read_file` | 파일 읽기 |
| `write_file` | 파일 쓰기 |
| `cuda_benchmark` | CUDA 행렬 연산 벤치마크 |
| `list_processes` | 프로세스 목록 확인 |

## 사용 예시 (Claude Code에서)

Claude Code 터미널에서 이렇게 요청하면 됩니다:

- "Jetson GPU 상태 확인해줘"
- "Jetson에서 PyTorch로 이미지 추론 실행해줘"
- "Jetson CUDA 벤치마크 돌려줘"
- "Jetson에 있는 파일 읽어줘"
