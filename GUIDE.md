**English** | [한국어](GUIDE.ko.md)

# Jetson MCP Agent Setup Guide

## ⚠️ Jetson Xavier Environment Constraints (Must Follow)

### Python Version Policy
- **System Python**: 3.8.10 (`/usr/bin/python3` → `python3.8`)
  - Default Python for JetPack R35.6.1. System packages depend on it
  - **Never upgrade or modify** — it will break the system
  - PyTorch 2.1.0 installed (NVIDIA JetPack wheel, cp38)
- **Python 3.10** (`/usr/local/bin/python3.10`)
  - Source-built installation, fully isolated from system
  - Used for MCP server since MCP SDK `requires-python: >=3.10`
  - **Do not upgrade this version either**

### Dual-Runtime Architecture
```
MCP Server:     Python 3.10 venv (~/<deploy-dir>/venv/)
                └─ mcp, uvicorn, httpx

PyTorch/CUDA:   Python 3.8 (system)
                └─ torch 2.1.0a0 (NVIDIA JetPack wheel)
                └─ run_python(), cuda_benchmark() invoke /usr/bin/python3.8
```

### JetPack Dependencies
- **JetPack**: R35.6.1 (L4T R35.6.1)
- **CUDA**: 11.4 (`/usr/local/cuda-11.4/`)
- **cuDNN**: 8.6.0
- **TensorRT**: 8.5.2
- **Do not arbitrarily upgrade versions**: JetPack, CUDA, Python, and PyTorch are all interdependent — upgrading any one individually can break the entire system

### Required Environment Variables
```bash
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Architecture

```
[Mac: Claude Code]  ──── Streamable HTTP ────▶  [Jetson Xavier: MCP Server]
                                                        │
                                                  ┌─────┴─────┐
                                                  │ CUDA GPU  │
                                                  │ PyTorch   │
                                                  │ Shell/Py  │
                                                  └───────────┘
```

## Deployment

### Method 1: Deploy Script (Recommended)

```bash
# Edit JETSON_HOST, JETSON_USER at the top of deploy.sh, then run
chmod +x deploy.sh
./deploy.sh
```

### Method 2: Manual Deployment

```bash
# 1. Transfer files
scp jetson_mcp_server.py requirements.txt <user>@<jetson-ip>:~/mcp-server/

# 2. Install and run on Jetson
ssh <user>@<jetson-ip>
cd ~/mcp-server
/usr/local/bin/python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python3 jetson_mcp_server.py --port 8765
```

## Connect from Claude Code

```bash
# Register MCP server
claude mcp add jetson-xavier --transport streamable-http http://<jetson-ip>:8765/mcp
```

Or add directly to your project's `.mcp.json`:

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

## Available Tools (17)

### System & Connectivity
| Tool | Description |
|------|-------------|
| `ping` | Server connection health check |
| `system_info` | OS, CPU, memory, disk info |
| `gpu_status` | CUDA/GPU status and utilization |
| `python_env` | Python versions and ML package list |
| `list_processes` | Process listing |

### Execution
| Tool | Description |
|------|-------------|
| `execute_command` | Shell command execution (dangerous commands blocked) |
| `run_python` | Python code execution (CUDA accelerated) |
| `read_file` | Read file |
| `write_file` | Write file |
| `cuda_benchmark` | CUDA matrix multiplication benchmark |

### Package Management
| Tool | Description |
|------|-------------|
| `install_package` | Safe install with JetPack compatibility check |
| `list_compatible_packages` | Compatible package version list |

### Task Queue (Async Jobs)
| Tool | Description |
|------|-------------|
| `submit_job` | Submit long-running job (background execution, automatic fan control) |
| `check_job` | Check job status / list all jobs |
| `get_result` | Retrieve completed job results |
| `get_log` | View real-time execution logs |

### Fan Cooling Control
| Tool | Description |
|------|-------------|
| `set_fan_profile` | Get/set fan profile (quiet/cool/aggressive) |

---

## Fan Cooling Control

### 3-Level Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `quiet` | Minimal noise. Fan starts at 50°C, stops when idle | Idle / low load |
| `cool` | Balanced mode. Fan starts at 35°C | Normal operation |
| `aggressive` | Cooling priority. Fan always on, max speed at 50°C | AI training / inference |

### Manual Control
Use the `set_fan_profile` tool:
- Call without arguments → view current profile
- `profile="aggressive"` → switch to that profile

### Automatic Control (Task Queue Integration)
When submitting a job via `submit_job`:
1. **Job start**: Fan automatically switches to `aggressive` mode (default)
2. **Job complete**: Fan reverts to previous profile
3. Use `fan_profile` parameter to specify per-job profile
4. Set `fan_profile=""` to run job without changing fan settings

> ⚠️ **Noise warning**: `aggressive` mode significantly increases fan noise.

---

## Usage Examples (in Claude Code)

Just ask naturally in the Claude Code terminal:

- "Check Jetson GPU status"
- "Run image inference with PyTorch on Jetson"
- "Run Jetson CUDA benchmark"
- "Read a file on Jetson"
- "Check Jetson fan status"
- "Switch Jetson fan to aggressive"
- "Submit a ResNet training job" (automatically switches fan to aggressive)
