#!/usr/bin/env python3
"""
Jetson Xavier MCP Server
========================
Mac의 Claude Code에서 Jetson Xavier의 CUDA 자원을 활용할 수 있도록
MCP(Model Context Protocol) 서버를 제공합니다.

Transport: Streamable HTTP (원격 접속 지원)
"""

import json
import subprocess
import shlex
import os
import sys
import asyncio
import platform
from datetime import datetime
from typing import Any

# MCP SDK
from mcp.server.fastmcp import FastMCP

# ── 서버 초기화 ──────────────────────────────────────────────
mcp = FastMCP(
    "Jetson Xavier",
    instructions="Jetson Xavier CUDA 가속 및 시스템 관리 MCP 서버",
    host="0.0.0.0",
    port=8765,
)


# ── 유틸리티 ─────────────────────────────────────────────────

# ── 환경 상수 ────────────────────────────────────────────────
# Jetson Xavier 이원화 구조:
#   MCP 서버: Python 3.10 (venv) — MCP SDK requires-python >=3.10
#   PyTorch/CUDA: Python 3.8 (시스템) — JetPack R35.6.1 전용 wheel
PYTHON38 = "/usr/bin/python3.8"
CUDA_ENV = "export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"


async def run_cmd(cmd: str, timeout: int = 60) -> dict:
    """쉘 커맨드를 비동기로 실행하고 결과를 반환합니다."""
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace").strip(),
            "stderr": stderr.decode("utf-8", errors="replace").strip(),
        }
    except asyncio.TimeoutError:
        proc.kill()
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


# ── Tool: 시스템 정보 ────────────────────────────────────────
@mcp.tool()
async def system_info() -> dict:
    """Jetson Xavier의 시스템 정보를 반환합니다 (OS, CPU, 메모리, 디스크)."""
    results = {}

    # 호스트명 & OS
    results["hostname"] = platform.node()
    results["os"] = (await run_cmd("cat /etc/os-release | head -5"))["stdout"]

    # CPU
    results["cpu"] = (await run_cmd("lscpu | grep -E 'Model name|CPU\\(s\\)|Architecture'"))["stdout"]

    # 메모리
    results["memory"] = (await run_cmd("free -h"))["stdout"]

    # 디스크
    results["disk"] = (await run_cmd("df -h / /home 2>/dev/null"))["stdout"]

    # 업타임
    results["uptime"] = (await run_cmd("uptime"))["stdout"]

    return results


# ── Tool: GPU / CUDA 상태 ────────────────────────────────────
@mcp.tool()
async def gpu_status() -> dict:
    """Jetson의 GPU/CUDA 상태를 확인합니다 (tegrastats, jetson_clocks 등)."""
    results = {}

    # CUDA 버전 (nvcc가 PATH에 없을 수 있으므로 절대경로 사용)
    cuda_ver = await run_cmd("/usr/local/cuda/bin/nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null")
    results["cuda_version"] = cuda_ver["stdout"] or "CUDA not found"

    # JetPack 버전
    jetpack = await run_cmd("cat /etc/nv_tegra_release 2>/dev/null || dpkg -l nvidia-jetpack 2>/dev/null | tail -1")
    results["jetpack"] = jetpack["stdout"] or "JetPack info not found"

    # GPU 사용률 (tegrastats 1회 스냅샷)
    tegra = await run_cmd("timeout 2 tegrastats --interval 1000 2>/dev/null | head -1")
    results["tegrastats"] = tegra["stdout"] or "tegrastats not available"

    # GPU 주파수
    gpu_freq = await run_cmd("cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>/dev/null || echo 'N/A'")
    results["gpu_freq"] = gpu_freq["stdout"]

    return results


# ── Tool: Python 환경 확인 ───────────────────────────────────
@mcp.tool()
async def python_env() -> dict:
    """설치된 Python 버전과 주요 ML 패키지 목록을 반환합니다."""
    results = {}

    results["python_version"] = (await run_cmd("python3 --version"))["stdout"]
    results["pip_version"] = (await run_cmd("python3 -m pip --version 2>/dev/null"))["stdout"]

    # 주요 ML 패키지 확인 (Python 3.8에 설치되어 있음)
    packages = ["torch", "torchvision", "tensorflow", "numpy", "opencv-python", "jetson-inference", "onnxruntime"]
    pkg_check = await run_cmd(
        f"{PYTHON38} -m pip list 2>/dev/null | grep -iE '({'|'.join(packages)})'"
    )
    results["ml_packages"] = pkg_check["stdout"] or "No ML packages found"

    # CUDA 사용 가능 여부 (PyTorch — Python 3.8)
    cuda_check = await run_cmd(
        f'{CUDA_ENV} && {PYTHON38} -c "import torch; print(f\'CUDA available: {{torch.cuda.is_available()}}, Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}\')" 2>/dev/null'
    )
    results["pytorch_cuda"] = cuda_check["stdout"] or cuda_check["stderr"]

    return results


# ── Tool: 셸 커맨드 실행 ────────────────────────────────────
@mcp.tool()
async def execute_command(command: str, timeout: int = 120) -> dict:
    """
    Jetson Xavier에서 셸 커맨드를 실행합니다.

    Args:
        command: 실행할 셸 커맨드
        timeout: 타임아웃(초), 기본 120초
    """
    # 보안: 위험한 명령어 차단
    dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/sd"]
    for d in dangerous:
        if d in command:
            return {"exit_code": -1, "stdout": "", "stderr": f"Blocked dangerous command: {d}"}

    return await run_cmd(command, timeout=min(timeout, 600))


# ── Tool: Python 스크립트 실행 ───────────────────────────────
@mcp.tool()
async def run_python(code: str, timeout: int = 300) -> dict:
    """
    Jetson Xavier에서 Python 코드를 실행합니다. CUDA 가속 작업에 사용하세요.

    Args:
        code: 실행할 Python 코드
        timeout: 타임아웃(초), 기본 300초
    """
    # 임시 파일에 코드 저장 후 Python 3.8(PyTorch/CUDA)으로 실행
    tmp_path = "/tmp/mcp_python_task.py"
    with open(tmp_path, "w") as f:
        f.write(code)

    result = await run_cmd(f"{CUDA_ENV} && {PYTHON38} {tmp_path}", timeout=min(timeout, 600))

    # 임시 파일 정리
    try:
        os.remove(tmp_path)
    except:
        pass

    return result


# ── Tool: 파일 읽기 ─────────────────────────────────────────
@mcp.tool()
async def read_file(path: str) -> dict:
    """
    Jetson Xavier에서 파일 내용을 읽어옵니다.

    Args:
        path: 읽을 파일의 절대 경로
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(1_000_000)  # 최대 1MB
        return {"success": True, "content": content, "size": os.path.getsize(path)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Tool: 파일 쓰기 ─────────────────────────────────────────
@mcp.tool()
async def write_file(path: str, content: str) -> dict:
    """
    Jetson Xavier에 파일을 작성합니다.

    Args:
        path: 작성할 파일의 절대 경로
        content: 파일 내용
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "path": path, "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Tool: CUDA 벤치마크 ─────────────────────────────────────
@mcp.tool()
async def cuda_benchmark(matrix_size: int = 1024) -> dict:
    """
    간단한 CUDA 행렬 연산 벤치마크를 실행합니다.

    Args:
        matrix_size: 행렬 크기 (기본 1024x1024)
    """
    code = f"""
import time
try:
    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available")
    else:
        device = torch.device('cuda')
        print(f"Device: {{torch.cuda.get_device_name(0)}}")
        print(f"Matrix size: {matrix_size}x{matrix_size}")

        # Warmup
        a = torch.randn({matrix_size}, {matrix_size}, device=device)
        b = torch.randn({matrix_size}, {matrix_size}, device=device)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"10x matmul time: {{elapsed:.4f}}s")
        print(f"Avg per matmul: {{elapsed/10*1000:.2f}}ms")
        print(f"GPU Memory: {{torch.cuda.memory_allocated()/1024**2:.1f}}MB / {{torch.cuda.get_device_properties(0).total_memory/1024**2:.0f}}MB")
except ImportError:
    print("PyTorch not installed")
"""
    return await run_python(code, timeout=60)


# ── Tool: 프로세스 목록 ─────────────────────────────────────
@mcp.tool()
async def list_processes(filter: str = "") -> dict:
    """
    실행 중인 프로세스 목록을 반환합니다.

    Args:
        filter: 필터링할 문자열 (예: 'python', 'cuda')
    """
    if filter:
        cmd = f"ps aux | head -1 && ps aux | grep -i {shlex.quote(filter)} | grep -v grep"
    else:
        cmd = "ps aux --sort=-%mem | head -20"
    return await run_cmd(cmd)


# ── Tool: 핑 테스트 ─────────────────────────────────────────
@mcp.tool()
async def ping() -> dict:
    """MCP 서버 연결 상태를 확인합니다 (health check)."""
    return {
        "status": "ok",
        "server": "Jetson Xavier MCP Server",
        "hostname": platform.node(),
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
    }


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jetson Xavier MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="포트 (기본: 8765)")
    args = parser.parse_args()

    # MCP SDK 1.26.0: host/port는 FastMCP 생성자에서 설정
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    print(f"🚀 Jetson Xavier MCP Server starting on {args.host}:{args.port}")
    mcp.run(transport="streamable-http")
