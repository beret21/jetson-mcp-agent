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
import uuid
import traceback
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
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
WORKSPACE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WORKSPACE_INDEX = os.path.join(WORKSPACE_ROOT, ".workspace.json")
DUCKDB_PATH = os.path.join(WORKSPACE_ROOT, "analytics.duckdb")
FAN_CONFIG_PATH = "/etc/nvfancontrol.conf"
FAN_PROFILES_INFO = {
    "quiet": "소음 최소. 50°C부터 팬 시작, 유휴 시 팬 정지.",
    "cool": "균형 모드. 35°C부터 팬 시작, 일반 운영에 적합.",
    "aggressive": "냉각 우선. 팬 항상 동작, 50°C에서 최대 속도. AI 학습/추론용.",
}


async def run_cmd(cmd: str, timeout: int = 60) -> dict:
    """쉘 커맨드를 비동기로 실행하고 결과를 반환합니다."""
    proc = None
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
        if proc:
            try:
                proc.kill()
                await proc.wait()  # 좀비 프로세스 방지
            except ProcessLookupError:
                pass
        return {"exit_code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


# ── 팬 제어 헬퍼 ─────────────────────────────────────────────
async def _get_fan_profile() -> str:
    """현재 팬 프로파일을 반환합니다 (nvfancontrol.conf에서 읽기)."""
    result = await run_cmd(f"grep 'FAN_DEFAULT_PROFILE' {FAN_CONFIG_PATH}")
    line = result["stdout"].strip()
    return line.split()[-1] if line else "unknown"


async def _set_fan_profile(profile: str) -> dict:
    """팬 프로파일을 변경합니다 (nvfancontrol.conf 수정 + 서비스 재시작)."""
    if profile not in FAN_PROFILES_INFO:
        return {"error": f"Invalid profile: {profile}. Use: {', '.join(FAN_PROFILES_INFO)}"}
    # 1. 설정 파일의 FAN_DEFAULT_PROFILE 변경
    sed_cmd = f"sudo sed -i 's/FAN_DEFAULT_PROFILE .*/FAN_DEFAULT_PROFILE {profile}/' {FAN_CONFIG_PATH}"
    result = await run_cmd(sed_cmd)
    if result["exit_code"] != 0:
        return {"error": f"Failed to update config: {result['stderr']}"}
    # 2. nvfancontrol 서비스 재시작하여 적용
    restart = await run_cmd("sudo systemctl restart nvfancontrol")
    if restart["exit_code"] != 0:
        return {"error": f"Failed to restart nvfancontrol: {restart['stderr']}"}
    return {"profile": profile, "description": FAN_PROFILES_INFO[profile]}


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
    dangerous = ["rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/sd", ":(){ :|:", "chmod -R 777 /", "chown -R", "> /dev/null", "shutdown", "reboot", "init 0", "halt"]
    cmd_lower = command.lower().strip()
    for d in dangerous:
        if d in cmd_lower:
            return {"exit_code": -1, "stdout": "", "stderr": f"Blocked dangerous command pattern: {d}"}

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
    # 고유 파일명으로 동시 호출 충돌 방지
    tmp_path = f"/tmp/mcp_python_{uuid.uuid4().hex[:8]}.py"
    with open(tmp_path, "w") as f:
        f.write(code)

    try:
        result = await run_cmd(f"{CUDA_ENV} && {PYTHON38} {tmp_path}", timeout=min(timeout, 600))
    finally:
        # 임시 파일 정리 (성공/실패 무관)
        try:
            os.remove(tmp_path)
        except OSError:
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
        file_size = os.path.getsize(path)
        # 바이너리 파일 감지 (이미지, 모델 등)
        binary_exts = {".bin", ".pt", ".pth", ".onnx", ".pb", ".h5", ".pkl",
                       ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
                       ".mp4", ".avi", ".wav", ".mp3", ".so", ".o", ".a"}
        ext = os.path.splitext(path)[1].lower()
        if ext in binary_exts:
            return {
                "success": True,
                "content": f"(바이너리 파일 — 내용 표시 불가)",
                "size": file_size,
                "type": "binary",
                "extension": ext,
            }
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(1_000_000)  # 최대 1MB
        truncated = file_size > 1_000_000
        result = {"success": True, "content": content, "size": file_size}
        if truncated:
            result["truncated"] = True
            result["message"] = f"파일이 1MB를 초과하여 잘렸습니다. 전체 크기: {file_size:,} bytes"
        return result
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


# ── Tool: 팬 냉각 제어 ──────────────────────────────────────
@mcp.tool()
async def set_fan_profile(profile: str = "") -> dict:
    """
    Jetson Xavier 팬 냉각 프로파일을 조회하거나 변경합니다.

    Args:
        profile: 변경할 프로파일 (quiet/cool/aggressive). 비우면 현재 상태 조회.
            - quiet: 소음 최소. 50°C부터 팬 시작, 유휴 시 팬 정지.
            - cool: 균형 모드. 35°C부터 팬 시작, 일반 운영에 적합.
            - aggressive: 냉각 우선. 팬 항상 동작, 50°C에서 최대 속도. AI 학습/추론용.
    """
    current = await _get_fan_profile()

    if not profile:
        return {
            "current_profile": current,
            "description": FAN_PROFILES_INFO.get(current, "unknown"),
            "available_profiles": FAN_PROFILES_INFO,
        }

    if profile == current:
        return {"message": f"이미 '{profile}' 모드입니다.", "profile": current}

    result = await _set_fan_profile(profile)
    if "error" in result:
        return result

    return {
        "previous_profile": current,
        "current_profile": profile,
        "description": FAN_PROFILES_INFO[profile],
        "message": f"팬 프로파일이 '{current}' → '{profile}'로 변경되었습니다.",
    }


# ── Tool: Jetson 안전 패키지 설치 ────────────────────────────
# NVIDIA JetPack 호환 버전을 확인하고 안전하게 설치
NVIDIA_WHEEL_URLS = {
    "jp/v512": "https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/",
}

# JetPack R35.6.1 호환 버전 매핑
JETSON_COMPATIBLE = {
    "torch": {"version": "2.1.0a0+41361538.nv23.06", "method": "nvidia_wheel", "url_key": "jp/v512"},
    "torchvision": {"version": "0.16.0", "method": "no_deps"},
    "torchaudio": {"version": "2.1.0", "method": "no_deps"},
    "numpy": {"version": "1.24.4", "method": "pip"},
    "onnxruntime-gpu": {"version": "1.17.0", "method": "pip"},
    "opencv-python": {"version": "4.8.1.78", "method": "pip"},
    "scipy": {"version": "1.10.1", "method": "pip"},
    "scikit-learn": {"version": "1.3.2", "method": "pip"},
    "pandas": {"version": "2.0.3", "method": "pip"},
    "matplotlib": {"version": "3.7.5", "method": "pip"},
}


@mcp.tool()
async def install_package(package: str, version: str = "", force: bool = False) -> dict:
    """
    Jetson Xavier에 Python 패키지를 안전하게 설치합니다.
    NVIDIA JetPack 호환 버전을 자동으로 확인하고 적절한 방법으로 설치합니다.

    Args:
        package: 설치할 패키지 이름 (예: 'torchvision', 'numpy')
        version: 특정 버전 지정 (비어있으면 호환 버전 자동 선택)
        force: True이면 호환성 경고를 무시하고 설치 (위험)
    """
    # 호환 버전 확인
    compat = JETSON_COMPATIBLE.get(package)

    if compat and not version:
        version = compat["version"]
        method = compat["method"]
    elif compat and version != compat["version"] and not force:
        return {
            "error": f"'{package}=={version}'은 JetPack R35.6.1과 호환되지 않을 수 있습니다.",
            "compatible_version": compat["version"],
            "message": f"호환 버전: {compat['version']}. force=True로 강제 설치 가능하지만 권장하지 않습니다.",
        }
    else:
        method = "pip"

    pkg_spec = f"{package}=={version}" if version else package

    # 설치 방법에 따라 실행
    if method == "nvidia_wheel":
        url = NVIDIA_WHEEL_URLS.get(compat["url_key"], "")
        cmd = f"{CUDA_ENV} && {PYTHON38} -m pip install --no-cache-dir -f {url} {pkg_spec}"
    elif method == "no_deps":
        cmd = f"{CUDA_ENV} && {PYTHON38} -m pip install --no-cache-dir --no-deps {pkg_spec}"
    else:
        cmd = f"{CUDA_ENV} && {PYTHON38} -m pip install --no-cache-dir {pkg_spec}"

    result = await run_cmd(cmd, timeout=300)

    # 설치 후 CUDA 상태 확인 (torch 관련 패키지일 때)
    if package in ("torch", "torchvision", "torchaudio"):
        cuda_check = await run_cmd(
            f'{CUDA_ENV} && {PYTHON38} -c "import torch; print(torch.cuda.is_available())"'
        )
        result["cuda_still_working"] = cuda_check["stdout"].strip() == "True"

    return result


@mcp.tool()
async def list_compatible_packages() -> dict:
    """Jetson Xavier JetPack R35.6.1과 호환되는 패키지 버전 목록을 반환합니다."""
    installed = {}
    for pkg in JETSON_COMPATIBLE:
        check = await run_cmd(
            f'{PYTHON38} -c "import importlib; m=importlib.import_module(\'{pkg.replace("-", "_")}\'); print(getattr(m, \'__version__\', \'unknown\'))" 2>/dev/null'
        )
        installed[pkg] = check["stdout"].strip() if check["exit_code"] == 0 else "not installed"

    return {
        "jetpack": "R35.6.1",
        "cuda": "11.4",
        "python": "3.8.10",
        "compatible_packages": {
            pkg: {
                "recommended": info["version"],
                "installed": installed.get(pkg, "unknown"),
                "install_method": info["method"],
            }
            for pkg, info in JETSON_COMPATIBLE.items()
        },
    }


# ── Task Queue ───────────────────────────────────────────────
# 비동기 작업 관리: 작업 제출 → 백그라운드 실행 → 나중에 결과 확인

os.makedirs(JOBS_DIR, exist_ok=True)


def _job_path(job_id: str) -> str:
    """작업 JSON 파일 경로를 반환합니다."""
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _save_job(job: dict) -> None:
    """작업 상태를 JSON 파일로 안전하게 저장합니다 (atomic write)."""
    path = _job_path(job["id"])
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # atomic — 동시 쓰기 시 파일 깨짐 방지


def _load_job(job_id: str) -> dict | None:
    """저장된 작업을 불러옵니다."""
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _log_path(job_id: str) -> str:
    """작업 로그 파일 경로를 반환합니다."""
    return os.path.join(JOBS_DIR, f"{job_id}.log")


def _read_log(job_id: str, tail: int = 20) -> str:
    """작업 로그의 마지막 N줄을 반환합니다."""
    path = _log_path(job_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:]).strip()


async def _run_job_with_log(job_id: str, cmd: str, timeout: int) -> dict:
    """커맨드를 실행하면서 stdout/stderr를 로그 파일에 실시간 기록합니다."""
    log_file = _log_path(job_id)
    output_lines = []  # try 블록 밖에서 선언하여 except에서도 접근 가능
    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # stderr를 stdout에 합침
        )

        with open(log_file, "w", encoding="utf-8") as lf:
            async def read_stream():
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    output_lines.append(decoded)
                    lf.write(decoded)
                    lf.flush()

            await asyncio.wait_for(read_stream(), timeout=timeout)

        await proc.wait()
        return {
            "exit_code": proc.returncode,
            "stdout": "".join(output_lines).strip(),
            "stderr": "",
        }
    except asyncio.TimeoutError:
        if proc:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        return {"exit_code": -1, "stdout": "".join(output_lines).strip(), "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
        return {"exit_code": -1, "stdout": "".join(output_lines).strip(), "stderr": str(e)}


async def _run_job(job_id: str) -> None:
    """백그라운드에서 작업을 실행합니다. 팬 프로파일 자동 전환 포함."""
    job = _load_job(job_id)
    if not job:
        return

    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    _save_job(job)

    # 팬 프로파일 자동 전환
    fan_profile = job.get("fan_profile", "aggressive")
    previous_fan = None
    if fan_profile:
        try:
            previous_fan = await _get_fan_profile()
            if previous_fan != fan_profile:
                await _set_fan_profile(fan_profile)
                job["fan_switched"] = {"from": previous_fan, "to": fan_profile}
                _save_job(job)
        except Exception:
            pass  # 팬 제어 실패해도 작업은 계속 진행

    try:
        job_type = job["type"]
        timeout = job.get("timeout", 3600)

        if job_type == "python":
            tmp_path = f"/tmp/mcp_job_{job_id}.py"
            with open(tmp_path, "w") as f:
                f.write(job["code"])
            result = await _run_job_with_log(
                job_id,
                f"{CUDA_ENV} && {PYTHON38} {tmp_path}",
                timeout=min(timeout, 7200),
            )
            try:
                os.remove(tmp_path)
            except:
                pass

        elif job_type == "shell":
            result = await _run_job_with_log(
                job_id, job["command"], timeout=min(timeout, 7200)
            )

        else:
            result = {"exit_code": -1, "stdout": "", "stderr": f"Unknown job type: {job_type}"}

        job["status"] = "completed" if result["exit_code"] == 0 else "failed"
        job["result"] = result

    except Exception as e:
        job["status"] = "failed"
        job["result"] = {"exit_code": -1, "stdout": "", "stderr": traceback.format_exc()}

    # 팬 프로파일 복귀
    if previous_fan and previous_fan != fan_profile:
        try:
            await _set_fan_profile(previous_fan)
            job["fan_restored"] = previous_fan
        except Exception:
            pass

    job["finished_at"] = datetime.now().isoformat()
    _save_job(job)


@mcp.tool()
async def submit_job(
    name: str,
    type: str = "python",
    code: str = "",
    command: str = "",
    timeout: int = 3600,
    fan_profile: str = "aggressive",
) -> dict:
    """
    장시간 작업을 제출하고 즉시 job_id를 반환합니다. 작업은 백그라운드에서 실행됩니다.

    Args:
        name: 작업 이름 (예: 'ResNet 학습', '데이터 전처리')
        type: 작업 유형 — 'python' 또는 'shell'
        code: Python 코드 (type='python'일 때)
        command: 셸 커맨드 (type='shell'일 때)
        timeout: 최대 실행 시간(초), 기본 3600초 (1시간), 최대 7200초 (2시간)
        fan_profile: 작업 중 팬 프로파일 (quiet/cool/aggressive). 기본 aggressive.
                     빈 문자열이면 팬 프로파일을 변경하지 않습니다.
    """
    if type == "python" and not code:
        return {"error": "type='python'이면 code가 필요합니다."}
    if type == "shell" and not command:
        return {"error": "type='shell'이면 command가 필요합니다."}
    if type not in ("python", "shell"):
        return {"error": "type은 'python' 또는 'shell'만 가능합니다."}
    if fan_profile and fan_profile not in FAN_PROFILES_INFO:
        return {"error": f"fan_profile은 {', '.join(FAN_PROFILES_INFO)} 또는 빈 문자열만 가능합니다."}

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "name": name,
        "type": type,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "timeout": min(timeout, 7200),
    }
    if type == "python":
        job["code"] = code
    else:
        job["command"] = command
    if fan_profile:
        job["fan_profile"] = fan_profile

    _save_job(job)

    # 백그라운드에서 실행 (현재 이벤트 루프에 태스크 추가)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    loop.create_task(_run_job(job_id))

    response = {
        "job_id": job_id,
        "name": name,
        "status": "queued",
        "message": f"작업이 제출되었습니다. check_job('{job_id}')로 상태를 확인하세요.",
    }
    if fan_profile:
        response["fan_notice"] = f"⚠️ 작업 중 팬이 '{fan_profile}' 모드로 전환됩니다. 소음이 증가할 수 있습니다."
    return response


@mcp.tool()
async def check_job(job_id: str = "", list_all: bool = False) -> dict:
    """
    작업 상태를 확인합니다.

    Args:
        job_id: 확인할 작업 ID (특정 작업 조회)
        list_all: True이면 모든 작업 목록 반환
    """
    if list_all:
        jobs = []
        for fname in sorted(os.listdir(JOBS_DIR), reverse=True):
            if fname.endswith(".json"):
                with open(os.path.join(JOBS_DIR, fname), "r") as f:
                    job = json.load(f)
                jobs.append({
                    "id": job["id"],
                    "name": job["name"],
                    "status": job["status"],
                    "submitted_at": job.get("submitted_at", ""),
                    "finished_at": job.get("finished_at", ""),
                })
        return {"total": len(jobs), "jobs": jobs}

    if not job_id:
        return {"error": "job_id를 입력하거나 list_all=True로 전체 목록을 확인하세요."}

    job = _load_job(job_id)
    if not job:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}

    info = {
        "id": job["id"],
        "name": job["name"],
        "status": job["status"],
        "submitted_at": job.get("submitted_at", ""),
        "started_at": job.get("started_at", ""),
        "finished_at": job.get("finished_at", ""),
    }

    # 실행 시간 계산
    if job.get("started_at") and job.get("finished_at"):
        start = datetime.fromisoformat(job["started_at"])
        end = datetime.fromisoformat(job["finished_at"])
        info["elapsed_seconds"] = (end - start).total_seconds()
    elif job.get("started_at") and job["status"] == "running":
        start = datetime.fromisoformat(job["started_at"])
        info["running_seconds"] = (datetime.now() - start).total_seconds()
        info["recent_log"] = _read_log(job["id"], tail=10)
        if job.get("fan_switched"):
            info["fan_profile"] = job["fan_switched"]["to"]

    if job.get("fan_restored"):
        info["fan_restored"] = job["fan_restored"]

    return info


@mcp.tool()
async def get_result(job_id: str) -> dict:
    """
    완료된 작업의 결과를 가져옵니다.

    Args:
        job_id: 결과를 가져올 작업 ID
    """
    job = _load_job(job_id)
    if not job:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}

    if job["status"] == "queued":
        return {"status": "queued", "message": "아직 실행 대기 중입니다."}
    if job["status"] == "running":
        return {"status": "running", "message": "아직 실행 중입니다."}

    return {
        "id": job["id"],
        "name": job["name"],
        "status": job["status"],
        "elapsed_seconds": (
            (datetime.fromisoformat(job["finished_at"]) - datetime.fromisoformat(job["started_at"])).total_seconds()
            if job.get("started_at") and job.get("finished_at") else None
        ),
        "result": job.get("result", {}),
    }


@mcp.tool()
async def get_log(job_id: str, tail: int = 50) -> dict:
    """
    작업의 실행 로그를 가져옵니다. 실행 중인 작업의 진행 상태를 확인할 때 유용합니다.

    Args:
        job_id: 로그를 가져올 작업 ID
        tail: 가져올 마지막 줄 수 (기본 50줄)
    """
    job = _load_job(job_id)
    if not job:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}

    log_content = _read_log(job_id, tail=min(tail, 500))

    info = {
        "id": job_id,
        "name": job["name"],
        "status": job["status"],
        "log": log_content or "(로그 없음)",
    }

    if job.get("started_at") and job["status"] == "running":
        start = datetime.fromisoformat(job["started_at"])
        info["running_seconds"] = (datetime.now() - start).total_seconds()

    return info


# ── 워크스페이스 헬퍼 ──────────────────────────────────────────

def _resolve_workspace_path(name: str) -> str:
    """워크스페이스 상대 경로를 절대 경로로 변환 (경로 탈출 방지)."""
    resolved = os.path.realpath(os.path.join(WORKSPACE_ROOT, name))
    if not resolved.startswith(os.path.realpath(WORKSPACE_ROOT)):
        raise ValueError(f"Path traversal detected: {name}")
    return resolved


def _load_workspace_index() -> dict:
    """워크스페이스 인덱스를 로드합니다."""
    if os.path.exists(WORKSPACE_INDEX):
        with open(WORKSPACE_INDEX, "r") as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(), "versions": {}}


def _save_workspace_index(index: dict) -> None:
    """워크스페이스 인덱스를 저장합니다."""
    with open(WORKSPACE_INDEX, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


async def _dir_size(path: str) -> int:
    """디렉토리 전체 크기(bytes)를 반환합니다."""
    result = await run_cmd(f"du -sb {shlex.quote(path)} 2>/dev/null")
    if result["exit_code"] == 0 and result["stdout"]:
        return int(result["stdout"].split()[0])
    return 0


def _human_size(size_bytes: int) -> str:
    """바이트를 사람이 읽기 쉬운 형태로 변환합니다."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


# ── Tool: 워크스페이스 관리 ───────────────────────────────────
@mcp.tool()
async def workspace(
    action: str,
    name: str = "",
    source: str = "",
    description: str = "",
    pattern: str = "",
) -> dict:
    """
    EDA 데이터 워크스페이스를 관리합니다. 데이터 버전 관리, 포크, 정리 등.

    Args:
        action: 수행할 작업
            - "init": 워크스페이스 초기화 (디렉토리 구조 생성)
            - "status": 전체 현황 (디스크 사용량, 버전 목록)
            - "list": 특정 버전 파일 목록 (name으로 지정, pattern으로 필터)
            - "fork": source에서 name으로 데이터 복사 (새 버전 생성)
            - "diff": source와 name 두 버전의 파일 차이 비교
            - "info": 특정 버전 상세 정보
            - "delete": 버전 삭제 (raw는 보호)
        name: 대상 버전/디렉토리 이름 (예: "v1", "raw", "results")
        source: fork/diff에서 원본 버전 이름 (예: "raw", "v1")
        description: fork 생성 시 버전 설명
        pattern: list에서 파일 필터 glob 패턴 (예: "*.csv")
    """
    try:
        if action == "init":
            return await _ws_init()
        elif action == "status":
            return await _ws_status()
        elif action == "list":
            return await _ws_list(name, pattern)
        elif action == "fork":
            return await _ws_fork(name, source, description)
        elif action == "diff":
            return await _ws_diff(name, source)
        elif action == "info":
            return await _ws_info(name)
        elif action == "delete":
            return await _ws_delete(name)
        else:
            return {"error": f"Unknown action: {action}. Use: init, status, list, fork, diff, info, delete"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Workspace error: {traceback.format_exc()}"}


async def _ws_init() -> dict:
    """워크스페이스 디렉토리 구조를 생성합니다."""
    dirs = ["raw", "results"]
    created = []
    for d in dirs:
        path = os.path.join(WORKSPACE_ROOT, d)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            created.append(d)

    index = _load_workspace_index()
    if "raw" not in index["versions"]:
        index["versions"]["raw"] = {
            "created": datetime.now().isoformat(),
            "source": None,
            "description": "원본 데이터 (보호됨)",
            "protected": True,
        }
    _save_workspace_index(index)

    return {
        "workspace_root": WORKSPACE_ROOT,
        "created_dirs": created or "(이미 존재)",
        "message": "워크스페이스가 초기화되었습니다.",
    }


async def _ws_status() -> dict:
    """워크스페이스 전체 현황을 반환합니다."""
    if not os.path.exists(WORKSPACE_ROOT):
        return {"error": "워크스페이스가 초기화되지 않았습니다. workspace(action='init')을 먼저 실행하세요."}

    # 디스크 사용량
    disk = await run_cmd(f"df -h {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    disk_info = {}
    if disk["exit_code"] == 0:
        parts = disk["stdout"].split()
        if len(parts) >= 5:
            disk_info = {"total": parts[1], "used": parts[2], "available": parts[3], "usage_percent": parts[4]}

    # 버전 목록
    index = _load_workspace_index()
    versions = []
    for vname, vmeta in index.get("versions", {}).items():
        vpath = os.path.join(WORKSPACE_ROOT, vname)
        if os.path.exists(vpath):
            size = await _dir_size(vpath)
            file_count = sum(1 for f in os.listdir(vpath) if not f.startswith("."))
            versions.append({
                "name": vname,
                "files": file_count,
                "size": _human_size(size),
                "created": vmeta.get("created", ""),
                "source": vmeta.get("source"),
                "description": vmeta.get("description", ""),
                "protected": vmeta.get("protected", False),
            })

    # 인덱스에 없지만 존재하는 디렉토리도 표시
    for d in sorted(os.listdir(WORKSPACE_ROOT)):
        dpath = os.path.join(WORKSPACE_ROOT, d)
        if os.path.isdir(dpath) and not d.startswith(".") and d not in index.get("versions", {}):
            size = await _dir_size(dpath)
            file_count = sum(1 for f in os.listdir(dpath) if not f.startswith("."))
            versions.append({
                "name": d,
                "files": file_count,
                "size": _human_size(size),
                "created": "",
                "source": None,
                "description": "(인덱스에 미등록)",
            })

    # DuckDB 상태
    db_info = None
    if os.path.exists(DUCKDB_PATH):
        db_size = os.path.getsize(DUCKDB_PATH)
        db_info = {"path": DUCKDB_PATH, "size": _human_size(db_size)}

    ws_size = await _dir_size(WORKSPACE_ROOT)

    return {
        "workspace_root": WORKSPACE_ROOT,
        "total_size": _human_size(ws_size),
        "disk": disk_info,
        "versions": versions,
        "duckdb": db_info,
    }


async def _ws_list(name: str, pattern: str) -> dict:
    """특정 버전의 파일 목록을 반환합니다."""
    if not name:
        return {"error": "name을 지정하세요 (예: 'raw', 'v1')"}

    vpath = _resolve_workspace_path(name)
    if not os.path.exists(vpath):
        return {"error": f"버전 '{name}'이 존재하지 않습니다."}

    if pattern:
        import glob as globmod
        files_full = globmod.glob(os.path.join(vpath, pattern))
        files = [os.path.basename(f) for f in files_full]
    else:
        files = [f for f in os.listdir(vpath) if not f.startswith(".")]

    file_details = []
    for fname in sorted(files):
        fpath = os.path.join(vpath, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            detail = {
                "name": fname,
                "size": _human_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
            # CSV/Parquet 행 수 빠르게 확인
            if fname.endswith(".csv"):
                wc = await run_cmd(f"wc -l < {shlex.quote(fpath)}")
                if wc["exit_code"] == 0:
                    detail["rows"] = int(wc["stdout"].strip()) - 1  # 헤더 제외
            file_details.append(detail)
        elif os.path.isdir(fpath):
            file_details.append({"name": fname + "/", "type": "directory"})

    return {"version": name, "path": vpath, "files": file_details, "total": len(file_details)}


async def _ws_fork(name: str, source: str, description: str) -> dict:
    """source에서 name으로 데이터를 복사하여 새 버전을 생성합니다."""
    if not name:
        return {"error": "name을 지정하세요 (예: 'v1', 'v2')"}
    if not source:
        return {"error": "source를 지정하세요 (예: 'raw', 'v1')"}

    src_path = _resolve_workspace_path(source)
    dst_path = _resolve_workspace_path(name)

    if not os.path.exists(src_path):
        return {"error": f"원본 '{source}'가 존재하지 않습니다."}
    if os.path.exists(dst_path):
        return {"error": f"대상 '{name}'이 이미 존재합니다. 다른 이름을 사용하세요."}

    # 디스크 잔여 공간 확인
    src_size = await _dir_size(src_path)
    disk = await run_cmd(f"df --output=avail -B1 {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    if disk["exit_code"] == 0:
        avail = int(disk["stdout"].strip())
        if src_size > avail * 0.9:  # 90% 이상 사용 시 거부
            return {"error": f"디스크 공간 부족. 필요: {_human_size(src_size)}, 가용: {_human_size(avail)}"}

    # 복사
    result = await run_cmd(f"cp -r {shlex.quote(src_path)} {shlex.quote(dst_path)}", timeout=300)
    if result["exit_code"] != 0:
        return {"error": f"복사 실패: {result['stderr']}"}

    # 메타데이터 저장
    meta = {
        "name": name,
        "created": datetime.now().isoformat(),
        "source": source,
        "description": description or f"Forked from {source}",
    }
    with open(os.path.join(dst_path, "metadata.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 인덱스 업데이트
    index = _load_workspace_index()
    index["versions"][name] = meta
    _save_workspace_index(index)

    new_size = await _dir_size(dst_path)
    disk2 = await run_cmd(f"df --output=avail -B1 {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    remaining = int(disk2["stdout"].strip()) if disk2["exit_code"] == 0 else 0

    return {
        "message": f"'{source}' → '{name}' 포크 완료",
        "version": name,
        "size": _human_size(new_size),
        "disk_remaining": _human_size(remaining),
        "description": meta["description"],
    }


async def _ws_diff(name: str, source: str) -> dict:
    """두 버전의 파일 차이를 비교합니다."""
    if not name or not source:
        return {"error": "name과 source를 모두 지정하세요."}

    src_path = _resolve_workspace_path(source)
    dst_path = _resolve_workspace_path(name)

    if not os.path.exists(src_path):
        return {"error": f"'{source}'가 존재하지 않습니다."}
    if not os.path.exists(dst_path):
        return {"error": f"'{name}'이 존재하지 않습니다."}

    src_files = {f for f in os.listdir(src_path) if not f.startswith(".")}
    dst_files = {f for f in os.listdir(dst_path) if not f.startswith(".")}

    only_in_source = sorted(src_files - dst_files)
    only_in_target = sorted(dst_files - src_files)
    common = sorted(src_files & dst_files)

    changed = []
    unchanged = []
    for fname in common:
        sf = os.path.join(src_path, fname)
        df = os.path.join(dst_path, fname)
        if os.path.isfile(sf) and os.path.isfile(df):
            s_size = os.path.getsize(sf)
            d_size = os.path.getsize(df)
            if s_size != d_size:
                changed.append({"file": fname, f"{source}_size": _human_size(s_size), f"{name}_size": _human_size(d_size)})
            else:
                unchanged.append(fname)
        else:
            unchanged.append(fname)

    return {
        "source": source,
        "target": name,
        f"only_in_{source}": only_in_source,
        f"only_in_{name}": only_in_target,
        "changed": changed,
        "unchanged_count": len(unchanged),
    }


async def _ws_info(name: str) -> dict:
    """특정 버전의 상세 정보를 반환합니다."""
    if not name:
        return {"error": "name을 지정하세요."}

    vpath = _resolve_workspace_path(name)
    if not os.path.exists(vpath):
        return {"error": f"'{name}'이 존재하지 않습니다."}

    index = _load_workspace_index()
    meta = index.get("versions", {}).get(name, {})

    # lineage 추적
    lineage = [name]
    current = name
    while True:
        src = index.get("versions", {}).get(current, {}).get("source")
        if src and src in index.get("versions", {}):
            lineage.insert(0, src)
            current = src
        else:
            break

    size = await _dir_size(vpath)
    file_count = sum(1 for f in os.listdir(vpath) if not f.startswith("."))

    return {
        "name": name,
        "path": vpath,
        "size": _human_size(size),
        "files": file_count,
        "created": meta.get("created", ""),
        "source": meta.get("source"),
        "description": meta.get("description", ""),
        "protected": meta.get("protected", False),
        "lineage": " → ".join(lineage),
    }


async def _ws_delete(name: str) -> dict:
    """버전을 삭제합니다."""
    if not name:
        return {"error": "name을 지정하세요."}

    index = _load_workspace_index()
    meta = index.get("versions", {}).get(name, {})

    if meta.get("protected"):
        return {"error": f"'{name}'은 보호된 디렉토리입니다. 삭제할 수 없습니다."}

    vpath = _resolve_workspace_path(name)
    if not os.path.exists(vpath):
        return {"error": f"'{name}'이 존재하지 않습니다."}

    size = await _dir_size(vpath)

    result = await run_cmd(f"rm -rf {shlex.quote(vpath)}")
    if result["exit_code"] != 0:
        return {"error": f"삭제 실패: {result['stderr']}"}

    # 인덱스에서 제거
    if name in index.get("versions", {}):
        del index["versions"][name]
        _save_workspace_index(index)

    return {
        "message": f"'{name}' 삭제 완료",
        "freed": _human_size(size),
    }


# ── Tool: 데이터 업로드 (Mac → Jetson) ────────────────────────
@mcp.tool()
async def upload_data(
    filename: str,
    content_base64: str = "",
    content_text: str = "",
    dest: str = "raw",
    overwrite: bool = False,
) -> dict:
    """
    Mac에서 Jetson으로 데이터를 업로드합니다.
    텍스트 파일은 content_text, 바이너리 파일은 content_base64를 사용합니다.

    Args:
        filename: 저장할 파일명 (예: 'data.csv', 'model.pt')
        content_base64: base64 인코딩된 파일 내용 (바이너리 파일용)
        content_text: 텍스트 파일 내용 (CSV, JSON, Python 등)
        dest: 저장 디렉토리 (워크스페이스 상대 경로, 기본: 'raw')
        overwrite: True이면 기존 파일 덮어쓰기
    """
    import base64

    if not filename:
        return {"error": "filename을 지정하세요."}
    if not content_base64 and not content_text:
        return {"error": "content_base64 또는 content_text 중 하나를 제공하세요."}

    # 경로 해석
    try:
        dest_path = _resolve_workspace_path(dest)
    except ValueError as e:
        return {"error": str(e)}

    os.makedirs(dest_path, exist_ok=True)
    file_path = os.path.join(dest_path, filename)

    # 덮어쓰기 방지
    if os.path.exists(file_path) and not overwrite:
        existing_size = os.path.getsize(file_path)
        return {
            "error": f"'{filename}'이 이미 존재합니다 ({_human_size(existing_size)}). overwrite=True로 덮어쓰기 가능.",
            "existing_path": file_path,
        }

    # 디스크 공간 확인
    disk = await run_cmd(f"df --output=avail -B1 {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    if disk["exit_code"] == 0:
        avail = int(disk["stdout"].strip())
        if avail < 100 * 1024 * 1024:  # 100MB 미만이면 경고
            return {"error": f"디스크 공간 부족. 가용: {_human_size(avail)}"}

    try:
        if content_base64:
            # 바이너리 파일: base64 디코딩
            data = base64.b64decode(content_base64)
            with open(file_path, "wb") as f:
                f.write(data)
            file_size = len(data)
        else:
            # 텍스트 파일: 직접 쓰기
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_text)
            file_size = len(content_text.encode("utf-8"))

        return {
            "success": True,
            "path": file_path,
            "filename": filename,
            "size": _human_size(file_size),
            "size_bytes": file_size,
            "dest": dest,
            "message": f"'{filename}'이 {dest}/에 저장되었습니다.",
        }
    except Exception as e:
        return {"error": f"파일 저장 실패: {str(e)}"}


# ── Tool: URL 다운로드 ───────────────────────────────────────
@mcp.tool()
async def fetch_url(
    url: str,
    dest: str = "",
    filename: str = "",
    extract: bool = False,
    timeout: int = 600,
) -> dict:
    """
    Jetson Xavier에서 URL로부터 파일을 다운로드합니다.
    Kaggle, HuggingFace, 공개 데이터셋 등에 활용합니다.

    Args:
        url: 다운로드할 URL
        dest: 저장 디렉토리 (기본: data/raw/). 워크스페이스 상대 경로 가능.
        filename: 저장 파일명 (기본: URL에서 추출)
        extract: True이면 tar.gz/zip 자동 해제
        timeout: 다운로드 타임아웃(초), 기본 600초
    """
    # URL 검증
    if not url.startswith(("http://", "https://")):
        return {"error": "http:// 또는 https:// URL만 지원합니다."}

    # 대상 경로 결정
    if dest:
        try:
            dest_path = _resolve_workspace_path(dest)
        except ValueError as e:
            return {"error": str(e)}
    else:
        dest_path = os.path.join(WORKSPACE_ROOT, "raw")

    os.makedirs(dest_path, exist_ok=True)

    # 파일명 결정
    if not filename:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        filename = unquote(os.path.basename(parsed.path)) or "download"

    file_path = os.path.join(dest_path, filename)

    # wget으로 다운로드 (대용량 파일 메모리 효율)
    import time
    start_time = time.time()
    cmd = f"wget -q --show-progress -O {shlex.quote(file_path)} {shlex.quote(url)}"
    result = await run_cmd(cmd, timeout=min(timeout, 3600))

    if result["exit_code"] != 0:
        # wget 실패 시 curl 시도
        cmd = f"curl -fSL -o {shlex.quote(file_path)} {shlex.quote(url)}"
        result = await run_cmd(cmd, timeout=min(timeout, 3600))

    if result["exit_code"] != 0:
        return {"error": f"다운로드 실패: {result['stderr']}"}

    elapsed = time.time() - start_time
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    response = {
        "success": True,
        "path": file_path,
        "size": _human_size(file_size),
        "elapsed_seconds": round(elapsed, 1),
    }

    # 압축 해제
    if extract and os.path.exists(file_path):
        extract_dir = os.path.join(dest_path, os.path.splitext(filename)[0])
        if filename.endswith((".tar.gz", ".tgz")):
            ext_cmd = f"tar -xzf {shlex.quote(file_path)} -C {shlex.quote(dest_path)}"
        elif filename.endswith(".tar.bz2"):
            ext_cmd = f"tar -xjf {shlex.quote(file_path)} -C {shlex.quote(dest_path)}"
        elif filename.endswith(".zip"):
            ext_cmd = f"unzip -o {shlex.quote(file_path)} -d {shlex.quote(dest_path)}"
        elif filename.endswith(".gz") and not filename.endswith(".tar.gz"):
            ext_cmd = f"gunzip -f {shlex.quote(file_path)}"
        else:
            ext_cmd = None

        if ext_cmd:
            ext_result = await run_cmd(ext_cmd, timeout=300)
            response["extracted"] = ext_result["exit_code"] == 0
            if ext_result["exit_code"] != 0:
                response["extract_error"] = ext_result["stderr"]

    return response


# ── Tool: 데이터 통계 ────────────────────────────────────────
@mcp.tool()
async def data_stats(path: str, sample_rows: int = 5) -> dict:
    """
    데이터 파일의 기본 통계를 반환합니다. CSV, Parquet, JSON 지원.

    Args:
        path: 데이터 파일 경로. 워크스페이스 상대 경로 가능 (예: "v1/sales.csv")
        sample_rows: 미리보기 행 수 (기본 5행)
    """
    # 경로 해석: 절대 경로가 아니면 워크스페이스 상대 경로
    if not os.path.isabs(path):
        try:
            path = _resolve_workspace_path(path)
        except ValueError as e:
            return {"error": str(e)}

    if not os.path.exists(path):
        return {"error": f"파일이 존재하지 않습니다: {path}"}

    ext = os.path.splitext(path)[1].lower()
    sample_rows = min(sample_rows, 20)

    code = f"""
import json, sys
try:
    import pandas as pd
    path = {repr(path)}
    ext = {repr(ext)}
    sample = {sample_rows}

    if ext == '.csv':
        df = pd.read_csv(path, nrows=10000)
    elif ext == '.tsv':
        df = pd.read_csv(path, sep='\\t', nrows=10000)
    elif ext in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    elif ext == '.json':
        df = pd.read_json(path)
    else:
        print(json.dumps({{"error": "지원하지 않는 형식: " + ext}}))
        sys.exit(0)

    # 전체 행 수 (CSV는 파일 전체 카운트)
    total_rows = len(df)
    if ext == '.csv' and total_rows >= 10000:
        import subprocess
        wc = subprocess.run(['wc', '-l', path], capture_output=True, text=True)
        if wc.returncode == 0:
            total_rows = int(wc.stdout.split()[0]) - 1

    result = {{
        "shape": list(df.shape),
        "total_rows": total_rows,
        "columns": list(df.columns),
        "dtypes": {{col: str(dtype) for col, dtype in df.dtypes.items()}},
        "null_counts": df.isnull().sum().to_dict(),
        "head": df.head(sample).to_dict(orient='records'),
    }}

    # 수치 컬럼 기본 통계
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().to_dict()
        result["describe"] = desc

    print(json.dumps(result, default=str, ensure_ascii=False))

except ImportError:
    print(json.dumps({{"error": "pandas가 설치되어 있지 않습니다."}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result = await run_python(code, timeout=60)
    if result["exit_code"] != 0:
        return {"error": f"분석 실패: {result['stderr'] or result['stdout']}"}

    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


# ── Tool: DuckDB 쿼리 ────────────────────────────────────────
@mcp.tool()
async def db_query(sql: str, limit: int = 100) -> dict:
    """
    DuckDB에서 SQL 쿼리를 실행합니다. 시계열 분석, 집계, Parquet/CSV 직접 쿼리 지원.

    Args:
        sql: 실행할 SQL 쿼리. Parquet/CSV 직접 쿼리 가능:
             SELECT * FROM read_csv_auto('data/raw/sales.csv')
             SELECT * FROM 'data/v1/output.parquet'
        limit: 최대 반환 행 수 (기본 100)
    """
    limit = min(limit, 10000)

    code = f"""
import json, sys, os
try:
    import duckdb
    db_path = {repr(DUCKDB_PATH)}
    ws_root = {repr(WORKSPACE_ROOT)}

    # 워크스페이스 디렉토리를 기준으로 상대 경로 사용 가능하도록
    os.chdir(ws_root)

    con = duckdb.connect(db_path)
    sql = {repr(sql)}
    limit = {limit}

    result = con.execute(sql).fetchdf()
    total_rows = len(result)

    if total_rows > limit:
        result = result.head(limit)

    output = {{
        "columns": list(result.columns),
        "total_rows": total_rows,
        "returned_rows": len(result),
        "data": result.to_dict(orient='records'),
    }}
    con.close()
    print(json.dumps(output, default=str, ensure_ascii=False))

except ImportError:
    print(json.dumps({{"error": "duckdb가 설치되어 있지 않습니다. install_package('duckdb')로 설치하세요."}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result = await run_python(code, timeout=120)
    if result["exit_code"] != 0:
        return {"error": f"쿼리 실패: {result['stderr'] or result['stdout']}"}

    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


# ── Tool: DuckDB 데이터 적재 ─────────────────────────────────
@mcp.tool()
async def db_ingest(
    source: str,
    table: str,
    mode: str = "create",
) -> dict:
    """
    CSV/Parquet 파일을 DuckDB 테이블로 적재합니다.

    Args:
        source: 데이터 파일 경로. 워크스페이스 상대 경로 가능 (예: "raw/sales.csv")
        table: 생성할 테이블 이름
        mode: 적재 모드
            - "create": 새 테이블 생성 (이미 있으면 에러)
            - "replace": 기존 테이블 교체
            - "append": 기존 테이블에 추가
    """
    if not source or not table:
        return {"error": "source와 table을 모두 지정하세요."}
    if mode not in ("create", "replace", "append"):
        return {"error": "mode는 create, replace, append 중 하나입니다."}

    # 경로 해석
    if not os.path.isabs(source):
        try:
            source = _resolve_workspace_path(source)
        except ValueError as e:
            return {"error": str(e)}

    if not os.path.exists(source):
        return {"error": f"파일이 존재하지 않습니다: {source}"}

    ext = os.path.splitext(source)[1].lower()

    code = f"""
import json, sys, os
try:
    import duckdb
    db_path = {repr(DUCKDB_PATH)}
    source = {repr(source)}
    table = {repr(table)}
    mode = {repr(mode)}
    ext = {repr(ext)}

    con = duckdb.connect(db_path)

    if ext == '.csv':
        read_fn = f"read_csv_auto('{{source}}')"
    elif ext in ('.parquet', '.pq'):
        read_fn = f"read_parquet('{{source}}')"
    elif ext == '.json':
        read_fn = f"read_json_auto('{{source}}')"
    else:
        print(json.dumps({{"error": "지원하지 않는 형식: " + ext}}))
        sys.exit(0)

    if mode == "create":
        con.execute(f"CREATE TABLE {{table}} AS SELECT * FROM {{read_fn}}")
    elif mode == "replace":
        con.execute(f"CREATE OR REPLACE TABLE {{table}} AS SELECT * FROM {{read_fn}}")
    elif mode == "append":
        con.execute(f"INSERT INTO {{table}} SELECT * FROM {{read_fn}}")

    # 결과 확인
    count = con.execute(f"SELECT COUNT(*) FROM {{table}}").fetchone()[0]
    cols = [desc[0] for desc in con.execute(f"SELECT * FROM {{table}} LIMIT 0").description]
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]

    con.close()
    print(json.dumps({{
        "success": True,
        "table": table,
        "rows": count,
        "columns": cols,
        "mode": mode,
        "all_tables": tables,
    }}))

except ImportError:
    print(json.dumps({{"error": "duckdb가 설치되어 있지 않습니다. install_package('duckdb')로 설치하세요."}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result = await run_python(code, timeout=300)
    if result["exit_code"] != 0:
        return {"error": f"적재 실패: {result['stderr'] or result['stdout']}"}

    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


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
    print(f"   Python: {platform.python_version()} | PID: {os.getpid()}")
    print(f"   Jobs dir: {JOBS_DIR}")
    print(f"   Workspace: {WORKSPACE_ROOT}")
    mcp.run(transport="streamable-http")
