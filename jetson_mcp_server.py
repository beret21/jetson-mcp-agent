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
FAN_PROFILE_PATH = "/sys/devices/platform/thermal_fan_est/fan_profile"
FAN_PROFILES_INFO = {
    "quiet": "소음 최소. 50°C부터 팬 시작, 유휴 시 팬 정지.",
    "cool": "균형 모드. 35°C부터 팬 시작, 일반 운영에 적합.",
    "aggressive": "냉각 우선. 팬 항상 동작, 50°C에서 최대 속도. AI 학습/추론용.",
}


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


# ── 팬 제어 헬퍼 ─────────────────────────────────────────────
async def _get_fan_profile() -> str:
    """현재 팬 프로파일을 반환합니다."""
    result = await run_cmd(f"cat {FAN_PROFILE_PATH}")
    return result["stdout"].strip()


async def _set_fan_profile(profile: str) -> dict:
    """팬 프로파일을 변경합니다."""
    if profile not in FAN_PROFILES_INFO:
        return {"error": f"Invalid profile: {profile}. Use: {', '.join(FAN_PROFILES_INFO)}"}
    result = await run_cmd(f"sudo sh -c 'echo {profile} > {FAN_PROFILE_PATH}'")
    if result["exit_code"] != 0:
        return {"error": f"Failed to set fan profile: {result['stderr']}"}
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
    """작업 상태를 JSON 파일로 저장합니다."""
    with open(_job_path(job["id"]), "w") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


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
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # stderr를 stdout에 합침
        )

        output_lines = []
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
        proc.kill()
        return {"exit_code": -1, "stdout": "".join(output_lines).strip(), "stderr": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


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
    asyncio.get_event_loop().create_task(_run_job(job_id))

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
