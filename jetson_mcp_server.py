#!/usr/bin/env python3
"""
Jetson Xavier MCP Server
========================
Mac의 Claude Code에서 Jetson Xavier의 CUDA 자원을 활용할 수 있도록
MCP(Model Context Protocol) 서버를 제공합니다.

8개 그룹 도구:
  system    — 시스템 상태 관찰
  execute   — 코드 실행 (shell, python, benchmark)
  file      — 파일 I/O
  device    — 디바이스 관리 (팬, 패키지)
  job       — 장시간 작업 관리
  workspace — 데이터 버전 관리
  data      — 데이터 I/O + 정제
  xai       — XAI 설명·분석 계층

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
import base64
from datetime import datetime
from typing import Any

# MCP SDK
from mcp.server.fastmcp import FastMCP

# ── 서버 초기화 ──────────────────────────────────────────────
mcp = FastMCP(
    "Jetson Xavier",
    instructions="Jetson Xavier CUDA 가속 및 시스템 관리 MCP 서버. 8개 그룹 도구 제공.",
    host="0.0.0.0",
    port=8765,
)


# ══════════════════════════════════════════════════════════════
#  환경 상수
# ══════════════════════════════════════════════════════════════

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

# JetPack R35.6.1 호환 버전 매핑
NVIDIA_WHEEL_URLS = {
    "jp/v512": "https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/",
}
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


# ══════════════════════════════════════════════════════════════
#  공통 유틸리티
# ══════════════════════════════════════════════════════════════

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
                await proc.wait()
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


def _human_size(size_bytes: int) -> str:
    """바이트를 사람이 읽기 쉬운 형태로 변환합니다."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


async def _dir_size(path: str) -> int:
    """디렉토리 전체 크기(bytes)를 반환합니다."""
    result = await run_cmd(f"du -sb {shlex.quote(path)} 2>/dev/null")
    if result["exit_code"] == 0 and result["stdout"]:
        return int(result["stdout"].split()[0])
    return 0


# ══════════════════════════════════════════════════════════════
#  1. system — 시스템 상태 관찰
# ══════════════════════════════════════════════════════════════

async def _system_info(compact: bool) -> dict:
    results = {}
    results["hostname"] = platform.node()
    results["os"] = (await run_cmd("cat /etc/os-release | head -5"))["stdout"]
    results["cpu"] = (await run_cmd("lscpu | grep -E 'Model name|CPU\\(s\\)|Architecture'"))["stdout"]
    results["memory"] = (await run_cmd("free -h"))["stdout"]
    results["disk"] = (await run_cmd("df -h / /home 2>/dev/null"))["stdout"]
    results["uptime"] = (await run_cmd("uptime"))["stdout"]
    if compact:
        mem = await run_cmd("free -h | awk '/Mem:/{print $3\"/\"$2}'")
        disk_use = await run_cmd("df -h / | awk 'NR==2{print $5}'")
        return {
            "hostname": results["hostname"],
            "cpu": results["cpu"].split("\n")[0].split(":")[-1].strip() if results["cpu"] else "",
            "ram": mem["stdout"],
            "disk": disk_use["stdout"],
        }
    return results


async def _gpu_status(compact: bool) -> dict:
    results = {}
    cuda_ver = await run_cmd("/usr/local/cuda/bin/nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null")
    results["cuda_version"] = cuda_ver["stdout"] or "CUDA not found"
    jetpack = await run_cmd("cat /etc/nv_tegra_release 2>/dev/null || dpkg -l nvidia-jetpack 2>/dev/null | tail -1")
    results["jetpack"] = jetpack["stdout"] or "JetPack info not found"
    tegra = await run_cmd("timeout 2 tegrastats --interval 1000 2>/dev/null | head -1")
    results["tegrastats"] = tegra["stdout"] or "tegrastats not available"
    gpu_freq = await run_cmd("cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>/dev/null || echo 'N/A'")
    results["gpu_freq"] = gpu_freq["stdout"]
    if compact:
        return {"cuda": "11.4", "gpu_freq": results["gpu_freq"], "tegra": results["tegrastats"][:80]}
    return results


async def _python_env(compact: bool) -> dict:
    results = {}
    results["python_version"] = (await run_cmd("python3 --version"))["stdout"]
    results["pip_version"] = (await run_cmd("python3 -m pip --version 2>/dev/null"))["stdout"]
    packages = ["torch", "torchvision", "tensorflow", "numpy", "opencv-python", "jetson-inference", "onnxruntime"]
    pkg_check = await run_cmd(
        f"{PYTHON38} -m pip list 2>/dev/null | grep -iE '({'|'.join(packages)})'"
    )
    results["ml_packages"] = pkg_check["stdout"] or "No ML packages found"
    cuda_check = await run_cmd(
        f'{CUDA_ENV} && {PYTHON38} -c "import torch; print(f\'CUDA: {{torch.cuda.is_available()}}, Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}\')" 2>/dev/null'
    )
    results["pytorch_cuda"] = cuda_check["stdout"] or cuda_check["stderr"]
    if compact:
        return {"python": results["python_version"], "cuda_ok": "True" in results["pytorch_cuda"]}
    return results


async def _list_processes(filter: str, compact: bool) -> dict:
    if filter:
        cmd = f"ps aux | head -1 && ps aux | grep -i {shlex.quote(filter)} | grep -v grep"
    else:
        cmd = "ps aux --sort=-%mem | head -20"
    result = await run_cmd(cmd)
    if compact:
        lines = result["stdout"].split("\n")
        return {"count": max(0, len(lines) - 1), "preview": "\n".join(lines[:5])}
    return result


@mcp.tool()
async def system(action: str, filter: str = "", compact: bool = False) -> dict:
    """
    Jetson Xavier 시스템 정보를 조회합니다.

    Args:
        action: 수행할 작업
            - "info": OS, CPU, 메모리, 디스크, 업타임
            - "gpu": CUDA, JetPack, tegrastats, GPU 주파수
            - "python": Python 버전, ML 패키지, CUDA 상태
            - "ping": 서버 연결 상태 (health check)
            - "processes": 실행 중인 프로세스 목록
        filter: processes에서 필터링 문자열 (예: 'python')
        compact: True이면 핵심 수치만 반환
    """
    try:
        if action == "info":
            return await _system_info(compact)
        elif action == "gpu":
            return await _gpu_status(compact)
        elif action == "python":
            return await _python_env(compact)
        elif action == "ping":
            return {
                "status": "ok",
                "server": "Jetson Xavier MCP Server",
                "hostname": platform.node(),
                "timestamp": datetime.now().isoformat(),
                "python": platform.python_version(),
                "tools": 8,
            }
        elif action == "processes":
            return await _list_processes(filter, compact)
        else:
            return {"error": f"Unknown action: {action}. Use: info, gpu, python, ping, processes"}
    except Exception as e:
        return {"error": f"system error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  2. execute — 코드 실행
# ══════════════════════════════════════════════════════════════

async def _execute_shell(command: str, timeout: int, compact: bool) -> dict:
    dangerous = ["rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/sd", ":(){ :|:",
                  "chmod -R 777 /", "chown -R", "> /dev/null", "shutdown", "reboot", "init 0", "halt"]
    cmd_lower = command.lower().strip()
    for d in dangerous:
        if d in cmd_lower:
            return {"exit_code": -1, "stdout": "", "stderr": f"Blocked dangerous command pattern: {d}"}
    result = await run_cmd(command, timeout=min(timeout, 600))
    if compact and result["exit_code"] == 0:
        lines = result["stdout"].split("\n")
        return {"ok": True, "lines": len(lines), "preview": "\n".join(lines[:5])}
    return result


async def _execute_python(code: str, timeout: int, compact: bool) -> dict:
    tmp_path = f"/tmp/mcp_python_{uuid.uuid4().hex[:8]}.py"
    with open(tmp_path, "w") as f:
        f.write(code)
    try:
        result = await run_cmd(f"{CUDA_ENV} && {PYTHON38} {tmp_path}", timeout=min(timeout, 600))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if compact and result["exit_code"] == 0:
        lines = result["stdout"].split("\n")
        return {"ok": True, "lines": len(lines), "output": "\n".join(lines[-10:])}
    return result


async def _execute_benchmark(matrix_size: int, compact: bool) -> dict:
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
        a = torch.randn({matrix_size}, {matrix_size}, device=device)
        b = torch.randn({matrix_size}, {matrix_size}, device=device)
        torch.cuda.synchronize()
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
    return await _execute_python(code, timeout=60, compact=compact)


@mcp.tool()
async def execute(action: str, command: str = "", code: str = "",
                  timeout: int = 300, matrix_size: int = 1024,
                  compact: bool = False) -> dict:
    """
    Jetson Xavier에서 코드를 실행합니다.

    Args:
        action: 수행할 작업
            - "shell": 셸 커맨드 실행 (command 필요)
            - "python": Python 코드 실행 — CUDA 가속 (code 필요)
            - "benchmark": CUDA 행렬 연산 벤치마크 (matrix_size 선택)
        command: shell에서 실행할 커맨드
        code: python에서 실행할 코드
        timeout: 타임아웃(초), 기본 300초
        matrix_size: benchmark 행렬 크기 (기본 1024)
        compact: True이면 핵심 결과만 반환
    """
    try:
        if action == "shell":
            if not command:
                return {"error": "command를 지정하세요."}
            return await _execute_shell(command, timeout, compact)
        elif action == "python":
            if not code:
                return {"error": "code를 지정하세요."}
            return await _execute_python(code, timeout, compact)
        elif action == "benchmark":
            return await _execute_benchmark(matrix_size, compact)
        else:
            return {"error": f"Unknown action: {action}. Use: shell, python, benchmark"}
    except Exception as e:
        return {"error": f"execute error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  3. file — 파일 I/O
# ══════════════════════════════════════════════════════════════

async def _file_read(path: str, compact: bool) -> dict:
    try:
        file_size = os.path.getsize(path)
        binary_exts = {".bin", ".pt", ".pth", ".onnx", ".pb", ".h5", ".pkl",
                       ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
                       ".mp4", ".avi", ".wav", ".mp3", ".so", ".o", ".a"}
        ext = os.path.splitext(path)[1].lower()
        if ext in binary_exts:
            return {"success": True, "content": "(바이너리 파일)", "size": file_size,
                    "type": "binary", "extension": ext}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(1_000_000)
        truncated = file_size > 1_000_000
        if compact:
            lines = content.split("\n")
            return {"size": _human_size(file_size), "lines": len(lines),
                    "preview": "\n".join(lines[:10]), "truncated": truncated}
        result = {"success": True, "content": content, "size": file_size}
        if truncated:
            result["truncated"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _file_write(path: str, content: str, compact: bool) -> dict:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if compact:
            return {"ok": True, "size": _human_size(len(content))}
        return {"success": True, "path": path, "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def file(action: str, path: str = "", content: str = "",
               compact: bool = False) -> dict:
    """
    Jetson Xavier 파일을 읽거나 씁니다.

    Args:
        action: 수행할 작업
            - "read": 파일 읽기 (바이너리 자동 감지, 최대 1MB)
            - "write": 파일 쓰기 (디렉토리 자동 생성)
        path: 파일 절대 경로
        content: write 시 파일 내용
        compact: True이면 메타정보만 반환
    """
    try:
        if action == "read":
            if not path:
                return {"error": "path를 지정하세요."}
            return await _file_read(path, compact)
        elif action == "write":
            if not path or not content:
                return {"error": "path와 content를 지정하세요."}
            return await _file_write(path, content, compact)
        else:
            return {"error": f"Unknown action: {action}. Use: read, write"}
    except Exception as e:
        return {"error": f"file error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  4. device — 디바이스 관리 (팬, 패키지)
# ══════════════════════════════════════════════════════════════

async def _get_fan_profile() -> str:
    result = await run_cmd(f"grep 'FAN_DEFAULT_PROFILE' {FAN_CONFIG_PATH}")
    line = result["stdout"].strip()
    return line.split()[-1] if line else "unknown"


async def _set_fan_profile(profile: str) -> dict:
    if profile not in FAN_PROFILES_INFO:
        return {"error": f"Invalid profile: {profile}. Use: {', '.join(FAN_PROFILES_INFO)}"}
    sed_cmd = f"sudo sed -i 's/FAN_DEFAULT_PROFILE .*/FAN_DEFAULT_PROFILE {profile}/' {FAN_CONFIG_PATH}"
    result = await run_cmd(sed_cmd)
    if result["exit_code"] != 0:
        return {"error": f"Failed to update config: {result['stderr']}"}
    restart = await run_cmd("sudo systemctl restart nvfancontrol")
    if restart["exit_code"] != 0:
        return {"error": f"Failed to restart nvfancontrol: {restart['stderr']}"}
    return {"profile": profile, "description": FAN_PROFILES_INFO[profile]}


async def _device_fan(profile: str, compact: bool) -> dict:
    current = await _get_fan_profile()
    if not profile:
        if compact:
            return {"fan": current}
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
    if compact:
        return {"fan": f"{current}→{profile}"}
    return {
        "previous_profile": current,
        "current_profile": profile,
        "description": FAN_PROFILES_INFO[profile],
        "message": f"팬 프로파일이 '{current}' → '{profile}'로 변경되었습니다.",
    }


async def _device_install(package: str, version: str, force: bool, compact: bool) -> dict:
    if not package:
        return {"error": "package를 지정하세요."}
    compat = JETSON_COMPATIBLE.get(package)
    if compat and not version:
        version = compat["version"]
        method = compat["method"]
    elif compat and version != compat["version"] and not force:
        return {
            "error": f"'{package}=={version}'은 JetPack R35.6.1과 호환되지 않을 수 있습니다.",
            "compatible_version": compat["version"],
            "message": f"호환 버전: {compat['version']}. force=True로 강제 설치 가능.",
        }
    else:
        method = "pip"

    pkg_spec = f"{package}=={version}" if version else package
    if method == "nvidia_wheel":
        url = NVIDIA_WHEEL_URLS.get(compat["url_key"], "")
        cmd = f"{CUDA_ENV} && {PYTHON38} -m pip install --no-cache-dir -f {url} {pkg_spec}"
    elif method == "no_deps":
        cmd = f"{CUDA_ENV} && {PYTHON38} -m pip install --no-cache-dir --no-deps {pkg_spec}"
    else:
        cmd = f"{CUDA_ENV} && {PYTHON38} -m pip install --no-cache-dir {pkg_spec}"

    result = await run_cmd(cmd, timeout=300)
    if package in ("torch", "torchvision", "torchaudio"):
        cuda_check = await run_cmd(f'{CUDA_ENV} && {PYTHON38} -c "import torch; print(torch.cuda.is_available())"')
        result["cuda_still_working"] = cuda_check["stdout"].strip() == "True"
    if compact:
        return {"ok": result["exit_code"] == 0, "package": pkg_spec}
    return result


async def _device_packages(compact: bool) -> dict:
    installed = {}
    for pkg in JETSON_COMPATIBLE:
        check = await run_cmd(
            f'{PYTHON38} -c "import importlib; m=importlib.import_module(\'{pkg.replace("-", "_")}\'); print(getattr(m, \'__version__\', \'unknown\'))" 2>/dev/null'
        )
        installed[pkg] = check["stdout"].strip() if check["exit_code"] == 0 else "not installed"

    if compact:
        return {pkg: installed.get(pkg, "?") for pkg in JETSON_COMPATIBLE}
    return {
        "jetpack": "R35.6.1", "cuda": "11.4", "python": "3.8.10",
        "compatible_packages": {
            pkg: {"recommended": info["version"], "installed": installed.get(pkg, "unknown"), "install_method": info["method"]}
            for pkg, info in JETSON_COMPATIBLE.items()
        },
    }


@mcp.tool()
async def device(action: str, profile: str = "", package: str = "",
                 version: str = "", force: bool = False,
                 compact: bool = False) -> dict:
    """
    Jetson Xavier 디바이스를 관리합니다 (팬 냉각, 패키지 설치).

    Args:
        action: 수행할 작업
            - "fan": 팬 프로파일 조회/변경 (quiet/cool/aggressive)
            - "install": Python 패키지 안전 설치 (JetPack 호환 확인)
            - "packages": JetPack 호환 패키지 목록
        profile: fan에서 프로파일 지정 (비우면 현재 조회)
        package: install에서 패키지명
        version: install에서 버전 (비우면 호환 버전 자동)
        force: install에서 호환성 경고 무시
        compact: True이면 핵심만 반환
    """
    try:
        if action == "fan":
            return await _device_fan(profile, compact)
        elif action == "install":
            return await _device_install(package, version, force, compact)
        elif action == "packages":
            return await _device_packages(compact)
        else:
            return {"error": f"Unknown action: {action}. Use: fan, install, packages"}
    except Exception as e:
        return {"error": f"device error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  5. job — 장시간 작업 관리
# ══════════════════════════════════════════════════════════════

os.makedirs(JOBS_DIR, exist_ok=True)


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _save_job(job: dict) -> None:
    path = _job_path(job["id"])
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_job(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _log_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.log")


def _read_log(job_id: str, tail: int = 20) -> str:
    path = _log_path(job_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:]).strip()


async def _run_job_with_log(job_id: str, cmd: str, timeout: int) -> dict:
    log_file = _log_path(job_id)
    output_lines = []
    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
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
        return {"exit_code": proc.returncode, "stdout": "".join(output_lines).strip(), "stderr": ""}
    except asyncio.TimeoutError:
        if proc:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        return {"exit_code": -1, "stdout": "".join(output_lines).strip(), "stderr": f"Timed out after {timeout}s"}
    except Exception as e:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
        return {"exit_code": -1, "stdout": "".join(output_lines).strip(), "stderr": str(e)}


async def _run_job(job_id: str) -> None:
    job = _load_job(job_id)
    if not job:
        return
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    _save_job(job)

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
            pass

    try:
        job_type = job["type"]
        timeout = job.get("timeout", 3600)
        if job_type == "python":
            tmp_path = f"/tmp/mcp_job_{job_id}.py"
            with open(tmp_path, "w") as f:
                f.write(job["code"])
            result = await _run_job_with_log(job_id, f"{CUDA_ENV} && {PYTHON38} {tmp_path}", timeout=min(timeout, 7200))
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        elif job_type == "shell":
            result = await _run_job_with_log(job_id, job["command"], timeout=min(timeout, 7200))
        else:
            result = {"exit_code": -1, "stdout": "", "stderr": f"Unknown job type: {job_type}"}
        job["status"] = "completed" if result["exit_code"] == 0 else "failed"
        job["result"] = result
    except Exception as e:
        job["status"] = "failed"
        job["result"] = {"exit_code": -1, "stdout": "", "stderr": traceback.format_exc()}

    if previous_fan and previous_fan != fan_profile:
        try:
            await _set_fan_profile(previous_fan)
            job["fan_restored"] = previous_fan
        except Exception:
            pass

    job["finished_at"] = datetime.now().isoformat()
    _save_job(job)


async def _job_submit(name, type, code, command, timeout, fan_profile, compact) -> dict:
    if type == "python" and not code:
        return {"error": "type='python'이면 code가 필요합니다."}
    if type == "shell" and not command:
        return {"error": "type='shell'이면 command가 필요합니다."}
    if type not in ("python", "shell"):
        return {"error": "type은 'python' 또는 'shell'만 가능합니다."}
    if fan_profile and fan_profile not in FAN_PROFILES_INFO:
        return {"error": f"fan_profile은 {', '.join(FAN_PROFILES_INFO)} 또는 빈 문자열만 가능합니다."}

    job_id = uuid.uuid4().hex[:12]
    job_data = {
        "id": job_id, "name": name, "type": type, "status": "queued",
        "submitted_at": datetime.now().isoformat(), "timeout": min(timeout, 7200),
    }
    if type == "python":
        job_data["code"] = code
    else:
        job_data["command"] = command
    if fan_profile:
        job_data["fan_profile"] = fan_profile
    _save_job(job_data)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    loop.create_task(_run_job(job_id))

    if compact:
        return {"job_id": job_id, "status": "queued"}
    return {
        "job_id": job_id, "name": name, "status": "queued",
        "message": f"작업이 제출되었습니다. job(action='check', job_id='{job_id}')로 상태를 확인하세요.",
        "fan_notice": f"작업 중 팬이 '{fan_profile}' 모드로 전환됩니다." if fan_profile else None,
    }


async def _job_check(job_id, list_all, compact) -> dict:
    if list_all:
        jobs = []
        for fname in sorted(os.listdir(JOBS_DIR), reverse=True):
            if fname.endswith(".json"):
                with open(os.path.join(JOBS_DIR, fname), "r") as f:
                    j = json.load(f)
                jobs.append({"id": j["id"], "name": j["name"], "status": j["status"],
                             "submitted_at": j.get("submitted_at", ""), "finished_at": j.get("finished_at", "")})
        if compact:
            return {"total": len(jobs), "jobs": [{"id": j["id"], "status": j["status"]} for j in jobs]}
        return {"total": len(jobs), "jobs": jobs}

    if not job_id:
        return {"error": "job_id를 입력하거나 list_all=True로 전체 목록을 확인하세요."}

    j = _load_job(job_id)
    if not j:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}

    info = {"id": j["id"], "name": j["name"], "status": j["status"],
            "submitted_at": j.get("submitted_at", ""), "started_at": j.get("started_at", ""),
            "finished_at": j.get("finished_at", "")}
    if j.get("started_at") and j.get("finished_at"):
        start = datetime.fromisoformat(j["started_at"])
        end = datetime.fromisoformat(j["finished_at"])
        info["elapsed_seconds"] = (end - start).total_seconds()
    elif j.get("started_at") and j["status"] == "running":
        start = datetime.fromisoformat(j["started_at"])
        info["running_seconds"] = (datetime.now() - start).total_seconds()
        info["recent_log"] = _read_log(j["id"], tail=10)
    if compact:
        return {"id": j["id"], "status": j["status"],
                "elapsed": info.get("elapsed_seconds", info.get("running_seconds"))}
    return info


async def _job_result(job_id, compact) -> dict:
    j = _load_job(job_id)
    if not j:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}
    if j["status"] in ("queued", "running"):
        return {"status": j["status"], "message": "아직 실행 중입니다." if j["status"] == "running" else "실행 대기 중입니다."}
    elapsed = None
    if j.get("started_at") and j.get("finished_at"):
        elapsed = (datetime.fromisoformat(j["finished_at"]) - datetime.fromisoformat(j["started_at"])).total_seconds()
    if compact:
        stdout = j.get("result", {}).get("stdout", "")
        lines = stdout.split("\n")
        return {"status": j["status"], "elapsed": elapsed, "output": "\n".join(lines[-10:])}
    return {"id": j["id"], "name": j["name"], "status": j["status"],
            "elapsed_seconds": elapsed, "result": j.get("result", {})}


async def _job_log(job_id, tail, compact) -> dict:
    j = _load_job(job_id)
    if not j:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}
    log_content = _read_log(job_id, tail=min(tail, 500))
    if compact:
        lines = log_content.split("\n") if log_content else []
        return {"status": j["status"], "lines": len(lines), "tail": "\n".join(lines[-5:])}
    info = {"id": job_id, "name": j["name"], "status": j["status"], "log": log_content or "(로그 없음)"}
    if j.get("started_at") and j["status"] == "running":
        info["running_seconds"] = (datetime.now() - datetime.fromisoformat(j["started_at"])).total_seconds()
    return info


@mcp.tool()
async def job(action: str, job_id: str = "", name: str = "",
              type: str = "python", code: str = "", command: str = "",
              timeout: int = 3600, fan_profile: str = "aggressive",
              list_all: bool = False, tail: int = 50,
              compact: bool = False) -> dict:
    """
    장시간 작업을 관리합니다 (제출, 상태 확인, 결과 조회).

    Args:
        action: 수행할 작업
            - "submit": 작업 제출 — 백그라운드 실행 후 job_id 반환
            - "check": 작업 상태 확인 (job_id 또는 list_all=True)
            - "result": 완료된 작업 결과 조회
            - "log": 작업 실행 로그 (실시간 확인 가능)
        job_id: check/result/log에서 작업 ID
        name: submit에서 작업 이름
        type: submit에서 python 또는 shell
        code: submit에서 Python 코드
        command: submit에서 셸 커맨드
        timeout: submit에서 최대 실행 시간(초), 기본 3600
        fan_profile: submit에서 팬 프로파일 (기본 aggressive)
        list_all: check에서 True이면 전체 목록
        tail: log에서 마지막 줄 수 (기본 50)
        compact: True이면 핵심만 반환
    """
    try:
        if action == "submit":
            return await _job_submit(name, type, code, command, timeout, fan_profile, compact)
        elif action == "check":
            return await _job_check(job_id, list_all, compact)
        elif action == "result":
            if not job_id:
                return {"error": "job_id를 지정하세요."}
            return await _job_result(job_id, compact)
        elif action == "log":
            if not job_id:
                return {"error": "job_id를 지정하세요."}
            return await _job_log(job_id, tail, compact)
        else:
            return {"error": f"Unknown action: {action}. Use: submit, check, result, log"}
    except Exception as e:
        return {"error": f"job error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  6. workspace — 데이터 버전 관리 (기존 유지)
# ══════════════════════════════════════════════════════════════

def _resolve_workspace_path(name: str) -> str:
    resolved = os.path.realpath(os.path.join(WORKSPACE_ROOT, name))
    if not resolved.startswith(os.path.realpath(WORKSPACE_ROOT)):
        raise ValueError(f"Path traversal detected: {name}")
    return resolved


def _load_workspace_index() -> dict:
    if os.path.exists(WORKSPACE_INDEX):
        with open(WORKSPACE_INDEX, "r") as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(), "versions": {}}


def _save_workspace_index(index: dict) -> None:
    with open(WORKSPACE_INDEX, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


async def _ws_init() -> dict:
    dirs = ["raw", "results"]
    created = []
    for d in dirs:
        path = os.path.join(WORKSPACE_ROOT, d)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            created.append(d)
    index = _load_workspace_index()
    if "raw" not in index["versions"]:
        index["versions"]["raw"] = {"created": datetime.now().isoformat(), "source": None,
                                     "description": "원본 데이터 (보호됨)", "protected": True}
    _save_workspace_index(index)
    return {"workspace_root": WORKSPACE_ROOT, "created_dirs": created or "(이미 존재)", "message": "워크스페이스가 초기화되었습니다."}


async def _ws_status(compact: bool) -> dict:
    if not os.path.exists(WORKSPACE_ROOT):
        return {"error": "워크스페이스가 초기화되지 않았습니다. workspace(action='init')을 먼저 실행하세요."}
    disk = await run_cmd(f"df -h {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    disk_info = {}
    if disk["exit_code"] == 0:
        parts = disk["stdout"].split()
        if len(parts) >= 5:
            disk_info = {"total": parts[1], "used": parts[2], "available": parts[3], "usage_percent": parts[4]}
    index = _load_workspace_index()
    versions = []
    for vname, vmeta in index.get("versions", {}).items():
        vpath = os.path.join(WORKSPACE_ROOT, vname)
        if os.path.exists(vpath):
            size = await _dir_size(vpath)
            file_count = sum(1 for f in os.listdir(vpath) if not f.startswith("."))
            versions.append({"name": vname, "files": file_count, "size": _human_size(size),
                            "created": vmeta.get("created", ""), "source": vmeta.get("source"),
                            "description": vmeta.get("description", ""), "protected": vmeta.get("protected", False)})
    for d in sorted(os.listdir(WORKSPACE_ROOT)):
        dpath = os.path.join(WORKSPACE_ROOT, d)
        if os.path.isdir(dpath) and not d.startswith(".") and d not in index.get("versions", {}):
            size = await _dir_size(dpath)
            file_count = sum(1 for f in os.listdir(dpath) if not f.startswith("."))
            versions.append({"name": d, "files": file_count, "size": _human_size(size), "description": "(인덱스에 미등록)"})
    db_info = None
    if os.path.exists(DUCKDB_PATH):
        db_info = {"path": DUCKDB_PATH, "size": _human_size(os.path.getsize(DUCKDB_PATH))}
    ws_size = await _dir_size(WORKSPACE_ROOT)
    if compact:
        return {"size": _human_size(ws_size), "versions": len(versions), "disk": disk_info.get("usage_percent", "?")}
    return {"workspace_root": WORKSPACE_ROOT, "total_size": _human_size(ws_size),
            "disk": disk_info, "versions": versions, "duckdb": db_info}


async def _ws_list(name, pattern, compact) -> dict:
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
            detail = {"name": fname, "size": _human_size(stat.st_size),
                      "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()}
            if fname.endswith(".csv"):
                wc = await run_cmd(f"wc -l < {shlex.quote(fpath)}")
                if wc["exit_code"] == 0:
                    detail["rows"] = int(wc["stdout"].strip()) - 1
            file_details.append(detail)
        elif os.path.isdir(fpath):
            file_details.append({"name": fname + "/", "type": "directory"})
    if compact:
        return {"version": name, "total": len(file_details),
                "files": [f["name"] for f in file_details[:20]]}
    return {"version": name, "path": vpath, "files": file_details, "total": len(file_details)}


async def _ws_fork(name, source, description) -> dict:
    if not name:
        return {"error": "name을 지정하세요."}
    if not source:
        return {"error": "source를 지정하세요."}
    src_path = _resolve_workspace_path(source)
    dst_path = _resolve_workspace_path(name)
    if not os.path.exists(src_path):
        return {"error": f"원본 '{source}'가 존재하지 않습니다."}
    if os.path.exists(dst_path):
        return {"error": f"대상 '{name}'이 이미 존재합니다."}
    src_size = await _dir_size(src_path)
    disk = await run_cmd(f"df --output=avail -B1 {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    if disk["exit_code"] == 0:
        avail = int(disk["stdout"].strip())
        if src_size > avail * 0.9:
            return {"error": f"디스크 공간 부족. 필요: {_human_size(src_size)}, 가용: {_human_size(avail)}"}
    result = await run_cmd(f"cp -r {shlex.quote(src_path)} {shlex.quote(dst_path)}", timeout=300)
    if result["exit_code"] != 0:
        return {"error": f"복사 실패: {result['stderr']}"}
    meta = {"name": name, "created": datetime.now().isoformat(), "source": source,
            "description": description or f"Forked from {source}"}
    with open(os.path.join(dst_path, "metadata.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    index = _load_workspace_index()
    index["versions"][name] = meta
    _save_workspace_index(index)
    new_size = await _dir_size(dst_path)
    return {"message": f"'{source}' → '{name}' 포크 완료", "version": name,
            "size": _human_size(new_size), "description": meta["description"]}


async def _ws_diff(name, source) -> dict:
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
    changed, unchanged = [], []
    for fname in common:
        sf, df = os.path.join(src_path, fname), os.path.join(dst_path, fname)
        if os.path.isfile(sf) and os.path.isfile(df):
            s_size, d_size = os.path.getsize(sf), os.path.getsize(df)
            if s_size != d_size:
                changed.append({"file": fname, f"{source}_size": _human_size(s_size), f"{name}_size": _human_size(d_size)})
            else:
                unchanged.append(fname)
        else:
            unchanged.append(fname)
    return {"source": source, "target": name, f"only_in_{source}": only_in_source,
            f"only_in_{name}": only_in_target, "changed": changed, "unchanged_count": len(unchanged)}


async def _ws_info(name) -> dict:
    if not name:
        return {"error": "name을 지정하세요."}
    vpath = _resolve_workspace_path(name)
    if not os.path.exists(vpath):
        return {"error": f"'{name}'이 존재하지 않습니다."}
    index = _load_workspace_index()
    meta = index.get("versions", {}).get(name, {})
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
    return {"name": name, "path": vpath, "size": _human_size(size), "files": file_count,
            "created": meta.get("created", ""), "source": meta.get("source"),
            "description": meta.get("description", ""), "protected": meta.get("protected", False),
            "lineage": " → ".join(lineage)}


async def _ws_delete(name) -> dict:
    if not name:
        return {"error": "name을 지정하세요."}
    index = _load_workspace_index()
    meta = index.get("versions", {}).get(name, {})
    if meta.get("protected"):
        return {"error": f"'{name}'은 보호된 디렉토리입니다."}
    vpath = _resolve_workspace_path(name)
    if not os.path.exists(vpath):
        return {"error": f"'{name}'이 존재하지 않습니다."}
    size = await _dir_size(vpath)
    result = await run_cmd(f"rm -rf {shlex.quote(vpath)}")
    if result["exit_code"] != 0:
        return {"error": f"삭제 실패: {result['stderr']}"}
    if name in index.get("versions", {}):
        del index["versions"][name]
        _save_workspace_index(index)
    return {"message": f"'{name}' 삭제 완료", "freed": _human_size(size)}


@mcp.tool()
async def workspace(action: str, name: str = "", source: str = "",
                    description: str = "", pattern: str = "",
                    compact: bool = False) -> dict:
    """
    EDA 데이터 워크스페이스를 관리합니다. 데이터 버전 관리, 포크, 비교.

    Args:
        action: 수행할 작업
            - "init": 워크스페이스 초기화 (디렉토리 구조 생성)
            - "status": 전체 현황 (디스크 사용량, 버전 목록)
            - "list": 특정 버전 파일 목록 (name, pattern)
            - "fork": source에서 name으로 데이터 복사
            - "diff": source와 name 두 버전 비교
            - "info": 특정 버전 상세 정보
            - "delete": 버전 삭제 (raw는 보호)
        name: 대상 버전/디렉토리 이름
        source: fork/diff에서 원본 버전 이름
        description: fork 생성 시 설명
        pattern: list에서 파일 필터 (예: "*.csv")
        compact: True이면 핵심만 반환
    """
    try:
        if action == "init":
            return await _ws_init()
        elif action == "status":
            return await _ws_status(compact)
        elif action == "list":
            return await _ws_list(name, pattern, compact)
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
        return {"error": f"workspace error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  7. data — 데이터 I/O + 정제
# ══════════════════════════════════════════════════════════════

async def _data_upload(filename, content_base64, content_text, dest, overwrite, compact) -> dict:
    if not filename:
        return {"error": "filename을 지정하세요."}
    if not content_base64 and not content_text:
        return {"error": "content_base64 또는 content_text 중 하나를 제공하세요."}
    try:
        dest_path = _resolve_workspace_path(dest or "raw")
    except ValueError as e:
        return {"error": str(e)}
    os.makedirs(dest_path, exist_ok=True)
    file_path = os.path.join(dest_path, filename)
    if os.path.exists(file_path) and not overwrite:
        return {"error": f"'{filename}'이 이미 존재합니다. overwrite=True로 덮어쓰기 가능."}
    disk = await run_cmd(f"df --output=avail -B1 {shlex.quote(WORKSPACE_ROOT)} | tail -1")
    if disk["exit_code"] == 0 and int(disk["stdout"].strip()) < 100 * 1024 * 1024:
        return {"error": f"디스크 공간 부족."}
    try:
        if content_base64:
            data = base64.b64decode(content_base64)
            with open(file_path, "wb") as f:
                f.write(data)
            file_size = len(data)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_text)
            file_size = len(content_text.encode("utf-8"))
        if compact:
            return {"ok": True, "file": filename, "size": _human_size(file_size)}
        return {"success": True, "path": file_path, "filename": filename,
                "size": _human_size(file_size), "size_bytes": file_size, "dest": dest or "raw"}
    except Exception as e:
        return {"error": f"파일 저장 실패: {str(e)}"}


async def _data_fetch(url, dest, filename, extract, timeout, compact) -> dict:
    if not url or not url.startswith(("http://", "https://")):
        return {"error": "http:// 또는 https:// URL만 지원합니다."}
    if dest:
        try:
            dest_path = _resolve_workspace_path(dest)
        except ValueError as e:
            return {"error": str(e)}
    else:
        dest_path = os.path.join(WORKSPACE_ROOT, "raw")
    os.makedirs(dest_path, exist_ok=True)
    if not filename:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        filename = unquote(os.path.basename(parsed.path)) or "download"
    file_path = os.path.join(dest_path, filename)
    import time
    start_time = time.time()
    cmd = f"wget -q --show-progress -O {shlex.quote(file_path)} {shlex.quote(url)}"
    result = await run_cmd(cmd, timeout=min(timeout, 3600))
    if result["exit_code"] != 0:
        cmd = f"curl -fSL -o {shlex.quote(file_path)} {shlex.quote(url)}"
        result = await run_cmd(cmd, timeout=min(timeout, 3600))
    if result["exit_code"] != 0:
        return {"error": f"다운로드 실패: {result['stderr']}"}
    elapsed = time.time() - start_time
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    response = {"success": True, "path": file_path, "size": _human_size(file_size), "elapsed_seconds": round(elapsed, 1)}
    if extract and os.path.exists(file_path):
        if filename.endswith((".tar.gz", ".tgz")):
            ext_cmd = f"tar -xzf {shlex.quote(file_path)} -C {shlex.quote(dest_path)}"
        elif filename.endswith(".zip"):
            ext_cmd = f"unzip -o {shlex.quote(file_path)} -d {shlex.quote(dest_path)}"
        else:
            ext_cmd = None
        if ext_cmd:
            ext_result = await run_cmd(ext_cmd, timeout=300)
            response["extracted"] = ext_result["exit_code"] == 0
    if compact:
        return {"ok": True, "file": filename, "size": _human_size(file_size)}
    return response


async def _data_stats(path, sample_rows, compact) -> dict:
    if not os.path.isabs(path):
        try:
            path = _resolve_workspace_path(path)
        except ValueError as e:
            return {"error": str(e)}
    if not os.path.exists(path):
        return {"error": f"파일이 존재하지 않습니다: {path}"}
    ext = os.path.splitext(path)[1].lower()
    sample_rows = min(sample_rows, 20)
    compact_flag = "True" if compact else "False"
    code = f"""
import json, sys
try:
    import pandas as pd
    path = {repr(path)}
    ext = {repr(ext)}
    sample = {sample_rows}
    compact = {compact_flag}

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

    total_rows = len(df)
    if ext == '.csv' and total_rows >= 10000:
        import subprocess
        wc = subprocess.run(['wc', '-l', path], capture_output=True, text=True)
        if wc.returncode == 0:
            total_rows = int(wc.stdout.split()[0]) - 1

    if compact:
        null_total = int(df.isnull().sum().sum())
        result = {{"rows": total_rows, "cols": len(df.columns), "nulls": null_total,
                  "columns": list(df.columns)}}
    else:
        result = {{
            "shape": list(df.shape), "total_rows": total_rows,
            "columns": list(df.columns),
            "dtypes": {{col: str(dtype) for col, dtype in df.dtypes.items()}},
            "null_counts": df.isnull().sum().to_dict(),
            "head": df.head(sample).to_dict(orient='records'),
        }}
        num_cols = [c for c in df.columns if df[c].dtype.kind in ('i', 'f')]
        if num_cols:
            desc = dict()
            for c in num_cols:
                s = df[c].dropna()
                desc[c] = {{"count": len(s), "mean": float(s.mean()), "std": float(s.std()),
                           "min": float(s.min()), "25%": float(s.quantile(0.25)),
                           "50%": float(s.quantile(0.5)), "75%": float(s.quantile(0.75)),
                           "max": float(s.max())}}
            result["describe"] = desc

    print(json.dumps(result, default=str, ensure_ascii=False))
except ImportError:
    print(json.dumps({{"error": "pandas가 설치되어 있지 않습니다."}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result = await _execute_python(code, timeout=60, compact=False)
    if result["exit_code"] != 0:
        return {"error": f"분석 실패: {result['stderr'] or result['stdout']}"}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


async def _data_query(sql, limit, compact) -> dict:
    limit = min(limit, 10000)
    compact_flag = "True" if compact else "False"
    code = f"""
import json, sys, os
try:
    import duckdb
    os.chdir({repr(WORKSPACE_ROOT)})
    con = duckdb.connect({repr(DUCKDB_PATH)})
    result = con.execute({repr(sql)}).fetchdf()
    total_rows = len(result)
    if total_rows > {limit}:
        result = result.head({limit})
    compact = {compact_flag}
    if compact:
        output = {{"rows": total_rows, "cols": len(result.columns), "columns": list(result.columns)}}
    else:
        output = {{"columns": list(result.columns), "total_rows": total_rows,
                  "returned_rows": len(result), "data": result.to_dict(orient='records')}}
    con.close()
    print(json.dumps(output, default=str, ensure_ascii=False))
except ImportError:
    print(json.dumps({{"error": "duckdb가 설치되어 있지 않습니다."}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result = await _execute_python(code, timeout=120, compact=False)
    if result["exit_code"] != 0:
        return {"error": f"쿼리 실패: {result['stderr'] or result['stdout']}"}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


async def _data_ingest(source, table, mode, compact) -> dict:
    if not source or not table:
        return {"error": "source와 table을 모두 지정하세요."}
    if mode not in ("create", "replace", "append"):
        return {"error": "mode는 create, replace, append 중 하나입니다."}
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
    con = duckdb.connect({repr(DUCKDB_PATH)})
    source = {repr(source)}
    table = {repr(table)}
    mode = {repr(mode)}
    ext = {repr(ext)}
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
    count = con.execute(f"SELECT COUNT(*) FROM {{table}}").fetchone()[0]
    cols = [desc[0] for desc in con.execute(f"SELECT * FROM {{table}} LIMIT 0").description]
    con.close()
    print(json.dumps({{"success": True, "table": table, "rows": count, "columns": cols, "mode": mode}}))
except ImportError:
    print(json.dumps({{"error": "duckdb가 설치되어 있지 않습니다."}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    result = await _execute_python(code, timeout=300, compact=False)
    if result["exit_code"] != 0:
        return {"error": f"적재 실패: {result['stderr'] or result['stdout']}"}
    try:
        parsed = json.loads(result["stdout"])
        if compact and parsed.get("success"):
            return {"ok": True, "table": table, "rows": parsed["rows"]}
        return parsed
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


@mcp.tool()
async def data(action: str, path: str = "", url: str = "",
               filename: str = "", content_base64: str = "",
               content_text: str = "", dest: str = "",
               overwrite: bool = False, extract: bool = False,
               timeout: int = 600, sample_rows: int = 5,
               sql: str = "", limit: int = 100,
               source: str = "", table: str = "", mode: str = "create",
               compact: bool = False) -> dict:
    """
    데이터 전송, 저장, 통계 분석을 수행합니다.

    Args:
        action: 수행할 작업
            - "upload": Mac→Jetson 파일 업로드 (filename + content_text/content_base64)
            - "fetch": URL에서 데이터 다운로드 (url)
            - "stats": 데이터 기본 통계 — Detection 단계 (path)
            - "query": DuckDB SQL 쿼리 (sql)
            - "ingest": CSV/Parquet → DuckDB 테이블 적재 (source, table)
        path: stats에서 데이터 파일 경로
        url: fetch에서 다운로드 URL
        filename: upload에서 저장 파일명
        content_base64: upload에서 바이너리 파일 (base64)
        content_text: upload에서 텍스트 파일
        dest: upload/fetch에서 저장 디렉토리 (기본: raw)
        overwrite: upload에서 덮어쓰기 허용
        extract: fetch에서 압축 자동 해제
        timeout: fetch 타임아웃(초)
        sample_rows: stats에서 미리보기 행 수
        sql: query에서 SQL 쿼리
        limit: query에서 최대 반환 행 수
        source: ingest에서 소스 파일 경로
        table: ingest에서 테이블 이름
        mode: ingest에서 create/replace/append
        compact: True이면 핵심만 반환
    """
    try:
        if action == "upload":
            return await _data_upload(filename, content_base64, content_text, dest, overwrite, compact)
        elif action == "fetch":
            return await _data_fetch(url, dest, filename, extract, timeout, compact)
        elif action == "stats":
            if not path:
                return {"error": "path를 지정하세요."}
            return await _data_stats(path, sample_rows, compact)
        elif action == "query":
            if not sql:
                return {"error": "sql을 지정하세요."}
            return await _data_query(sql, limit, compact)
        elif action == "ingest":
            return await _data_ingest(source, table, mode, compact)
        else:
            return {"error": f"Unknown action: {action}. Use: upload, fetch, stats, query, ingest"}
    except Exception as e:
        return {"error": f"data error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  8. xai — XAI 설명·분석 계층
# ══════════════════════════════════════════════════════════════

def _build_xai_code(path: str, columns: str, method: str, threshold: float,
                    focus: str, compact: bool, action: str) -> str:
    """XAI 분석용 Python 코드를 생성합니다."""
    return f"""
import json, sys
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np

    path = {repr(path)}
    columns_str = {repr(columns)}
    method = {repr(method)}
    threshold = {threshold}
    focus = {repr(focus)}
    compact = {repr(compact)}
    xai_action = {repr(action)}

    # 데이터 로드
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'csv':
        df = pd.read_csv(path)
    elif ext == 'tsv':
        df = pd.read_csv(path, sep='\\t')
    elif ext in ('parquet', 'pq'):
        df = pd.read_parquet(path)
    elif ext == 'json':
        df = pd.read_json(path)
    else:
        print(json.dumps({{"error": f"지원하지 않는 형식: {{ext}}"}}))
        sys.exit(0)

    # 컬럼 필터
    if columns_str:
        target_cols = [c.strip() for c in columns_str.split(',')]
        df_analysis = df[[c for c in target_cols if c in df.columns]]
    else:
        df_analysis = df

    num_cols = [c for c in df_analysis.columns if df_analysis[c].dtype.kind in ('i', 'f')]
    result = {{"rows": len(df), "cols": len(df.columns), "num_cols": len(num_cols)}}

    # ── 상관관계 분석 ──
    if xai_action in ('explain', 'correlate') and len(num_cols) >= 2:
        corr_matrix = df_analysis[num_cols].corr()
        strong_corrs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > threshold:
                    strength = "강한 양의 상관" if r > 0 else "강한 음의 상관"
                    strong_corrs.append({{
                        "col1": num_cols[i], "col2": num_cols[j],
                        "r": round(float(r), 3), "strength": strength
                    }})
        result["correlations"] = sorted(strong_corrs, key=lambda x: abs(x["r"]), reverse=True)

        # 다중공선성 경고
        if len(strong_corrs) > 3:
            result["multicollinearity_warning"] = f"{{len(strong_corrs)}}개 강한 상관쌍 발견. 다중공선성 주의."

    # ── 이상치 탐지 ──
    if xai_action in ('explain', 'outliers') and num_cols:
        outliers = dict()
        for col in num_cols:
            series = df_analysis[col].dropna()
            if len(series) < 10:
                continue
            if method in ('auto', 'iqr'):
                Q1, Q3 = series.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                mask = (series < lower) | (series > upper)
            elif method == 'zscore':
                z = np.abs((series - series.mean()) / series.std())
                mask = z > 3
                lower, upper = series.mean() - 3*series.std(), series.mean() + 3*series.std()
            else:
                continue

            count = int(mask.sum())
            if count > 0:
                mean_with = float(series.mean())
                mean_without = float(series[~mask].mean())
                impact = abs(mean_with - mean_without) / abs(mean_with) * 100 if mean_with != 0 else 0
                outliers[col] = {{
                    "count": count,
                    "percent": round(count / len(series) * 100, 1),
                    "range": [round(float(lower), 2), round(float(upper), 2)],
                    "impact_pct": round(impact, 1),
                    "narrative": f"{{col}}에서 {{count}}건({{round(count/len(series)*100,1)}}%) 이상치 발견. 평균을 {{round(impact,1)}}% 왜곡."
                }}
        result["outliers"] = outliers

    # ── 분포 프로파일링 ──
    if xai_action in ('explain', 'profile') and num_cols:
        distributions = dict()
        for col in num_cols:
            series = df_analysis[col].dropna()
            if len(series) < 5:
                continue
            s = float(series.skew())
            k = float(series.kurtosis())
            if abs(s) < 0.5:
                shape = "정규분포"
            elif s > 0:
                shape = "우편향 (양의 왜도)"
            else:
                shape = "좌편향 (음의 왜도)"
            distributions[col] = {{
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "skewness": round(s, 3),
                "kurtosis": round(k, 3),
                "shape": shape,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }}
        result["distributions"] = distributions

    # ── 결측 패턴 ──
    if xai_action in ('explain', 'profile'):
        null_info = dict()
        for col in df_analysis.columns:
            null_count = int(df_analysis[col].isnull().sum())
            if null_count > 0:
                pct = round(null_count / len(df) * 100, 1)
                null_info[col] = {{"count": null_count, "percent": pct}}
        result["missing"] = null_info
        if null_info:
            # MCAR 간이 판단: 결측 비율이 전체적으로 유사하면 MCAR 추정
            pcts = [v["percent"] for v in null_info.values()]
            if max(pcts) - min(pcts) < 5:
                result["missing_pattern"] = "MCAR 추정 (결측이 무작위로 분포)"
            else:
                result["missing_pattern"] = "MAR/MNAR 가능성 (결측 비율 불균등)"

    # ── 카디널리티 (profile) ──
    if xai_action == 'profile':
        cardinality = dict()
        for col in df_analysis.columns:
            unique = df_analysis[col].nunique()
            ratio = round(unique / len(df) * 100, 1)
            if ratio > 95:
                label = "고유값 (ID성)"
            elif ratio < 1:
                label = "저카디널리티 (범주형)"
            else:
                label = "혼합"
            cardinality[col] = {{"unique": unique, "ratio_pct": ratio, "label": label}}
        result["cardinality"] = cardinality

    # ── 자연어 요약 생성 ──
    summary_parts = []
    summary_parts.append(f"데이터셋: {{len(df)}}행 × {{len(df.columns)}}열 (수치 {{len(num_cols)}}열)")

    if result.get("missing"):
        null_cols = list(result["missing"].keys())
        summary_parts.append(f"결측: {{', '.join(null_cols[:3])}} 등 {{len(null_cols)}}개 컬럼")

    if result.get("correlations"):
        top_corrs = result["correlations"][:3]
        pairs = [f"{{c['col1']}}↔{{c['col2']}}(r={{c['r']}})" for c in top_corrs]
        summary_parts.append(f"강한 상관: {{', '.join(pairs)}}")

    if result.get("outliers"):
        out_cols = list(result["outliers"].keys())
        total_outliers = sum(v["count"] for v in result["outliers"].values())
        summary_parts.append(f"이상치: {{', '.join(out_cols[:3])}}에서 총 {{total_outliers}}건 발견")

    if result.get("distributions"):
        skewed = [col for col, d in result["distributions"].items() if abs(d["skewness"]) > 1]
        if skewed:
            summary_parts.append(f"편향 분포: {{', '.join(skewed[:3])}}")

    result["summary"] = ". ".join(summary_parts) + "."

    # ── 인사이트 (설명 가능한 핵심 발견) ──
    insights = []
    if result.get("correlations"):
        for c in result["correlations"][:2]:
            insights.append(f"{{c['col1']}}과 {{c['col2']}}는 {{c['strength']}}(r={{c['r']}})을 보입니다.")
    if result.get("outliers"):
        for col, info in list(result["outliers"].items())[:2]:
            insights.append(info["narrative"])
    result["insights"] = insights

    # ── compact 모드 ──
    if compact:
        alerts = len(result.get("correlations", [])) + len(result.get("outliers", dict()))
        compact_result = {{"summary": result["summary"], "alerts": alerts, "insights": result.get("insights", [])}}
        print(json.dumps(compact_result, default=str, ensure_ascii=False))
    else:
        print(json.dumps(result, default=str, ensure_ascii=False))

except ImportError as e:
    print(json.dumps({{"error": f"필요한 패키지가 없습니다: {{e}}"}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""


async def _xai_analyze(path, columns, method, threshold, focus, compact, action) -> dict:
    """XAI 분석을 Python 3.8에서 실행합니다."""
    if not os.path.isabs(path):
        try:
            path = _resolve_workspace_path(path)
        except ValueError as e:
            return {"error": str(e)}
    if not os.path.exists(path):
        return {"error": f"파일이 존재하지 않습니다: {path}"}
    code = _build_xai_code(path, columns, method, threshold, focus, compact, action)
    result = await _execute_python(code, timeout=120, compact=False)
    if result["exit_code"] != 0:
        return {"error": f"XAI 분석 실패: {result['stderr'] or result['stdout']}"}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


async def _xai_trace(job_id, compact) -> dict:
    """완료된 학습 작업의 결과를 해석합니다."""
    j = _load_job(job_id)
    if not j:
        return {"error": f"작업 '{job_id}'를 찾을 수 없습니다."}
    if j["status"] not in ("completed", "failed"):
        return {"status": j["status"], "message": "작업이 아직 완료되지 않았습니다."}

    stdout = j.get("result", {}).get("stdout", "")
    log_content = _read_log(job_id, tail=100)
    full_output = log_content or stdout

    # 학습 메트릭 파싱
    code = f"""
import json, re, sys

output = {repr(full_output)}
lines = output.strip().split('\\n')

metrics = dict()
epochs = []
losses = []
accuracies = []

for line in lines:
    # epoch/loss/accuracy 패턴 매칭
    epoch_match = re.search(r'[Ee]poch\\s*[:\\[]?\\s*(\\d+)', line)
    loss_match = re.search(r'[Ll]oss[:\\s]+([\\d.]+)', line)
    acc_match = re.search(r'[Aa]cc(?:uracy)?[:\\s]+([\\d.]+)%?', line)

    if epoch_match:
        epochs.append(int(epoch_match.group(1)))
    if loss_match:
        losses.append(float(loss_match.group(1)))
    if acc_match:
        val = float(acc_match.group(1))
        accuracies.append(val if val <= 1 else val)

    # 혼동행렬 파싱
    cm_match = re.search(r'TP[=:\s]+(\d+).*?FP[=:\s]+(\d+).*?FN[=:\s]+(\d+).*?TN[=:\s]+(\d+)', line)
    if cm_match:
        metrics['confusion_matrix'] = {{'TP': int(cm_match.group(1)), 'FP': int(cm_match.group(2)),
                                         'FN': int(cm_match.group(3)), 'TN': int(cm_match.group(4))}}

    prec_match = re.search(r'[Pp]recision[=:\s]+([\\d.]+)%?', line)
    rec_match = re.search(r'[Rr]ecall[=:\s]+([\\d.]+)%?', line)
    f1_match = re.search(r'[Ff]1[=:\s]+([\\d.]+)%?', line)
    if prec_match:
        metrics['precision'] = float(prec_match.group(1))
    if rec_match:
        metrics['recall'] = float(rec_match.group(1))
    if f1_match:
        metrics['f1'] = float(f1_match.group(1))

    # features/params 파싱
    feat_match = re.search(r'[Ff]eatures?[=:\s]+(\d+)', line)
    param_match = re.search(r'[Pp]arams?[=:\s]+([\d,]+)', line)
    if feat_match:
        metrics['features_used'] = int(feat_match.group(1))
    if param_match:
        metrics['params'] = int(param_match.group(1).replace(',', ''))

result = {{"job_id": {repr(job_id)}, "status": {repr(j['status'])}}}

if epochs:
    result["total_epochs"] = max(epochs)
if losses:
    result["loss"] = {{"first": round(losses[0], 4), "last": round(losses[-1], 4),
                       "trend": "하강" if losses[-1] < losses[0] else "상승 또는 정체"}}
if accuracies:
    result["accuracy"] = {{"first": round(accuracies[0], 2), "last": round(accuracies[-1], 2),
                           "best": round(max(accuracies), 2)}}

# 수렴 판단
summary_parts = []
if accuracies:
    last_acc = accuracies[-1]
    summary_parts.append(f"최종 정확도: {{last_acc}}%")
    if len(accuracies) >= 3:
        recent = accuracies[-3:]
        if max(recent) - min(recent) < 1.0:
            summary_parts.append("수렴 정체 (plateau). 더 많은 epoch 필요.")
            result["convergence"] = "plateau"
        elif recent[-1] > recent[0]:
            summary_parts.append("아직 개선 중. 추가 학습 권장.")
            result["convergence"] = "improving"
        else:
            summary_parts.append("과적합 가능성. Early stopping 검토.")
            result["convergence"] = "overfitting"
if losses:
    if losses[-1] < losses[0] * 0.5:
        summary_parts.append(f"Loss가 {{round((1-losses[-1]/losses[0])*100)}}% 감소.")

result["summary"] = " ".join(summary_parts) if summary_parts else "학습 메트릭을 파싱할 수 없습니다."

# 혼동행렬/precision/recall/f1/features 추가
for key in ('confusion_matrix', 'precision', 'recall', 'f1', 'features_used', 'params'):
    if key in metrics:
        result[key] = metrics[key]

if {repr(compact)}:
    print(json.dumps({{"summary": result["summary"],
                       "accuracy": result.get("accuracy", dict()).get("last"),
                       "convergence": result.get("convergence", "unknown")}}))
else:
    print(json.dumps(result, default=str, ensure_ascii=False))
"""
    result = await _execute_python(code, timeout=30, compact=False)
    if result["exit_code"] != 0:
        return {"error": f"trace 분석 실패: {result['stderr'] or result['stdout']}"}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"output": result["stdout"]}


async def _xai_diagnose(job_id, path, columns, threshold, compact) -> dict:
    """학습 결과 + 데이터 특성을 결합한 종합 진단. EDA 반복 루프의 핵심."""
    # 1. 학습 결과 파싱 (trace 재사용)
    trace_result = await _xai_trace(job_id, False)
    if "error" in trace_result:
        return trace_result

    # 2. 데이터 파일 경로 확인
    if not os.path.isabs(path):
        try:
            path = _resolve_workspace_path(path)
        except ValueError as e:
            return {"error": str(e)}
    if not os.path.exists(path):
        return {"error": f"파일이 존재하지 않습니다: {path}"}

    # 3. Python 3.8 진단 스크립트 생성
    used_cols_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else []
    code = f"""
import json, sys, traceback
try:
    import pandas as pd
    import numpy as np

    df = pd.read_csv({repr(path)}, nrows=50000)
    num_cols = [c for c in df.columns if df[c].dtype.kind in ('i', 'f')]
    all_cols = list(df.columns)
    used_cols = {repr(used_cols_list)} if {repr(bool(used_cols_list))} else num_cols[:20]

    result = {{}}

    # 사용/미사용 컬럼 분석
    result['unused_columns'] = {{
        'available': len(all_cols),
        'numeric_available': len(num_cols),
        'used': len(used_cols),
        'unused_numeric': [c for c in num_cols if c not in used_cols][:15]
    }}

    # 다중공선성 분석
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        mc_pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                r = abs(corr.iloc[i, j])
                if r >= {threshold}:
                    mc_pairs.append({{
                        'col1': num_cols[i], 'col2': num_cols[j],
                        'r': round(float(r), 4),
                        'action': 'drop_one' if r > 0.95 else 'monitor'
                    }})
        mc_pairs.sort(key=lambda x: -x['r'])
        result['multicollinearity'] = mc_pairs[:20]
    else:
        result['multicollinearity'] = []

    # 저분산 컬럼
    low_var = []
    for c in num_cols:
        if df[c].std() < 1e-6:
            low_var.append(c)
    result['low_variance'] = low_var

    # 편향된 분포
    skewed = []
    for c in num_cols:
        s = float(df[c].skew()) if df[c].std() > 0 else 0
        if abs(s) > 2:
            skewed.append({{'col': c, 'skewness': round(s, 2), 'action': 'log_transform' if s > 2 else 'reflect_log'}})
    result['skewed_features'] = skewed[:10]

    # 클래스 불균형 분석 (마지막 컬럼이 타겟이라 가정, 또는 object/category 컬럼)
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() <= 10]
    class_imbalance = {{'detected': False}}
    for tc in reversed(cat_cols):
        vc = df[tc].value_counts()
        if len(vc) >= 2:
            ratio = float(vc.min()) / float(vc.max())
            if ratio < 0.5:
                class_imbalance = {{
                    'detected': True,
                    'target_column': tc,
                    'distribution': {{str(k): int(v) for k, v in vc.items()}},
                    'ratio': round(ratio, 3),
                    'action': 'class_weight' if ratio > 0.2 else 'oversampling'
                }}
                break
    result['class_imbalance'] = class_imbalance

    # 규칙 기반 추천 생성
    recommendations = []
    priority = 1

    # 다중공선성 추천
    perfect_mc = [p for p in result['multicollinearity'] if p['r'] > 0.95]
    if perfect_mc:
        drop_cols = [p['col2'] for p in perfect_mc[:5]]
        recommendations.append({{
            'priority': priority,
            'category': 'multicollinearity',
            'action': 'drop_redundant',
            'description': f"완전 다중공선성 {{len(perfect_mc)}}쌍 발견. 중복 컬럼 제거 필요.",
            'columns_to_drop': drop_cols,
            'expected_impact': '모델 안정성 대폭 향상'
        }})
        priority += 1

    # 미사용 컬럼 추천
    unused_n = result['unused_columns']['numeric_available'] - result['unused_columns']['used']
    if unused_n > 5:
        recommendations.append({{
            'priority': priority,
            'category': 'feature_selection',
            'action': 'add_features',
            'description': f"{{unused_n}}개 수치형 컬럼 미사용. 추가 피처 탐색 권장.",
            'candidate_columns': result['unused_columns']['unused_numeric'][:10],
            'expected_impact': '모델 표현력 향상'
        }})
        priority += 1

    # 편향 분포 추천
    if result['skewed_features']:
        recommendations.append({{
            'priority': priority,
            'category': 'distribution',
            'action': 'transform_skewed',
            'description': f"{{len(result['skewed_features'])}}개 컬럼 심한 편향. 로그/sqrt 변환 권장.",
            'columns': [s['col'] for s in result['skewed_features']],
            'expected_impact': '정규성 개선, 모델 학습 안정화'
        }})
        priority += 1

    # 클래스 불균형 추천
    if class_imbalance['detected']:
        recommendations.append({{
            'priority': priority,
            'category': 'class_imbalance',
            'action': class_imbalance['action'],
            'description': f"클래스 불균형 (비율 {{class_imbalance['ratio']}}). {{class_imbalance['action']}} 적용 권장.",
            'expected_impact': 'FN 감소, 소수 클래스 탐지율 향상'
        }})
        priority += 1

    # 피처 엔지니어링 추천 (시계열 패턴)
    velocity_cols = [c for c in num_cols if 'velocity' in c.lower() or 'speed' in c.lower()]
    accel_cols = [c for c in num_cols if 'accel' in c.lower()]
    if velocity_cols or accel_cols:
        recommendations.append({{
            'priority': priority,
            'category': 'feature_engineering',
            'action': 'add_rolling_stats',
            'description': '시계열 센서 데이터에 rolling mean/std 추가로 시간적 패턴 포착 권장.',
            'columns': (velocity_cols + accel_cols)[:6],
            'window_sizes': [5, 10, 50],
            'expected_impact': '시간적 패턴 포착, 분류 성능 향상'
        }})
        priority += 1

    # StandardScaler 추천
    ranges = []
    for c in num_cols[:20]:
        r = float(df[c].max() - df[c].min())
        if r > 0:
            ranges.append(r)
    if ranges and max(ranges) / (min(ranges) + 1e-10) > 100:
        recommendations.append({{
            'priority': priority,
            'category': 'preprocessing',
            'action': 'standardize',
            'description': '피처 스케일 차이 큼. StandardScaler 적용 권장.',
            'expected_impact': '그래디언트 안정화, 학습 속도 향상'
        }})

    result['recommendations'] = recommendations

    # 심각도 판단
    n_critical = len(perfect_mc)
    if n_critical >= 3 or (class_imbalance['detected'] and class_imbalance.get('ratio', 1) < 0.3):
        result['severity'] = 'critical'
    elif n_critical >= 1 or len(result['skewed_features']) >= 3:
        result['severity'] = 'warning'
    else:
        result['severity'] = 'info'

    # 요약
    parts = []
    parts.append(f"{{len(all_cols)}}컬럼 중 {{len(used_cols)}}개 사용")
    if perfect_mc:
        parts.append(f"완전 다중공선성 {{len(perfect_mc)}}쌍")
    if class_imbalance['detected']:
        parts.append(f"클래스 불균형 (비율 {{class_imbalance['ratio']}})")
    if result['skewed_features']:
        parts.append(f"심한 편향 {{len(result['skewed_features'])}}컬럼")
    result['summary'] = ". ".join(parts) + "."

    print(json.dumps(result, default=str, ensure_ascii=False))

except Exception as e:
    print(json.dumps({{"error": traceback.format_exc()}}))
"""

    # 진단 스크립트 실행
    diag_result = await _execute_python(code, timeout=120, compact=False)
    if diag_result["exit_code"] != 0:
        return {"error": f"diagnose 실패: {diag_result['stderr'] or diag_result['stdout']}"}
    try:
        data_diag = json.loads(diag_result["stdout"])
    except json.JSONDecodeError:
        return {"error": f"diagnose JSON 파싱 실패: {diag_result['stdout'][:500]}"}

    # trace 결과와 결합
    final = {
        "iteration_context": {
            "job_id": job_id,
            "accuracy": trace_result.get("accuracy", {}).get("last"),
            "convergence": trace_result.get("convergence", "unknown"),
            "loss_trend": trace_result.get("loss", {}).get("trend"),
            "confusion_matrix": trace_result.get("confusion_matrix"),
            "features_used": trace_result.get("features_used"),
        },
        "data_issues": {
            "multicollinearity": data_diag.get("multicollinearity", []),
            "low_variance": data_diag.get("low_variance", []),
            "skewed_features": data_diag.get("skewed_features", []),
            "unused_columns": data_diag.get("unused_columns", {}),
            "class_imbalance": data_diag.get("class_imbalance", {}),
        },
        "recommendations": data_diag.get("recommendations", []),
        "severity": data_diag.get("severity", "info"),
        "summary": data_diag.get("summary", ""),
    }

    if compact:
        return {
            "severity": final["severity"],
            "summary": final["summary"],
            "recommendations": len(final["recommendations"]),
            "accuracy": final["iteration_context"]["accuracy"],
            "convergence": final["iteration_context"]["convergence"],
        }
    return final


async def _xai_compare(job_ids_str, compact) -> dict:
    """여러 반복 학습 결과를 비교하고 중단 여부를 판단합니다."""
    job_ids = [jid.strip() for jid in job_ids_str.split(",") if jid.strip()]
    if len(job_ids) < 2:
        return {"error": "최소 2개 이상의 job_id가 필요합니다. 쉼표로 구분하세요."}

    iterations = []
    for idx, jid in enumerate(job_ids):
        trace = await _xai_trace(jid, False)
        if "error" in trace:
            iterations.append({"job_id": jid, "iteration": idx + 1, "error": trace["error"]})
            continue
        iterations.append({
            "job_id": jid,
            "iteration": idx + 1,
            "accuracy": trace.get("accuracy", {}).get("last"),
            "convergence": trace.get("convergence", "unknown"),
            "loss_first": trace.get("loss", {}).get("first"),
            "loss_last": trace.get("loss", {}).get("last"),
            "features_used": trace.get("features_used"),
            "confusion_matrix": trace.get("confusion_matrix"),
            "total_epochs": trace.get("total_epochs"),
        })

    # 정확도 추이 계산
    accuracies = [it.get("accuracy") for it in iterations]
    deltas = [None]
    for i in range(1, len(accuracies)):
        if accuracies[i] is not None and accuracies[i-1] is not None:
            deltas.append(round(accuracies[i] - accuracies[i-1], 2))
        else:
            deltas.append(None)

    valid_acc = [a for a in accuracies if a is not None]
    best_idx = accuracies.index(max(valid_acc)) + 1 if valid_acc else 0
    best_acc = max(valid_acc) if valid_acc else 0

    # 추세 판단
    if len(valid_acc) >= 2:
        if valid_acc[-1] > valid_acc[-2] + 1:
            trend = "improving"
        elif valid_acc[-1] < valid_acc[-2] - 1:
            trend = "degrading"
        else:
            trend = "plateau"
    else:
        trend = "unknown"

    # 중단 추천
    should_stop = False
    reason = ""
    if best_acc >= 95:
        should_stop = True
        reason = f"목표 정확도 달성 ({best_acc}%). 추가 반복 불필요."
    elif len(valid_acc) >= 3:
        recent_deltas = [d for d in deltas[-2:] if d is not None]
        if recent_deltas and all(abs(d) < 1.0 for d in recent_deltas):
            should_stop = True
            reason = "2회 연속 <1%p 개선. 수렴 정체로 판단."
    elif len(valid_acc) >= 2 and valid_acc[-1] < valid_acc[-2] - 5:
        should_stop = True
        reason = f"정확도 하락 ({valid_acc[-2]}% → {valid_acc[-1]}%). 이전 반복 결과 사용 권장."

    if not should_stop:
        if len(iterations) >= 5:
            should_stop = True
            reason = "최대 반복 횟수(5회) 도달."
        else:
            last_delta = deltas[-1] if deltas[-1] is not None else 0
            reason = f"+{last_delta}%p 개선. 추가 반복 권장." if last_delta > 0 else "추가 반복으로 개선 시도 권장."

    progression = {
        "accuracy_delta": deltas,
        "best_iteration": best_idx,
        "best_accuracy": best_acc,
        "trend": trend,
    }

    stop_rec = {
        "should_stop": should_stop,
        "reason": reason,
    }

    # 요약 생성
    acc_str = " → ".join(f"{a}%" if a is not None else "N/A" for a in accuracies)
    total_delta = round(valid_acc[-1] - valid_acc[0], 2) if len(valid_acc) >= 2 else 0
    summary = f"{len(iterations)}회 반복: {acc_str} (총 {'+' if total_delta >= 0 else ''}{total_delta}%p)"

    result = {
        "iterations": iterations,
        "progression": progression,
        "stop_recommendation": stop_rec,
        "summary": summary,
    }

    if compact:
        return {
            "summary": summary,
            "best_accuracy": best_acc,
            "trend": trend,
            "should_stop": should_stop,
            "reason": reason,
        }
    return result


@mcp.tool()
async def xai(action: str, path: str = "", job_id: str = "",
              job_ids: str = "", columns: str = "", method: str = "auto",
              threshold: float = 0.7, focus: str = "all",
              compact: bool = False) -> dict:
    """
    XAI(설명가능AI) 분석 — 데이터와 모델 결과를 사람이 이해할 수 있게 설명합니다.
    EDA Reasoning Loop: Detection → Reasoning → Narrative → Action → 반복.

    Args:
        action: 수행할 작업
            - "explain": 종합 XAI 분석 (상관관계 + 이상치 + 분포 + 자연어 요약)
            - "correlate": 컬럼 간 상관관계 분석 (Pearson/Spearman, 다중공선성)
            - "outliers": 이상치 탐지 + 영향도 분석
            - "profile": 데이터 프로파일링 (분포, 편향, 결측 패턴, 카디널리티)
            - "trace": 학습 작업 결과 해석 (loss/accuracy 추세, 수렴 판단)
            - "diagnose": 학습 결과 + 데이터 특성 종합 진단 (피처 엔지니어링 추천 포함)
            - "compare": 여러 반복 학습 결과 비교 (정확도 추이, 중단 판단)
        path: 분석 대상 데이터 파일 (워크스페이스 상대경로 가능)
        job_id: trace/diagnose에서 분석할 작업 ID
        job_ids: compare에서 비교할 작업 ID 목록 (쉼표 구분, 시간순)
        columns: 특정 컬럼만 분석 (쉼표 구분, 비우면 전체)
        method: 이상치 탐지 방법 (auto/iqr/zscore)
        threshold: 상관관계 임계값 (기본 0.7)
        focus: 분석 초점 (all/correlation/outlier/distribution/missing)
        compact: True이면 핵심 인사이트만 반환
    """
    try:
        if action in ("explain", "correlate", "outliers", "profile"):
            if not path:
                return {"error": "path를 지정하세요."}
            return await _xai_analyze(path, columns, method, threshold, focus, compact, action)
        elif action == "trace":
            if not job_id:
                return {"error": "job_id를 지정하세요."}
            return await _xai_trace(job_id, compact)
        elif action == "diagnose":
            if not job_id or not path:
                return {"error": "diagnose에는 job_id와 path가 모두 필요합니다."}
            return await _xai_diagnose(job_id, path, columns, threshold, compact)
        elif action == "compare":
            if not job_ids:
                return {"error": "compare에는 job_ids(쉼표 구분)가 필요합니다."}
            return await _xai_compare(job_ids, compact)
        else:
            return {"error": f"Unknown action: {action}. Use: explain, correlate, outliers, profile, trace, diagnose, compare"}
    except Exception as e:
        return {"error": f"xai error: {traceback.format_exc()}"}


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jetson Xavier MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="포트 (기본: 8765)")
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    print(f"🚀 Jetson Xavier MCP Server starting on {args.host}:{args.port}")
    print(f"   Python: {platform.python_version()} | PID: {os.getpid()}")
    print(f"   Tools: 8 groups (system, execute, file, device, job, workspace, data, xai)")
    print(f"   Workspace: {WORKSPACE_ROOT}")
    mcp.run(transport="streamable-http")
