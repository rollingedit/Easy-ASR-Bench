from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .dependency_specs import CORE_IMPORTS
from .ctranslate2_probe import ctranslate2_cuda_available


@dataclass(frozen=True)
class DependencyGroup:
    modules: tuple[str, ...]
    requirement_file: str
    description: str


@dataclass(frozen=True)
class LlamaCppWheelDecision:
    accelerator: str
    extra_index_url: str
    reason: str
    supported: bool
    repair_commands: dict[str, str]
    pip_args: tuple[str, ...] = ()
    env: dict[str, str] | None = None


DEPENDENCY_GROUPS = {
    "core": DependencyGroup(
        tuple(CORE_IMPORTS.values()),
        "requirements/core.txt",
        "base app, media conversion, scoring, and reports",
    ),
    "onnx": DependencyGroup(
        ("onnxruntime", "tokenizers", "jinja2"),
        "requirements/onnx.txt",
        "ONNX ASR model support using CPU-safe defaults",
    ),
    "transformers_cpu": DependencyGroup(
        ("torch", "transformers", "safetensors", "sentencepiece", "google.protobuf", "torchaudio"),
        "requirements/transformers_cpu.txt",
        "Hugging Face safetensors ASR support with common audio/tokenizer extras",
    ),
    "faster_whisper": DependencyGroup(
        ("faster_whisper", "ctranslate2"),
        "requirements/faster_whisper.txt",
        "faster-whisper / CTranslate2 ASR support",
    ),
    "openai_whisper": DependencyGroup(
        ("whisper", "torch"),
        "requirements/openai_whisper.txt",
        "OpenAI Whisper .pt support when explicitly trusted or allowlisted",
    ),
    "whisper_cpp": DependencyGroup(
        ("pywhispercpp",),
        "requirements/whisper_cpp.txt",
        "whisper.cpp GGML model support",
    ),
    "llama_cpp": DependencyGroup(
        ("llama_cpp",),
        "requirements/llama_cpp.txt",
        "GGUF ASR+mmproj and text LLM reference/correction support",
    ),
}


REQUIREMENT_FILES = {name: group.requirement_file for name, group in DEPENDENCY_GROUPS.items()}


LLAMA_CPP_CUDA_WHEEL_TAGS = ("cu118", "cu121", "cu122", "cu123", "cu124", "cu125", "cu130", "cu132")
LLAMA_CPP_CUDA_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
LLAMA_CPP_CUDA_WHEEL_PROBE_URL = f"{LLAMA_CPP_CUDA_WHEEL_INDEX}/llama-cpp-python/"
LLAMA_CPP_VULKAN_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/vulkan"


CUDA_INSTALL_OVERRIDES = {
    "onnx": {
        "requirement_files": ["requirements/onnx_cuda.txt"],
        "description": "ONNX Runtime GPU package for CUDA 12.x",
    },
    "faster_whisper": {
        "requirement_files": ["requirements/faster_whisper_cuda.txt"],
        "description": "faster-whisper/CTranslate2 plus NVIDIA CUDA 12 cuBLAS/cuDNN Python wheels",
    },
    "llama_cpp": {
        "requirement_files": ["requirements/llama_cpp_cuda_cu124.txt"],
        "description": "llama-cpp-python CUDA prebuilt wheel for GGUF reference LLMs",
    },
    "openai_whisper": {
        "requirement_files": ["requirements/torch_cuda_cu128.txt", "requirements/openai_whisper.txt"],
        "description": "PyTorch CUDA 12.8 wheel plus OpenAI Whisper package",
    },
    "transformers_cpu": {
        "requirement_files": ["requirements/torch_cuda_cu128.txt", "requirements/transformers_cpu.txt"],
        "description": "PyTorch CUDA 12.8 wheel plus Transformers ASR packages",
    },
}

CUDA_REPAIR_MARKERS = {
    "faster_whisper": ("nvidia.cublas.lib", "nvidia.cudnn.lib"),
}

ACCELERATOR_OVERRIDES = {
    ("onnx", "cuda"): CUDA_INSTALL_OVERRIDES["onnx"],
    ("onnx", "directml"): {
        "requirement_files": ["requirements/onnx_directml.txt"],
        "description": "ONNX Runtime DirectML package for Windows GPU acceleration",
    },
    ("onnx", "openvino"): {
        "requirement_files": ["requirements/onnx_openvino.txt"],
        "description": "ONNX Runtime OpenVINO package for Intel CPU/iGPU/NPU acceleration",
    },
    ("transformers_cpu", "cuda"): CUDA_INSTALL_OVERRIDES["transformers_cpu"],
    ("openai_whisper", "cuda"): CUDA_INSTALL_OVERRIDES["openai_whisper"],
    ("faster_whisper", "cuda"): CUDA_INSTALL_OVERRIDES["faster_whisper"],
    ("llama_cpp", "cuda"): CUDA_INSTALL_OVERRIDES["llama_cpp"],
    ("llama_cpp", "vulkan"): {
        "requirement_files": ["requirements/llama_cpp_vulkan.txt"],
        "description": "llama-cpp-python Vulkan prebuilt wheel for GGUF reference LLMs",
    },
}


def missing_modules(group: str) -> list[str]:
    metadata = DEPENDENCY_GROUPS.get(group)
    if metadata is None:
        return []
    return [module for module in metadata.modules if not module_available(module)]


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def missing_modules_for_config(group: str, config: dict) -> list[str]:
    missing = missing_modules(group)
    decision = acceleration_install_decision(config, group)
    accelerator = decision.get("accelerator")
    if not decision["use_accelerator"]:
        return missing
    if accelerator == "cuda" and group in {"transformers_cpu", "openai_whisper"} and not torch_cuda_available():
        missing.append("torch CUDA wheel")
    elif accelerator == "cuda" and group == "onnx" and not onnx_provider_available("CUDAExecutionProvider"):
        missing.append("onnxruntime CUDAExecutionProvider")
    elif accelerator == "directml" and group == "onnx" and not onnx_provider_available("DmlExecutionProvider"):
        missing.append("onnxruntime DirectML provider")
    elif accelerator == "openvino" and group == "onnx" and not onnx_provider_available("OpenVINOExecutionProvider"):
        missing.append("onnxruntime OpenVINO provider")
    elif accelerator == "cuda" and group == "faster_whisper":
        for module in CUDA_REPAIR_MARKERS[group]:
            if not module_available(module):
                missing.append(module)
        if not ctranslate2_cuda_available():
            missing.append("CTranslate2 CUDA backend")
    elif accelerator in {"cuda", "vulkan"} and group == "llama_cpp" and not llama_cpp_gpu_capable():
        missing.append("llama-cpp-python GPU offload build")
    return sorted(set(missing))


def torch_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def onnx_cuda_provider_available() -> bool:
    return onnx_provider_available("CUDAExecutionProvider")


def onnx_provider_available(provider: str) -> bool:
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort

        return provider in list(ort.get_available_providers())
    except Exception:
        return False


def llama_cpp_cuda_capable() -> bool:
    return llama_cpp_gpu_capable()


def llama_cpp_gpu_capable() -> bool:
    try:
        if importlib.util.find_spec("llama_cpp") is None:
            return False
    except (ModuleNotFoundError, ValueError):
        return False
    try:
        from llama_cpp import llama_supports_gpu_offload

        return bool(llama_supports_gpu_offload())
    except Exception:
        return False


def group_available(group: str) -> bool:
    return not missing_modules(group)


def install_group(group: str, project_root: Path, use_cuda: bool = False) -> None:
    if group not in DEPENDENCY_GROUPS:
        raise KeyError(f"Unknown dependency group: {group}")
    requirement_files = [DEPENDENCY_GROUPS[group].requirement_file]
    override = CUDA_INSTALL_OVERRIDES.get(group) if use_cuda else None
    if override:
        requirement_files = list(override["requirement_files"])
    for requirement_file in requirement_files:
        req = project_root / requirement_file
        if not req.exists():
            raise FileNotFoundError(f"Missing dependency requirement file: {req}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def install_group_for_config(group: str, project_root: Path, config: dict, log_path: Path | None = None) -> dict:
    decision = acceleration_install_decision(config, group)
    requirement_files = [DEPENDENCY_GROUPS[group].requirement_file]
    pip_args: list[str] | None = None
    env = None
    if decision["use_accelerator"]:
        if group == "llama_cpp" and decision["accelerator"] in {"cuda", "vulkan"}:
            if not llama_cpp_wheel_index_available(str(decision.get("extra_index_url", ""))):
                fallback_url = str(decision.get("extra_index_url", ""))
                requirement_files = [DEPENDENCY_GROUPS[group].requirement_file]
                decision = {
                    **decision,
                    "use_accelerator": False,
                    "accelerator_fallback": "cpu",
                    "accelerator_fallback_reason": (
                        f"llama-cpp-python {decision['accelerator']} wheel index was unavailable at {fallback_url}; "
                        "installing the CPU package instead of falling back to a local source build."
                    ),
                    "requirement_files": requirement_files,
                }
            else:
                pip_args = list(decision.get("pip_args", []))
                env = {**os.environ, **decision.get("env", {})} if decision.get("env") else None
                requirement_files = []
        else:
            override = ACCELERATOR_OVERRIDES[(group, decision["accelerator"])]
            requirement_files = list(override["requirement_files"])
            env = {**os.environ, **override.get("env", {})}
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", newline="\n")
        log_handle.write(f"Installing dependency group {group}\n")
        if pip_args:
            log_handle.write(f"pip args: {' '.join(pip_args)}\n")
        else:
            log_handle.write(f"Requirement files: {', '.join(requirement_files)}\n")
    try:
        commands: list[list[str]] = []
        if pip_args:
            commands.append([sys.executable, "-m", "pip", "install", *pip_args])
        else:
            for requirement_file in requirement_files:
                req = project_root / requirement_file
                if not req.exists():
                    raise FileNotFoundError(f"Missing dependency requirement file: {req}")
                commands.append([sys.executable, "-m", "pip", "install", "-r", str(req)])
        for command in commands:
            if log_handle is None:
                subprocess.check_call(command, env=env)
            else:
                log_handle.write(f"\n> {' '.join(command)}\n")
                completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log_handle.write(completed.stdout or "")
                if completed.returncode != 0:
                    raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)
    finally:
        if log_handle is not None:
            log_handle.close()
    return decision


def recovery_command(group: str, use_cuda: bool = False) -> str:
    metadata = DEPENDENCY_GROUPS.get(group)
    if metadata is None:
        return ""
    requirement_files = [metadata.requirement_file]
    if use_cuda and group in CUDA_INSTALL_OVERRIDES:
        requirement_files = list(CUDA_INSTALL_OVERRIDES[group]["requirement_files"])
    return " && ".join(f'"{sys.executable}" -m pip install -r {requirement_file}' for requirement_file in requirement_files)


def _pip_repair_commands(pip_args: tuple[str, ...], env: dict[str, str] | None = None) -> dict[str, str]:
    quoted_args = " ".join(pip_args)
    base = f'"{sys.executable}" -m pip install {quoted_args}'.strip()
    if not env:
        return {"powershell": f'& {base}', "cmd": base}
    ps_prefix = "; ".join(f'$env:{key}="{value}"' for key, value in env.items())
    cmd_prefix = " && ".join(f'set "{key}={value}"' for key, value in env.items())
    return {
        "powershell": f"{ps_prefix}; & {base}",
        "cmd": f"{cmd_prefix} && {base}",
    }


def _requirement_repair_commands(requirement_files: list[str], env: dict[str, str] | None = None) -> dict[str, str]:
    pip_args = tuple(part for requirement_file in requirement_files for part in ("-r", requirement_file))
    return _pip_repair_commands(pip_args, env)


def nvidia_driver_version() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""


def _driver_major(driver_version: str) -> int:
    try:
        return int(str(driver_version).split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def llama_cpp_cuda_tag_for_driver(driver_version: str) -> str:
    major = _driver_major(driver_version)
    if major >= 580:
        return "cu132"
    if major >= 575:
        return "cu130"
    if major >= 555:
        return "cu125"
    if major >= 550:
        return "cu124"
    if major >= 545:
        return "cu123"
    if major >= 535:
        return "cu122"
    if major >= 530:
        return "cu121"
    return "cu118"


def resolve_llama_cpp_wheel(config: dict, accelerator: str) -> LlamaCppWheelDecision:
    if sys.version_info >= (3, 13):
        return LlamaCppWheelDecision(
            accelerator=accelerator,
            extra_index_url="",
            reason="llama-cpp-python prebuilt accelerator wheels are not assumed safe for Python 3.13+ in this installer; CPU package is safer.",
            supported=False,
            repair_commands=_requirement_repair_commands([DEPENDENCY_GROUPS["llama_cpp"].requirement_file]),
        )
    if accelerator == "cuda":
        driver = nvidia_driver_version()
        tag = str(config.get("dependency_install", {}).get("llama_cpp_cuda_tag") or llama_cpp_cuda_tag_for_driver(driver))
        if tag not in LLAMA_CPP_CUDA_WHEEL_TAGS:
            tag = "cu124"
        index_url = f"https://abetlen.github.io/llama-cpp-python/whl/{tag}"
        pip_args = ("--extra-index-url", index_url, "llama-cpp-python")
        return LlamaCppWheelDecision(
            accelerator="cuda",
            extra_index_url=index_url,
            reason=f"llama-cpp-python CUDA prebuilt wheel index selected as {tag}" + (f" from NVIDIA driver {driver}" if driver else "."),
            supported=True,
            repair_commands=_pip_repair_commands(pip_args),
            pip_args=pip_args,
        )
    if accelerator == "vulkan":
        index_url = LLAMA_CPP_VULKAN_WHEEL_INDEX
        pip_args = ("--extra-index-url", index_url, "llama-cpp-python")
        if vulkan_detected():
            return LlamaCppWheelDecision(
                accelerator="vulkan",
                extra_index_url=index_url,
                reason="Vulkan runtime detected; trying the llama-cpp-python Vulkan prebuilt wheel before any source build.",
                supported=True,
                repair_commands=_pip_repair_commands(pip_args),
                pip_args=pip_args,
            )
        if config.get("dependency_install", {}).get("allow_vulkan_source_build", False) and vulkan_build_tooling_detected():
            env = {"CMAKE_ARGS": "-DGGML_VULKAN=on", "FORCE_CMAKE": "1"}
            return LlamaCppWheelDecision(
                accelerator="vulkan",
                extra_index_url="",
                reason="Vulkan source build explicitly enabled and Vulkan SDK/build tools were detected.",
                supported=True,
                repair_commands=_requirement_repair_commands(["requirements/llama_cpp_vulkan.txt"], env),
                pip_args=("-r", "requirements/llama_cpp_vulkan.txt"),
                env=env,
            )
        return LlamaCppWheelDecision(
            accelerator="vulkan",
            extra_index_url=index_url,
            reason="Vulkan runtime was not detected; CPU package is safer.",
            supported=False,
            repair_commands=_requirement_repair_commands([DEPENDENCY_GROUPS["llama_cpp"].requirement_file]),
        )
    return LlamaCppWheelDecision(
        accelerator=accelerator,
        extra_index_url="",
        reason=f"No llama-cpp-python wheel resolver exists for {accelerator}.",
        supported=False,
        repair_commands=_requirement_repair_commands([DEPENDENCY_GROUPS["llama_cpp"].requirement_file]),
    )


def llama_cpp_cuda_wheel_index_available(timeout_seconds: int = 10) -> bool:
    return llama_cpp_wheel_index_available(LLAMA_CPP_CUDA_WHEEL_INDEX, timeout_seconds)


def llama_cpp_wheel_index_available(index_url: str, timeout_seconds: int = 10) -> bool:
    if not index_url:
        return True
    probe_url = f"{index_url.rstrip('/')}/llama-cpp-python/"
    try:
        request = urllib.request.Request(probe_url, headers={"User-Agent": "Easy-ASR-Bench-dependency-probe"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            return 200 <= status < 400
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return False


def recovery_command_for_config(group: str, config: dict) -> str:
    decision = acceleration_install_decision(config, group)
    if not decision["use_accelerator"]:
        return recovery_command(group)
    if group == "llama_cpp" and decision["accelerator"] in {"cuda", "vulkan"}:
        commands = decision.get("repair_commands") or {}
        if commands:
            return f"PowerShell: {commands.get('powershell', '')}\ncmd.exe: {commands.get('cmd', '')}"
    override = ACCELERATOR_OVERRIDES[(group, decision["accelerator"])]
    commands = _requirement_repair_commands(override["requirement_files"], override.get("env"))
    return commands["cmd"]


def nvidia_gpu_detected() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run([nvidia_smi, "-L"], text=True, capture_output=True, timeout=10)
    except Exception:
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def windows_gpu_detected() -> bool:
    if sys.platform != "win32":
        return False
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return False
    if completed.returncode != 0:
        return False
    names = [line.strip().lower() for line in completed.stdout.splitlines() if line.strip()]
    return any("microsoft basic" not in name and "remote" not in name for name in names)


def amd_gpu_detected() -> bool:
    if sys.platform != "win32":
        return False
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return False
    output = completed.stdout.lower()
    return completed.returncode == 0 and any(marker in output for marker in ["amd", "radeon", "rx ", "ryzen ai"])


def intel_gpu_or_npu_detected() -> bool:
    if sys.platform != "win32":
        return False
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return False
    return completed.returncode == 0 and "intel" in completed.stdout.lower()


def vulkan_detected() -> bool:
    vulkaninfo = shutil.which("vulkaninfo")
    if not vulkaninfo:
        return False
    try:
        completed = subprocess.run([vulkaninfo, "--summary"], text=True, capture_output=True, timeout=10)
    except Exception:
        return False
    return completed.returncode == 0


def vulkan_sdk_detected() -> bool:
    if os.environ.get("VULKAN_SDK"):
        return True
    return shutil.which("glslc") is not None


def vulkan_build_tooling_detected() -> bool:
    return vulkan_detected() and vulkan_sdk_detected()


def cuda_requested(config: dict) -> bool:
    runtime = config.get("runtime", {})
    provider = str(runtime.get("provider", "auto")).lower()
    return provider in {"auto", "cuda"} or bool(runtime.get("prefer_gpu", True))


def cuda_install_decision(config: dict, group: str) -> dict:
    decision = acceleration_install_decision(config, group)
    return {
        "use_cuda": decision["use_accelerator"] and decision.get("accelerator") == "cuda",
        "reason": decision["reason"],
        "requirement_files": decision.get("requirement_files", []),
    }


def acceleration_install_decision(config: dict, group: str) -> dict:
    accelerator = preferred_accelerator(config, group)
    if accelerator is None:
        return {"use_accelerator": False, "accelerator": None, "reason": "No supported accelerator was detected for this dependency group."}
    key = (group, accelerator)
    if key not in ACCELERATOR_OVERRIDES:
        return {"use_accelerator": False, "accelerator": accelerator, "reason": f"No {accelerator} package set exists for this dependency group."}
    if accelerator == "cuda":
        if not cuda_requested(config):
            return {"use_accelerator": False, "accelerator": accelerator, "reason": "CUDA is not requested or preferred in config.json."}
        if not config.get("dependency_install", {}).get("allow_cuda_install", False):
            return {"use_accelerator": False, "accelerator": accelerator, "reason": "CUDA package installation is disabled by dependency_install.allow_cuda_install."}
        if not nvidia_gpu_detected():
            return {"use_accelerator": False, "accelerator": accelerator, "reason": "No NVIDIA GPU was detected with nvidia-smi."}
    if group == "llama_cpp" and accelerator in {"cuda", "vulkan"}:
        wheel = resolve_llama_cpp_wheel(config, accelerator)
        if not wheel.supported:
            return {"use_accelerator": False, "accelerator": accelerator, "reason": wheel.reason}
        if not config.get("dependency_install", {}).get("allow_accelerator_install", True):
            return {"use_accelerator": False, "accelerator": accelerator, "reason": "Accelerator package installation is disabled."}
        return {
            "use_accelerator": True,
            "accelerator": accelerator,
            "reason": wheel.reason,
            "requirement_files": list(ACCELERATOR_OVERRIDES[key]["requirement_files"]),
            "extra_index_url": wheel.extra_index_url,
            "pip_args": list(wheel.pip_args),
            "repair_commands": wheel.repair_commands,
            "env": wheel.env or {},
        }
    if not config.get("dependency_install", {}).get("allow_accelerator_install", True):
        return {"use_accelerator": False, "accelerator": accelerator, "reason": "Accelerator package installation is disabled."}
    override = ACCELERATOR_OVERRIDES[key]
    return {
        "use_accelerator": True,
        "accelerator": accelerator,
        "reason": override["description"],
        "requirement_files": override["requirement_files"],
    }


def preferred_accelerator(config: dict, group: str) -> str | None:
    runtime = config.get("runtime", {})
    provider = str(runtime.get("provider", "auto")).lower()
    if provider in {"cuda", "directml", "openvino", "vulkan"}:
        return provider
    if nvidia_gpu_detected():
        return "cuda"
    if group == "onnx":
        if intel_gpu_or_npu_detected():
            return "openvino"
        if windows_gpu_detected():
            return "directml"
    if group == "llama_cpp" and vulkan_detected():
        return "vulkan"
    return None


def _legacy_cuda_install_decision(config: dict, group: str) -> dict:
    if group not in CUDA_INSTALL_OVERRIDES:
        return {"use_cuda": False, "reason": "No CUDA-specific package set exists for this dependency group."}
    if not cuda_requested(config):
        return {"use_cuda": False, "reason": "CUDA is not requested or preferred in config.json."}
    if not config.get("dependency_install", {}).get("allow_cuda_install", False):
        return {"use_cuda": False, "reason": "CUDA package installation is disabled by dependency_install.allow_cuda_install."}
    if not nvidia_gpu_detected():
        return {"use_cuda": False, "reason": "No NVIDIA GPU was detected with nvidia-smi."}
    return {
        "use_cuda": True,
        "reason": CUDA_INSTALL_OVERRIDES[group]["description"],
        "requirement_files": CUDA_INSTALL_OVERRIDES[group]["requirement_files"],
    }


def dependency_status(config: dict | None = None) -> dict[str, dict]:
    status = {}
    for group, metadata in DEPENDENCY_GROUPS.items():
        missing = missing_modules_for_config(group, config) if config is not None else missing_modules(group)
        cuda_recovery = recovery_command(group, use_cuda=True) if group in CUDA_INSTALL_OVERRIDES else ""
        if group == "llama_cpp":
            cuda_recovery = "Resolved dynamically from NVIDIA driver/Python runtime; use accelerator_recovery_command from setup.bat --doctor."
        status[group] = {
            "available": not missing,
            "missing": missing,
            "description": metadata.description,
            "requirement_file": metadata.requirement_file,
            "recovery_command": recovery_command(group),
            "cuda_recovery_command": cuda_recovery,
            "accelerator_recovery_command": recovery_command_for_config(group, config) if config is not None else "",
        }
    return status


def cuda_diagnostics() -> dict:
    torch_installed = module_available("torch")
    onnxruntime_installed = module_available("onnxruntime")
    diagnostics = {
        "torch_installed": torch_installed,
        "torch_cuda_available": False,
        "torch_cuda_version": None,
        "torch_gpu_names": [],
        "onnxruntime_installed": onnxruntime_installed,
        "onnxruntime_providers": [],
        "onnx_cuda_available": False,
        "onnx_directml_available": False,
        "onnx_openvino_available": False,
        "messages": [],
        "nvidia_gpu_detected": nvidia_gpu_detected(),
        "amd_gpu_detected": amd_gpu_detected(),
        "intel_gpu_or_npu_detected": intel_gpu_or_npu_detected(),
        "windows_gpu_detected": windows_gpu_detected(),
        "vulkan_detected": vulkan_detected(),
        "vulkan_sdk_detected": vulkan_sdk_detected(),
        "vulkan_build_tooling_detected": vulkan_build_tooling_detected(),
        "ctranslate2_cuda_available": ctranslate2_cuda_available(),
    }
    if diagnostics["torch_installed"]:
        try:
            import torch

            diagnostics["torch_cuda_available"] = bool(torch.cuda.is_available())
            diagnostics["torch_cuda_version"] = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                diagnostics["torch_gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except Exception as exc:  # pragma: no cover - defensive around third-party import failures.
            diagnostics["messages"].append(f"Torch CUDA check failed: {exc}")
    if diagnostics["onnxruntime_installed"]:
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
            diagnostics["onnxruntime_providers"] = providers
            diagnostics["onnx_cuda_available"] = "CUDAExecutionProvider" in providers
            diagnostics["onnx_directml_available"] = "DmlExecutionProvider" in providers
            diagnostics["onnx_openvino_available"] = "OpenVINOExecutionProvider" in providers
        except Exception as exc:  # pragma: no cover - defensive around third-party import failures.
            diagnostics["messages"].append(f"ONNX Runtime provider check failed: {exc}")
    if diagnostics["torch_installed"] and not diagnostics["torch_cuda_available"]:
        diagnostics["messages"].append("Torch is installed, but Torch CUDA is not available. Transformers models will run on CPU.")
    if diagnostics["onnxruntime_installed"] and not diagnostics["onnx_cuda_available"]:
        diagnostics["messages"].append("ONNX Runtime is installed, but CUDAExecutionProvider is not available. ONNX models will run on CPUExecutionProvider.")
    if diagnostics["vulkan_detected"] and not diagnostics["vulkan_sdk_detected"]:
        diagnostics["messages"].append("Vulkan runtime is visible. GGUF Vulkan setup will try the prebuilt llama-cpp-python Vulkan wheel first; local source builds require the Vulkan SDK.")
    if module_available("ctranslate2") and not diagnostics["ctranslate2_cuda_available"]:
        diagnostics["messages"].append("CTranslate2 is installed, but its CUDA backend is not available. faster-whisper will run on CPU.")
    return diagnostics
