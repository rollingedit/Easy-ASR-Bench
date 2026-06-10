from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib.util
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .dependency_specs import CORE_IMPORTS
from .dependency_specs import parse_requirement_packages
from .ctranslate2_probe import ctranslate2_cuda_available
from .disk_space import format_bytes, require_disk_space


VC_REDIST_PACKAGE_ID = "Microsoft.VCRedist.2015+.x64"
VC_REDIST_REPAIR_COMMAND = f"winget install -e --id {VC_REDIST_PACKAGE_ID} --accept-package-agreements --accept-source-agreements"
VC_REDIST_REGISTRY_KEYS = (
    r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
)

_LLAMA_CPP_DLL_HANDLES: list[object] = []


def prepare_llama_cpp_dll_search_path() -> None:
    if sys.platform != "win32":
        return
    try:
        import site
    except Exception:
        return
    roots = [Path(base) for base in site.getsitepackages() + [site.getusersitepackages()]]
    for root in roots:
        for rel in ("llama_cpp/lib", "torch/lib", "nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
            path = root / rel
            if path.exists() and hasattr(os, "add_dll_directory"):
                try:
                    _LLAMA_CPP_DLL_HANDLES.append(os.add_dll_directory(str(path)))
                except OSError:
                    continue
    for root in roots:
        llama_lib = root / "llama_cpp" / "lib"
        if not llama_lib.exists():
            continue
        for name in ("ggml-base.dll", "ggml-cuda.dll", "ggml-cpu.dll", "ggml.dll", "llama.dll"):
            dll = llama_lib / name
            if dll.exists():
                try:
                    _LLAMA_CPP_DLL_HANDLES.append(ctypes.CDLL(str(dll)))
                except OSError:
                    continue


@dataclass(frozen=True)
class DependencyGroup:
    modules: tuple[str, ...]
    requirement_file: str
    description: str
    install_kind: str = "pip"


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
    "python_packaging": DependencyGroup(
        ("pip", "setuptools", "pkg_resources"),
        "requirements/python_packaging.txt",
        "Python packaging tools needed for dependency repair and native package imports",
    ),
    "core": DependencyGroup(
        tuple(CORE_IMPORTS.values()),
        "requirements/core.txt",
        "base app, media conversion, scoring, and reports",
    ),
    "media_tools": DependencyGroup(
        (),
        "requirements/core.txt",
        "FFmpeg media conversion/probing executable provided by imageio-ffmpeg",
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
        ("faster_whisper", "ctranslate2", "pkg_resources"),
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
        "GGUF text LLM reference/correction support",
    ),
    "llama_mtmd": DependencyGroup(
        (),
        "",
        "GGUF ASR+mmproj native MTMD runtime support",
        install_kind="native_tool",
    ),
}


REQUIREMENT_FILES = {name: group.requirement_file for name, group in DEPENDENCY_GROUPS.items()}


LLAMA_CPP_CUDA_WHEEL_TAGS = ("cu118", "cu121", "cu122", "cu123", "cu124", "cu125", "cu130", "cu132")
LLAMA_CPP_PACKAGE_SPEC = "llama-cpp-python==0.3.28"
LLAMA_CPP_CUDA_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
LLAMA_CPP_CUDA_WHEEL_PROBE_URL = f"{LLAMA_CPP_CUDA_WHEEL_INDEX}/llama-cpp-python/"
LLAMA_CPP_CPU_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
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
    "faster_whisper": (
        ("nvidia.cublas.lib", "nvidia.cublas.bin"),
        ("nvidia.cudnn.lib", "nvidia.cudnn.bin"),
    ),
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

ACCELERATOR_PACKAGE_CONFLICTS = {
    ("onnx", "cuda"): ("onnxruntime", "onnxruntime-directml", "onnxruntime-openvino"),
    ("onnx", "directml"): ("onnxruntime", "onnxruntime-gpu", "onnxruntime-openvino"),
    ("onnx", "openvino"): ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"),
}

ACCELERATOR_REPLACED_REQUIREMENTS = {
    ("onnx", "cuda"): ("onnxruntime",),
    ("onnx", "directml"): ("onnxruntime",),
    ("onnx", "openvino"): ("onnxruntime",),
}

ACCELERATOR_FORCE_REINSTALL = {
    ("onnx", "cuda"),
    ("onnx", "directml"),
    ("onnx", "openvino"),
}

ONNX_PROVIDER_COMPATIBILITY = {
    ("onnx", "cuda"): {
        "package": "onnxruntime-gpu",
        "provider": "CUDAExecutionProvider",
        "config_key": "onnxruntime_cuda_compatibility_versions",
        "versions": ("1.24.0", "1.23.2", "1.22.0", "1.21.1", "1.20.2", "1.19.2", "1.18.1", "1.17.3"),
    },
    ("onnx", "directml"): {
        "package": "onnxruntime-directml",
        "provider": "DmlExecutionProvider",
        "config_key": "onnxruntime_directml_compatibility_versions",
        "versions": ("1.24.4", "1.24.3", "1.24.2", "1.24.1", "1.23.0", "1.22.0", "1.21.1", "1.20.1", "1.19.2", "1.18.1", "1.17.3"),
    },
    ("onnx", "openvino"): {
        "package": "onnxruntime-openvino",
        "provider": "OpenVINOExecutionProvider",
        "config_key": "onnxruntime_openvino_compatibility_versions",
        "versions": ("1.24.1", "1.24.0", "1.23.0", "1.22.0", "1.21.1", "1.20.1", "1.19.2", "1.18.1", "1.17.3"),
    },
}

DEPENDENCY_INSTALL_SPACE_ESTIMATES = {
    ("core", None): 600 * 1024 * 1024,
    ("media_tools", None): 600 * 1024 * 1024,
    ("onnx", None): 1500 * 1024 * 1024,
    ("onnx", "cuda"): 2500 * 1024 * 1024,
    ("onnx", "directml"): 1500 * 1024 * 1024,
    ("onnx", "openvino"): 2500 * 1024 * 1024,
    ("transformers_cpu", None): 6000 * 1024 * 1024,
    ("transformers_cpu", "cuda"): 12000 * 1024 * 1024,
    ("openai_whisper", None): 5000 * 1024 * 1024,
    ("openai_whisper", "cuda"): 12000 * 1024 * 1024,
    ("faster_whisper", None): 1500 * 1024 * 1024,
    ("faster_whisper", "cuda"): 3500 * 1024 * 1024,
    ("llama_cpp", None): 1500 * 1024 * 1024,
    ("llama_cpp", "cpu"): 1500 * 1024 * 1024,
    ("llama_cpp", "cuda"): 3000 * 1024 * 1024,
    ("llama_cpp", "vulkan"): 3000 * 1024 * 1024,
}


def missing_modules(group: str) -> list[str]:
    metadata = DEPENDENCY_GROUPS.get(group)
    if metadata is None:
        return []
    if group == "media_tools":
        status = media_tools_status()
        return [] if status["available"] else list(status["missing"])
    if metadata.install_kind == "native_tool":
        if group == "llama_mtmd" and not gguf_asr_runtime_available():
            return ["llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler"]
        return []
    missing = _missing_import_modules(metadata)
    missing.extend(requirement_version_issues([metadata.requirement_file]))
    return sorted(set(missing))


def _missing_import_modules(metadata: DependencyGroup) -> list[str]:
    return [module for module in metadata.modules if not module_available(module)]


def requirement_version_issues(requirement_files: list[str], ignored_packages: set[str] | None = None) -> list[str]:
    project_root = Path(__file__).resolve().parent.parent
    issues: list[str] = []
    ignored = {package.lower() for package in (ignored_packages or set())}
    for requirement_file in requirement_files:
        path = project_root / requirement_file
        if not path.exists():
            continue
        for requirement in parse_requirement_packages(path.read_text(encoding="utf-8")):
            try:
                from packaging.requirements import Requirement
            except Exception:
                continue
            try:
                parsed = Requirement(requirement.raw)
            except Exception:
                continue
            if parsed.name.lower() in ignored:
                continue
            if not parsed.specifier:
                continue
            try:
                installed = importlib.metadata.version(parsed.name)
            except importlib.metadata.PackageNotFoundError:
                continue
            if installed not in parsed.specifier:
                issues.append(f"{parsed.name}{parsed.specifier} (installed {installed})")
    return issues


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def media_tools_status() -> dict:
    status = {
        "available": False,
        "ffmpeg_path": "",
        "ffprobe_path": "",
        "ffprobe_available": False,
        "probe_method": "ffmpeg_fallback",
        "missing": [],
        "messages": [],
        "repair_command": recovery_command("core"),
    }
    if not module_available("imageio_ffmpeg"):
        return {
            **status,
            "missing": ["imageio_ffmpeg"],
            "messages": ["imageio-ffmpeg is not importable; media conversion cannot be bootstrapped."],
        }
    try:
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        return {
            **status,
            "missing": ["imageio_ffmpeg ffmpeg executable"],
            "messages": [f"imageio-ffmpeg could not resolve an ffmpeg executable: {type(exc).__name__}: {exc}"],
        }
    ffprobe = ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe")
    status.update(
        {
            "available": ffmpeg.exists(),
            "ffmpeg_path": str(ffmpeg),
            "ffprobe_path": str(ffprobe),
            "ffprobe_available": ffprobe.exists(),
        }
    )
    if not ffmpeg.exists():
        status["missing"] = ["ffmpeg executable"]
        status["messages"].append(f"imageio-ffmpeg resolved {ffmpeg}, but the executable does not exist.")
    elif not ffprobe.exists():
        status["messages"].append("ffprobe is not beside ffmpeg; Easy ASR Bench will use FFmpeg stream diagnostics as fallback.")
    return status


def distribution_installed(package: str) -> bool:
    try:
        importlib.metadata.version(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def missing_modules_for_config(group: str, config: dict) -> list[str]:
    metadata = DEPENDENCY_GROUPS.get(group)
    if metadata is None:
        return []
    if group == "media_tools":
        status = media_tools_status()
        return [] if status["available"] else list(status["missing"])
    if metadata.install_kind == "native_tool":
        if group == "llama_mtmd" and not gguf_asr_runtime_available(config):
            return ["llama-mtmd-cli or llama-cpp-python Qwen3ASRChatHandler"]
        return []
    decision = acceleration_install_decision(config, group)
    accelerator = decision.get("accelerator")
    ignored_base_requirements = (
        set(ACCELERATOR_REPLACED_REQUIREMENTS.get((group, str(accelerator)), ()))
        if decision["use_accelerator"]
        else set()
    )
    missing = _missing_import_modules(metadata)
    missing.extend(requirement_version_issues([metadata.requirement_file], ignored_packages=ignored_base_requirements))
    if not decision["use_accelerator"]:
        return missing
    for package in ACCELERATOR_PACKAGE_CONFLICTS.get((group, str(accelerator)), ()):
        if distribution_installed(package):
            missing.append(f"{package} conflicts with {accelerator} package")
    missing.extend(requirement_version_issues(list(decision.get("requirement_files", []))))
    if accelerator == "cuda" and group in {"transformers_cpu", "openai_whisper"} and not torch_cuda_available():
        missing.append("torch CUDA wheel")
    elif accelerator == "cuda" and group == "onnx" and not onnx_provider_available("CUDAExecutionProvider"):
        missing.append("onnxruntime CUDAExecutionProvider")
    elif accelerator == "directml" and group == "onnx" and not onnx_provider_available("DmlExecutionProvider"):
        missing.append("onnxruntime DirectML provider")
    elif (
        accelerator == "openvino"
        and group == "onnx"
        and provider_explicitly_requested(config, "openvino")
        and not onnx_provider_available("OpenVINOExecutionProvider")
    ):
        missing.append("onnxruntime OpenVINO provider")
    elif accelerator == "cuda" and group == "faster_whisper":
        for alternatives in CUDA_REPAIR_MARKERS[group]:
            if not any(module_available(module) for module in alternatives):
                missing.append(" or ".join(alternatives))
        if not ctranslate2_cuda_available():
            missing.append("CTranslate2 CUDA backend")
    elif (
        accelerator in {"cuda", "vulkan"}
        and group == "llama_cpp"
        and (accelerator == "cuda" or provider_explicitly_requested(config, accelerator))
        and not llama_cpp_gpu_capable()
    ):
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
    providers, _message = onnxruntime_available_providers()
    return provider in providers


def _windows_error_mode_prefix() -> str:
    return (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    import ctypes\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
    )


def _run_json_python_probe(code: str, timeout: int = 15) -> tuple[dict, str]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _windows_error_mode_prefix() + code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
        return {}, detail
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1]), ""
    except (json.JSONDecodeError, IndexError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def onnxruntime_available_providers() -> tuple[list[str], str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return [], "onnxruntime is not installed."
    payload, error = _run_json_python_probe(
        """
import json
try:
    import onnxruntime as ort
    print(json.dumps({"ok": True, "providers": list(ort.get_available_providers())}))
except Exception as exc:
    print(json.dumps({"ok": False, "providers": [], "error": str(exc), "error_type": type(exc).__name__}))
"""
    )
    if not payload:
        return [], f"ONNX Runtime provider probe failed in isolated process: {error}"
    providers = list(payload.get("providers", []) or [])
    if payload.get("ok"):
        return providers, ""
    return providers, f"ONNX Runtime provider probe failed: {payload.get('error_type', 'Error')}: {payload.get('error', '')}"


def visual_cpp_redistributable_status(include_winget: bool = False) -> dict:
    status = {
        "installed": False,
        "version": "",
        "source": "",
        "repair_command": VC_REDIST_REPAIR_COMMAND,
        "details": [],
    }
    if sys.platform != "win32":
        return {**status, "source": "not_windows", "details": ["Visual C++ Redistributable probing is only applicable on Windows."]}
    try:
        import winreg
    except Exception as exc:
        status["details"].append(f"winreg unavailable: {type(exc).__name__}: {exc}")
    else:
        for key_path in VC_REDIST_REGISTRY_KEYS:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                    installed = int(winreg.QueryValueEx(key, "Installed")[0])
                    version = str(winreg.QueryValueEx(key, "Version")[0])
            except OSError as exc:
                status["details"].append(f"{key_path}: {type(exc).__name__}: {exc}")
                continue
            if installed:
                return {
                    **status,
                    "installed": True,
                    "version": version,
                    "source": "registry",
                    "details": [f"{key_path}: Installed={installed}; Version={version}"],
                }
    if include_winget:
        try:
            completed = subprocess.run(
                ["winget", "list", "-e", "--id", VC_REDIST_PACKAGE_ID],
                text=True,
                capture_output=True,
                timeout=20,
            )
        except Exception as exc:
            status["details"].append(f"winget list failed: {type(exc).__name__}: {exc}")
        else:
            output = "\n".join(line.rstrip() for line in (completed.stdout + completed.stderr).splitlines() if line.strip())
            status["details"].append(output[-2000:])
            if completed.returncode == 0 and VC_REDIST_PACKAGE_ID.lower() in output.lower():
                version = ""
                for line in output.splitlines():
                    if VC_REDIST_PACKAGE_ID in line:
                        parts = line.split()
                        version = next((part for part in parts if part[:2].isdigit() and "." in part), "")
                        break
                return {**status, "installed": True, "version": version, "source": "winget", "details": status["details"]}
    return status


def _huggingface_cache_dir() -> str:
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return hub_cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home) / "hub")
    try:
        from huggingface_hub import constants

        return str(constants.HF_HUB_CACHE)
    except Exception:
        return str(Path.home() / ".cache" / "huggingface" / "hub")


def _symlink_supported_in_temp() -> tuple[bool | None, str]:
    if sys.platform != "win32":
        return True, "not_windows"
    try:
        with tempfile.TemporaryDirectory(prefix="easy_asr_hf_symlink_probe_") as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.txt"
            link = temp_path / "link.txt"
            source.write_text("probe", encoding="utf-8")
            os.symlink(source, link)
            return link.exists(), "temp_probe"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def huggingface_cache_status() -> dict:
    disabled_warning = os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING", "").strip().lower() in {"1", "true", "yes", "on"}
    symlink_supported, probe_detail = _symlink_supported_in_temp()
    status = {
        "available": module_available("huggingface_hub"),
        "cache_dir": _huggingface_cache_dir(),
        "platform": sys.platform,
        "symlink_supported": symlink_supported,
        "symlink_probe": probe_detail,
        "symlink_warning_disabled": disabled_warning,
        "severity": "ok",
        "messages": [],
        "repair_guidance": "",
    }
    if sys.platform == "win32" and symlink_supported is False and not disabled_warning:
        status["severity"] = "note"
        status["repair_guidance"] = (
            "Hugging Face downloads still work, but the cache may duplicate files and use more disk space. "
            "Enable Windows Developer Mode or run setup from an elevated shell to allow symlinks, or move HF_HOME/HF_HUB_CACHE to a roomy disk."
        )
        status["messages"].append(
            "Hugging Face Hub cache symlinks are not available for this user. This is a cache-space/performance note, not an ASR runtime failure."
        )
    elif sys.platform == "win32" and symlink_supported is False:
        status["severity"] = "note"
        status["messages"].append(
            "Hugging Face Hub cache symlinks are not available, but HF_HUB_DISABLE_SYMLINKS_WARNING suppresses the upstream warning."
        )
    return status


def llama_cpp_cuda_capable() -> bool:
    return llama_cpp_gpu_capable()


def llama_cpp_gpu_capable() -> bool:
    try:
        if importlib.util.find_spec("llama_cpp") is None:
            return False
    except (ModuleNotFoundError, ValueError):
        return False
    code = (
        "import os\n"
        "import ctypes\n"
        "from pathlib import Path\n"
        "_dll_handles = []\n"
        "_preloaded_dlls = []\n"
        "try:\n"
        "    import site\n"
        "    roots = [Path(base) for base in site.getsitepackages() + [site.getusersitepackages()]]\n"
        "    for root in roots:\n"
        "        for rel in ('llama_cpp/lib', 'torch/lib', 'nvidia/cublas/bin', 'nvidia/cudnn/bin', 'nvidia/cuda_nvrtc/bin'):\n"
        "            path = root / rel\n"
        "            if path.exists() and hasattr(os, 'add_dll_directory'):\n"
        "                _dll_handles.append(os.add_dll_directory(str(path)))\n"
        "    for root in roots:\n"
        "        llama_lib = root / 'llama_cpp' / 'lib'\n"
        "        if not llama_lib.exists():\n"
        "            continue\n"
        "        for name in ('ggml-base.dll', 'ggml-cuda.dll', 'ggml-cpu.dll', 'ggml.dll', 'llama.dll'):\n"
        "            dll = llama_lib / name\n"
        "            if dll.exists():\n"
        "                _preloaded_dlls.append(ctypes.CDLL(str(dll)))\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    from llama_cpp import llama_supports_gpu_offload\n"
        "    print('EASY_ASR_LLAMA_GPU_OFFLOAD=' + ('1' if llama_supports_gpu_offload() else '0'))\n"
        "except Exception:\n"
        "    print('EASY_ASR_LLAMA_GPU_OFFLOAD=0')\n"
    )
    try:
        completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
    except Exception:
        return False
    return "EASY_ASR_LLAMA_GPU_OFFLOAD=1" in completed.stdout


LLAMA_MTMD_CLI_NAMES = ("llama-mtmd-cli.exe", "llama-mtmd-cli")
LLAMA_CPP_WINGET_PACKAGE_ID = "ggml.llamacpp"
LLAMA_MTMD_REPAIR_COMMAND = (
    f"winget install -e --id {LLAMA_CPP_WINGET_PACKAGE_ID} "
    "--accept-package-agreements --accept-source-agreements"
)
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def llama_cpp_qwen3_asr_handler_available() -> bool:
    try:
        prepare_llama_cpp_dll_search_path()
        from llama_cpp.llama_chat_format import Qwen3ASRChatHandler  # noqa: F401

        return True
    except Exception:
        return False


def probe_llama_mtmd_cli_path(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return {"ok": False, "path": str(candidate), "reason": "missing"}
    last_error = ""
    for attempt in range(6):
        try:
            completed = subprocess.run(
                [str(candidate), "--help"],
                cwd=candidate.parent,
                text=True,
                capture_output=True,
                timeout=15,
            )
        except PermissionError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 5:
                time.sleep(1.0)
                continue
            return {"ok": False, "path": str(candidate), "reason": "permission_denied", "error": last_error, "attempts": attempt + 1}
        except Exception as exc:
            return {"ok": False, "path": str(candidate), "reason": "probe_failed", "error": f"{type(exc).__name__}: {exc}", "attempts": attempt + 1}
        break
    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    ok = completed.returncode == 0 or "llama" in output.lower() or "usage" in output.lower()
    return {
        "ok": ok,
        "path": str(candidate),
        "reason": "probe_ok" if ok else "probe_exit_nonzero",
        "attempts": attempt + 1,
        "exit_code": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-500:],
        "stderr_tail": (completed.stderr or "")[-500:],
    }


def _llama_mtmd_stage_dir(config: dict | None, source: Path) -> Path:
    root = _native_tools_cache_root(config)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source.parent.name) or "llama_mtmd"
    return root / "llama_mtmd" / safe_name


def _native_tools_cache_root(config: dict | None) -> Path:
    cache_root = ""
    if config:
        folders = config.get("folders", {})
        if isinstance(folders, dict):
            cache_root = str(folders.get("cache") or "")
    root = (Path(cache_root) if cache_root else Path.cwd() / "Cache").resolve()
    return root / "native_tools"


def _stage_llama_mtmd_runtime_copy(path: str | Path, config: dict | None = None) -> dict:
    source = Path(path)
    if not source.exists() or not source.is_file():
        return {"ok": False, "reason": "source_missing", "source_path": str(source)}
    destination_dir = _llama_mtmd_stage_dir(config, source)
    try:
        if source.parent.resolve() == destination_dir.resolve():
            return {"ok": True, "path": str(source), "source_path": str(source), "reason": "already_staged"}
    except OSError:
        pass
    copy_failures: list[dict] = []
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in source.parent.iterdir():
        if not item.is_file():
            continue
        destination = destination_dir / item.name
        try:
            shutil.copy2(item, destination)
        except Exception as exc:
            if not destination.exists():
                copy_failures.append({"path": str(item), "error": f"{type(exc).__name__}: {exc}"})
    staged = destination_dir / source.name
    if not staged.exists():
        return {
            "ok": False,
            "reason": "staged_executable_missing",
            "source_path": str(source),
            "stage_dir": str(destination_dir),
            "copy_failures": copy_failures,
        }
    return {
        "ok": True,
        "path": str(staged),
        "source_path": str(source),
        "stage_dir": str(destination_dir),
        "reason": "staged_copy",
        "copy_failures": copy_failures,
    }


def _llama_cpp_release_asset_priority(name: str, *, prefer_vulkan: bool) -> tuple[int, str]:
    lowered = name.lower()
    if not (lowered.endswith(".zip") and "bin-win" in lowered and "x64" in lowered):
        return (100, lowered)
    if lowered.startswith("cudart-") or "server" in lowered:
        return (90, lowered)
    if prefer_vulkan and "vulkan" in lowered:
        return (0, lowered)
    if "cpu" in lowered:
        return (1, lowered)
    if "avx2" in lowered:
        return (2, lowered)
    if all(marker not in lowered for marker in ("cuda", "hip", "sycl", "kompute")):
        return (3, lowered)
    return (20, lowered)


def _select_llama_cpp_windows_asset(release: dict, *, prefer_vulkan: bool) -> dict | None:
    assets = [asset for asset in release.get("assets", []) if isinstance(asset, dict)]
    candidates = []
    for asset in assets:
        name = str(asset.get("name") or "")
        url = str(asset.get("browser_download_url") or "")
        priority = _llama_cpp_release_asset_priority(name, prefer_vulkan=prefer_vulkan)
        if priority[0] < 20 and url:
            candidates.append((priority, asset))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = destination / member.filename
            try:
                target.resolve().relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip member path: {member.filename}") from exc
        archive.extractall(destination)


def _download_url(url: str, destination: Path, *, timeout_seconds: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Easy-ASR-Bench-native-tool-repair"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _install_llama_mtmd_portable_release(config: dict | None, log_handle=None) -> dict:
    cache_root = _native_tools_cache_root(config) / "llama_mtmd"
    downloads = cache_root / "downloads"
    try:
        request = urllib.request.Request(LLAMA_CPP_RELEASES_API, headers={"User-Agent": "Easy-ASR-Bench-native-tool-repair"})
        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": "release_lookup_failed", "error": f"{type(exc).__name__}: {exc}", "api_url": LLAMA_CPP_RELEASES_API}
    prefer_vulkan = vulkan_detected()
    asset = _select_llama_cpp_windows_asset(release, prefer_vulkan=prefer_vulkan)
    if asset is None:
        return {
            "ok": False,
            "reason": "no_windows_x64_asset",
            "release_tag": str(release.get("tag_name") or ""),
            "asset_names": [str(item.get("name") or "") for item in release.get("assets", []) if isinstance(item, dict)],
        }
    name = str(asset.get("name") or "llama.cpp-windows-x64.zip")
    url = str(asset.get("browser_download_url") or "")
    tag = str(release.get("tag_name") or "latest")
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).stem)
    archive_path = downloads / name
    extract_dir = cache_root / f"github_{safe_tag}_{safe_name}"
    if log_handle is not None:
        log_handle.write(f"\nFalling back to official llama.cpp GitHub release asset: {tag} / {name}\n")
    try:
        if not archive_path.exists():
            _download_url(url, archive_path)
        _safe_extract_zip(archive_path, extract_dir)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "portable_release_install_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "release_tag": tag,
            "asset_name": name,
            "asset_url": url,
            "archive_path": str(archive_path),
            "extract_dir": str(extract_dir),
        }
    cli = next((path for cli_name in LLAMA_MTMD_CLI_NAMES for path in extract_dir.rglob(cli_name) if path.is_file()), None)
    if cli is None:
        return {"ok": False, "reason": "llama_mtmd_cli_missing_in_release_asset", "release_tag": tag, "asset_name": name, "extract_dir": str(extract_dir)}
    probe = probe_llama_mtmd_cli_path(cli)
    return {
        "ok": bool(probe.get("ok")),
        "reason": "portable_release_probe_ok" if probe.get("ok") else "portable_release_probe_failed",
        "path": str(cli),
        "probe": probe,
        "release_tag": tag,
        "asset_name": name,
        "asset_url": url,
        "archive_path": str(archive_path),
        "extract_dir": str(extract_dir),
        "prefer_vulkan": prefer_vulkan,
    }


def _cached_llama_mtmd_status_after_transient_denial(config: dict | None, rejected: list[dict]) -> dict | None:
    denied_paths = []
    for item in rejected:
        if item.get("reason") != "permission_denied":
            continue
        raw_path = str(item.get("path") or "")
        if not raw_path:
            continue
        try:
            denied_paths.append(str(Path(raw_path).resolve()).lower())
        except OSError:
            denied_paths.append(raw_path.lower())
    if not denied_paths:
        return None
    log_dirs: list[Path] = []
    if config:
        folders = config.get("folders", {})
        if isinstance(folders, dict) and folders.get("logs"):
            log_dirs.append(Path(str(folders["logs"])))
    log_dirs.append(Path.cwd() / "Logs")
    log_dirs.append(Path(__file__).resolve().parent.parent / "Logs")
    seen_dirs: set[str] = set()
    for log_dir in log_dirs:
        try:
            resolved_dir = str(log_dir.resolve()).lower()
        except OSError:
            resolved_dir = str(log_dir).lower()
        if resolved_dir in seen_dirs:
            continue
        seen_dirs.add(resolved_dir)
        path = log_dir / "dependency_resolution_llama_mtmd.json"
        if not path.exists():
            continue
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runtime_path = str(saved.get("runtime_path") or "")
        if not runtime_path:
            continue
        try:
            resolved_runtime_path = str(Path(runtime_path).resolve()).lower()
        except OSError:
            resolved_runtime_path = runtime_path.lower()
        if (
            saved.get("schema") == "easy_asr_bench.runtime_resolution.v1"
            and saved.get("dependency_group") == "llama_mtmd"
            and saved.get("backend_verified") is True
            and saved.get("backend_probe_kind") == "llama_mtmd_runtime_probe"
            and Path(runtime_path).exists()
            and Path(runtime_path).is_file()
        ):
            path_was_denied = resolved_runtime_path in denied_paths
            if not path_was_denied:
                alternate_probe = probe_llama_mtmd_cli_path(runtime_path)
                if not alternate_probe.get("ok"):
                    continue
            return {
                "path": runtime_path,
                "source_resolution_path": str(path),
                "resolution": saved,
                "path_was_denied": path_was_denied,
            }
    return None


def llama_mtmd_cli_status(config: dict | None = None) -> dict:
    configured = ""
    if config:
        llama_cpp_config = config.get("llama_cpp", {})
        if isinstance(llama_cpp_config, dict):
            configured = str(llama_cpp_config.get("mtmd_cli_path", "") or "")
    candidates: list[tuple[str, str]] = []
    if configured:
        candidates.append(("config", configured))
    cache_roots = [_native_tools_cache_root(config) / "llama_mtmd"]
    default_cache_root = (Path.cwd() / "Cache" / "native_tools" / "llama_mtmd").resolve()
    if default_cache_root not in cache_roots:
        cache_roots.append(default_cache_root)
    for index, cache_root in enumerate(cache_roots):
        if cache_root.exists():
            source = "native_tools_cache" if index == 0 else "project_native_tools_cache"
            for name in LLAMA_MTMD_CLI_NAMES:
                candidates.extend((source, str(path)) for path in cache_root.glob(f"github*/**/{name}"))
    for name in LLAMA_MTMD_CLI_NAMES:
        found = shutil.which(name)
        if found:
            candidates.append(("path", found))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        packages_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if packages_root.exists():
            for name in LLAMA_MTMD_CLI_NAMES:
                candidates.extend(("winget_packages", str(path)) for path in packages_root.glob(f"ggml.llamacpp*/*{name}"))
    checked: list[str] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for source, candidate in candidates:
        if not candidate:
            continue
        try:
            resolved_candidate = str(Path(candidate).resolve())
        except OSError:
            resolved_candidate = str(candidate)
        if resolved_candidate.lower() in seen:
            continue
        seen.add(resolved_candidate.lower())
        checked.append(candidate)
        path = Path(candidate)
        if path.exists() and path.is_file():
            probe = probe_llama_mtmd_cli_path(path)
            if not probe.get("ok"):
                rejected.append({"source": source, **probe})
                if probe.get("reason") == "permission_denied":
                    staged = _stage_llama_mtmd_runtime_copy(path, config)
                    if staged.get("ok"):
                        staged_probe = probe_llama_mtmd_cli_path(staged["path"])
                        if staged_probe.get("ok"):
                            return {
                                "available": True,
                                "path": str(staged["path"]),
                                "source": "staged_copy_after_permission_denied",
                                "probe": staged_probe,
                                "staged_runtime": staged,
                                "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                                "checked": checked,
                                "rejected": rejected,
                                "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
                            }
                        rejected.append({"source": "staged_copy", **staged_probe, "staged_runtime": staged})
                    else:
                        rejected.append({"source": "staged_copy", "ok": False, **staged})
                continue
            if config and isinstance(config.get("folders"), dict) and config["folders"].get("cache") and source != "config":
                staged = _stage_llama_mtmd_runtime_copy(path, config)
                if staged.get("ok"):
                    staged_probe = probe_llama_mtmd_cli_path(staged["path"])
                    if staged_probe.get("ok"):
                        return {
                            "available": True,
                            "path": str(staged["path"]),
                            "source": "staged_copy",
                            "probe": staged_probe,
                            "source_probe": probe,
                            "staged_runtime": staged,
                            "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                            "checked": checked,
                            "rejected": rejected,
                            "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
                        }
            return {
                "available": True,
                "path": str(path),
                "source": source,
                "probe": probe,
                "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                "checked": checked,
                "rejected": rejected,
                "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
            }
    cached = _cached_llama_mtmd_status_after_transient_denial(config, rejected)
    if cached is not None:
        staged = _stage_llama_mtmd_runtime_copy(cached["path"], config)
        path = cached["path"]
        source = "cached_runtime_resolution_after_permission_denied"
        staged_probe = {}
        if staged.get("ok"):
            staged_probe = probe_llama_mtmd_cli_path(staged["path"])
            if staged_probe.get("ok"):
                path = str(staged["path"])
                source = "staged_cached_runtime_resolution_after_permission_denied"
        if cached.get("path_was_denied") and source != "staged_cached_runtime_resolution_after_permission_denied":
            rejected.append({"source": source, "ok": False, "reason": "cached_runtime_denied_and_staging_failed", "path": path, "staged_runtime": staged})
            return {
                "available": False,
                "path": "",
                "source": "",
                "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                "checked": checked,
                "rejected": rejected,
                "cached_runtime_resolution": cached["resolution"],
                "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
            }
        return {
            "available": True,
            "path": path,
            "source": source,
            "probe": {
                "ok": True,
                "reason": "cached_runtime_resolution_after_transient_permission_denied",
                "source_resolution_path": cached["source_resolution_path"],
                "staged_probe": staged_probe,
            },
            "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
            "checked": checked,
            "rejected": rejected,
            "staged_runtime": staged,
            "cached_runtime_resolution": cached["resolution"],
            "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
        }
    return {
        "available": False,
        "path": "",
        "source": "",
        "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
        "checked": checked,
        "rejected": rejected,
        "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
    }


def gguf_asr_runtime_available(config: dict | None = None) -> bool:
    return llama_cpp_qwen3_asr_handler_available() or bool(llama_mtmd_cli_status(config).get("available"))


def group_available(group: str) -> bool:
    return not missing_modules(group)


def install_group(group: str, project_root: Path, use_cuda: bool = False) -> None:
    if group not in DEPENDENCY_GROUPS:
        raise KeyError(f"Unknown dependency group: {group}")
    if group == "llama_mtmd":
        if not gguf_asr_runtime_available():
            subprocess.check_call(
                [
                    "winget",
                    "install",
                    "-e",
                    "--id",
                    LLAMA_CPP_WINGET_PACKAGE_ID,
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ]
            )
        return
    requirement_files = [DEPENDENCY_GROUPS[group].requirement_file]
    override = CUDA_INSTALL_OVERRIDES.get(group) if use_cuda else None
    if override:
        requirement_files = list(override["requirement_files"])
    for requirement_file in requirement_files:
        req = project_root / requirement_file
        if not req.exists():
            raise FileNotFoundError(f"Missing dependency requirement file: {req}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def _run_dependency_command(command: list[str], env: dict[str, str] | None, log_handle) -> None:
    if log_handle is None:
        subprocess.check_call(command, env=env)
        return
    log_handle.write(f"\n> {' '.join(command)}\n")
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_handle.write(completed.stdout or "")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)


def _llama_cpp_cpu_wheel_pip_args() -> list[str]:
    return ["--upgrade", "--force-reinstall", "--extra-index-url", LLAMA_CPP_CPU_WHEEL_INDEX, LLAMA_CPP_PACKAGE_SPEC]


def _repair_onnx_provider_compatibility(group: str, decision: dict, config: dict, env: dict[str, str] | None, log_handle) -> dict:
    accelerator = str(decision.get("accelerator"))
    metadata = ONNX_PROVIDER_COMPATIBILITY.get((group, accelerator))
    if metadata is None:
        return decision
    provider = str(metadata["provider"])
    providers, provider_error = onnxruntime_available_providers()
    if provider in providers:
        return {**decision, "provider_probe_error": provider_error}
    versions = config.get("dependency_install", {}).get(str(metadata["config_key"])) or list(metadata["versions"])
    errors: list[str] = []
    for version in versions:
        package_spec = f"{metadata['package']}=={version}"
        command = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-deps", package_spec]
        try:
            _run_dependency_command(command, env, log_handle)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{package_spec}: install failed with exit code {exc.returncode}")
            continue
        providers, provider_error = onnxruntime_available_providers()
        if provider in providers:
            _run_dependency_command([sys.executable, "-m", "pip", "install", package_spec], env, log_handle)
            return {
                **decision,
                "provider_compatibility_repair": package_spec,
                "provider_probe_error": "",
            }
        errors.append(f"{package_spec}: {provider_error or f'{provider} not listed in {providers}'}")
    return {
        **decision,
        "provider_compatibility_failed": errors,
        "provider_probe_error": errors[-1] if errors else provider_error,
    }


def _dependency_install_space_estimate(group: str, decision: dict) -> int | None:
    accelerator = decision.get("accelerator") if decision.get("use_accelerator") else None
    if group == "llama_cpp" and decision.get("cpu_wheel_fallback"):
        accelerator = "cpu"
    return DEPENDENCY_INSTALL_SPACE_ESTIMATES.get((group, accelerator)) or DEPENDENCY_INSTALL_SPACE_ESTIMATES.get((group, None))


def _check_dependency_install_disk_space(group: str, project_root: Path, config: dict, decision: dict) -> dict:
    estimate = _dependency_install_space_estimate(group, decision)
    allow_low_space = bool(config.get("dependency_install", {}).get("allow_low_disk_space_install", False))
    check = require_disk_space(project_root, estimate, label=f"{group} dependency install", allow_low_space=allow_low_space)
    return {
        "path": str(check.path),
        "required_bytes": check.required_bytes,
        "required": format_bytes(check.required_bytes),
        "free_bytes": check.free_bytes,
        "free": format_bytes(check.free_bytes),
        "ok": check.ok,
        "reason": check.reason,
        "override": allow_low_space and not check.ok,
    }


def install_group_for_config(group: str, project_root: Path, config: dict, log_path: Path | None = None) -> dict:
    if group == "llama_mtmd":
        log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", newline="\n")
            log_handle.write("Installing native dependency group llama_mtmd\n")
        try:
            if gguf_asr_runtime_available(config):
                return {"use_accelerator": False, "accelerator": None, "native_tool": "llama-mtmd-cli", "already_available": True}
            command = [
                "winget",
                "install",
                "-e",
                "--id",
                LLAMA_CPP_WINGET_PACKAGE_ID,
                "--accept-package-agreements",
                "--accept-source-agreements",
            ]
            winget_error = ""
            try:
                _run_dependency_command(command, None, log_handle)
            except Exception as exc:
                winget_error = f"{type(exc).__name__}: {exc}"
                if log_handle is not None:
                    log_handle.write(f"\nwinget llama.cpp repair failed: {winget_error}\n")
            post_install_status = llama_mtmd_cli_status(config)
            portable_repair = {}
            if not post_install_status.get("available"):
                portable_repair = _install_llama_mtmd_portable_release(config, log_handle=log_handle)
                post_install_status = llama_mtmd_cli_status(config)
                if not post_install_status.get("available") and portable_repair.get("ok"):
                    post_install_status = {
                        "available": True,
                        "path": str(portable_repair.get("path") or ""),
                        "source": "github_portable_release",
                        "probe": portable_repair.get("probe", {}),
                        "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                        "checked": [],
                        "rejected": [],
                        "portable_repair": portable_repair,
                        "qwen3_asr_handler_available": llama_cpp_qwen3_asr_handler_available(),
                    }
            if not post_install_status.get("available") and winget_error:
                raise OSError(winget_error)
            return {
                "use_accelerator": False,
                "accelerator": None,
                "native_tool": "llama-mtmd-cli",
                "repair_command": LLAMA_MTMD_REPAIR_COMMAND,
                "winget_error": winget_error,
                "portable_repair": portable_repair,
                "post_install_status": post_install_status,
            }
        finally:
            if log_handle is not None:
                log_handle.close()
    decision = acceleration_install_decision(config, group)
    requirement_files = [DEPENDENCY_GROUPS[group].requirement_file]
    pip_args: list[str] | None = None
    env = None
    if decision["use_accelerator"]:
        if group == "llama_cpp" and decision["accelerator"] in {"cuda", "vulkan"}:
            if not llama_cpp_wheel_index_available(str(decision.get("extra_index_url", ""))):
                fallback_url = str(decision.get("extra_index_url", ""))
                pip_args = _llama_cpp_cpu_wheel_pip_args()
                requirement_files = []
                decision = {
                    **decision,
                    "use_accelerator": False,
                    "accelerator_fallback": "cpu",
                    "accelerator_fallback_reason": (
                        f"llama-cpp-python {decision['accelerator']} wheel index was unavailable at {fallback_url}; "
                        "installing the prebuilt CPU wheel instead of falling back to a local source build."
                    ),
                    "cpu_wheel_index": LLAMA_CPP_CPU_WHEEL_INDEX,
                    "cpu_wheel_fallback": True,
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
    elif group == "llama_cpp" and sys.version_info < (3, 13):
        pip_args = _llama_cpp_cpu_wheel_pip_args()
        requirement_files = []
        decision = {
            **decision,
            "cpu_wheel_index": LLAMA_CPP_CPU_WHEEL_INDEX,
            "cpu_wheel_fallback": True,
            "reason": f"{decision.get('reason', '')} Using the official prebuilt CPU wheel index instead of a local source build.".strip(),
        }
    disk_space_check = _check_dependency_install_disk_space(group, project_root, config, decision)
    decision = {**decision, "disk_space_check": disk_space_check}
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", newline="\n")
        log_handle.write(f"Installing dependency group {group}\n")
        log_handle.write(
            "Disk space check: "
            f"required={disk_space_check['required']} free={disk_space_check['free']} "
            f"path={disk_space_check['path']} reason={disk_space_check['reason']}"
            + (" override=true" if disk_space_check["override"] else "")
            + "\n"
        )
        if pip_args:
            log_handle.write(f"pip args: {' '.join(pip_args)}\n")
        else:
            log_handle.write(f"Requirement files: {', '.join(requirement_files)}\n")
    try:
        commands: list[list[str]] = []
        for package in ACCELERATOR_PACKAGE_CONFLICTS.get((group, str(decision.get("accelerator"))), ()):
            if distribution_installed(package):
                commands.append([sys.executable, "-m", "pip", "uninstall", "-y", package])
        if pip_args:
            install_command = [sys.executable, "-m", "pip", "install", *pip_args]
            commands.append(install_command)
        else:
            for requirement_file in requirement_files:
                req = project_root / requirement_file
                if not req.exists():
                    raise FileNotFoundError(f"Missing dependency requirement file: {req}")
                install_command = [sys.executable, "-m", "pip", "install", "-r", str(req)]
                commands.append(install_command)
        for command in commands:
            _run_dependency_command(command, env, log_handle)
        if decision["use_accelerator"]:
            decision = _repair_onnx_provider_compatibility(group, decision, config, env, log_handle)
    finally:
        if log_handle is not None:
            log_handle.close()
    return decision


def recovery_command(group: str, use_cuda: bool = False) -> str:
    metadata = DEPENDENCY_GROUPS.get(group)
    if metadata is None:
        return ""
    if metadata.install_kind == "native_tool":
        if group == "llama_mtmd":
            return LLAMA_MTMD_REPAIR_COMMAND
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
        pip_args = ("--upgrade", "--force-reinstall", "--no-deps", "--index-url", index_url, LLAMA_CPP_PACKAGE_SPEC)
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
        pip_args = ("--upgrade", "--force-reinstall", "--no-deps", "--index-url", index_url, LLAMA_CPP_PACKAGE_SPEC)
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
                repair_commands=_requirement_repair_commands(["requirements/llama_cpp_vulkan_source_build.txt"], env),
                pip_args=("-r", "requirements/llama_cpp_vulkan_source_build.txt"),
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
    metadata = DEPENDENCY_GROUPS.get(group)
    if metadata and metadata.install_kind == "native_tool":
        return recovery_command(group)
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


def provider_explicitly_requested(config: dict, provider: str) -> bool:
    runtime = config.get("runtime", {})
    return str(runtime.get("provider", "auto")).lower() == provider


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
            "install_kind": metadata.install_kind,
            "recovery_command": recovery_command(group),
            "cuda_recovery_command": cuda_recovery,
            "accelerator_recovery_command": recovery_command_for_config(group, config) if config is not None else "",
        }
        if group == "llama_mtmd":
            status[group]["runtime_status"] = llama_mtmd_cli_status(config)
        if group == "media_tools":
            status[group]["runtime_status"] = media_tools_status()
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
        "visual_cpp_redistributable": visual_cpp_redistributable_status(),
        "huggingface_cache": huggingface_cache_status(),
        "ctranslate2_cuda_available": ctranslate2_cuda_available(),
        "llama_mtmd_cli": llama_mtmd_cli_status(),
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
        providers, provider_error = onnxruntime_available_providers()
        diagnostics["onnxruntime_providers"] = providers
        diagnostics["onnx_cuda_available"] = "CUDAExecutionProvider" in providers
        diagnostics["onnx_directml_available"] = "DmlExecutionProvider" in providers
        diagnostics["onnx_openvino_available"] = "OpenVINOExecutionProvider" in providers
        if provider_error:
            diagnostics["messages"].append(provider_error)
    if diagnostics["torch_installed"] and not diagnostics["torch_cuda_available"]:
        diagnostics["messages"].append("Torch is installed, but Torch CUDA is not available. Transformers models will run on CPU.")
    if diagnostics["onnxruntime_installed"] and not diagnostics["onnx_cuda_available"]:
        diagnostics["messages"].append("ONNX Runtime is installed, but CUDAExecutionProvider is not available. ONNX models will run on CPUExecutionProvider.")
    if diagnostics["vulkan_detected"] and not diagnostics["vulkan_sdk_detected"]:
        diagnostics["messages"].append("Vulkan runtime is visible. GGUF Vulkan setup will try the prebuilt llama-cpp-python Vulkan wheel first; local source builds require the Vulkan SDK.")
    if sys.platform == "win32" and not diagnostics["visual_cpp_redistributable"]["installed"]:
        diagnostics["messages"].append("Microsoft Visual C++ 2015-2022 Redistributable x64 was not detected. Native ASR backends such as CTranslate2, ONNX Runtime, and llama-cpp-python may fail to import until it is installed.")
    for message in diagnostics.get("huggingface_cache", {}).get("messages", []):
        diagnostics["messages"].append(message)
    if module_available("ctranslate2") and not diagnostics["ctranslate2_cuda_available"]:
        diagnostics["messages"].append("CTranslate2 is installed, but its CUDA backend is not available. faster-whisper will run on CPU.")
    return diagnostics
