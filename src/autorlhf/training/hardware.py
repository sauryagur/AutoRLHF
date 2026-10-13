"""Hardware preflight.

``torch`` is imported lazily inside the functions so that ``import
autorlhf.training.jobs`` (and therefore the TUI) stays fast.  Nothing here ever
raises because an optional dependency is missing: the result is degraded and the
reason is reported in :attr:`HardwareInfo.warnings`.

Measured versus estimated: every value read from the OS or from ``torch`` is a
measurement and is documented as such; per-run memory estimates produced by
:func:`estimate_resources` are modelled, and the caveats are listed in
:attr:`HardwareInfo.estimate_notes`.
"""

from __future__ import annotations

import importlib
import os
import platform as platform_module
import sys
from typing import Any

from autorlhf.config.presets import estimate_training_memory
from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.types import HardwareInfo, ResourceEstimate

#: Libraries whose versions are recorded in the manifest and shown in the TUI.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "wandb",
)

_MIB = 1024 * 1024


# --------------------------------------------------------------------------- #
# system memory
# --------------------------------------------------------------------------- #


def _ram_from_proc_meminfo() -> tuple[int | None, int | None]:
    """``(total_mib, available_mib)`` from ``/proc/meminfo``, or ``(None, None)``."""

    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            values: dict[str, int] = {}
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        values[key.strip()] = int(parts[0])
                    except ValueError:
                        continue
    except OSError:
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    return (
        None if total is None else total // 1024,
        None if available is None else available // 1024,
    )


def _ram_from_sysconf() -> tuple[int | None, int | None]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None, None
    if pages <= 0 or page_size <= 0:
        return None, None
    total_mib = (pages * page_size) // _MIB
    return total_mib, None


def system_memory_mib() -> tuple[int | None, int | None]:
    """Measured ``(total_mib, available_mib)`` host RAM; ``None`` when unknown."""

    if sys.platform.startswith("linux"):
        total, available = _ram_from_proc_meminfo()
        if total is not None:
            return total, available
    total, available = _ram_from_sysconf()
    if total is not None:
        return total, available
    try:  # psutil is optional and never a dependency of the training engine
        psutil = importlib.import_module("psutil")
        memory = psutil.virtual_memory()
        return int(memory.total // _MIB), int(memory.available // _MIB)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# accelerators
# --------------------------------------------------------------------------- #


def _device_type(torch: Any) -> str:
    if getattr(torch.version, "hip", None):
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
        return "mps"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and getattr(xpu, "is_available", lambda: False)():
        return "xpu"
    return "cpu"


def _preferred_device_name(torch: Any, device_type: str) -> tuple[str, int]:
    """``(name, count)`` for the detected device type."""

    if device_type in ("cuda", "rocm"):
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else ""
        if device_type == "rocm" and name:
            name = f"{name} (ROCm)"
        return name, count
    if device_type == "mps":
        return "Apple Metal (MPS)", 1
    if device_type == "xpu":
        xpu = torch.xpu
        count = xpu.device_count()
        name = xpu.get_device_name(0) if count else ""
        return name, count
    return platform_module.machine() or "cpu", 0


def _vram_mib(torch: Any, device_type: str) -> tuple[int | None, int | None]:
    """Measured ``(total_mib, free_mib)``, summed over devices."""

    if device_type not in ("cuda", "rocm"):
        return None, None
    try:
        count = torch.cuda.device_count()
    except Exception:  # pragma: no cover - defensive
        return None, None
    if not count:
        return None, None
    totals = 0
    frees = 0
    for index in range(count):
        try:
            free, total = torch.cuda.mem_get_info(index)
        except Exception:
            return None, None
        totals += int(total)
        frees += int(free)
    return totals // _MIB, frees // _MIB


def _bitsandbytes_status(
    torch: Any, device_type: str, *, with_bitsandbytes: bool, warnings: list[str]
) -> tuple[bool, str | None]:
    if not with_bitsandbytes:
        return False, None
    try:
        bnb = importlib.import_module("bitsandbytes")
    except Exception as exc:
        warnings.append(
            f"bitsandbytes is not importable ({type(exc).__name__}); QLoRA is unavailable."
        )
        return False, None
    version = getattr(bnb, "__version__", None)
    if device_type in ("cuda", "rocm"):
        return True, version
    warnings.append(
        f"bitsandbytes {version or '?'} is installed but cannot quantize without a CUDA/ROCm "
        "GPU; QLoRA stays disabled."
    )
    return False, version


def _flash_attention_available() -> bool:
    try:
        importlib.import_module("flash_attn")
    except Exception:
        return False
    return True


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def detect_hardware(*, with_bitsandbytes: bool = True) -> HardwareInfo:
    """Inspect this machine.  Never raises; missing parts become warnings."""

    warnings: list[str] = []
    estimate_notes = [
        "device, VRAM and RAM figures are measured at inspection time",
        "per-run memory requirements are modelled, not measured",
    ]

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        total_ram, available_ram = system_memory_mib()
        warnings.append(
            f"torch is not importable ({type(exc).__name__}: {exc}); "
            "no hardware inspection was possible."
        )
        return HardwareInfo(
            device_type="cpu",
            device_name=platform_module.machine() or "cpu",
            device_count=0,
            total_ram_mib=total_ram,
            available_ram_mib=available_ram,
            cpu_count=os.cpu_count() or 1,
            platform=sys.platform,
            estimate_notes=estimate_notes,
            warnings=warnings,
        )

    device_type = _device_type(torch)
    name, count = _preferred_device_name(torch, device_type)
    total_vram, free_vram = _vram_mib(torch, device_type)
    total_ram, available_ram = system_memory_mib()

    if device_type in ("cuda", "rocm"):
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16 = False
        fp16 = True
    elif device_type == "mps":
        # MPS supports fp16; bf16 depends on the macOS/torch build.
        fp16 = True
        bf16 = bool(getattr(torch.backends.mps, "is_macos_13_or_greater", False))
    elif device_type == "xpu":
        fp16 = True
        bf16 = True
    else:
        fp16 = False
        bf16 = False

    bnb_available, bnb_version = _bitsandbytes_status(
        torch, device_type, with_bitsandbytes=with_bitsandbytes, warnings=warnings
    )
    flash_attention = _flash_attention_available()
    if flash_attention and device_type == "cpu":
        warnings.append(
            "flash-attn is installed but needs an accelerator; attention falls back to sdpa."
        )
        flash_attention = False

    if device_type == "cpu":
        warnings.append(
            "No accelerator detected: LoRA on a small model is the only runnable method. "
            "QLoRA and full fine-tuning are disabled with a specific reason."
        )
    if total_vram is not None:
        estimate_notes.append(
            "VRAM totals are measured via torch.cuda.mem_get_info; other processes may "
            "already be using part of it."
        )
    if total_ram is None:
        warnings.append("host RAM could not be measured; memory estimates are less reliable.")

    if device_type == "cpu":
        try:
            importlib.import_module("triton")
        except Exception:
            pass
        else:
            warnings.append(
                "triton is importable but has no active driver here; TRL's fused log-prob "
                "kernel is replaced by a pure-PyTorch fallback."
            )

    return HardwareInfo(
        device_type=device_type,
        device_name=name,
        device_count=count,
        total_vram_mib=total_vram,
        free_vram_mib=free_vram,
        total_ram_mib=total_ram,
        available_ram_mib=available_ram,
        cpu_count=os.cpu_count() or 1,
        platform=sys.platform,
        torch_version=getattr(torch, "__version__", None),
        cuda_version=getattr(getattr(torch, "version", None), "cuda", None),
        bf16_supported=bf16,
        fp16_supported=fp16,
        bitsandbytes_available=bnb_available,
        bitsandbytes_version=bnb_version,
        flash_attention_available=flash_attention,
        packages=_package_versions(),
        estimate_notes=estimate_notes,
        warnings=warnings,
    )


def estimate_resources(
    hw: HardwareInfo,
    hp: Hyperparameters,
    *,
    param_count: int | None,
    gpu_count: int = 1,
) -> ResourceEstimate:
    """Model the memory this configuration needs.

    The arithmetic lives in :func:`autorlhf.config.presets.estimate_training_memory`
    so there is exactly one implementation; this wrapper only adds the measured
    hardware context as caveats.
    """

    effective_gpus = max(1, gpu_count or hw.device_count or 1)
    estimate = estimate_training_memory(hp, param_count=param_count, gpu_count=effective_gpus)
    notes = list(estimate.assumptions)
    if not hw.has_accelerator:
        notes.append(
            "no accelerator detected: VRAM is not the binding constraint, host RAM is; "
            "these figures describe what the same job would need on a GPU."
        )
    if hw.total_ram_mib is not None and estimate.required_ram_mib > hw.total_ram_mib:
        notes.append(
            f"estimated host RAM need ({estimate.required_ram_mib} MiB) exceeds measured "
            f"system RAM ({hw.total_ram_mib} MiB)."
        )
    return estimate.model_copy(update={"assumptions": notes})


__all__ = [
    "TRACKED_PACKAGES",
    "detect_hardware",
    "estimate_resources",
    "system_memory_mib",
]
