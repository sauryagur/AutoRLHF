"""Hardware preflight: feasibility gating, estimates, and lazy torch import."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.enums import TrainingMethod
from autorlhf.core.types import HardwareInfo
from autorlhf.training.hardware import detect_hardware, estimate_resources


def _cpu_hw(**overrides) -> HardwareInfo:
    base = dict(
        device_type="cpu",
        device_name="test-cpu",
        device_count=0,
        total_ram_mib=15 * 1024,
        available_ram_mib=8 * 1024,
        cpu_count=12,
        platform="linux",
    )
    base.update(overrides)
    return HardwareInfo(**base)


def _cuda_hw(**overrides) -> HardwareInfo:
    base = dict(
        device_type="cuda",
        device_name="NVIDIA Test",
        device_count=1,
        total_vram_mib=24 * 1024,
        free_vram_mib=22 * 1024,
        total_ram_mib=64 * 1024,
        available_ram_mib=48 * 1024,
        cpu_count=16,
        bf16_supported=True,
        fp16_supported=True,
        bitsandbytes_available=True,
    )
    base.update(overrides)
    return HardwareInfo(**base)


def test_lora_supported_on_cpu_but_qlora_and_fft_are_not() -> None:
    hw = _cpu_hw()
    ok, reason = hw.supports(TrainingMethod.LORA)
    assert ok and reason == ""

    ok, reason = hw.supports(TrainingMethod.QLORA)
    assert not ok
    assert "GPU" in reason

    ok, reason = hw.supports(TrainingMethod.FFT)
    assert not ok
    assert "GPU" in reason


def test_qlora_without_bitsandbytes_is_refused_with_reason() -> None:
    hw = _cuda_hw(bitsandbytes_available=False)
    ok, reason = hw.supports(TrainingMethod.QLORA)
    assert not ok
    assert "bitsandbytes" in reason


def test_qlora_feasible_on_cuda_with_bitsandbytes() -> None:
    hw = _cuda_hw()
    assert hw.has_accelerator
    assert hw.supports(TrainingMethod.QLORA) == (True, "")
    assert hw.supports(TrainingMethod.FFT) == (True, "")


def test_qlora_refused_on_mps() -> None:
    hw = _cuda_hw(device_type="mps", device_name="Apple M2", total_vram_mib=None)
    ok, reason = hw.supports(TrainingMethod.QLORA)
    assert not ok
    assert "MPS" in reason


def test_detect_hardware_on_this_machine() -> None:
    hw = detect_hardware()
    # This project's dev machine has no accelerator; the detection must say so
    # rather than guess.
    assert hw.device_type in {"cpu", "cuda", "rocm", "mps", "xpu"}
    assert hw.torch_version
    assert hw.cpu_count >= 1
    assert hw.total_ram_mib is None or hw.total_ram_mib > 0
    assert "torch" in hw.packages
    assert hw.estimate_notes, "caveats must be recorded so the UI can label estimates"
    if hw.device_type == "cpu":
        assert not hw.has_accelerator
        assert hw.supports(TrainingMethod.QLORA)[0] is False
        assert hw.supports(TrainingMethod.FFT)[0] is False
        assert hw.supports(TrainingMethod.LORA)[0] is True
        assert any("No accelerator" in warning for warning in hw.warnings)


def test_detect_hardware_without_bitsandbytes_probe() -> None:
    hw = detect_hardware(with_bitsandbytes=False)
    assert hw.bitsandbytes_available is False
    assert hw.bitsandbytes_version is None


def test_estimate_resources_delegates_and_annotates() -> None:
    hp = Hyperparameters(
        method=TrainingMethod.LORA, lora_target_modules=["q_proj"], max_seq_length=512
    )
    estimate = estimate_resources(_cpu_hw(), hp, param_count=1_000_000)
    assert estimate.total_parameters == 1_000_000
    assert estimate.required_ram_mib > 0
    assert estimate.assumptions
    assert any("no accelerator" in note for note in estimate.assumptions)


def test_estimate_resources_unknown_param_count_is_labelled() -> None:
    estimate = estimate_resources(_cpu_hw(), Hyperparameters(), param_count=None)
    assert estimate.total_parameters is None
    assert any("model size unknown" in note for note in estimate.assumptions)


def _run_python(code: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importing_jobs_is_fast_and_torch_free() -> None:
    payload = _run_python(
        """
        import json, sys, time
        start = time.perf_counter()
        import autorlhf.training.jobs  # noqa: F401
        elapsed = time.perf_counter() - start
        heavy = [name for name in ("torch", "trl", "peft", "transformers") if name in sys.modules]
        print(json.dumps({"seconds": elapsed, "heavy": heavy}))
        """
    )
    assert payload["heavy"] == []
    assert payload["seconds"] < 1.0


def test_importing_hardware_does_not_import_torch() -> None:
    payload = _run_python(
        """
        import json, sys
        import autorlhf.training.hardware  # noqa: F401
        print(json.dumps({"torch": "torch" in sys.modules}))
        """
    )
    assert payload["torch"] is False
