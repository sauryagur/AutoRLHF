"""Typed error hierarchy.

Every user-facing failure carries an actionable message; the TUI renders
``str(exc)`` and the ``hint`` attribute verbatim, so keep both short and
specific.  Never put credentials in an error message.
"""

from __future__ import annotations


class AutorlhfError(Exception):
    """Base class for all expected failures."""

    #: Short, actionable next step shown next to the error in the TUI.
    hint: str = ""
    #: Whether retrying the same operation could succeed.
    retryable: bool = False

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        if hint:
            self.hint = hint


# --- configuration / credentials ------------------------------------------- #


class ConfigError(AutorlhfError, ValueError):
    """Invalid or contradictory run configuration."""


class CredentialError(AutorlhfError):
    """Missing, malformed, or rejected API credential."""


class DoctorError(AutorlhfError):
    """Environment/dependency diagnostic failed."""


# --- model / tokenizer ------------------------------------------------------ #


class ModelLoadError(AutorlhfError):
    """Could not load the requested Hugging Face model."""


class UnsupportedArchitectureError(ModelLoadError):
    """Model is not a supported causal LM for ``AutoModelForCausalLM``."""


class ChatTemplateError(AutorlhfError):
    """Tokenizer has no usable chat template and the fallback was declined."""


# --- hardware / quantization ------------------------------------------------ #


class HardwareError(AutorlhfError):
    """Requested training method is not runnable on the detected hardware."""


class OutOfMemoryError(AutorlhfError):
    """CUDA/accelerator OOM during generation or training."""


class QuantizationUnavailableError(HardwareError):
    """``bitsandbytes`` (or a compatible backend) is not usable here."""


# --- remote services ------------------------------------------------------- #


class RemoteAPIError(AutorlhfError):
    """Non-retryable error returned by OpenRouter (chat or decisions)."""


class RateLimitedError(RemoteAPIError):
    """HTTP 429.  Retryable."""

    def __init__(self, message: str, retry_after: float | None = None, hint: str = "") -> None:
        super().__init__(message, hint)
        self.retry_after = retry_after


class JudgeError(RemoteAPIError):
    """A Jev decision could not be obtained or parsed.

    Never let this collapse into a score of zero — callers must record the
    failure and retry or surface it.
    """


class GenerationError(RemoteAPIError):
    """A text-generation call failed."""


# --- data / training ------------------------------------------------------- #


class InsufficientDataError(AutorlhfError):
    """Not enough validated examples/pairs to run the requested objective."""


class CheckpointError(AutorlhfError):
    """A checkpoint directory exists but is incomplete or unreadable."""


class TargetModulesError(AutorlhfError):
    """LoRA target modules did not match any trainable module."""


class TrainingError(AutorlhfError):
    """Training failed for a reason we can explain."""


class CacheError(AutorlhfError):
    """Cache read/write failure."""


__all__ = [
    "AutorlhfError",
    "CacheError",
    "ChatTemplateError",
    "CheckpointError",
    "ConfigError",
    "CredentialError",
    "DoctorError",
    "GenerationError",
    "HardwareError",
    "InsufficientDataError",
    "JudgeError",
    "ModelLoadError",
    "OutOfMemoryError",
    "QuantizationUnavailableError",
    "RateLimitedError",
    "RemoteAPIError",
    "TargetModulesError",
    "TrainingError",
    "UnsupportedArchitectureError",
]
