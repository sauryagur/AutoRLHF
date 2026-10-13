"""Optional W&B (training metrics) and Weave (LLM / evaluation traces).

Both integrations are **independently toggleable and never required**.  Neither
library is imported at module import time: ``wandb`` / ``weave`` are pulled in
lazily inside the functions that actually use them, so a machine without the
``tracking`` extra — or a run with tracking disabled — behaves exactly as if the
package did not exist.

Failure is always *degrading*, never fatal: an auth or connectivity problem
becomes a ``warning`` event on the bus plus a :class:`TrackingState` entry, and
training continues locally.

No credential is ever logged, printed, persisted to SQLite, or written into an
export.  The values are read from :class:`~autorlhf.config.settings.SecretStore`
by the libraries themselves (via the process environment) and are handled by
:mod:`autorlhf.core.logging` redaction everywhere else.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.events import EventBus
from autorlhf.core.logging import get_logger

logger = get_logger("autorlhf.integrations.tracking")

#: Shown whenever Weave is requested without recorded consent.  It must state the
#: egress explicitly — this is the whole reason consent is required.
WEAVE_EGRESS_NOTICE = (
    "Weave tracing is disabled because consent was not recorded. Enabling it would "
    "transmit prompts, model responses, and Jev judgements to an external service "
    "(Weights & Biases / Weave). No data has left this machine. To enable, set "
    "weave_consent=True on the run config or re-run `autorlhf --setup` and accept the "
    "notice."
)


class TrackingState(BaseModel):
    """What actually happened for a given run's tracking setup."""

    wandb_enabled: bool = False
    weave_enabled: bool = False
    wandb_project: str = ""
    weave_project: str = ""
    wandb_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _wandb_config_excerpt(config: RunConfig, run_id: str) -> dict[str, Any]:
    """A small, credential-free W&B config payload."""

    return {
        "autorlhf_run_id": run_id,
        "slug": config.slug,
        "method": config.method.value,
        "objective": config.objective.value,
        "preset_id": config.training.preset_id,
        "model_id": config.model.model_id,
        "model_revision": config.model.resolved_revision_label(),
        "judge_model": config.generation.judge_model,
        "teacher_model": config.generation.teacher_model,
        "feedback_mode": config.feedback_mode.value,
    }


class TrainingTracker:
    """Thin, failure-tolerant wrapper around optional tracking backends."""

    def __init__(
        self,
        *,
        config: RunConfig,
        env: Environment,
        run_id: str,
        bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self.run_id = run_id
        self.bus = bus
        self._state = TrackingState(
            wandb_project=config.wandb_project or env.settings.wandb_project,
            weave_project=config.weave_project or env.settings.weave_project,
        )
        self._wandb_module: Any | None = None
        self._wandb_run: Any | None = None
        self._weave_client: Any | None = None

    # -- introspection ----------------------------------------------------- #

    @property
    def state(self) -> TrackingState:
        return self._state

    @property
    def enabled(self) -> bool:
        """True when at least one backend is genuinely active."""

        return self._state.wandb_enabled or self._state.weave_enabled

    @property
    def report_to(self) -> list[str]:
        """``report_to`` value for TRL — only backends that really initialised."""

        return ["wandb"] if self._state.wandb_enabled else []

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> TrackingState:
        """Initialise the enabled backends.  Never raises; degrades with warnings."""

        if self.config.wandb_enabled:
            self._start_wandb()
        if self.config.weave_enabled:
            self._start_weave()
        return self._state

    def _warn(self, message: str, *, remediation: str = "") -> None:
        self._state.warnings.append(message)
        logger.warning(message)
        if self.bus is not None:
            self.bus.emit(
                "warning",
                message,
                run_id=self.run_id,
                phase="tracking",
                remediation=remediation,
            )

    def _start_wandb(self) -> None:
        project = self._state.wandb_project or "autorlhf"
        try:
            import wandb
        except ImportError as exc:
            self._warn(
                f"W&B is enabled but the 'wandb' package is not installed ({exc}).",
                remediation="Install the optional extra: uv sync --extra tracking",
            )
            return

        settings = None
        try:
            from wandb import Settings as WandbSettings

            settings = WandbSettings(silent=True)
        except Exception:  # pragma: no cover - older/newer wandb shapes
            settings = None

        # The API key is read by wandb from the environment; supply the secret
        # store's merged view only for the duration of init.
        env_vars = self.env.env_with_secrets()
        saved: dict[str, str | None] = {}
        for key in ("WANDB_API_KEY", "WANDB_ENTITY"):
            value = env_vars.get(key)
            if value:
                saved[key] = os.environ.get(key)
                os.environ[key] = value
        try:
            kwargs: dict[str, Any] = {
                "project": project,
                "name": self.run_id,
                "reinit": True,
                "config": _wandb_config_excerpt(self.config, self.run_id),
            }
            if settings is not None:
                kwargs["settings"] = settings
            run = wandb.init(**kwargs)
        except Exception as exc:
            self._warn(
                f"W&B init failed: {exc}",
                remediation=(
                    "Check WANDB_API_KEY (`autorlhf --setup`), network access, and the "
                    "project name; training continues locally without W&B."
                ),
            )
            return
        finally:
            for key, previous in saved.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

        self._wandb_module = wandb
        self._wandb_run = run
        self._state.wandb_enabled = True
        self._state.wandb_url = getattr(run, "url", None)
        if self.bus is not None:
            self.bus.emit(
                "log",
                f"W&B logging to project '{project}'",
                run_id=self.run_id,
                phase="tracking",
                project=project,
                url=self._state.wandb_url,
            )

    def _start_weave(self) -> None:
        client = weave_client(self.env, config=self.config)
        if client is None:
            notice = (
                WEAVE_EGRESS_NOTICE
                if not self.config.weave_consent
                else ("Weave tracing could not be initialised; see logs above.")
            )
            self._warn(notice)
            return
        self._weave_client = client
        self._state.weave_enabled = True


    def finish(self, *, summary: dict | None = None) -> None:
        """Flush and close any active backends.  Never raises."""

        if self._wandb_run is not None and self._wandb_module is not None:
            try:
                if summary:
                    for key, value in summary.items():
                        self._wandb_run.summary[key] = value
                self._wandb_module.finish()
            except Exception as exc:
                self._warn(f"W&B finish failed: {exc}")
            finally:
                self._wandb_run = None
                self._wandb_module = None

        if self._weave_client is not None:
            try:
                import weave

                weave.finish()
            except Exception as exc:
                self._warn(f"Weave finish failed: {exc}")
            finally:
                self._weave_client = None


# --------------------------------------------------------------------------- #
# Weave (LLM / evaluation tracing) helpers
# --------------------------------------------------------------------------- #


def weave_client(env: Environment, *, config: RunConfig | None = None) -> Any | None:
    """Initialise a Weave client, or return ``None``.

    Returns ``None`` (never raises) when Weave is disabled, when consent has not
    been recorded, or when the package cannot be initialised.  The consent case
    logs :data:`WEAVE_EGRESS_NOTICE` so the reason — external data egress — is
    recorded explicitly.
    """

    if config is None or not config.weave_enabled:
        return None
    if not config.weave_consent:
        logger.warning(WEAVE_EGRESS_NOTICE)
        return None
    try:
        import weave
    except ImportError as exc:
        logger.warning("Weave is enabled but the 'weave' package is not installed: %s", exc)
        return None
    project = config.weave_project or env.settings.weave_project or "autorlhf"
    try:
        return weave.init(project)
    except Exception as exc:
        logger.warning("Weave init failed: %s", exc)
        return None


def weave_log(name: str, payload: dict) -> None:
    """Attach ``payload`` to the current Weave call, if there is one.

    A safe no-op without Weave or outside a traced call — the OpenRouter SDK is
    not auto-patched.
    """

    try:
        import weave
    except ImportError:
        return
    try:
        call = weave.get_current_call()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("weave_log could not read the current call: %s", exc)
        return
    if call is None:
        return
    try:
        attributes = getattr(call, "attributes", None)
        if isinstance(attributes, dict):
            attributes[name] = payload
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("weave_log could not attach '%s': %s", name, exc)


__all__ = [
    "WEAVE_EGRESS_NOTICE",
    "TrackingState",
    "TrainingTracker",
    "weave_client",
    "weave_log",
]
