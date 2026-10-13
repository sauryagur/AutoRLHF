"""Optional, independently toggleable experiment-tracking integrations.

Nothing in this package imports :mod:`wandb` or :mod:`weave` at import time.
Both libraries are pulled in lazily *only* when the corresponding integration is
actually enabled, so an install without the ``tracking`` extra — and every
default run — starts and trains exactly as if the package did not exist.

``integrations`` is imported *by* :mod:`autorlhf.training` and the TUI, never the
other way round.
"""

from __future__ import annotations

from autorlhf.integrations.tracking import (
    WEAVE_EGRESS_NOTICE,
    TrackingState,
    TrainingTracker,
    weave_client,
    weave_log,
)

__all__ = [
    "WEAVE_EGRESS_NOTICE",
    "TrackingState",
    "TrainingTracker",
    "weave_client",
    "weave_log",
]
