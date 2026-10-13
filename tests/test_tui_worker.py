"""The worker drives the frozen interfaces in the documented order."""

from __future__ import annotations

from pathlib import Path

from test_tui_support import StubJob, make_run, run_python, stub_hooks

from autorlhf.config.settings import Environment
from autorlhf.core.enums import RunStatus
from autorlhf.storage.runs import RunPaths
from autorlhf.tui.worker import CALL_ORDER, PipelineHooks, PipelineWorker


def build_worker(app_paths, store, bus, run_config, *, record=None, hooks=None, **kwargs):
    """Create a stored run plus a worker wired to the stub interfaces."""

    run_id, run_paths = make_run(store, app_paths, run_config, status=RunStatus.CREATED)
    record = {} if record is None else record
    worker = PipelineWorker(
        run_id=run_id,
        env=Environment(app_paths),
        store=store,
        paths=run_paths,
        config=run_config,
        bus=bus,
        hooks=hooks or stub_hooks(record),
        **kwargs,
    )
    return worker, record


def test_worker_calls_interfaces_in_documented_order(app_paths, store, bus, run_config):
    worker, record = build_worker(app_paths, store, bus, run_config, interactive=True)
    worker.run()

    assert worker.calls == list(CALL_ORDER)
    assert worker.bundle == "bundle"
    assert worker.baseline == "baseline-report"
    assert (worker.final, worker.comparison) == ("final-report", "comparison-report")
    assert worker.error is None

    ctx = record["ctx"]
    assert ctx.closed is True
    assert ctx.kwargs["interactive"] is True
    assert ctx.kwargs["paths"] == worker.paths
    assert ctx.kwargs["store"] is store

    job = record["job"]
    assert job.kwargs["cancel_event"] is ctx.cancel_event
    assert job.kwargs["pause_event"] is ctx.pause_event
    assert job.kwargs["plan"] is worker.plan
    assert job.kwargs["run_id"] == worker.run_id
    assert job.resume_from is None

    # the adapter the trainer produced is what the final evaluation loads
    assert record["adapter_path"] == Path("/tmp/stub-adapter")
    assert record["checkpoint_objective"] is worker.plan.objective


def test_worker_passes_a_resume_checkpoint_when_one_exists(app_paths, store, bus, run_config):
    checkpoint = Path("/tmp/checkpoint-10")
    record = {"resume_from": checkpoint}
    hooks = stub_hooks(record)
    worker, record = build_worker(app_paths, store, bus, run_config, record=record, hooks=hooks)
    worker.run()
    assert record["job"].resume_from == checkpoint


def test_worker_honours_the_resume_flag_for_dataset_preparation(
    app_paths, store, bus, run_config
):
    worker, record = build_worker(app_paths, store, bus, run_config, resume=True)
    worker.run()
    assert record["prepare_dataset_resume"] is True

    worker, record = build_worker(app_paths, store, bus, run_config, resume=False)
    worker.run()
    assert record["prepare_dataset_resume"] is False


def test_worker_emits_a_phase_event(app_paths, store, bus, run_config):
    worker, _record = build_worker(app_paths, store, bus, run_config)
    worker.run()
    kinds = [event.kind for event in bus.history()]
    assert "phase" in kinds


def test_worker_uses_the_base_model_when_no_adapter_was_written(app_paths, store, bus, run_config):
    record: dict = {}
    hooks = stub_hooks(record)
    hooks = PipelineHooks(
        hooks.make_context, hooks.make_runner, _no_adapter_job(record), hooks.find_resume_checkpoint
    )
    worker, _record = build_worker(app_paths, store, bus, run_config, hooks=hooks)
    worker.run()
    assert record["adapter_path"] is None
    assert worker.calls == list(CALL_ORDER)


def _no_adapter_job(record: dict):
    def make_job(**kwargs):
        job = StubJob(**kwargs)
        job.adapter_path = None
        record["job"] = job
        return job

    return make_job


def test_worker_forwards_pause_resume_and_stop_to_the_job(app_paths, store, bus, run_config):
    record: dict = {}
    hooks = stub_hooks(record)
    holder: dict = {}

    def make_job(**kwargs):
        job = StubJob(**kwargs)
        record["job"] = job

        def steer() -> None:
            holder["worker"].request_pause()
            holder["worker"].request_resume()
            holder["worker"].request_stop()

        job.on_run = steer
        return job

    hooks = PipelineHooks(
        hooks.make_context, hooks.make_runner, make_job, hooks.find_resume_checkpoint
    )
    worker, _ = build_worker(app_paths, store, bus, run_config, hooks=hooks)
    holder["worker"] = worker
    worker.run()

    job = record["job"]
    assert job.paused is False  # paused then resumed
    assert job.stopped is True
    assert worker.cancel_event.is_set() is True


def test_worker_reports_status_without_a_job(app_paths, store, bus, run_config):
    worker, _record = build_worker(app_paths, store, bus, run_config)
    status = worker.status
    assert status["run_id"] == worker.run_id
    assert status["calls"] == []
    assert status["error"] is None
    assert status["paused"] is False
    assert isinstance(worker.paths, RunPaths)


def test_importing_the_worker_does_not_import_textual_torch_or_the_pipeline():
    result = run_python(
        "import sys\n"
        "import autorlhf.tui.worker as w\n"
        "assert w.CALL_ORDER == ('prepare_dataset', 'run_review', 'run_baseline_eval',\n"
        "                        'build_training_plan', 'TrainingJob.run', 'run_final_eval')\n"
        "for name in ('autorlhf.pipeline.context', 'torch', 'textual'):\n"
        "    assert name not in sys.modules, name\n"
        "assert callable(w.default_hooks)\n"
    )
    assert result.returncode == 0, result.stderr
