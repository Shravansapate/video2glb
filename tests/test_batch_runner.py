from __future__ import annotations

from pathlib import Path
import time

import pytest

import convert
from src.pipeline import batch_runner
from src.pipeline.batch_runner import BatchJob, BatchProcessResult, ProcessAttempt


def _attempt(tmp_path: Path, *, number: int, returncode: int, timed_out: bool = False, retryable: bool = False):
    return ProcessAttempt(
        attempt=number,
        returncode=returncode,
        timed_out=timed_out,
        duration_seconds=0.01,
        log_path=tmp_path / f"attempt-{number}.log",
        retryable=retryable,
    )


def test_recursive_discovery_is_case_insensitive_and_excludes_output(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "B.MP4").write_bytes(b"b")
    (tmp_path / "nested" / "a.mp4").write_bytes(b"a")
    (tmp_path / "nested" / "ignore.txt").write_text("x", encoding="utf-8")
    (tmp_path / "output" / "debug.mp4").write_bytes(b"debug")

    videos = batch_runner.discover_mp4_files(
        tmp_path,
        recursive=True,
        excluded_roots=(tmp_path / "output",),
    )

    assert [path.name for path in videos] == ["B.MP4", "a.mp4"]


def test_duplicate_stems_get_stable_distinct_output_directories(tmp_path: Path):
    first = tmp_path / "one" / "Change.mp4"
    second = tmp_path / "two" / "Change.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    names = batch_runner.assign_output_directory_names(
        [first, second],
        input_dir=tmp_path,
        uppercase=True,
    )
    repeated = batch_runner.assign_output_directory_names(
        [first, second],
        input_dir=tmp_path,
        uppercase=True,
    )

    assert names == repeated
    assert names[first].startswith("CHANGE__")
    assert names[second].startswith("CHANGE__")
    assert names[first] != names[second]
    assert "/" not in names[first] and "\\" not in names[first]


def test_unique_flat_stem_preserves_existing_output_name(tmp_path: Path):
    video = tmp_path / "Change.mp4"
    video.write_bytes(b"video")

    names = batch_runner.assign_output_directory_names(
        [video],
        input_dir=tmp_path,
        uppercase=True,
    )

    assert names[video] == "CHANGE"


def test_transient_failed_attempt_is_retried_and_exit_two_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    returncodes = iter([1, 0, 2])

    def fake_run(_command, *, attempt_number, **_kwargs):
        return _attempt(
            tmp_path,
            number=attempt_number,
            returncode=next(returncodes),
            retryable=True,
        )

    monkeypatch.setattr(batch_runner, "_run_command_once", fake_run)
    monkeypatch.setattr(batch_runner.time, "sleep", lambda _seconds: None)
    jobs = [
        BatchJob("a", ("python", "a"), "a"),
        BatchJob("b", ("python", "b"), "b"),
    ]

    results = batch_runner.run_isolated_jobs(
        jobs,
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        max_workers=1,
        timeout_seconds=10,
        retries=2,
    )

    assert [attempt.returncode for attempt in results[0].attempts] == [1, 0]
    assert [attempt.returncode for attempt in results[1].attempts] == [2]


def test_parallel_completion_still_returns_deterministic_job_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(command, *, attempt_number, **_kwargs):
        if command[-1] == "slow":
            time.sleep(0.03)
        return _attempt(tmp_path, number=attempt_number, returncode=0)

    monkeypatch.setattr(batch_runner, "_run_command_once", fake_run)
    jobs = [
        BatchJob("slow", ("worker", "slow"), "slow"),
        BatchJob("fast", ("worker", "fast"), "fast"),
    ]
    completed: list[str] = []

    results = batch_runner.run_isolated_jobs(
        jobs,
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        max_workers=2,
        timeout_seconds=10,
        retries=0,
        on_result=lambda result: completed.append(result.key),
    )

    assert completed == ["fast", "slow"]
    assert [result.key for result in results] == ["slow", "fast"]


def test_timeout_is_recorded_as_failure_code_124(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class TimedOutProcess:
        def wait(self, timeout=None):
            raise batch_runner.subprocess.TimeoutExpired(["worker"], timeout)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", lambda *_args, **_kwargs: TimedOutProcess())
    monkeypatch.setattr(batch_runner, "_terminate_process_tree", lambda _process: None)

    attempt = batch_runner._run_command_once(
        ("worker",),
        cwd=tmp_path,
        log_path=tmp_path / "timeout.log",
        timeout_seconds=0.01,
        attempt_number=1,
    )

    assert attempt.returncode == 124
    assert attempt.timed_out is True
    assert attempt.error == "Conversion exceeded 0.01 seconds."


def test_process_success_cannot_pass_without_intact_release_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    video = tmp_path / "input" / "Change.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    output = tmp_path / "output" / "CHANGE"
    output.mkdir(parents=True)
    source_hash = convert.sha256_file(video)
    (output / "Change.qc.json").write_text(
        '{"technical_qc":"PASS"}', encoding="utf-8"
    )
    (output / "Change.metadata.json").write_text(
        (
            '{"technical_qc":"PASS","production":{"production_eligible":true,'
            '"engineering_candidate":true},"release_evidence":{'
            f'"source_video":{{"sha256":"{source_hash}"}},'
            '"avatar":{"sha256":"avatar-hash"}}}'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(convert, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(convert, "evaluate_published_release", lambda *_args, **_kwargs: (False, ["invalid release"]))
    monkeypatch.setattr(convert, "evaluate_published_engineering_candidate", lambda *_args, **_kwargs: False)
    result = BatchProcessResult(
        key="Change.mp4",
        attempts=(_attempt(tmp_path, number=1, returncode=0),),
    )

    classified = convert.classify_batch_artifact(
        video=video,
        output_dir_name="CHANGE",
        expected_source_sha256=source_hash,
        expected_avatar_sha256="avatar-hash",
        trusted_signers_path=tmp_path / "trusted.json",
        process_result=result,
    )

    assert classified["status"] == "FAIL"
    assert classified["production_eligible"] is False
    assert classified["engineering_candidate"] is False


@pytest.mark.parametrize(
    ("workers", "timeout", "retries"),
    [(0, 10.0, 0), (9, 10.0, 0), (1, 0.0, 0), (1, 86_401.0, 0), (1, 10.0, 4)],
)
def test_batch_resource_limits_are_enforced(
    tmp_path: Path,
    workers: int,
    timeout: float,
    retries: int,
):
    with pytest.raises(ValueError):
        batch_runner.run_isolated_jobs(
            [],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            max_workers=workers,
            timeout_seconds=timeout,
            retries=retries,
        )
