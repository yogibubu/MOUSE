from __future__ import annotations

from matrix_gaussian import jobs


def test_status_stays_running_after_intermediate_normal_termination(
    tmp_path, monkeypatch
):
    (tmp_path / "gauin.gjf").write_text("#p test\n", encoding="utf-8")
    (tmp_path / "gauin.log").write_text(
        "Normal termination of Gaussian DV\n"
        "Entering the next requested calculation stage\n",
        encoding="utf-8",
    )
    (tmp_path / jobs.PID_FILE).write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "_pid_is_running", lambda pid: pid == 12345)

    status = jobs.gaussian_job_status(tmp_path)

    assert status.status == "running"
    assert status.normal_termination is False
    assert status.pid == 12345


def test_status_completes_after_launcher_exits(tmp_path, monkeypatch):
    (tmp_path / "gauin.gjf").write_text("#p test\n", encoding="utf-8")
    (tmp_path / "gauin.log").write_text(
        "Normal termination of Gaussian DV\n", encoding="utf-8"
    )
    (tmp_path / jobs.PID_FILE).write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "_pid_is_running", lambda pid: False)

    status = jobs.gaussian_job_status(tmp_path)

    assert status.status == "completed"
    assert status.normal_termination is True
