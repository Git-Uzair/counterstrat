"""Job-state persistence under the Windows reader/replace race.

The UI polls GET /api/jobs/{id} every second; Python readers hold files
without FILE_SHARE_DELETE, so os.replace on the state file raises
PermissionError (WinError 5) whenever a poll overlaps a save. Observed
2026-09-04: an ingest died at the 'serializing' save, and data/jobs held
leaked .tmp_* files from three different server pids.
"""

import threading
import time

from counterstrat.web.ingest import JobState, load_job_state, save_job_state


def test_save_survives_concurrent_reader(tmp_path):
    state = JobState(job_id="j1", stage="queued")
    save_job_state(tmp_path, state)
    target = tmp_path / "jobs" / "j1.json"

    # a poller mid-read: the handle must outlive this scope, held by the thread
    fh = open(target, encoding="utf-8")  # noqa: SIM115

    def release():
        time.sleep(0.15)
        fh.close()

    t = threading.Thread(target=release)
    t.start()
    try:
        state.stage = "done"
        save_job_state(tmp_path, state)  # must retry until the reader lets go
    finally:
        t.join()
        if not fh.closed:
            fh.close()
    loaded = load_job_state(tmp_path, "j1")
    assert loaded is not None and loaded.stage == "done"
    assert not list((tmp_path / "jobs").glob("*.tmp_*")), "leaked temp files"
