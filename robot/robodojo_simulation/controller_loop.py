"""Run EMERGE's controller in the thread that owns the Isaac application."""
from __future__ import annotations

import json
import time
from pathlib import Path

from robot.controller import _install_profile, _poll_once, _publish_runtime_state


def watch_driver_loop(driver, workspace: Path, *, driver_name="robodojo",
                      poll_interval=0.25, stop_when_terminal=True,
                      raise_on_interrupt=False, stop_file=None,
                      trajectory_file=None, **_unused):
    state_file = workspace / "ROBOT_STATE.md"
    _install_profile(driver, workspace)
    driver.load_environment()
    _publish_runtime_state(driver, state_file)
    try:
        while True:
            _poll_once(driver, workspace / "ACTION.md", state_file)
            driver.raise_if_infrastructure_error()
            if stop_file is not None and Path(stop_file).exists():
                driver.request_stop()
                _publish_runtime_state(driver, state_file)
                break
            if stop_when_terminal and driver.is_terminal():
                break
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        if raise_on_interrupt:
            raise
    finally:
        if trajectory_file:
            Path(trajectory_file).write_text(json.dumps({
                "event": "controller_finished", "driver": driver_name,
                "state": driver.get_runtime_state(),
            }) + "\n")
