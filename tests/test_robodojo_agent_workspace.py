"""Regressions for skill access and backend selection at the RoboDojo boundary."""

import ast
import json
import os
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Emerge.agent.context import ContextBuilder
from Emerge.agent.skills import BUILTIN_SKILLS_DIR
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.agent.tools.filesystem import ReadFileTool
from robot.mujoco_simulation.scene_io import default_robot_state_doc, save_robot_state_doc
from robot.robodojo_simulation.agent_workspace import install_agent_skills

ROOT = Path(__file__).resolve().parents[1]


def worker_functions(**overrides):
    # Importing the worker starts Isaac. Load only its actual workspace/launch
    # functions so these boundary checks can run without a GPU or simulator.
    tree = ast.parse((ROOT / "scripts/run_robodojo_agent_worker.py").read_text())
    selected = ast.Module(body=[
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in {"_prepare_workspace", "_start_agent_supervisor"}
    ], type_ignores=[])
    namespace = dict(
        Path=Path, Any=Any, shutil=shutil, os=os, threading=threading, time=time,
        subprocess=subprocess, EMERGE_ROOT=ROOT,
        install_agent_skills=install_agent_skills,
        default_robot_state_doc=default_robot_state_doc,
        save_robot_state_doc=save_robot_state_doc,
        ARGS=SimpleNamespace(task="align_blocks", layout_id=0, policy_seed=0,
                             agent_python="/usr/bin/python3", agent_message="test",
                             agent_config=None, episode_timeout_s=10),
    )
    namespace.update(overrides)
    exec(compile(selected, "run_robodojo_agent_worker.py", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.asyncio
async def test_every_advertised_skill_is_readable_in_restricted_episode(tmp_path, monkeypatch):
    workspace = tmp_path / "episode"
    worker_functions()._prepare_workspace(workspace)
    monkeypatch.setenv("EMERGE_POLICY_BACKEND", "vla")
    prompt = ContextBuilder(workspace).build_system_prompt()
    summary = ET.fromstring(prompt[prompt.index("<skills>"):prompt.index("</skills>") + 9])
    skills = {entry.findtext("name"): Path(entry.findtext("location")) for entry in summary}
    assert {"vla", "object-location", "task-verification", "planning"} <= skills.keys()
    assert "wam" not in skills
    assert "### Skill: vla" in prompt
    assert "### Skill: wam" not in prompt
    reader = ReadFileTool(workspace=workspace, allowed_dir=workspace)
    for path in skills.values():
        assert path.is_relative_to(workspace)
        assert not path.is_symlink()
        result = await reader.execute(path=str(path))
        assert not result.startswith("Error:"), result
    assert "outside allowed directory" in await reader.execute(
        path=str(BUILTIN_SKILLS_DIR / "vla/SKILL.md"))
    benchmark = tmp_path / "private_benchmark.py"
    benchmark.write_text("private benchmark data")
    assert "outside allowed directory" in await reader.execute(path=str(benchmark))


def test_skill_resources_copied_and_workspace_overrides_preserved(tmp_path):
    builtin = tmp_path / "builtin"
    source = builtin / "example"
    (source / "references").mkdir(parents=True)
    (source / "SKILL.md").write_text("Read references/guide.md")
    (source / "references/guide.md").write_text("guide")
    workspace = tmp_path / "workspace"
    install_agent_skills(workspace, builtin)
    copied = workspace / "skills/example"
    assert (copied / "references/guide.md").read_text() == "guide"
    (copied / "SKILL.md").write_text("user override")
    install_agent_skills(workspace, builtin)
    assert (copied / "SKILL.md").read_text() == "user override"
    assert (source / "SKILL.md").read_text() == "Read references/guide.md"


@pytest.mark.asyncio
async def test_vla_backend_hides_wam_and_rejects_it_before_dispatch(tmp_path, monkeypatch):
    worker_functions()._prepare_workspace(tmp_path)
    monkeypatch.setenv("EMERGE_POLICY_BACKEND", "vla")
    tool = EmbodiedActionTool(tmp_path)
    schema = json.dumps(tool.parameters)
    assert "vla_execute" in schema
    assert "wam_execute" not in schema
    assert "wam_execute" not in tool.description
    result = await tool.execute(action_type="wam_execute", parameters={}, reasoning="test")
    assert "disabled by EMERGE_POLICY_BACKEND=vla" in result
    assert (tmp_path / "ACTION.md").read_text() == ""
    assert not tool.active_action_ids


@pytest.mark.parametrize("inherited", [None, "wam", "both"])
def test_batch_supervisor_pins_backend_for_every_agent(tmp_path, monkeypatch, inherited):
    if inherited is None:
        monkeypatch.delenv("EMERGE_POLICY_BACKEND", raising=False)
    else:
        monkeypatch.setenv("EMERGE_POLICY_BACKEND", inherited)
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs)
        captured["command"] = command
        return SimpleNamespace(poll=lambda: 0, wait=lambda **kw: 0)

    worker = worker_functions(subprocess=SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT))
    worker._prepare_workspace(tmp_path)
    (tmp_path / "ROBOT_STATE.md").write_text('{"connected": true}')
    thread, state = worker._start_agent_supervisor(
        workspace=tmp_path, session_id="test", done_file=tmp_path / ".done",
        verdict_ready=threading.Event())
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert state["error"] is None
    assert state["returncode"] == 0
    assert captured["env"]["EMERGE_POLICY_BACKEND"] == "vla"
    assert "--restrict-to-workspace" in captured["command"]
    assert os.environ.get("EMERGE_POLICY_BACKEND") == inherited


def test_single_episode_agent_command_overrides_inherited_wam(tmp_path):
    script = (ROOT / "scripts/run_robodojo_agent_episode.sh").read_text()
    command = next(line for line in script.splitlines() if '"${agent_args[@]}"' in line)
    result = subprocess.run(
        ["bash", "-c", 'AGENT_PYTHON=printenv; agent_args=(EMERGE_POLICY_BACKEND); '
         'WORKSPACE="$1"; ' + command + '\nwait', "test", str(tmp_path)],
        env={**os.environ, "EMERGE_POLICY_BACKEND": "wam"}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "agent.log").read_text().strip() == "vla"
