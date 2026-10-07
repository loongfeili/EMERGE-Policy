"""Embodied action tool for executing robot actions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for lightweight test envs
    logger = logging.getLogger(__name__)

from Emerge.base import Tool
from Emerge.utils.action_queue import (
    append_action,
    empty_action_document,
    normalize_action_document,
    parse_action_markdown,
    pending_action_type,
    update_action_document,
    action_timestamp,
)

if TYPE_CHECKING:
    from Emerge.agent.visual_monitor import VisualInterruptCoordinator


_BASE_RESULT_TIMEOUT_S = 60.0
_VLA_PER_STEP_TIMEOUT_S = 15.0
_WAM_INFERENCE_TIMEOUT_S = 180.0
_WAM_PER_STEP_TIMEOUT_S = 2.0
_PROPOSAL_INFERENCE_TIMEOUT_S = 180.0

_AC_WM_MEDIATED_ACTIONS = frozenset({
    "vla_execute", "move_to_pose", "move_linear", "set_gripper", "follow_arc",
})
_AC_WM_INTERNAL_ACTIONS = frozenset({
    "ac_wm_select", "vla_propose", "rule_propose", "execute_action_chunk",
})
_AC_WM_MAX_ROWS = 64
_CHUNK_RESULT = re.compile(r"steps=(\d+), reason=([a-z_]+)")
_VISUAL_MONITOR_MARKER = "The visual monitor already verified"


class EmbodiedActionTool(Tool):
    """Validate embodied actions and dispatch them through the workspace queue."""

    @property
    def name(self) -> str:
        return "execute_robot_action"

    @property
    def description(self) -> str:
        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        if backend == "vla":
            policy_guidance = (
                "Use vla_execute for the current visually sensitive contact phase, "
                "with a phase-local instruction that omits future subgoals. "
            )
        elif backend == "wam":
            policy_guidance = (
                "Use wam_execute for the current visually sensitive contact phase. "
                "phase_instruction must describe exactly the next unfinished "
                "visual-contact phase, not the complete mission or a later phase. "
                "The evaluator preserves the full task separately and locks the "
                "configured conditioning mode. "
            )
        else:
            policy_guidance = (
                "Use the active model-policy backend for the current visually "
                "sensitive contact phase. VLA uses instruction; WAM keeps task and "
                "phase instructions separate. "
            )
        ac_wm_guidance = ""
        if self.ac_wm_subagent is not None:
            ac_wm_guidance = (
                "Robot-motion skills (vla_execute, move_to_pose, move_linear, set_gripper, follow_arc) "
                "are mediated by an internal AC-WM planner: the skill proposes controls, a world model "
                "and VLM judge evaluate them, and only selected controls execute. The result is AC-WM "
                "JSON; a rejected proposal did not move the robot, so replan from the latest observation. "
                "vla_execute keeps proposing and executing judged chunks until its step budget is used, "
                "the goal is reached, or a chunk is rejected. "
            )
        return (
            "Execute a physical action on the robot. "
            f"{ac_wm_guidance}"
            "Use explicit geometry-driven motion primitives for coarse approach and clear-space transport. "
            "Choose concrete poses, line segments, arc geometry, and gripper openings from the latest ROBOT_STATE.md or a successful object_location result. "
            f"{policy_guidance}"
            "Every tool call MUST include a non-empty `parameters` object. "
            "Do not call this tool with only `action_type` and `reasoning`."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        if backend == "vla":
            policy_actions = (
                "- 'vla_execute': Execute the current phase's natural-language "
                "instruction using the VLA policy"
            )
            policy_examples = (
                "- vla_execute: {instruction: 'pick up the red block', step: 40}\n"
                "For vla_execute, provide one phase-local instruction and a positive "
                "step budget."
            )
        elif backend == "wam":
            policy_actions = (
                "- 'wam_execute': Execute a bounded Cosmos Policy WAM action chunk "
                "using the evaluator-locked conditioning"
            )
            policy_examples = (
                "- wam_execute grasp: {phase_instruction: 'grasp and lift the red block', step: 48}\n"
                "- wam_execute placement: {phase_instruction: 'place the held red block in the basket and release it', step: 60}\n"
                "For wam_execute, provide exactly one current phase_instruction and a "
                "positive step budget. The evaluator supplies task_instruction."
            )
        else:
            policy_actions = (
                "- 'vla_execute': Execute a phase-local instruction with VLA\n"
                "- 'wam_execute': Execute a bounded Cosmos Policy WAM action chunk"
            )
            policy_examples = (
                "- vla_execute: {instruction: 'pick up the red block', step: 40}\n"
                "- wam_execute: {phase_instruction: 'grasp and lift the red block', step: 48}"
            )
        return {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": (
                        "The type of action to execute. Supported actions:\n"
                        "- 'move_to_pose': Move end-effector to target position and orientation\n"
                        "- 'move_linear': Move end-effector linearly (straight line) to target or by delta\n"
                        "- 'set_gripper': Set gripper opening (open/close command or specific width in meters)\n"
                        "- 'follow_arc': Move end-effector along an arc (circular motion)\n"
                        f"{policy_actions}"
                    ),
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "The parameters for the action. "
                        "Vector parameters prefer list form like [x,y,z], but keyed objects like {x,y,z} are accepted. "
                        "Simulation end-effector targets use metres; Panda reach is approximately 0.85 m from the current robot base pose in ROBOT_STATE.md, and negative x/y values can be valid. "
                        "Examples:\n"
                        "- move_to_pose: {position_m: [x,y,z], orientation_euler: [r,p,y]} or {orientation_quat: [x,y,z,w]}\n"
                        "- move_linear: {position_m: [x,y,z]} or {delta_m: [dx,dy,dz]}, optionally with orientation\n"
                        "- set_gripper: {opening_m: 0.08} or {command: 'open'} or {command: 'close'}\n"
                        "- follow_arc: {center: [x,y,z], axis: [x,y,z], radius_m: 0.1, angle_deg: 90}\n"
                        f"{policy_examples}\n"
                        "For move_to_pose, ALWAYS provide position_m and either orientation_euler or orientation_quat. "
                        "For move_linear, ALWAYS provide either position_m or delta_m. "
                        "For follow_arc, ALWAYS provide center, axis, radius_m, and one of angle_deg or angle_rad. "
                        "A model-policy action completes when its step budget is used or "
                        "the goal is reached earlier. Re-read ROBOT_STATE.md before "
                        "choosing the next action."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "The reasoning behind choosing this action.",
                },
            },
            "required": ["action_type", "parameters", "reasoning"],
        }

    def __init__(
        self,
        workspace: Path,
        visual_interrupts: VisualInterruptCoordinator | None = None,
        ac_wm_subagent=None,
    ):
        self.workspace = workspace
        self.visual_interrupts = visual_interrupts
        self.ac_wm_subagent = ac_wm_subagent
        self.active_action_ids: set[str] = set()
        self.on_event = None

    async def execute(
        self,
        action_type: str,
        parameters: dict[str, Any],
        reasoning: str,
    ) -> str:
        """Validate an action, mediating robot-motion skills through AC-WM when configured."""
        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        requested_backend = {
            "vla_execute": "vla",
            "wam_execute": "wam",
        }.get(action_type)
        if backend in {"vla", "wam"} and requested_backend not in {None, backend}:
            return f"Error: {action_type} is disabled by EMERGE_POLICY_BACKEND={backend}"
        if self.ac_wm_subagent is not None and action_type in _AC_WM_MEDIATED_ACTIONS:
            return await self._execute_action_through_ac_wm(action_type, parameters, reasoning)
        if action_type in _AC_WM_INTERNAL_ACTIONS:
            return "Error: internal AC-WM/skill action; request the current subgoal through a supported robot action."
        parameters = self._effective_parameters(action_type, parameters)
        return await self._dispatch_action(action_type, parameters)

    async def _dispatch_action(self, action_type: str, parameters: dict[str, Any]) -> str:
        """Enqueue a controller action without re-entering AC-WM mediation."""
        embodied_file = self.workspace / "EMBODIED.md"
        action_file = self.workspace / "ACTION.md"
        if not embodied_file.exists():
            return f"Error: {embodied_file.name} not found for the target robot. Cannot dispatch action."
        logger.info("Dispatching action: {} {}", action_type, parameters)
        accepted = self._accept_action(action_type, parameters, action_file)
        if isinstance(accepted, str):
            return accepted
        dispatch_message, action_id = accepted
        self.active_action_ids.add(action_id)
        if self.on_event:
            self.on_event("action.updated", {"action_id": action_id, "action_type": action_type, "status": "pending"})
        return await self._wait_for_action_result(
            action_file,
            action_id=action_id,
            dispatch_message=dispatch_message,
            timeout_s=self._result_timeout(action_type, parameters),
        )

    async def _execute_action_through_ac_wm(
        self, action_type: str, parameters: dict[str, Any], reasoning: str,
    ) -> str:
        if action_type == "vla_execute":
            instruction = str(parameters.get("instruction", parameters.get("prompt", ""))).strip()
            if not instruction:
                return "Error: vla_execute requires instruction or prompt"
            try:
                step = int(parameters.get("step", 40))
            except (TypeError, ValueError):
                return "Error: vla_execute step must be an integer"
            if step < 1:
                return "Error: vla_execute step must be positive"
            return await self._execute_vla_through_ac_wm(instruction, step, parameters, reasoning)

        try:
            step = max(1, min(int(parameters.get("steps", 32)), _AC_WM_MAX_ROWS))
        except (TypeError, ValueError):
            return f"Error: {action_type} steps must be an integer"
        instruction = str(reasoning or "").strip()
        if not instruction:
            instruction = f"Execute the {action_type} skill using its supplied target parameters."
        result = await self._run_ac_wm_selection(action_type, instruction, step, parameters, reasoning)
        return json.dumps({"agent": "ac-wm", **self._selection_payload(result)}, ensure_ascii=False, default=str)

    async def _execute_vla_through_ac_wm(
        self, instruction: str, step: int, parameters: dict[str, Any], reasoning: str,
    ) -> str:
        """Run judged VLA chunks until the step budget, the goal, or a rejection stops it."""
        remaining, executed, chunks = step, 0, []
        stop_reason, last = "budget_exhausted", None
        while remaining > 0:
            last = await self._run_ac_wm_selection("vla_execute", instruction, remaining, parameters, reasoning)
            output = last.output if isinstance(last.output, dict) else {}
            if last.status.value != "success":
                error = str(last.error or "")
                match = _CHUNK_RESULT.search(error)
                if match:
                    executed += int(match.group(1))
                    chunks.append({"selected_candidate_id": last.metadata.get("selected_candidate_id"),
                                   "steps": int(match.group(1)), "reason": match.group(2)})
                if _VISUAL_MONITOR_MARKER in error:
                    stop_reason = "visual_monitor_verified"
                elif match:
                    stop_reason = match.group(2)
                elif last.metadata.get("evaluations") and not last.metadata.get("selected_candidate_id"):
                    stop_reason = "ac_wm_rejected"
                else:
                    stop_reason = "ac_wm_failed"
                break
            dispatch = str(output.get("dispatch_result") or "")
            match = _CHUNK_RESULT.search(dispatch)
            steps, reason = (int(match.group(1)), match.group(2)) if match else (0, "unparsed_dispatch_result")
            executed += steps
            remaining -= steps
            chunks.append({"selected_candidate_id": output.get("selected_candidate_id"),
                           "score": output.get("score"), "steps": steps, "reason": reason})
            if _VISUAL_MONITOR_MARKER in dispatch:
                stop_reason = "visual_monitor_verified"
                break
            if reason != "chunk_completed" or steps <= 0:
                stop_reason = reason
                break

        if stop_reason in {"budget_exhausted", "goal_reached", "visual_monitor_verified"}:
            status = "success"
        else:
            status = "partial" if executed else "failed"
        summary = (
            f"AC-WM executed {executed}/{step} VLA steps in {len(chunks)} selected chunk(s); "
            f"stopped: {stop_reason}."
        )
        if stop_reason in {"ac_wm_rejected", "ac_wm_failed"}:
            summary += " The last proposal was not executed; replan from the latest observation."
        return json.dumps({
            "status": status,
            "agent": "ac-wm",
            "action_type": "vla_execute",
            "summary": summary,
            "steps_executed": executed,
            "requested_steps": step,
            "stop_reason": stop_reason,
            "chunks": chunks,
            "last_selection": self._selection_payload(last) if last is not None else None,
        }, ensure_ascii=False, default=str)

    async def _run_ac_wm_selection(
        self, action_type: str, instruction: str, step: int, parameters: dict[str, Any], reasoning: str,
    ):
        from Emerge.subagents.content import TextContent
        from Emerge.subagents.models import SubagentTask

        # No overall timeout: the proposal, the selection and the dispatch are
        # each bounded, and cancelling mid-dispatch would orphan a moving robot.
        task = SubagentTask(
            content=(TextContent(instruction),),
            input={"instruction": instruction, "step": step, "action_type": action_type,
                   "parameters": dict(parameters), "reasoning": reasoning,
                   "output_dir": str(self.workspace / "artifacts" / "ac-wm" / uuid4().hex)},
        )
        result = await self.ac_wm_subagent.run(task)
        self._record_ac_wm_decision(task, result)
        return result

    @staticmethod
    def _selection_payload(result) -> dict[str, Any]:
        return {
            "status": result.status.value,
            "summary": result.summary,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }

    def _record_ac_wm_decision(self, task, result) -> None:
        """Append one AC-WM verdict so an episode can be audited after the fact."""
        output = result.output if isinstance(result.output, dict) else {}
        entry = {
            "time": action_timestamp(),
            "task_id": task.task_id,
            "action_type": task.input.get("action_type"),
            "instruction": task.input.get("instruction"),
            "status": result.status.value,
            "summary": result.summary,
            "error": result.error,
            "selected_candidate_id": result.metadata.get("selected_candidate_id"),
            "dispatched": bool(result.metadata.get("dispatched")),
            "score": output.get("score"),
            "evaluations": result.metadata.get("evaluations", []),
            "dispatch_result": output.get("dispatch_result"),
        }
        path = self.workspace / "artifacts" / "ac-wm" / "decisions.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to record AC-WM decision: {}", exc)

    async def propose_action_candidates(self, task):
        """Ask the requested low-level skill for a non-executing action preview."""
        from Emerge.ac_wm.protocol import ActionCandidate, RolloutRequest

        instruction = str(task.input["instruction"])
        action_type = str(task.input["action_type"])
        if action_type == "vla_execute":
            step = int(task.input["step"])
            proposal_parameters = {"instruction": instruction, "horizon": min(step, _AC_WM_MAX_ROWS),
                                   "max_execute_steps": step}
            if "replan_steps" in task.input["parameters"]:
                proposal_parameters["replan_steps"] = task.input["parameters"]["replan_steps"]
            raw = await self._dispatch_action("vla_propose", proposal_parameters)
            marker, skill_name, source = "VLA_PROPOSAL:", "vla", "live_openpi_proposal"
            candidate_id = f"vla-proposal-{task.task_id[:10]}"
        else:
            raw = await self._dispatch_action(
                "rule_propose", {"skill_action_type": action_type, "parameters": task.input["parameters"]},
            )
            marker, skill_name, source = "RULE_PROPOSAL:", f"rule:{action_type}", "rule_controller_preview"
            candidate_id = f"{action_type}-proposal-{task.task_id[:10]}"
        if marker not in raw or _VISUAL_MONITOR_MARKER in raw:
            raise RuntimeError(raw)
        proposal, _ = json.JSONDecoder().raw_decode(raw.split(marker, 1)[1].lstrip())
        actions = tuple(tuple(float(value) for value in row) for row in proposal["actions"])
        metadata = {"execute_steps": int(proposal["execute_steps"]), "source": source,
                    "action_type": action_type, "observation_revision": proposal.get("observation_revision")}
        for key in ("control_space", "control_description", "observation_images"):
            if proposal.get(key):
                metadata[key] = proposal[key]
        candidate = ActionCandidate(candidate_id=candidate_id, skill_name=skill_name,
                                    actions=actions, metadata=metadata)
        task_description = (
            instruction + "\nSkill action: " + action_type
            + "\nParameters: " + json.dumps(task.input["parameters"], ensure_ascii=False)
        )
        return RolloutRequest(
            task=task_description,
            observation_path=str(proposal["observation_path"]),
            candidates=(candidate,),
            domain_name=str(proposal.get("domain_name") or "libero"),
            output_dir=str(task.input["output_dir"]),
        )

    async def dispatch_selected_candidate(self, task, selected) -> str:
        """Execute the exact proposal prefix that the world model evaluated."""
        execute_steps = max(1, min(int(selected.metadata.get("execute_steps", 1)), len(selected.actions)))
        parameters = {"actions": [list(row) for row in selected.actions[:execute_steps]],
                      "candidate_id": selected.candidate_id, "skill_name": selected.skill_name}
        if selected.metadata.get("control_space"):
            parameters["control_space"] = selected.metadata["control_space"]
        return await self._dispatch_action("execute_action_chunk", parameters)

    @staticmethod
    def _effective_parameters(
        action_type: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Record evaluator-locked WAM conditioning in the action queue."""
        effective = dict(parameters)
        if action_type != "wam_execute":
            return effective
        mode = os.environ.get("EMERGE_WAM_CONDITIONING_MODE", "").strip()
        task = os.environ.get("EMERGE_WAM_TASK_INSTRUCTION", "").strip()
        if mode not in {"task", "phase", "task_with_phase"}:
            return effective
        effective["conditioning_mode"] = mode
        if task:
            effective["task_instruction"] = task
        if mode == "task":
            effective.pop("instruction", None)
            effective.pop("prompt", None)
            effective.pop("phase_instruction", None)
            if task:
                effective["conditioning_instruction"] = task
        else:
            phase = str(effective.get("phase_instruction", "")).strip()
            if phase:
                effective["conditioning_instruction"] = phase
        return effective

    @staticmethod
    def _result_timeout(action_type: str, parameters: dict[str, Any]) -> float:
        """Backstop wait budget, scaled by the action's step count."""
        if action_type == "vla_execute":
            try:
                step = max(0, int(parameters.get("step", 0)))
            except (TypeError, ValueError):
                step = 0
            return _BASE_RESULT_TIMEOUT_S + step * _VLA_PER_STEP_TIMEOUT_S
        if action_type == "wam_execute":
            try:
                step = max(0, int(parameters.get("step", 0)))
            except (TypeError, ValueError):
                step = 0
            query_count = max(1, (step + 15) // 16)
            return (
                _BASE_RESULT_TIMEOUT_S
                + query_count * _WAM_INFERENCE_TIMEOUT_S
                + step * _WAM_PER_STEP_TIMEOUT_S
            )
        if action_type == "vla_propose":
            return _BASE_RESULT_TIMEOUT_S + _PROPOSAL_INFERENCE_TIMEOUT_S
        if action_type == "execute_action_chunk":
            rows = parameters.get("actions")
            count = len(rows) if isinstance(rows, list) else 0
            return _BASE_RESULT_TIMEOUT_S + count * _VLA_PER_STEP_TIMEOUT_S
        return _BASE_RESULT_TIMEOUT_S

    @staticmethod
    def _accept_action(action_type: str, parameters: dict[str, Any], action_file: Path) -> str | tuple[str, str]:
        """Write validated action to ACTION.md."""
        def append(document):
            existing = pending_action_type(document)
            if existing:
                raise ValueError(f"Action '{existing}' is still pending/running; wait before dispatching")
            return append_action(document, action_type=action_type, parameters=parameters)
        try:
            document = update_action_document(action_file, append)
        except ValueError as exc:
            return f"Error: {exc}"
        action_id = str(document["actions"][-1]["id"])
        return f"Action '{action_type}' validated and dispatched to hardware.", action_id

    async def _wait_for_action_result(
        self,
        action_file: Path,
        *,
        action_id: str,
        dispatch_message: str,
        timeout_s: float = _BASE_RESULT_TIMEOUT_S,
        poll_interval_s: float = 0.5,
    ) -> str:
        result_task = asyncio.create_task(
            self._poll_action_result(
                action_file,
                action_id=action_id,
                dispatch_message=dispatch_message,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
        )
        if self.visual_interrupts is None:
            return await result_task

        interrupt_task = asyncio.create_task(self.visual_interrupts.wait_for(action_id))
        try:
            done, _ = await asyncio.wait(
                (result_task, interrupt_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if result_task in done:
                return result_task.result()

            signal = interrupt_task.result()
            self._request_action_cancel(
                action_file,
                action_id,
                reason=f"visual monitor completed {signal.step_id}",
            )
            result = await result_task
            evidence = json.dumps(signal.evidence, ensure_ascii=False)
            return (
                f"{result} The visual monitor already verified the current subgoal "
                "as achieved. Do not call task_verification again; update PLAN.md "
                f"and continue. Visual monitor evidence: {evidence}"
            )
        finally:
            for task in (result_task, interrupt_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(result_task, interrupt_task, return_exceptions=True)

    async def _poll_action_result(
        self,
        action_file: Path,
        *,
        action_id: str,
        dispatch_message: str,
        timeout_s: float,
        poll_interval_s: float,
    ) -> str:
        deadline = time.monotonic() + timeout_s
        last_status = "pending"
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval_s)
            document = self._load_action_document(action_file)
            if document is None:
                continue
            for action in document.get("actions", []):
                if str(action.get("id")) != action_id:
                    continue
                status = str(action.get("status") or "pending").strip().lower()
                if self.on_event and status != last_status:
                    self.on_event("action.updated", {"action_id": action_id, **action})
                last_status = status
                if status in {"pending", "running"}:
                    break
                result = str(action.get("result", "")).strip()
                if status == "failed":
                    return f"Error: Robot action failed. {result}"
                if result:
                    return f"{dispatch_message} Execution {status}. Result: {result}"
                return f"{dispatch_message} Execution {status}."
        return (
            f"Error: {dispatch_message} Still pending after {timeout_s:.0f}s; the action "
            "remains queued in ACTION.md and the watchdog may be stalled. Do not "
            "re-dispatch or sleep; inspect the watchdog process."
        )

    @staticmethod
    def _request_action_cancel(
        action_file: Path,
        action_id: str,
        *,
        reason: str,
    ) -> None:
        def mark(document):
            for action in document.get("actions", []):
                if str(action.get("id")) == action_id and action.get("status") in {"pending", "running"}:
                    action["cancel_requested"] = True
                    action["cancel_reason"] = reason
                    action["cancel_requested_at"] = action_timestamp()
        update_action_document(action_file, mark)

    async def cancel_active(self, reason: str, timeout: float) -> dict:
        ids = sorted(self.active_action_ids)
        if not ids:
            return {"acknowledged": True, "action_ids": []}
        action_file = self.workspace / "ACTION.md"
        document = self._load_action_document(action_file)
        states = {str(a["id"]): a.get("status") for a in (document or {}).get("actions", [])}
        unfinished = [i for i in ids if states.get(i) not in {"completed", "failed", "cancelled"}]
        if not unfinished:
            return {"acknowledged": True, "action_ids": ids, "requested_action_ids": [], "states": states}
        for action_id in unfinished:
            self._request_action_cancel(action_file, action_id, reason=reason)
        deadline = time.monotonic() + timeout
        while True:
            document = self._load_action_document(action_file)
            states = {str(a["id"]): a.get("status") for a in (document or {}).get("actions", [])}
            confirmed = all(states.get(i) in {"completed", "failed", "cancelled"} for i in ids)
            if confirmed or time.monotonic() >= deadline:
                return {"acknowledged": confirmed, "action_ids": ids, "requested_action_ids": unfinished,
                        "states": {i: states.get(i, "unknown") for i in ids}}
            await asyncio.sleep(0.1)

    @staticmethod
    def _load_action_document(action_file: Path) -> dict[str, Any] | None:
        if not action_file.exists():
            return empty_action_document()
        content = action_file.read_text(encoding="utf-8").strip()
        if not content:
            return empty_action_document()
        payload = parse_action_markdown(content)
        if payload is None:
            return None
        return normalize_action_document(payload)
