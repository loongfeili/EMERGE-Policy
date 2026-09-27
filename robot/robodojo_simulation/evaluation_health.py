"""Pause a batch on deterministic provider failures without inventing scores."""
from __future__ import annotations

import json
import os
from pathlib import Path


def fatal_provider_error(result: dict) -> str | None:
    if result.get("finish_reason") != "error":
        return None
    text = json.dumps(result.get("error") or result, ensure_ascii=False).lower()
    groups = {
        "quota_exhausted": ("余额不足", "用户额度不足", "insufficient_quota", "insufficient_user_quota", "insufficient balance"),
        "authentication_failed": ("http 401", "http 403", "invalid_api_key", "authenticationerror"),
        "model_unavailable": ("model_not_found", "model is not supported", "model does not exist"),
    }
    return next((reason for reason, markers in groups.items() if any(m in text for m in markers)), None)


def pause_on_provider_error(workspace: Path, stop_file: Path) -> str | None:
    try:
        result = json.loads((workspace / "agent_run/result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    reason = fatal_provider_error(result)
    if reason:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation makes the first failing episode the diagnostic source.
        try:
            with stop_file.open("x") as stream:
                json.dump({"reason": reason, "workspace": str(workspace)}, stream)
        except FileExistsError:
            pass
    return reason


def worker_stop_file() -> Path | None:
    value = os.environ.get("ROBODOJO_EVAL_STOP_FILE")
    return Path(value) if value else None
