"""Prepare skill resources for workspace-restricted RoboDojo agents."""

import shutil
from pathlib import Path


def install_agent_skills(workspace: Path, builtin_skills: Path) -> None:
    """Copy VLA-compatible skill trees, preserving existing workspace overrides.

    Copies (rather than symlinks) keep both SKILL.md and its referenced files
    readable under the agent's workspace restriction. RoboDojo has no WAM driver.
    """
    destination = workspace / "skills"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(builtin_skills.iterdir()):
        if source.name == "wam" or not (source / "SKILL.md").is_file():
            continue
        target = destination / source.name
        if not target.exists():
            shutil.copytree(source, target)
