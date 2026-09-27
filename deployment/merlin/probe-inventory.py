"""Validate the complete official layout inventory even for a four-episode smoke run."""
import importlib.util
import json
from pathlib import Path

root = Path("/home/tiger/EMERGE-Policy")
spec = importlib.util.spec_from_file_location("eval_robodojo_agent", root / "scripts/eval_robodojo_agent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tasks = module._tasks(root / "configs/robodojo_tasks_arx_x5_seed0.txt", "all")
layouts = module._layouts_per_task(tasks, "native", robodojo_root=Path("/home/tiger/RoboDojo"), env_cfg="arx_x5", policy_seed=0)
total = sum(len(value) for value in layouts.values())
assert len(tasks) == 54 and total == 2100, (len(tasks), total)
report = {"task_entries": len(tasks), "episodes": total, "seed": 0, "per_task": {key: len(value) for key, value in layouts.items()}}
Path("/home/tiger/robodojo-setup/standard-inventory.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"task_entries": len(tasks), "episodes": total, "inventory_verified": True}))
