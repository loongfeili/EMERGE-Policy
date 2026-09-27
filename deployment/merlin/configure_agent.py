"""Create a private runtime config from Merlin env_map. Never print credentials."""
import json
import os
from pathlib import Path


def make_config(env):
    key = env.get("EMERGE_API_KEY", "").strip()
    if not key:
        raise ValueError("Missing EMERGE_API_KEY in Merlin env_map")
    base = env.get("EMERGE_API_BASE", "https://edge.lingsuan.org").rstrip("/")
    if not base.endswith("/responses"):
        base += "/responses" if base.endswith("/v1") else "/v1/responses"
    return {
        "agents": {"defaults": {
            "model": env.get("EMERGE_MODEL", "gpt-6-astra"),
            "provider": "custom",
            "maxToolIterations": int(env.get("EMERGE_MAX_ITERATIONS", "40")),
        }},
        "providers": {"custom": {"apiBase": base, "apiKey": key}},
        "visual_monitor": {"verificationTimeoutSeconds": 120},
    }


if __name__ == "__main__":
    config = make_config(os.environ)
    directory = Path("/home/tiger/.config/emerge-robodojo")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / "agent.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(config, stream)
    path.chmod(0o600)
    print(json.dumps({"configured": True, "model": config["agents"]["defaults"]["model"],
                      "api_base": config["providers"]["custom"]["apiBase"],
                      "credential_source": "Merlin env_map"}))
