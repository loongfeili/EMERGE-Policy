"""Create a private runtime config from Merlin env_map. Never print credentials."""
import json
import os
from pathlib import Path


def make_config(env):
    key = (env.get("EMERGE_API_KEY") or env.get("AZURE_OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Missing EMERGE_API_KEY or AZURE_OPENAI_API_KEY in Merlin env_map")
    provider = env.get("EMERGE_PROVIDER", "custom")
    if provider not in {"custom", "azure_openai", "responses"}:
        raise ValueError("EMERGE_PROVIDER must be custom, azure_openai, or responses")
    if provider == "responses":
        base = env.get("EMERGE_API_BASE", "https://aidp.bytedance.net/api/modelhub/online").rstrip("/")
        options = {"apiBase": base, "apiKey": key, "trustEnv": False,
                   "reasoningSummary": env.get("EMERGE_REASONING_SUMMARY", "auto")}
    elif provider == "azure_openai":
        base = env.get("EMERGE_API_BASE", "https://aidp.bytedance.net/api/modelhub/online/v2/crawl").rstrip("/")
        options = {
            "apiBase": base, "apiKey": key,
            "apiVersion": env.get("EMERGE_API_VERSION", "2024-02-01"),
            "maxTokensParameter": env.get("EMERGE_MAX_TOKENS_PARAMETER", "max_tokens"),
            # Asset proxies are for GitHub/NVIDIA; AIDP uses the internal network.
            "trustEnv": False,
        }
    else:
        base = env.get("EMERGE_API_BASE", "https://edge.lingsuan.org").rstrip("/")
        if not base.endswith("/responses"):
            base += "/responses" if base.endswith("/v1") else "/v1/responses"
        options = {"apiBase": base, "apiKey": key}
    return {
        "agents": {"defaults": {
            "model": env.get("EMERGE_MODEL", "gpt-6-astra"),
            "provider": provider,
            "maxToolIterations": int(env.get("EMERGE_MAX_ITERATIONS", "40")),
            "reasoningEffort": env.get("EMERGE_REASONING_EFFORT", "high" if provider == "responses" else None),
        }},
        "providers": {provider: options},
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
    provider = config["agents"]["defaults"]["provider"]
    print(json.dumps({"configured": True, "model": config["agents"]["defaults"]["model"],
                      "provider": provider, "api_base": config["providers"][provider]["apiBase"],
                      "credential_source": "Merlin env_map"}))
