"""Submit a sealed release using Merlin's env_map, with no post-launch credential daemon."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

API_ENV_KEYS = {"EMERGE_API_KEY", "AZURE_OPENAI_API_KEY", "EMERGE_PROVIDER", "EMERGE_API_BASE", "EMERGE_API_VERSION",
                "EMERGE_MAX_TOKENS_PARAMETER", "EMERGE_RESPONSES_PROXY", "EMERGE_MODEL",
                "EMERGE_REASONING_EFFORT", "EMERGE_REASONING_SUMMARY"}
SHARED_ENV_KEYS = {"EMERGE_ASSET_PROXY", "VSCODE_SSH_KEY", "EMERGE_RATE_LIMIT_TOKEN"}


def merge_private_environment(environment, values):
    result = dict(environment)
    # A credential explicitly supplied in the private file wins over either
    # alias exported by the caller's shell, including an obsolete API key.
    aliases = {"EMERGE_API_KEY", "AZURE_OPENAI_API_KEY"}
    if aliases.intersection(values):
        for key in aliases:
            result.pop(key, None)
    result.update(values)
    return result


def cli(command, payload, *, dry_run=False):
    # The file is private, temporary, and removed even on error. Neither the request
    # nor CLI error output (which may echo env_map) is copied into public artifacts.
    with tempfile.TemporaryDirectory(prefix="emerge-merlin-request-") as temporary:
        path = Path(temporary) / "request.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
        args = ["merlin-cli", "--control-plane", "cn-seed", *command, "--from-file", str(path)]
        if dry_run:
            args.append("--dry-run")
        # Merlin restricts --from-file to the caller's working directory.
        result = subprocess.run(args, cwd=temporary, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("Merlin command failed: " + " ".join(command) + "; inspect the platform with a read-only command")
    if dry_run:
        return {"dry_run": "accepted"}
    value = json.loads(result.stdout)
    if value.get("ResponseBase", {}).get("code", 0):
        raise RuntimeError("Merlin API rejected " + " ".join(command))
    return value


def make_request(base, release, config_name, env):
    cfg = json.loads((release / config_name).read_text())
    release_manifest = json.loads((release / "release.json").read_text())
    for name, digest in release_manifest["files"].items():
        if Path(name).name != name or hashlib.sha256((release / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Release integrity failure: " + name)
    template = base["job_config"]["job_template_config"]
    if template.get("image_meta", {}).get("image_vid") != cfg["image_vid"]:
        raise ValueError("Baseline image differs from the sealed runtime configuration")
    # Explicitly reconstruct variables; do not inherit credentials from another trial.
    variables = {k: v for k, v in template.get("env_map", {}).items()
                 if k.startswith(("ARNOLD_", "HDFS_", "CPP_HDFS_", "CRS_", "MARIANA_", "CUDA_", "TORCH_", "PYTORCH_", "NCCL_"))}
    pod_release = cfg["release_mount"]
    variables.update({"EMERGE_RELEASE": pod_release, "EMERGE_CONFIG": pod_release + "/" + config_name,
                      "EMERGE_MANIFEST_SHA256": hashlib.sha256((release / "release.json").read_bytes()).hexdigest(),
                      "EMERGE_STAGE_SHA256": release_manifest["files"]["stage-release.py"],
                      "ENABLE_SSH": "1"})
    allowed_keys = SHARED_ENV_KEYS
    if cfg["kind"] == "eval":
        variables.update({"EMERGE_MODEL": env.get("EMERGE_MODEL", "gpt-6-astra"),
                          "EMERGE_PROVIDER": env.get("EMERGE_PROVIDER", "custom")})
        allowed_keys = allowed_keys | API_ENV_KEYS
        if cfg.get('require_api_limiter'):
            variables['EMERGE_RATE_LIMIT_REQUIRED'] = '1'
    if (cfg.get('require_api_limiter') or cfg.get('api_limiter')) and not env.get('EMERGE_RATE_LIMIT_TOKEN'):
        raise ValueError('Shared API limiter access token is required for this release')
    for key in sorted(allowed_keys):
        if env.get(key):
            variables[key] = env[key]
    if cfg["kind"] == "eval" and not (variables.get("EMERGE_API_KEY") or variables.get("AZURE_OPENAI_API_KEY")):
        raise ValueError("Export EMERGE_API_KEY or AZURE_OPENAI_API_KEY before submission")
    script = "entrypoint.sh" if cfg["kind"] == "eval" else "infer-entrypoint.sh"
    digest = release_manifest["files"][script]
    # Verify the fixed HDFS launcher itself before executing it.
    template["entrypoint_full_script"] = (
        "set -euo pipefail\ncp " + shlex.quote(pod_release + "/" + script) + " /tmp/emerge-entrypoint.sh\n"
        + "printf '%s  %s\\n' " + shlex.quote(digest) + " /tmp/emerge-entrypoint.sh | sha256sum -c -\n"
        + "exec bash /tmp/emerge-entrypoint.sh\n")
    template["env_map"] = variables
    template["git_repo"] = {"repo_name": ""}  # entrypoint performs verified GitHub fetches
    name = "geometry_seg_infer" if cfg["kind"] == "infer" else cfg["run_id"]
    overrides = {k: base[k] for k in ("job_config", "attachments", "options", "namespace") if k in base}
    overrides.update(name=name, resource_config=cfg["resource_config"], attachments=cfg["attachments"])
    return {"source_sid": cfg["baseline_job"], "name": name, "overrides": overrides}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, help="Private mode-0600 JSON containing env_map values")
    parser.add_argument("--submit", action="store_true", help="Without this flag, validate only")
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("Receipt already exists; inspect its job instead of submitting twice")
    if Path(args.config).name != args.config:
        raise ValueError("Config must be a basename within the sealed release")
    cfg = json.loads((args.release / args.config).read_text())
    base = cli(["job-v2", "runs", "get-request-config"], {"sid": cfg["baseline_job"]})
    if "job_config" not in base:
        base = base["data"]
    environment = dict(os.environ)
    if args.env_file:
        if args.env_file.stat().st_mode & 0o077:
            raise ValueError("Private environment file must not be accessible by group/others")
        values = json.loads(args.env_file.read_text())
        allowed = API_ENV_KEYS | SHARED_ENV_KEYS
        if set(values) - allowed or not all(isinstance(value, str) for value in values.values()):
            raise ValueError("Unexpected keys/types in private environment file")
        environment = merge_private_environment(environment, values)
    request = make_request(base, args.release, args.config, environment)
    cli(["job-v2", "runs", "fork"], request, dry_run=True)
    safe = {"name": request["name"], "config": cfg, "environment_variable_names": sorted(request["overrides"]["job_config"]["job_template_config"]["env_map"])}
    if not args.submit:
        print(json.dumps({"dry_run": "accepted", **safe}, indent=2))
        return
    check = cli(["job-v2", "check-items", "check"], {"check_point": "create", "node_type": "merlin_job_run_seed", "job_run_context": request["overrides"]})["data"]
    bad = [(item["check_item_sid"], item["status"]) for item in check["results"] if item["status"] not in ("success", "skipped")]
    if bad:
        raise RuntimeError("Merlin precheck failed: " + str(bad))
    request["overrides"]["request_context"] = {"precheck_sid": check["sid"]}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # A submission marker survives a timeout; retry only after querying platform state.
    args.receipt.write_text(json.dumps({"state": "submitting", **safe}, indent=2))
    response = cli(["job-v2", "runs", "fork"], request)
    sid = response["created"]["sid"]
    receipt = {"state": "submitted", "job": sid, "url": "https://seed.bytedance.net/development/instance/jobs/" + sid, **safe}
    args.receipt.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ("state", "job", "url")}))


if __name__ == "__main__":
    main()
