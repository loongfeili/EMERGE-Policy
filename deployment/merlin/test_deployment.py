import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class DeploymentTests(unittest.TestCase):
    def test_verified_bundle_checks_out_without_network_and_rejects_tampering(self):
        import hashlib
        checkout = module("git_checkout")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
            git("init", "-q", "-b", "main")
            git("config", "user.email", "test@example.com")
            git("config", "user.name", "test")
            (source / "code.py").write_text("value = 1\n")
            git("add", "code.py")
            git("commit", "-qm", "source")
            bundle = Path(directory) / "source.bundle"
            git("bundle", "create", str(bundle), "main")
            spec = {"url": "https://unreachable.invalid/repo.git", "branch": "main",
                    "commit": git("rev-parse", "HEAD"), "bundle": str(bundle),
                    "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
            destination = Path(directory) / "checkout"
            checkout.checkout(spec, destination)
            self.assertEqual((destination / "code.py").read_text(), "value = 1\n")
            bundle.write_bytes(bundle.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                checkout.checkout(spec, destination)

    def test_inference_replicas_use_their_own_reserved_host_ports(self):
        serve = module("serve")
        # Two four-GPU workers can share one eight-GPU host/IP.
        environment = {"ARNOLD_WORKER_0_PORT": "9513,9236,9858,9906,10409",
                       "ARNOLD_WORKER_1_PORT": "11071,10579,9368,9285,10118"}
        models0, limiter = serve.allocated_ports(environment, 0, 4, with_limiter=True)
        models1, unused = serve.allocated_ports(environment, 1, 4)
        self.assertEqual(models0, [9513,9236,9858,9906])
        self.assertEqual(limiter, 10409)
        self.assertEqual(models1, [11071,10579,9368,9285])
        self.assertIsNone(unused)
        self.assertFalse(set(models0 + [limiter]) & set(models1))

    def test_inference_rejects_missing_or_unsafe_port_allocations(self):
        serve = module("serve")
        for value in ["", "8000,8001,8002,8003", "8000,8000,8002,8003,8004",
                      "8000,8001,8002,8003,70000", "8000,8001,8002,8003,no"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                serve.allocated_ports({"ARNOLD_WORKER_0_PORT": value}, 0, 4, with_limiter=True)

    def test_runtime_credentials_are_required_and_routed(self):
        configure = module("configure_agent")
        with self.assertRaisesRegex(ValueError, "EMERGE_API_KEY"):
            configure.make_config({})
        cfg = configure.make_config({"EMERGE_API_KEY": "test-only-placeholder"})
        self.assertEqual(cfg["agents"]["defaults"]["model"], "gpt-6-astra")
        self.assertEqual(cfg["providers"]["custom"]["apiBase"], "https://edge.lingsuan.org/v1/responses")
        self.assertNotIn("apiBaseFallbacks", cfg["providers"]["custom"])
        self.assertEqual(cfg["visual_monitor"]["verificationTimeoutSeconds"], 120)

    def test_aidp_config_uses_azure_chat_without_responses_suffix(self):
        cfg = module("configure_agent").make_config({
            "EMERGE_API_KEY": "test-only-placeholder", "EMERGE_PROVIDER": "azure_openai",
            "EMERGE_API_BASE": "https://example.test/api/modelhub/online/v2/crawl",
            "EMERGE_API_VERSION": "2024-02-01", "EMERGE_MAX_TOKENS_PARAMETER": "max_tokens",
            "EMERGE_REASONING_EFFORT": "low",
        })
        self.assertEqual(cfg["agents"]["defaults"]["provider"], "azure_openai")
        self.assertEqual(cfg["agents"]["defaults"]["reasoningEffort"], "low")
        options = cfg["providers"]["azure_openai"]
        self.assertEqual(options["apiBase"], "https://example.test/api/modelhub/online/v2/crawl")
        self.assertEqual(options["apiVersion"], "2024-02-01")
        self.assertEqual(options["maxTokensParameter"], "max_tokens")
        self.assertFalse(options["trustEnv"])
        self.assertNotIn("custom", cfg["providers"])

    def test_git_fetch_preserves_exact_ancestor_and_rejects_dirty_source(self):
        checkout = module("git_checkout")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
            git("init", "-q", "-b", "robodojo")
            git("config", "user.email", "test@example.com")
            git("config", "user.name", "test")
            (source / "code.py").write_text("version = 1\n")
            git("add", ".")
            git("commit", "-qm", "first")
            first = git("rev-parse", "HEAD")
            (source / "code.py").write_text("version = 2\n")
            git("commit", "-qam", "second")
            spec = {"url": str(source), "branch": "robodojo", "commit": first}
            destination = Path(directory) / "destination"
            checkout.checkout(spec, destination)
            self.assertEqual((destination / "code.py").read_text(), "version = 1\n")
            (destination / "code.py").write_text("local mutation\n")
            with self.assertRaisesRegex(ValueError, "local source"):
                checkout.checkout(spec, destination)
            with self.assertRaisesRegex(ValueError, "modified"):
                checkout.verify(spec, destination)

    def test_aidp_responses_config_accepts_azure_key_and_reasoning_settings(self):
        cfg = module("configure_agent").make_config({
            "AZURE_OPENAI_API_KEY": "test-only-placeholder", "EMERGE_PROVIDER": "responses",
            "EMERGE_API_BASE": "https://example.test/api/modelhub/online",
            "EMERGE_REASONING_EFFORT": "high", "EMERGE_REASONING_SUMMARY": "auto",
        })
        self.assertEqual(cfg["agents"]["defaults"]["provider"], "responses")
        self.assertEqual(cfg["agents"]["defaults"]["reasoningEffort"], "high")
        self.assertEqual(cfg["providers"]["responses"], {
            "apiBase": "https://example.test/api/modelhub/online", "apiKey": "test-only-placeholder",
            "reasoningSummary": "auto", "trustEnv": False,
        })

    def test_private_key_overrides_obsolete_shell_alias(self):
        launch = module("launch")
        aliases = ("EMERGE_API_KEY", "AZURE_OPENAI_API_KEY")
        for chosen, obsolete in (aliases, aliases[::-1]):
            environment = launch.merge_private_environment(
                {obsolete: "old-test-key", "EMERGE_ASSET_PROXY": "http://assets.invalid"},
                {chosen: "new-test-key", "EMERGE_PROVIDER": "responses"},
            )
            self.assertNotIn(obsolete, environment)
            self.assertEqual(environment["EMERGE_ASSET_PROXY"], "http://assets.invalid")
            config = module("configure_agent").make_config(environment)
            self.assertEqual(config["providers"]["responses"]["apiKey"], "new-test-key")

    def test_merlin_submission_uses_env_map_without_inheriting_secrets(self):
        launch = module("launch")
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            cfg = {"kind": "eval", "run_id": "test-run", "image_vid": "test-image", "release_mount": "/mnt/hdfs/cache/test",
                   "resource_config": {}, "attachments": [], "baseline_job": "example"}
            (release / "verify.json").write_text(json.dumps(cfg))
            for name in ["entrypoint.sh", "infer-entrypoint.sh", "stage-release.py"]:
                (release / name).write_text("# fixed script\n")
            (release / "release.json").write_text(json.dumps({"files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in release.iterdir()
            }}))
            base = {"job_config": {"job_template_config": {"image_meta": {"image_vid": "test-image"}, "env_map": {"OLD_API_KEY": "never-inherit", "ARNOLD_HDFS_NATIVE": "1"}}}}
            request = launch.make_request(base, release, "verify.json", {"EMERGE_API_KEY": "new-test-placeholder"})
            template = request["overrides"]["job_config"]["job_template_config"]
            self.assertNotIn("OLD_API_KEY", template["env_map"])
            self.assertEqual(template["env_map"]["EMERGE_API_KEY"], "new-test-placeholder")
            self.assertNotIn("new-test-placeholder", template["entrypoint_full_script"])
            aidp = {"EMERGE_API_KEY": "new-test-placeholder", "EMERGE_PROVIDER": "azure_openai",
                    "EMERGE_API_BASE": "https://example.test/api/modelhub/online/v2/crawl",
                    "EMERGE_API_VERSION": "2024-02-01", "EMERGE_MAX_TOKENS_PARAMETER": "max_tokens",
                    "EMERGE_MODEL": "gpt-6-astra", "EMERGE_REASONING_EFFORT": "low"}
            request = launch.make_request(base, release, "verify.json", aidp)
            variables = request["overrides"]["job_config"]["job_template_config"]["env_map"]
            for key, value in aidp.items():
                self.assertEqual(variables[key], value)
            aidp = {"AZURE_OPENAI_API_KEY": "new-test-placeholder", "EMERGE_PROVIDER": "responses",
                    "EMERGE_API_BASE": "https://example.test/api/modelhub/online",
                    "EMERGE_MODEL": "gpt-6-astra", "EMERGE_REASONING_EFFORT": "high",
                    "EMERGE_REASONING_SUMMARY": "auto"}
            request = launch.make_request(base, release, "verify.json", aidp)
            variables = request["overrides"]["job_config"]["job_template_config"]["env_map"]
            for key, value in aidp.items():
                self.assertEqual(variables[key], value)
            self.assertNotIn("EMERGE_API_KEY", variables)
            cfg["kind"] = "infer"
            (release / "verify.json").write_text(json.dumps(cfg))
            manifest = json.loads((release / "release.json").read_text())
            manifest["files"]["verify.json"] = hashlib.sha256((release / "verify.json").read_bytes()).hexdigest()
            (release / "release.json").write_text(json.dumps(manifest))
            request = launch.make_request(base, release, "verify.json", aidp)
            variables = request["overrides"]["job_config"]["job_template_config"]["env_map"]
            self.assertTrue(set(aidp).isdisjoint(variables))
            (release / "entrypoint.sh").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                launch.make_request(base, release, "verify.json", {})

    def test_scale_configuration_allocates_expected_cards(self):
        publish = module("publish")
        for nodes, gpus in [(1, 8), (4, 8), (8, 8), (2, 1)]:
            role = publish.resources("eval", nodes, gpus)["arnold_resource_config"]["roles"][0]
            self.assertEqual(role["num"] * role["gpu"], nodes * gpus)
        infer = publish.resources("infer", 2, 4)["arnold_resource_config"]
        self.assertEqual(infer["group_id"], 1894)
        self.assertEqual(infer["roles"][0]["queue_name"], "a100-sxm-80gb.hpccluster-ydfgrrp7ac9tiffwmqs7.ai")

    def test_git_fetch_retries_transient_failure_but_remains_bounded(self):
        checkout = module("git_checkout")
        failure = subprocess.CalledProcessError(128, ["git", "fetch"])
        with patch.object(checkout, "git", side_effect=[failure, failure, ""]) as git, patch.object(checkout.time, "sleep"):
            checkout.fetch(Path("test-repo"), "robodojo")
            self.assertEqual(git.call_count, 3)
        with patch.object(checkout, "git", side_effect=failure) as git, patch.object(checkout.time, "sleep"):
            with self.assertRaises(subprocess.CalledProcessError):
                checkout.fetch(Path("test-repo"), "robodojo")
            self.assertEqual(git.call_count, 5)

    def test_service_snapshot_survives_a_replaced_hdfs_manifest(self):
        discovery = module("wait-services")
        from contextlib import nullcontext
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared, local = root / "pool.json", root / "local/services.json"
            (root / "source-lock.json").write_text(json.dumps({
                "repositories": {"emerge": {"commit": "pinned-source"}}
            }))
            pool = {"emerge_commit": "pinned-source", "policy_urls": ["ws://policy"],
                    "vggt_urls": ["ws://vggt"], "sam3_urls": ["ws://sam"]}
            shared.write_text(json.dumps(pool) + "\x00")
            env = {"EMERGE_SERVICES_MANIFEST": str(shared), "EMERGE_RELEASE": str(root),
                   "EMERGE_LOCAL_SERVICES_MANIFEST": str(local)}
            opener = SimpleNamespace(open=lambda *a, **k: nullcontext(SimpleNamespace(status=200)))
            with patch.dict(discovery.os.environ, env), \
                    patch.object(discovery.urllib.request, "build_opener", return_value=opener), \
                    patch.object(discovery.time, "sleep", side_effect=lambda _: shared.write_text(json.dumps(pool))):
                discovery.wait_for_services()
            shared.write_text("a later incomplete HDFS read")
            self.assertEqual(json.loads(local.read_text()), pool)


if __name__ == "__main__":
    unittest.main()
