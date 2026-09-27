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
    def test_runtime_credentials_are_required_and_routed(self):
        configure = module("configure_agent")
        with self.assertRaisesRegex(ValueError, "EMERGE_API_KEY"):
            configure.make_config({})
        cfg = configure.make_config({"EMERGE_API_KEY": "test-only-placeholder"})
        self.assertEqual(cfg["agents"]["defaults"]["model"], "gpt-6-astra")
        self.assertEqual(cfg["providers"]["custom"]["apiBase"], "https://edge.lingsuan.org/v1/responses")
        self.assertNotIn("apiBaseFallbacks", cfg["providers"]["custom"])
        self.assertEqual(cfg["visual_monitor"]["verificationTimeoutSeconds"], 120)

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

    def test_merlin_submission_uses_env_map_without_inheriting_secrets(self):
        launch = module("launch")
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            cfg = {"kind": "eval", "run_id": "test-run", "image_vid": "test-image", "release_mount": "/mnt/hdfs/cache/test",
                   "resource_config": {}, "attachments": [], "baseline_job": "example"}
            (release / "verify.json").write_text(json.dumps(cfg))
            for name in ["entrypoint.sh", "stage-release.py"]:
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


if __name__ == "__main__":
    unittest.main()
