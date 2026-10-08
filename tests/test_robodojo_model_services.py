"""RoboDojo on the shared model-service and driver-lifecycle infrastructure."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from Emerge.config.schema import Config, ObjectLocationSubagentConfig
from external_model_server.model_service.contracts import ServiceError
from external_model_server.model_service.discovery import PinnedDiscovery, ServiceDiscovery
from external_model_server.schemas import OPENPI, SAM3, VGGT

ROOT = Path(__file__).resolve().parents[1]


def test_cluster_endpoints_pin_perception_services(monkeypatch):
    config = Config.model_validate({"subagents": {"objectLocation": {
        "vggtUrl": "ws://[fdbd::1]:9001", "sam3Url": "ws://10.0.0.2:9002"}}})
    discovery = config.subagents.object_location.discovery(config.model_services)
    assert isinstance(discovery, PinnedDiscovery)
    assert discovery.resolve(VGGT) == "ws://[fdbd::1]:9001"
    assert asyncio.run(discovery.resolve_async(SAM3)) == "ws://10.0.0.2:9002"

    scanned = []
    monkeypatch.setattr(ServiceDiscovery, "resolve", lambda self, expectation: scanned.append(expectation) or "ws://127.0.0.1:8000")
    assert discovery.resolve(OPENPI) == "ws://127.0.0.1:8000"
    assert scanned == [OPENPI]


def test_unpinned_perception_keeps_local_discovery(monkeypatch):
    discovery = ObjectLocationSubagentConfig().discovery()
    assert discovery.endpoints == {}
    monkeypatch.setattr(ServiceDiscovery, "resolve", lambda self, expectation: "ws://127.0.0.1:8001")
    assert discovery.resolve(VGGT) == "ws://127.0.0.1:8001"


def _probe_checks():
    spec = importlib.util.spec_from_file_location("probe_checks", ROOT / "deployment/merlin/perception_probe_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("error", [
    ServiceError("GEOMETRY_REJECTED",
                 "VGGT camera baselines disagree on metric depth scale: relative_mad=0.4100, limit=0.2500"),
    RuntimeError("RuntimeError from ws://vggt: VGGT camera baselines disagree on metric depth scale: "
                 "relative_mad=0.4100, limit=0.2500"),
])
def test_perception_probe_recognizes_geometry_rejections(error):
    rejection = _probe_checks().quality_rejection(error, "ws://vggt")
    assert rejection["metric"] == "baseline_scale_relative_mad" and rejection["value"] == pytest.approx(0.41)


def test_perception_probe_does_not_hide_service_failures():
    checks = _probe_checks()
    assert checks.quality_rejection(ServiceError("INFERENCE_FAILED", "Model inference failed; see server logs"),
                                    "ws://vggt") is None


def test_robodojo_driver_meets_the_driver_lifecycle(tmp_path):
    from robot.drivers.robodojo_driver import RoboDojoDriver

    driver = RoboDojoDriver(SimpleNamespace(sim=object(), task_name="stack_bowls"))
    published = []
    driver._publish_observation = lambda: published.append(True)
    driver.load_environment()
    assert published == [True]
    assert driver.get_scene_catalog() == {"root": "RoboDojo", "current": "stack_bowls", "entries": []}
    with pytest.raises(ValueError, match="evaluation worker"):
        driver.switch_scene("stack_bowls")
    with pytest.raises(RuntimeError, match="evaluation worker"):
        driver.reset_environment()
    driver.close()
    driver.close()
    assert driver.get_scene_catalog()["current"] is None
