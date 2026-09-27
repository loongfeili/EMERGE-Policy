"""Wait for this release's service manifest and every advertised endpoint."""
import json
import os
import time
import urllib.request
from pathlib import Path


def save_snapshot(path, services):
    """Pin the validated endpoints locally before downstream probes read them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(services, indent=2))
    os.replace(temporary, path)


def main():
    wait_for_services()


def wait_for_services():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    path = Path(os.environ["EMERGE_SERVICES_MANIFEST"])
    lock = json.loads((Path(os.environ["EMERGE_RELEASE"]) / "source-lock.json").read_text())
    deadline = time.monotonic() + int(os.environ.get("EMERGE_SERVICE_WAIT_SECONDS", "3600"))
    while True:
        try:
            services = json.loads(path.read_text())
            assert services["emerge_commit"] == lock["repositories"]["emerge"]["commit"], "Inference source version differs"
            for name in ("policy_urls", "vggt_urls", "sam3_urls"):
                assert services[name], f"Empty service pool: {name}"
                for url in services[name]:
                    with opener.open(url.replace("ws://", "http://") + "/healthz", timeout=5) as response:
                        assert response.status == 200
            save_snapshot(Path(os.environ["EMERGE_LOCAL_SERVICES_MANIFEST"]), services)
            print("All declared model services are healthy; pinned a local service snapshot", flush=True)
            return
        except (OSError, ValueError, KeyError, AssertionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)


if __name__ == "__main__":
    main()
