"""Wait for this release's service manifest and every advertised endpoint."""
import json
import os
import time
import urllib.request
from pathlib import Path

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
        print("All declared model services are healthy and use the pinned source commit")
        break
    except (OSError, ValueError, KeyError, AssertionError):
        if time.monotonic() >= deadline:
            raise
        time.sleep(5)
