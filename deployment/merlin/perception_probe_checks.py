"""Recognize model quality rejections without hiding infrastructure failures."""
import math
import re


def quality_rejection(error, server_url):
    if not isinstance(error, RuntimeError):
        return None
    # Model-service errors carry a code; the legacy client prefixed the server.
    prefix = f"RuntimeError from {server_url}: "
    message = str(error)
    if getattr(error, "code", None) == "GEOMETRY_REJECTED":
        pass
    elif message.startswith(prefix):
        message = message[len(prefix):]
    else:
        return None
    patterns = [
        ("baseline_scale_relative_mad", r"VGGT camera baselines disagree on metric depth scale: relative_mad=([0-9]+(?:\.[0-9]+)?), limit=([0-9]+(?:\.[0-9]+)?)"),
        ("camera_center_rms_m", r"VGGT predicted camera geometry is inconsistent with calibration: rms=([0-9]+(?:\.[0-9]+)?)m, limit=([0-9]+(?:\.[0-9]+)?)m"),
    ]
    for metric, pattern in patterns:
        match = re.fullmatch(pattern, message)
        if match:
            value, limit = map(float, match.groups())
            if math.isfinite(value) and math.isfinite(limit) and value > limit > 0:
                return {"metric": metric, "value": value, "limit": limit, "message": message}
    return None
