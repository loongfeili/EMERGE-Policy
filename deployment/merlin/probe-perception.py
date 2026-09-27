import os
"""Check service readiness on fixed real data; record live-scene quality separately."""
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from Emerge.subagents.object_location.tools.location.client import ModelServerClient
from Emerge.subagents.object_location.tools.location.pose import estimate_object_locations
from perception_probe_checks import quality_rejection

SETUP = Path('/home/tiger/robodojo-setup')
SERVICES = Path(os.environ['EMERGE_SERVICES_MANIFEST'])


def load_views(workspace):
    manifest = json.loads((workspace / 'artifacts/observations/observation.json').read_text())
    views = []
    for view in manifest['views']:
        rgb = np.array(Image.open(workspace / view['image_path']).convert('RGB'))
        intrinsics = np.asarray(view['intrinsics'])
        transform = np.asarray(view['T_world_camera'])
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.std() > 0, 'invalid rendered RGB'
        assert intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all()
        assert transform.shape == (4, 4) and np.isfinite(transform).all()
        views.append(dict(view, rgb=rgb, intrinsics=intrinsics, T_world_camera=transform))
    assert len(views) == 3
    return manifest, views


async def probe(workspace, services, *, reference):
    manifest, views = load_views(workspace)
    report = {'workspace': str(workspace), 'reference': reference}
    results = {}
    for name, payload in [('vggt', {'views': views, 'reference_view': manifest['reference_view']}), ('sam3', {'views': views, 'targets': [{'object_key': 'bowl', 'prompt': 'bowl'}], 'max_instances': 4})]:
        url = services[name + '_urls'][0]
        started = time.monotonic()
        try:
            result = await ModelServerClient(url, timeout=180).infer(payload)
        except RuntimeError as error:
            rejection = quality_rejection(error, url) if name == 'vggt' else None
            if reference or rejection is None:
                raise
            report[name] = {'seconds': time.monotonic() - started, 'quality_rejection': rejection}
            continue
        assert len(result['views']) == len(views)
        results[name] = result
        entry = {'seconds': time.monotonic() - started, 'views': len(result['views'])}
        if name == 'vggt':
            entry['alignment'] = result['alignment']
            entry['finite_depth_fraction'] = [float(np.isfinite(v['depth_m']).mean()) for v in result['views']]
            assert all(x == 1 for x in entry['finite_depth_fraction'])
        else:
            entry['detections'] = [{'view': v['view_name'], 'found': sum(t['found'] for t in v['targets'])} for v in result['views']]
            if reference:
                assert any(d['found'] for d in entry['detections']), 'reference SAM detection failed'
        report[name] = entry
    if 'vggt' not in results:
        report.update(status='PASS_WITH_GEOMETRY_REJECTION', geometry_usable=False, localization_details=[])
        return report
    located = estimate_object_locations(results['vggt'], results['sam3'], [{'object_key': 'bowl', 'prompt': 'bowl'}], point_conf_threshold=0.3, min_points=80, bbox_padding_pixels=4, view_center_tolerance_m=0.08, ray_consensus_tolerance_m=0.02, min_consistent_views=2, coordinate_frame='robodojo_env', max_localization_distance_m=3.0)
    assert located, 'empty localization response'
    for item in located:
        if item['found']:
            position = np.asarray(item['position_m'])
            assert item['frame'] == 'robodojo_env' and position.shape == (3,) and np.isfinite(position).all() and np.linalg.norm(position) <= 3.0, item
        else:
            assert isinstance(item.get('failure_reason'), str) and item['failure_reason'] and item.get('position_m') is None, item
    usable = all(v['found'] for v in located)
    if reference and not usable:
        # This real three-bowl reference has an established ambiguous cross-view match.
        assert all(v.get('failure_reason') == 'insufficient_consistent_views' for v in located), 'unexpected reference localization rejection'
    report.update(status='PASS' if usable else 'PASS_WITH_LOCALIZATION_REJECTION', geometry_usable=usable, localization_details=located)
    return report


async def main():
    services = json.loads(SERVICES.read_text())
    # A known real calibrated scene must pass the complete chain first. Unknown failures
    # and transport/model errors remain fatal on both reference and live observations.
    reference = await probe(SETUP / 'perception-reference', services, reference=True)
    live = await probe(Path('/home/tiger/emerge-smoke-bulk25'), services, reference=False)
    report = {'status': live['status'], 'reference': reference, 'live': live}
    (SETUP / 'perception-probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
