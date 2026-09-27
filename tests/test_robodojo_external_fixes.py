import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Emerge.subagents.object_location.tools.observation import ObservationStore
from Emerge.subagents.object_location.tools.location.pose import _validate_pose_geometry
from robot.robodojo_simulation.evaluation_health import fatal_provider_error, pause_on_provider_error
from robot.robodojo_simulation.observation_publisher import RoboDojoObservationPublisher


def test_frame_propagates_from_nonzero_origin_to_observation(tmp_path):
    image = np.zeros((480, 640, 3), np.uint8)
    k = np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]])
    camera = SimpleNamespace(get_intrinsics_matrix=lambda **kw: k,
        get_world_pose=lambda **kw: (np.array([10., 20., 2.]), np.array([1., 0, 0, 0])))
    env = SimpleNamespace(camera_manager=SimpleNamespace(camera_names=[['cam_head']], cameras=[[camera]]),
        env_origins=np.array([[10., 20., 0.]]))
    pub = RoboDojoObservationPublisher(env, workspace=tmp_path, camera_names=['cam_head'])
    pub.publish({'vision': {'cam_head': {'color': image}}})
    obs = ObservationStore(tmp_path).load()
    assert obs.coordinate_frame == 'robodojo_env'
    assert obs.max_localization_distance_m == 3.0
    np.testing.assert_allclose(obs.views[0].T_world_camera[:3, 3], [0, 0, 2])


@pytest.mark.parametrize('center,reason', [([-.142,30,-23.86], 'behind'), ([0,0,30], 'range'), ([np.nan,0,1], 'non_finite')])
def test_invalid_geometry_is_rejected(center, reason):
    pose = SimpleNamespace(center=np.array(center), extent=np.ones(3)*.1,
        rotation_matrix=np.eye(3), bbox_3d_corners=np.ones((8,3)))
    geo = SimpleNamespace(cameras=[SimpleNamespace(name='head', T_world_camera_observed=np.eye(4))])
    with pytest.raises(RuntimeError, match=reason):
        _validate_pose_geometry(pose,geometry=geo,selected_views={'head'},max_distance_m=3)


def test_valid_geometry_uses_camera_distance_not_world_origin():
    t=np.eye(4);t[:3,3]=[10,20,30]
    geo=SimpleNamespace(cameras=[SimpleNamespace(name='head',T_world_camera_observed=t)])
    pose=SimpleNamespace(center=np.array([10,20,31]),extent=np.ones(3)*.1,
        rotation_matrix=np.eye(3),bbox_3d_corners=np.ones((8,3)))
    _validate_pose_geometry(pose,geometry=geo,selected_views={'head'},max_distance_m=3)


@pytest.mark.parametrize('message,expected', [('HTTP 403: 余额不足','quota_exhausted'),('HTTP 401','authentication_failed'),
    ('model_not_found','model_unavailable'),('HTTP 520',None),('ReadTimeout',None)])
def test_fatal_errors_are_distinct_from_transient(message,expected):
    assert fatal_provider_error({'finish_reason':'error','error':{'message':message}})==expected
    assert fatal_provider_error({'finish_reason':'stop','error':message}) is None


def test_quota_failure_stops_next_persistent_batch_without_fake_results(tmp_path):
    workspace=tmp_path/'episode';(workspace/'agent_run').mkdir(parents=True)
    (workspace/'agent_run/result.json').write_text(json.dumps({'finish_reason':'error','error':{'message':'HTTP 403: 余额不足'}}))
    flag=tmp_path/'API_STOP.json'
    assert pause_on_provider_error(workspace,flag)=='quota_exhausted'
    spec=importlib.util.spec_from_file_location('eval_fixes',Path(__file__).parents[1]/'scripts/eval_robodojo_agent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    emitted=[]
    module._run_persistent_task_batch([{'task':'align_blocks'}], args=None,output_dir=tmp_path,
        device_id=0,slot_index=0,model_metadata={},on_result=emitted.append)
    assert emitted==[]
    assert not (tmp_path/'batches').exists()


def test_small_multitask_trial_uses_all_available_slots():
    spec=importlib.util.spec_from_file_location('eval_assignment',Path(__file__).parents[1]/'scripts/eval_robodojo_agent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    episodes=[{'task':task,'layout_id':4} for task in ['align','arrange','build','classify']]
    slots=module._persistent_slot_assignments(episodes,4)
    assert [len(slot) for slot in slots]==[1,1,1,1]
    assert sorted(x['task'] for slot in slots for x in slot)==sorted(x['task'] for x in episodes)
