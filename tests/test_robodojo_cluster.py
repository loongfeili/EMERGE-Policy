import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot.vla.robodojo_policy import encode_observation, unpack_joint_actions
from robot.vla.openpi_bridge import _parse_server_endpoint
from robot.robodojo_simulation.eval_bridge import DirectAgentTransport, direct_agent_transport
from robot.robodojo_simulation.action_controller import RoboDojoActionController

ROOT = Path(__file__).resolve().parents[1]

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_dual_arm_codec_preserves_camera_and_joint_order():
    image=np.zeros((480,640,3),np.uint8)
    obs={'vision':{k:{'color':image} for k in ('cam_head','cam_left_wrist','cam_right_wrist')},
         'state':np.arange(14,dtype=np.float32),'instruction':'stack the bowls'}
    encoded=encode_observation(obs)
    assert encoded['images']['cam_high'].shape==(3,480,640)
    np.testing.assert_array_equal(encoded['state'],np.arange(14))
    action=unpack_joint_actions(np.arange(14)[None,:])[0]
    np.testing.assert_array_equal(action['left_arm_joint_state'],np.arange(6))
    np.testing.assert_array_equal(action['right_arm_joint_state'],np.arange(7,13))
    assert action['left_ee_joint_state'][0]==6
    assert action['right_ee_joint_state'][0]==13
    with pytest.raises(ValueError):unpack_joint_actions(np.zeros((10,7)))


def test_ipv6_health_url_is_bracketed():
    assert _parse_server_endpoint('ws://[fdbd:dc03::73]:8000') == (
        'fdbd:dc03::73',8000,'http://[fdbd:dc03::73]:8000')


def test_transport_restored_on_environment_failure():
    original=object();module=SimpleNamespace(WsModelClient=original)
    with pytest.raises(RuntimeError):
        with direct_agent_transport(module):
            assert module.WsModelClient is DirectAgentTransport
            module.WsModelClient().call('infer')
    assert module.WsModelClient is original


def test_partial_assets_cannot_silently_shrink_standard_eval(tmp_path):
    m=load_script('eval_robodojo_agent')
    config=tmp_path/'task/RoboDojo/config';config.mkdir(parents=True)
    (config/'_task.yml').write_text('common:\n  eval_nums: 2\n')
    layouts=tmp_path/'Assets/Eval_Layout/RoboDojo/arx_x5/0';layouts.mkdir(parents=True)
    (layouts/'task_0.json').write_text('{}')
    with pytest.raises(ValueError,match='Missing required layouts'):
        m._layouts_per_task(['task'],'native',robodojo_root=tmp_path,env_cfg='arx_x5',policy_seed=0)


def test_official_partial_credit_is_not_binary_success():
    m=load_script('summarize_robodojo_eval')
    assert m._official_episode_score({'success':False,'official_score':.35})==35
    assert m._official_episode_score({'success':False}) is None


def test_cancel_prevents_policy_request_or_simulation_step():
    driver=RoboDojoActionController(SimpleNamespace(),policy_client=object())
    assert driver.execute('vla_execute',{'instruction':'stack','step':10},
                          cancel_check=lambda:'visual monitor stopped action').startswith('Interrupted:')


def test_perception_resize_preserves_projection_and_original_policy_images(tmp_path):
    from robot.robodojo_simulation.observation_publisher import RoboDojoObservationPublisher
    image=np.zeros((480,640,3),np.uint8)
    k=np.array([[400.,0,320],[0,400,240],[0,0,1]])
    camera=SimpleNamespace(get_intrinsics_matrix=lambda **kw:k,
                           get_world_pose=lambda **kw:(np.array([1.,2.,3.]),np.array([1.,0.,0.,0.])))
    env=SimpleNamespace(camera_manager=SimpleNamespace(camera_names=[['cam_head']],cameras=[[camera]]),
                        env_origins=np.array([[.5,1.,2.]]))
    pub=RoboDojoObservationPublisher(env,workspace=tmp_path,camera_names=['cam_head'])
    observation={'vision':{'cam_head':{'color':image}}}
    views=pub._capture(observation)
    assert views[0].get_rgb().shape==(392,518,3)
    point=np.array([.1,.2,1.])
    np.testing.assert_allclose(views[0].get_intrinsics()@point,
                              np.diag([518/640,392/480,1])@k@point)
    np.testing.assert_allclose(views[0].get_world_camera_transform()[:3,3],[.5,1.,1.])
    assert image.shape==(480,640,3)


def test_openpi_connection_uses_wire_protocol_and_closes():
    import threading
    from websockets.sync.server import serve
    from external_model_server.protocol import pack_message, unpack_message
    from robot.vla.openpi_bridge import _BoundedPolicyConnection
    def handler(ws):
        ws.send(pack_message({'service':'test-policy'}))
        request=unpack_message(ws.recv())
        ws.send(pack_message({'actions':request['state'][None,:]}))
    with serve(handler,'127.0.0.1',0) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        conn=_BoundedPolicyConnection('127.0.0.1',server.socket.getsockname()[1],2)
        try:
            out=conn.infer({'state':np.arange(14,dtype=np.float32)})
            np.testing.assert_array_equal(out['actions'],np.arange(14)[None,:])
        finally:
            conn.close();server.shutdown();thread.join(timeout=3)


def test_policy_replicas_are_selected_by_gpu():
    import importlib.util
    path = Path(__file__).parents[1] / "scripts/eval_robodojo_agent.py"
    spec=importlib.util.spec_from_file_location("eval_replicas", path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    urls="ws://[::1]:8000,ws://[::1]:8003"
    assert mod._policy_url_for_device(urls, 0)=="ws://[::1]:8000"
    assert mod._policy_url_for_device(urls, 7)=="ws://[::1]:8003"


def test_official_unstable_scene_is_excluded_not_a_policy_failure():
    m=load_script('summarize_robodojo_eval')
    results={str(i):{'task':'insert_key','layout_id':i,'success':False,
                    'official_score':0.5,'termination_reason':'official_failure'} for i in range(50)}
    results['0'].update(official_excluded=True,official_score=None,
                       termination_reason='official_excluded:unstable_scene')
    result=m.summarize(results,expected_episodes=50,shards={'.':50})
    assert result['official_excluded_episodes']==1
    assert result['infrastructure_errors']==0
    task=result['official_protocol_seed0']['task_scores']['insert_key']
    assert task['complete'] and task['score_percent']==50
    assert task['official_excluded_episodes']==1


def test_agent_budget_exhaustion_does_not_trigger_selective_retry():
    m=load_script('eval_robodojo_agent')
    assert m._termination_reason({'finished':True,'success':False,'agent_finish_reason':'max_iterations'},timed_out=False,llm_failed=False,returncode=1)=='official_failure:agent_budget_exhausted'
    assert m._termination_reason({'finished':True,'success':False},timed_out=False,llm_failed=True,returncode=1)=='infrastructure_error:llm'
