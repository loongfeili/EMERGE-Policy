"""AC-WM mediation against the RoboDojo dual-arm controller and evaluation."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import robot.robodojo_simulation.action_controller as controller_module
from Emerge.ac_wm import ObservationRollout
from Emerge.ac_wm.subagent import AcWmSubagent
from Emerge.agent.tools.embodied import EmbodiedActionTool
from robot.robodojo_simulation.action_controller import RoboDojoActionController
from robot.robodojo_simulation.observation_publisher import RoboDojoObservationPublisher

ARMS = ("left", "right")
ROOT = Path(__file__).resolve().parents[1]


def _proposal(result, marker):
    assert result.startswith(marker), result
    return json.loads(result[len(marker):])


class Cell:
    """Native RoboDojo protocol model: arms track each command exactly."""

    def __init__(self, tmp_path, *, success_after=None, fail_after=None):
        self.poses = {arm: np.array([.1, -.2, .8, 1., 0., 0., 0.]) for arm in ARMS}
        self.joints = {arm: np.arange(6, dtype=float) * .1 for arm in ARMS}
        self.grippers = {arm: .022 for arm in ARMS}
        self.actions = []
        self.success_after = success_after
        self.fail_after = fail_after
        robots = {arm: SimpleNamespace(arm=arm, gripper_scale=[0., .044], gripper_move={'sign': 1})
                  for arm in ARMS}
        manager = SimpleNamespace(
            get_robot_by_arm_name=lambda name: robots[name.removesuffix('_arm')],
            get_real_endpose=lambda robot, **kw: [self.poses[robot.arm].copy()],
            get_end_effector_real_val=lambda robot, **kw: [np.array([self.grippers[robot.arm]])],
            get_joint=lambda robot, **kw: [self.joints[robot.arm].copy()],
        )
        self.env = SimpleNamespace(robot_manager=manager, take_action=self.step, end_flag=[False],
                                   success=[False], get_obs=lambda: {'vision': {}})
        self.snapshots = []
        self.published = []
        frame = tmp_path / 'cam_head.png'
        cv2.imwrite(str(frame), np.full((8, 8, 3), 120, np.uint8))

        def snapshot(observation=None):
            self.snapshots.append(observation)
            return {'observation_path': str(tmp_path / 'cam_head.mp4'),
                    'observation_images': [str(frame)], 'observation_revision': len(self.snapshots)}

        self.publisher = SimpleNamespace(write_ac_wm_snapshot=snapshot, publish=self.published.append)

    def step(self, action):
        self.actions.append(action)
        for arm in ARMS:
            if f'{arm}_ee_pose' in action:
                self.poses[arm] = np.asarray(action[f'{arm}_ee_pose'], dtype=float).copy()
            if f'{arm}_arm_joint_state' in action:
                self.joints[arm] = np.asarray(action[f'{arm}_arm_joint_state'], dtype=float).copy()
            self.grippers[arm] = float(action[f'{arm}_ee_joint_state'][0]) * .044
        if self.success_after is not None and len(self.actions) >= self.success_after:
            self.env.end_flag[0] = self.env.success[0] = True
        if self.fail_after is not None and len(self.actions) >= self.fail_after:
            self.env.end_flag[0] = True

    def controller(self, policy=None, config=None):
        return RoboDojoActionController(self.env, config, policy_client=policy,
                                        observation_publisher=self.publisher)


def test_move_linear_preview_is_ee16_and_holds_the_other_arm(tmp_path):
    cell = Cell(tmp_path)
    result = cell.controller().execute('rule_propose', {
        'skill_action_type': 'move_linear',
        'parameters': {'arm': 'left', 'delta_m': [0., 0., .08], 'steps': 4},
    })
    proposal = _proposal(result, 'RULE_PROPOSAL:')
    rows = np.asarray(proposal['actions'])
    assert rows.shape == (4 + 20, 16)
    assert proposal['execute_steps'] == len(rows)
    assert (proposal['domain_name'], proposal['control_space']) == ('robodojo_ee', 'robodojo_ee16')
    assert 'quaternion xyzw' in proposal['control_description']
    np.testing.assert_allclose(rows[0, :3], [.1, -.2, .82])
    np.testing.assert_allclose(rows[3, :3], [.1, -.2, .88])
    np.testing.assert_allclose(rows[:, 3:7], np.tile([0., 0., 0., 1.], (len(rows), 1)))
    np.testing.assert_allclose(rows[:, 8:16], np.tile([.1, -.2, .8, 0., 0., 0., 1., .5], (len(rows), 1)))
    assert cell.actions == [] and cell.snapshots == [None]


def test_every_geometric_skill_previews_without_moving(tmp_path):
    cell = Cell(tmp_path)
    controller = cell.controller()
    cases = {
        'move_to_pose': ({'arm': 'right', 'position_m': [.2, -.2, .8], 'orientation_quat': [0, 0, 0, 1],
                          'attempts': 5}, (5, 16)),
        'follow_arc': ({'arm': 'left', 'center': [.1, -.3, .8], 'axis': [0, 0, 1], 'radius_m': .1,
                        'angle_deg': 90, 'steps': 4}, (24, 16)),
        'set_gripper': ({'arm': 'both', 'command': 'open', 'steps': 3}, (3, 14)),
    }
    for skill, (parameters, shape) in cases.items():
        proposal = _proposal(controller.execute('rule_propose', {
            'skill_action_type': skill, 'parameters': parameters}), 'RULE_PROPOSAL:')
        assert np.asarray(proposal['actions']).shape == shape, skill
    gripper_rows = np.asarray(proposal['actions'])
    np.testing.assert_allclose(gripper_rows[0], [*np.arange(6) * .1, 1., *np.arange(6) * .1, 1.])
    assert proposal['control_space'] == 'robodojo_joint14'
    assert cell.actions == []
    assert controller.execute('rule_propose', {'skill_action_type': 'recover', 'parameters': {}}).startswith('Failed:')


def test_selected_ee16_chunk_replays_native_wxyz_and_stops_on_arrival(tmp_path):
    cell = Cell(tmp_path)
    controller = cell.controller()
    proposal = _proposal(controller.execute('rule_propose', {
        'skill_action_type': 'move_linear',
        'parameters': {'arm': 'left', 'delta_m': [0., .05, 0.], 'steps': 4},
    }), 'RULE_PROPOSAL:')
    result = controller.execute('execute_action_chunk', {
        'actions': proposal['actions'], 'control_space': 'robodojo_ee16',
        'candidate_id': 'move', 'skill_name': 'rule:move_linear',
    })
    assert result.endswith('steps=4, reason=target_reached.'), result
    assert len(cell.actions) == 4
    np.testing.assert_allclose(cell.actions[0]['left_ee_pose'], [.1, -.1875, .8, 1., 0., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(cell.poses['left'][:3], [.1, -.15, .8], atol=1e-6)
    np.testing.assert_allclose(cell.poses['right'][:3], [.1, -.2, .8], atol=1e-6)


def test_ee16_chunk_that_cannot_arrive_fails_with_its_step_count(tmp_path):
    cell = Cell(tmp_path)
    cell.env.take_action = lambda action: cell.actions.append(action)
    row = [.3, -.2, .8, 0, 0, 0, 1, .5, .1, -.2, .8, 0, 0, 0, 1, .5]
    result = cell.controller().execute('execute_action_chunk', {
        'actions': [row] * 3, 'control_space': 'robodojo_ee16', 'skill_name': 'rule:move_to_pose'})
    assert result.startswith('Failed:') and result.endswith('steps=3, reason=not_converged'), result


def test_selected_joint14_chunk_is_verbatim_until_official_success(tmp_path):
    cell = Cell(tmp_path, success_after=2)
    rows = np.linspace(0, 1, 4 * 14).reshape(4, 14)
    result = cell.controller().execute('execute_action_chunk', {
        'actions': rows.tolist(), 'candidate_id': 'vla-1', 'skill_name': 'vla'})
    assert result.endswith('steps=2, reason=goal_reached.'), result
    np.testing.assert_allclose(cell.actions[1]['right_arm_joint_state'], rows[1, 7:13], rtol=1e-6)
    np.testing.assert_allclose(cell.actions[1]['left_ee_joint_state'], rows[1, 6:7], rtol=1e-6)

    failed = Cell(tmp_path, fail_after=1)
    result = failed.controller().execute('execute_action_chunk', {'actions': rows.tolist()})
    assert result.startswith('Failed:') and 'steps=1, reason=failed_terminal' in result


@pytest.mark.parametrize('params', [
    {'actions': [[0.] * 15]},
    {'actions': [[float('nan')] * 14]},
    {'actions': [[0.] * 14] * 65},
    {'actions': [[0.] * 16], 'control_space': 'robodojo_ee16'},
    {'actions': [[.1, -.2, .8, 0, 0, 0, 1, 1.5] * 2], 'control_space': 'robodojo_ee16'},
    {'actions': [[0.] * 14], 'control_space': 'cartesian'},
])
def test_invalid_selected_rows_are_rejected_before_any_motion(tmp_path, params):
    cell = Cell(tmp_path)
    assert cell.controller().execute('execute_action_chunk', params).startswith('Failed:')
    assert cell.actions == []


def test_interrupted_chunk_reports_steps_already_executed(tmp_path):
    cell = Cell(tmp_path)
    reason = lambda: 'visual monitor completed step_2' if len(cell.actions) >= 2 else None
    result = cell.controller().execute('execute_action_chunk', {'actions': [[0.] * 14] * 5},
                                       cancel_check=reason)
    assert result.startswith('Interrupted:') and result.endswith('steps=2, reason=interrupted')


def test_vla_proposal_infers_from_a_published_observation_without_stepping(tmp_path, monkeypatch):
    monkeypatch.setattr(controller_module, 'encode_observation', lambda observation: dict(observation))
    cell = Cell(tmp_path)
    seen = []
    policy = SimpleNamespace(infer=lambda element: seen.append(element) or {'actions': np.ones((50, 14))})
    proposal = _proposal(cell.controller(policy).execute('vla_propose', {
        'instruction': 'stack the bowls', 'horizon': 64, 'max_execute_steps': 7}), 'VLA_PROPOSAL:')
    assert np.asarray(proposal['actions']).shape == (50, 14)
    assert proposal['execute_steps'] == 7
    assert (proposal['domain_name'], proposal['control_space']) == ('robodojo_joint', 'robodojo_joint14')
    assert seen[0]['instruction'] == 'stack the bowls'
    assert cell.published[0]['instruction'] == 'stack the bowls'
    assert cell.snapshots[0]['instruction'] == 'stack the bowls'
    assert cell.actions == []

    proposal = _proposal(cell.controller(policy, {'ac_wm_execute_steps': 25}).execute('vla_propose', {
        'instruction': 'stack the bowls', 'horizon': 64, 'max_execute_steps': 100}), 'VLA_PROPOSAL:')
    assert proposal['execute_steps'] == 25


def test_publisher_snapshot_reuses_the_latest_frames(tmp_path):
    k = np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]])
    camera = SimpleNamespace(get_intrinsics_matrix=lambda **kw: k,
                             get_world_pose=lambda **kw: (np.array([0., 0., 2.]), np.array([1., 0, 0, 0])))
    names = ['cam_head', 'cam_left_wrist']
    env = SimpleNamespace(camera_manager=SimpleNamespace(camera_names=[names], cameras=[[camera, camera]]),
                          env_origins=np.zeros((1, 3)))
    publisher = RoboDojoObservationPublisher(env, workspace=tmp_path, camera_names=names)
    head = np.zeros((48, 64, 3), np.uint8)
    head[..., 0] = 200
    publisher.publish({'vision': {'cam_head': {'color': head},
                                  'cam_left_wrist': {'color': np.zeros((48, 64, 3), np.uint8)}}})
    snapshot = publisher.write_ac_wm_snapshot()
    assert snapshot['observation_revision'] == publisher.revision == 1
    assert [Path(path).name for path in snapshot['observation_images']] == ['cam_head.png', 'cam_left_wrist.png']
    saved = cv2.imread(snapshot['observation_images'][0])
    assert saved.shape == (48, 64, 3) and saved[0, 0, 2] == 200  # RGB red stored as BGR
    capture = cv2.VideoCapture(snapshot['observation_path'])
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()
    with pytest.raises(RuntimeError, match='reference camera'):
        RoboDojoObservationPublisher(env, workspace=tmp_path / 'fresh', camera_names=names).write_ac_wm_snapshot()


def test_driver_does_not_republish_after_a_non_stepping_proposal():
    from robot.drivers.robodojo_driver import RoboDojoDriver

    driver = object.__new__(RoboDojoDriver)
    driver._connected = True
    driver._environment = SimpleNamespace(sim=object())
    driver._actions = SimpleNamespace(execute=lambda *args, **kwargs: 'ok')
    published = []
    driver._publish_observation = lambda: published.append(True)
    for action_type in ('vla_propose', 'rule_propose'):
        driver.execute_action(action_type, {})
    assert published == []
    driver.execute_action('execute_action_chunk', {})
    assert published == [True]


class ScriptedAcWm:
    """Return one prepared verdict per selection, like the real subagent."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.tasks = []

    async def run(self, task):
        self.tasks.append(task)
        status, dispatch, error, metadata = self.verdicts.pop(0)
        output = {'selected_candidate_id': 'c', 'score': .5, 'dispatch_result': dispatch} if dispatch else None
        return SimpleNamespace(status=SimpleNamespace(value=status), summary='verdict', output=output,
                               error=error, metadata=metadata)


def _chunk(steps, reason='chunk_completed', suffix=''):
    return ('success', "Action 'execute_action_chunk' validated and dispatched to hardware. Execution completed. "
            f'Result: Selected candidate c from skill vla executed its exact proposed prefix: '
            f'steps={steps}, reason={reason}.{suffix}', None,
            {'selected_candidate_id': 'c', 'dispatched': True, 'evaluations': [{'score': .5}]})


def _run_vla(tmp_path, verdicts, step=30):
    scripted = ScriptedAcWm(verdicts)
    tool = EmbodiedActionTool(workspace=tmp_path, ac_wm_subagent=scripted)
    result = json.loads(asyncio.run(tool.execute('vla_execute', {'instruction': 'stack', 'step': step}, 'r')))
    return result, scripted


def test_vla_budget_is_spent_in_judged_chunks(tmp_path):
    result, scripted = _run_vla(tmp_path, [_chunk(10), _chunk(10), _chunk(10)])
    assert (result['status'], result['stop_reason'], result['steps_executed']) == ('success', 'budget_exhausted', 30)
    assert [task.input['step'] for task in scripted.tasks] == [30, 20, 10]
    log = (tmp_path / 'artifacts/ac-wm/decisions.jsonl').read_text().splitlines()
    assert len(log) == 3 and json.loads(log[0])['dispatched'] is True


@pytest.mark.parametrize('verdicts,expected', [
    ([_chunk(10), ('failed', None, 'no candidate', {'evaluations': [{'score': 0.}], 'dispatched': False})],
     ('partial', 'ac_wm_rejected', 10)),
    ([_chunk(10), ('failed', None, 'skill proposal failed: timeout', {})], ('partial', 'ac_wm_failed', 10)),
    ([_chunk(10), ('failed', None, 'Error: Robot action failed. Failed: RoboDojo selected chunk reached a failed '
                   'terminal state; steps=3, reason=failed_terminal',
                   {'selected_candidate_id': 'c', 'evaluations': [{'score': .5}], 'dispatched': False})],
     ('partial', 'failed_terminal', 13)),
    ([_chunk(4, 'goal_reached')], ('success', 'goal_reached', 4)),
    ([_chunk(6, 'interrupted', ' The visual monitor already verified the current subgoal')],
     ('success', 'visual_monitor_verified', 6)),
    ([('failed', None, 'no candidate', {'evaluations': [{'score': 0.}], 'dispatched': False})],
     ('failed', 'ac_wm_rejected', 0)),
])
def test_vla_loop_stops_for_the_first_non_continuing_chunk(tmp_path, verdicts, expected):
    result, scripted = _run_vla(tmp_path, verdicts)
    assert (result['status'], result['stop_reason'], result['steps_executed']) == expected
    assert len(scripted.tasks) == len(verdicts)


def test_robodojo_actions_flow_through_ac_wm_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(controller_module, 'encode_observation', lambda observation: dict(observation))
    cell = Cell(tmp_path)
    policy = SimpleNamespace(infer=lambda element: {'actions': np.zeros((50, 14))})
    controller = cell.controller(policy)
    judged = []

    def judge(task, candidate, rollout):
        judged.append((task, candidate, rollout))
        return .5, 'clear progress'

    tool = EmbodiedActionTool(workspace=tmp_path)

    async def dispatch(action_type, parameters):
        return controller.execute(action_type, parameters)

    tool._dispatch_action = dispatch
    tool.ac_wm_subagent = AcWmSubagent(
        provider=SimpleNamespace(get_default_model=lambda: 'judge'), rollout=ObservationRollout(), judge=judge,
        candidate_provider=tool.propose_action_candidates, dispatch=tool.dispatch_selected_candidate)

    result = json.loads(asyncio.run(tool.execute(
        'move_linear', {'arm': 'left', 'delta_m': [0., 0., .05], 'steps': 5}, 'lift the left gripper')))
    assert result['status'] == 'success', result
    assert result['output']['skill_name'] == 'rule:move_linear'
    assert 'reason=target_reached' in result['output']['dispatch_result']
    np.testing.assert_allclose(cell.poses['left'][:3], [.1, -.2, .85], atol=1e-6)
    assert judged[0][0].startswith('lift the left gripper')
    assert judged[0][1].metadata['control_space'] == 'robodojo_ee16'
    assert judged[0][2].metadata['frames']

    moved = len(cell.actions)
    result = json.loads(asyncio.run(tool.execute('vla_execute', {'instruction': 'stack', 'step': 20}, 'r')))
    assert (result['status'], result['steps_executed'], len(result['chunks'])) == ('success', 20, 2)
    assert len(cell.actions) - moved == 20
    assert judged[-1][1].metadata['control_space'] == 'robodojo_joint14'


def _eval_module():
    spec = importlib.util.spec_from_file_location('eval_ac_wm', ROOT / 'scripts/eval_robodojo_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    return SimpleNamespace(**{'ac_wm': None, 'ac_wm_rollout': None, 'ac_wm_judge_model': None,
                              'policy_baseline': False, **overrides})


def test_eval_ac_wm_switch_is_exported_to_every_agent(tmp_path):
    module = _eval_module()
    env = {}
    assert module._configure_ac_wm(_args(), env) == {'ac_wm': {'enabled': False}}
    assert env['EMERGE_AC_WM'] == '0'

    env = {'EMERGE_AC_WM': '1'}
    metadata = module._configure_ac_wm(_args(ac_wm_judge_model='judge'), env)['ac_wm']
    assert (metadata['enabled'], metadata['rollout'], metadata['judge_model']) == (True, 'observation', 'judge')
    assert (env['EMERGE_AC_WM_ROLLOUT'], env['EMERGE_AC_WM_JUDGE_MODEL']) == ('observation', 'judge')

    with pytest.raises(ValueError, match='policy-baseline'):
        module._configure_ac_wm(_args(ac_wm=True, policy_baseline=True), {})
    with pytest.raises(ValueError, match='COSMOS_PYTHON'):
        module._configure_ac_wm(_args(ac_wm=True, ac_wm_rollout='cosmos'), {})
    python = tmp_path / 'python'
    python.write_text('')
    metadata = module._configure_ac_wm(_args(ac_wm=True, ac_wm_rollout='cosmos'), {
        'COSMOS_PYTHON': str(python), 'COSMOS_CHECKPOINT': 'ckpt',
        'COSMOS_DOMAIN_MAP': 'robodojo_joint=a,robodojo_ee=b'})['ac_wm']
    assert metadata['cosmos_domain_map'] == {'robodojo_joint': 'a', 'robodojo_ee': 'b'}


def test_eval_records_each_episode_ac_wm_verdicts(tmp_path):
    module = _eval_module()
    log = tmp_path / 'artifacts/ac-wm/decisions.jsonl'
    log.parent.mkdir(parents=True)
    log.write_text('\n'.join(json.dumps(item) for item in [
        {'action_type': 'vla_execute', 'status': 'success', 'dispatched': True, 'score': .5,
         'selected_candidate_id': 'a', 'evaluations': [{}]},
        {'action_type': 'vla_execute', 'status': 'failed', 'dispatched': False,
         'selected_candidate_id': None, 'evaluations': [{}]},
        {'action_type': 'move_linear', 'status': 'failed', 'dispatched': False, 'evaluations': []},
    ]) + '\nnot json\n')
    assert module._ac_wm_stats(tmp_path, {}) == {}
    stats = module._ac_wm_stats(tmp_path, {'ac_wm': {'enabled': True}})['ac_wm_stats']
    assert stats == {'decisions': 3, 'dispatched': 1, 'rejected': 1, 'failed': 1, 'mean_selected_score': .5,
                     'by_action': {'vla_execute': 2, 'move_linear': 1}}
