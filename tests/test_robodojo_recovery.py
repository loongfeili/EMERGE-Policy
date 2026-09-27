from types import SimpleNamespace

import numpy as np
import pytest

from robot.robodojo_simulation.action_controller import RoboDojoActionController


@pytest.mark.parametrize('failure', ['Failed: move_linear did not converge', 'Interrupted: cancelled'])
def test_failed_retreat_is_not_reported_as_recovered_or_followed_by_park(monkeypatch, failure):
    controller = RoboDojoActionController(SimpleNamespace(end_flag=[False]))
    calls = []
    def move(params):
        calls.append(params)
        return failure
    monkeypatch.setattr(controller, '_move_linear', move)
    assert controller.execute('recover', {'arm': 'left', 'mode': 'park'}) == failure
    assert len(calls) == 1


@pytest.mark.parametrize('success', [False, True])
def test_terminal_during_gripper_command_stops_recovery(monkeypatch, success):
    env = SimpleNamespace(end_flag=[False], success=[success])
    controller = RoboDojoActionController(env)
    def grip(params):
        env.end_flag[0] = True
        return controller._terminal_action_result('set_gripper')
    monkeypatch.setattr(controller, '_set_gripper', grip)
    monkeypatch.setattr(controller, '_move_linear', lambda p: pytest.fail('stepped after terminal'))
    result = controller.execute('recover', {'arm': 'right'})
    assert ('official task success' in result) if success else result.startswith('Failed:')
    assert 'lifted' not in result


@pytest.mark.parametrize('clearance', [float('nan'), float('inf'), -0.1, 0])
def test_invalid_recovery_height_is_rejected_before_gripper_action(monkeypatch, clearance):
    controller = RoboDojoActionController(SimpleNamespace())
    monkeypatch.setattr(controller, '_set_gripper', lambda p: pytest.fail('invalid request moved robot'))
    assert controller.execute('recover', {'arm': 'left', 'clearance_m': clearance}).startswith('Failed:')


def test_recovery_uses_live_pose_and_checks_arrival_with_native_quaternion_order():
    # A minimal native protocol model: apply actual wxyz targets, then read them
    # back through the manager. This exercises the relative target/frame boundary.
    poses = {arm: np.array([.1, -.2, .8, 1., 0., 0., 0.]) for arm in ('left', 'right')}
    robots = {arm: SimpleNamespace(arm=arm, gripper_scale=[0., .044], gripper_move={'sign': 1})
              for arm in poses}
    manager = SimpleNamespace(
        get_robot_by_arm_name=lambda name: robots[name.removesuffix('_arm')],
        get_real_endpose=lambda robot, **kwargs: [poses[robot.arm].copy()],
        get_end_effector_real_val=lambda robot, **kwargs: [np.array([.022])],
    )
    actions = []
    def step(action):
        actions.append(action)
        for arm in poses:
            poses[arm] = action[arm + '_ee_pose'].copy()
    controller = RoboDojoActionController(SimpleNamespace(robot_manager=manager, take_action=step,
                                                          end_flag=[False]))
    result = controller.execute('recover', {'arm': 'left', 'mode': 'park', 'steps': 2})
    assert not result.startswith('Failed:')
    np.testing.assert_allclose(poses['left'], [.1, -.35, .92, 1., 0., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(poses['right'], [.1, -.2, .8, 1., 0., 0., 0.], atol=1e-6)
    assert len(actions) == 4


def test_policy_error_is_latched_only_for_its_episode(monkeypatch):
    from robot.vla.openpi_bridge import Pi05Client
    from robot.drivers.robodojo_driver import RoboDojoDriver
    client = Pi05Client('ws://localhost:8000')
    monkeypatch.setattr(client, 'health_check', lambda: True)
    attempts = 0
    def infer(obs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError('injected timeout')
        return {'actions': np.zeros((10, 14))}
    transport = SimpleNamespace(infer=infer, close=lambda: None)
    monkeypatch.setattr(client, '_ensure_client', lambda: transport)
    with pytest.raises(TimeoutError):
        client.infer({})
    client.infer({})
    assert 'injected timeout' in client.last_error
    driver = object.__new__(RoboDojoDriver)
    driver._infrastructure_error = None
    driver._policy_client = client
    with pytest.raises(RuntimeError, match='injected timeout'):
        driver.raise_if_infrastructure_error()
    client.begin_episode()
    client.infer({})
    driver.raise_if_infrastructure_error()
    assert client.last_error is None
