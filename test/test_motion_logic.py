"""Offline regression checks: no ROS node or robot connections are created."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from risl_ur.custom_libraries.gripper_geometry import (
    GRIPPER_TABLE, GripperGeometry, gripper_reach_to_width, gripper_width_to_reach,
)


@pytest.mark.parametrize('width,reach', GRIPPER_TABLE.items())
def test_calibration_knots(width, reach):
    assert gripper_width_to_reach(width) == reach
    assert gripper_reach_to_width(reach) == width


def test_calibration_clamps_and_rejects_nonfinite():
    assert gripper_width_to_reach(1100) == GRIPPER_TABLE[1000]
    assert gripper_reach_to_width(1) == 0
    assert gripper_reach_to_width(0) == 1000
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            gripper_width_to_reach(value)
        with pytest.raises(ValueError):
            gripper_reach_to_width(value)


@pytest.mark.parametrize('yaw', [-170, -90, 0, 45, 170])
@pytest.mark.parametrize('offset', [0.003, 0.01])
def test_planar_pose_roundtrip_ignores_z(yaw, offset):
    geometry = GripperGeometry(offset)
    pose = ([0.1, -0.4, 0.141 + offset], [0, 180, yaw])
    p1, p2 = geometry.EE_pose_to_pushers_2D(pose)
    p1[2], p2[2] = 1, -1
    recovered = geometry.pushers_to_EE_pose_2D(p1, p2)
    np.testing.assert_allclose(recovered, pose, atol=1e-10)
    point, span, angle = geometry.EE_pose_to_pointspan(pose)
    np.testing.assert_allclose(
        geometry.pointspan_to_EE_pose_2D(point, span, angle), pose, atol=1e-10)


def test_explicit_width_overrides_height_and_coincident_points_use_given_yaw():
    geometry = GripperGeometry()
    p1, p2 = geometry.EE_pose_to_pushers_2D(([0, 0, 1], [0, 180, 90]), 800)
    np.testing.assert_allclose(p1, [0, 0.04, 0], atol=1e-12)
    np.testing.assert_allclose(p2, [0, -0.04, 0], atol=1e-12)
    assert geometry.pushers_to_EE_pose_2D([0, 0, 1], [0, 0, -1], 42)[1][2] == 42
    assert geometry.pushers_to_EE_pose_2D([0, 0, 0], [0.12, 0, 0]) is None
    with pytest.raises(ValueError):
        geometry.pointspan_to_EE_pose_2D([0, 0, 0], 0.12, 0)


@pytest.fixture
def handlers():
    pytest.importorskip('rclpy')
    from builtin_interfaces.msg import Time
    from risl_ur.custom_libraries.gripper_utils import GripperHandler
    from risl_ur.custom_libraries.pusher_utils import PusherHandler
    node = Mock()
    node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    node.create_publisher.side_effect = lambda *args: Mock()
    return GripperHandler(node, vertical_offset=0.003), PusherHandler(node)


@pytest.mark.parametrize('command,width', [('600', 600), ('open', 1100), ('close', 0)])
def test_manual_width_is_used_for_pusher_and_tip_publishing(handlers, command, width):
    gripper, pusher = handlers
    gripper.publish(command)
    assert gripper.previous_gripper_cmd == width
    pose = ([0.1, -0.4, 0.2], [0, 180, 0])
    gripper.publish_tip_points(*pose)
    pusher.update(*pose, gripper=gripper)
    left = pusher.pusher_pub_l.publish.call_args.args[0]
    right = pusher.pusher_pub_r.publish.call_args.args[0]
    assert left.point.x - right.point.x == pytest.approx(width * 0.0001)
    tips = gripper.tip_pub.publish.call_args.args[0].polygon.points
    assert tips[2].x == pytest.approx(left.point.x)
    assert tips[3].x == pytest.approx(right.point.x)
    assert tips[2].z == pytest.approx(0.2 - 0.003 - gripper_width_to_reach(width))
    assert left.header.frame_id == 'base'


def test_default_recommendations_and_incomplete_data(handlers):
    _, pusher = handlers
    assert len(pusher.recommended_subscriptions) == 4
    assert not pusher.pusher_recommendations_exist()
    with pytest.raises(ValueError, match='incomplete'):
        pusher.extract_pusher_locations()
    from geometry_msgs.msg import PointStamped, Vector3Stamped
    for number in (1, 2):
        point = PointStamped()
        point.point.x = number * 0.02
        normal = Vector3Stamped()
        normal.vector.y = 1.0
        pusher.push_recommend_callback(point, f'pusher_{number}_position')
        pusher.push_recommend_callback(normal, f'pusher_{number}_normal')
    assert pusher.pusher_recommendations_exist()
    assert pusher.extract_pusher_locations() == ([0.02, 0, 0], [0.04, 0, 0], 0)


def test_waypoints_keep_retreat_and_lower_heights_separate(handlers, monkeypatch):
    from risl_ur.custom_libraries import gripper_utils
    gripper, _ = handlers
    monkeypatch.setattr(gripper_utils, 'move', lambda pos, ori, duration: {
        'position': pos, 'orientation': ori, 'duration': duration})
    pose = [[0.1, -0.4, 0.144], [0, 180, 0]]
    moves = gripper.pointspan_to_acts_safe(pose, [0.2, -0.4, 0], 0.06, 0, 3)
    assert moves[0]['position'][2] == 0.144
    assert moves[2]['position'][2] == 0.303
    assert moves[3]['type'] == 'gripper'
    assert moves[-2]['position'][2] == pytest.approx(0.144)
    assert moves[-1]['position'][2] == pytest.approx(0.144)
    assert pose == [[0.1, -0.4, 0.144], [0, 180, 0]]


@pytest.fixture
def executor(monkeypatch):
    pytest.importorskip('rclpy')
    from risl_ur.custom_libraries import motion_utils
    monkeypatch.setattr(motion_utils.rclpy, 'spin_until_future_complete', Mock())
    executor = motion_utils.MotionExecutor.__new__(motion_utils.MotionExecutor)
    executor.node = Mock()
    executor.controller_manager = Mock()
    executor.position_controller = 'position'
    executor.velocity_controller = 'velocity'
    executor.force_controller = 'force'
    executor.passthrough_controller = 'passthrough'
    executor.joint_names = ['joint']
    executor.traj_client = Mock()
    executor.start_force_client = Mock()
    return executor


@pytest.mark.parametrize('succeeded', [True, False])
def test_fractional_trajectory_duration_and_controller_result(executor, succeeded):
    goal_handle = Mock(accepted=True)
    goal_handle.get_result_async.return_value.result.return_value = SimpleNamespace(
        status=4 if succeeded else 6, result=SimpleNamespace(error_code=0))
    executor.traj_client.send_goal_async.return_value.result.return_value = goal_handle
    assert executor.send_joint_angles([0.0], 1.25) is succeeded
    goal = executor.traj_client.send_goal_async.call_args.args[0]
    duration = goal.trajectory.points[0].time_from_start
    assert (duration.sec, duration.nanosec) == (1, 250_000_000)


def test_force_failure_and_exception_always_stop(executor):
    executor.send_cartesian_force_step = Mock(return_value=False)
    executor.stop_force_mode = Mock(return_value=True)
    assert executor.send_cartesian_force([0] * 6, 1) is False
    executor.stop_force_mode.assert_called_once()
    executor.send_cartesian_force_step.side_effect = RuntimeError('service failed')
    with pytest.raises(RuntimeError, match='service failed'):
        executor.send_cartesian_force([0] * 6, 1)
    assert executor.stop_force_mode.call_count == 2


def test_force_trajectory_stops_after_first_failed_step(executor, monkeypatch):
    from risl_ur.custom_libraries import motion_utils
    monkeypatch.setattr(motion_utils.rclpy, 'ok', lambda: True)
    executor.send_cartesian_force_step = Mock(return_value=False)
    executor.stop_force_mode = Mock(return_value=True)
    assert executor.send_cartesian_force_trajectory([[0] * 6] * 3, 0.1) is False
    executor.send_cartesian_force_step.assert_called_once()
    executor.stop_force_mode.assert_called_once()


def test_failed_ik_does_not_create_an_unusable_action(monkeypatch):
    pytest.importorskip('rclpy')
    from risl_ur.custom_libraries import actionlibraries
    monkeypatch.setattr(actionlibraries, 'compute_ik', lambda *args: None)
    with pytest.raises(ValueError, match='IK failed'):
        actionlibraries.move([0, 0, 0], [0, 180, 0], 1)
    actions = {'action_note': 'ignored', 'act2': {}}
    actionlibraries.append_new_act(actions, {'type': 'gripper', 'cmd': 600})
    assert actions['act3']['cmd'] == 600


@pytest.mark.parametrize('yaw', [-170, -90, 0, 45, 170])
def test_maximum_pointspan_survives_floating_point_roundoff(yaw):
    position, orientation = GripperGeometry().pointspan_to_EE_pose_2D(
        [0.5, -0.5, 0], 0.11, yaw)
    assert position[2] == pytest.approx(0.124)
    assert orientation[2] == yaw


def test_query_preserves_command_state_and_numeric_commands_match_driver(handlers):
    gripper, _ = handlers
    gripper.publish('600.0')
    assert gripper.gripper_pub.publish.call_args.args[0].data == '600'
    gripper.publish('query')
    assert gripper.previous_gripper_cmd == 600
    assert gripper.gripper_pub.publish.call_args.args[0].data == 'query'
    with pytest.raises(ValueError):
        gripper.publish('600.5')


def test_failed_controller_switch_does_not_send_a_goal(executor):
    executor.controller_manager.switch_to_controller.return_value = False
    assert executor.send_joint_angles([0.0], 1) is False
    executor.traj_client.send_goal_async.assert_not_called()


def test_async_force_reuses_the_same_execution_path(executor, monkeypatch):
    from risl_ur.custom_libraries import motion_utils
    thread = Mock()
    monkeypatch.setattr(motion_utils.threading, 'Thread', thread)
    assert executor.send_cartesian_force_async([1.0] * 6, 0.5) is True
    assert thread.call_args.kwargs['target'] == executor.send_cartesian_force
    assert thread.call_args.kwargs['args'] == ([1.0] * 6, 0.5)
    thread.return_value.start.assert_called_once()
