# 🤖 UR5e Custom IK Control (ROS 2)

This **ROS 2 package** provides custom inverse kinematics (IK) control for the **UR5e** robot, eliminating the need for MoveIt. It enables direct joint control and has been tested on **ROS 2 Humble**.

## ✨ Features

- Custom inverse kinematics implementation
- Direct joint control of the UR5e robot
- Compatible with ROS 2 Humble
- Support for tool communication (e.g., grippers)

## 📦 Dependencies

Ensure the **UR ROS 2 driver** is installed:

```bash
sudo apt-get install ros-humble-ur
```
## 📂 Pakage Structure 

```risl_ur/
└── risl_ur/
    ├── custom_libraries/
    │   └── inverse_kinematics.py
    ├── scripts/
    │   └── <all scripts go here>
    ├── resource/
    ├── test/
    ├── setup.cfg
    ├── setup.py
    └── package.xml
```

## 🛠️ Setup
1. Connect the UR5e robot to your network. 

2. In one terminal **source ROS Humble**. Launch the UR ROS 2 driver with tool communication enabled:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=192.168.1.111 \
  use_tool_communication:=true \
  tool_baud_rate:=1000000 \
  use_mock_hardware:=false \
  tool_voltage:=24 \
  tool_parity:=2 \
  tool_stop_bits:=1 \
  tool_rx_idle_chars:=1.5 \
  tool_tx_idle_chars:=3.5
```

3. Start controllers, if not done already. Run `ros2 control set_controller_state scaled_joint_trajectory_controller active`

## 🔧 Testing

To test the package:

1. In a second terminal, **source Humble, overlay the local workspace** and navigate to the package directory:
  ```
   cd <your_ros2_ws>/src/risl_ur/risl_ur/scripts
```

2. Run the test script:
```
  python3 testcartesian.py
```

## Scripts and Their Functions

`testcartesian.py`: Pick and Place routine with preset locations

`testgripperhover_1.py`: Defunct method to include gripper width in waypoint definitions.

`testgripperhover_2_checkpushers.py`: Test script to verify calculations for gripper finger locations. 

`testgripperhover_2_sendpushers.py`: Test script for defining waypoints as gripper finger locations.

`testgripperhover_2.py`: Modern method of adaptively adjusting gripper width based on EE height above table.

`testobjectfollow.py --target [target_name]`: EE follows user-defined object at set height above table. Connects to `risl_camera_localizer`.

`testobjectprepush.py`: EE sets up gripper for push, based on recommended pusher positions. Connects to `risl_camera_localizer`.

`testspinaround.py`: rotates the end-effector 360 degrees to verify that IK keeps all other joint angles the same.  


## Demonstration

The following video represents a culmination of capabilities from this package and `risl_camera_localizer`. Using the pusher recommendation system from that package, two contour points are published over ROS2 and read by a subscriber node. These two points, representing the desired location of the gripper fingertips, are converted to exact end-effector position and orientation. As the object moves over time, these points may change and necessitate a switch. A sequence of moves is then built such that the gripper retracts before switching position (as to not collide with the object), and automatically adjusts its raise/lower rate (as to keep the gripper from colliding with the table.) 

![Prepush demonstration](./media/trimmed20prepush.gif)

## Motion helper ownership and units

`custom_libraries/gripper_geometry.py` owns the calibration table and all planar
pose/pusher/pointspan conversions. `GripperGeometry` can be used without ROS;
`GripperHandler` inherits the same methods, including `EE_pose_to_pushers_2D`.
Positions, reach, spans and vertical offsets are in meters, Euler angles are in
degrees, and width commands are in tenths of a millimeter. Pusher 1 lies along
positive yaw from the tool center. Planar conversions ignore Z when measuring
pusher span and return projected pusher positions at Z=0.

`GripperHandler` owns command publication and the last commanded width.
`PusherHandler.update(position, euler, gripper=gripper)` uses that same geometry
and command state, including when automatic resizing is disabled. These published
positions are estimates from commands, not measured finger positions. Default
recommendation subscriptions use `/vision/recommended_pusher_{1,2}/{position,normal}`;
pass an explicit topic list to override them, or `[]` to disable subscriptions.

The measured reach table ends at 100 mm opening. Reach interpolation keeps that
endpoint for larger openings, although the driver accepts commands up to 110 mm.
Retreat/raise/lower planning lives in `GripperHandler.pointspan_to_acts_safe`;
`actionlibraries` builds action dictionaries, `MotionExecutor` sends them through
ROS controllers, and `ControllerManager` handles controller services.

Run the offline motion regressions from the workspace root (no robot required):

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/src/risl_ur:$PYTHONPATH" python3 -m pytest -q src/risl_ur/test/test_motion_logic.py
```
