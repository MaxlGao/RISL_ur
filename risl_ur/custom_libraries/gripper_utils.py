import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PolygonStamped, Point32
from risl_ur.custom_libraries.generallibraries import pose_text
from risl_ur.custom_libraries.actionlibraries import move, gripper_width

# Re-export these names for existing users of gripper_utils.
from risl_ur.custom_libraries.gripper_geometry import (
    GRIPPER_TABLE, GRIPPER_FINGER_OFFSET, GripperGeometry, gripper_width_to_reach,
)

class GripperHandler(GripperGeometry):
    """Contains a live gripper width adjuster that widens the gripper as the ee gets close to the table. 
    A closed gripper will protrude further, causing collision."""
    def __init__(self, node: Node, vertical_offset = 0.01):
        super().__init__(vertical_offset)
        self.node = node
        self.gripper_pub = self.node.create_publisher(String, "/gripper_command", 10)
        self.tip_pub = self.node.create_publisher(PolygonStamped, "/gripper/tip_points", 10)
        self.previous_gripper_cmd = 0
        self.action_tolerance_mm = 0.5 # Only move gripper if queried width is more than this amount different
        self.last_ee_position = None
        self.last_ee_euler = None
        self.tip_publish_timer = self.node.create_timer(0.05, self.publish_tip_points_from_last_ee_position)
        # offset = 0.000 means gripper tips always on table (dangerous)
        print("Created gripper")

    def update(self, ee_position, ee_euler=None):
        # Set the gripper width to the planned ee position
        # Increasing the virtual floor reduces the interpreted height.
        interpreted_height = ee_position[2] - self.vertical_offset
        width = self.height_to_gripper_width(interpreted_height)

        # Only move gripper if different from last command.
        if abs(self.previous_gripper_cmd - width) > 10*self.action_tolerance_mm:
            self.publish(str(width))

        self.last_ee_position = ee_position
        self.last_ee_euler = ee_euler
        self.publish_tip_points(ee_position, ee_euler)

    def publish(self, cmd: str):
        """Publish a driver command and remember its commanded (unmeasured) width."""
        cmd = str(cmd).lower()
        if cmd == "query":
            self.gripper_pub.publish(String(data=cmd))
            return
        if cmd == "open":
            width = 1100
        elif cmd == "close":
            width = 0
        else:
            width = float(cmd)
        if not np.isfinite(width) or not 0 <= width <= 1100 or not float(width).is_integer():
            raise ValueError("Gripper opening must be between 0 and 1100 tenths of mm")
        msg = String()
        msg.data = cmd if cmd in ("open", "close") else str(int(width))
        self.gripper_pub.publish(msg)
        self.previous_gripper_cmd = width

    def publish_tip_points_from_last_ee_position(self):
        if self.last_ee_position is None:
            return
        self.publish_tip_points(self.last_ee_position, self.last_ee_euler)

    def publish_tip_points(self, ee_position, ee_euler):
        """Publish the two gripper tip points (finger endpoints) as a PolygonStamped.

        This can be used for visualization / localization to see where the controller
        thinks the gripper finger tips are located.

        ee_position: [x, y, z]
        ee_euler:   [roll, pitch, yaw] (degrees) (unused if None)
        """
        # Use only the XY pose and yaw for finger direction; assume 0 Z for contact plane
        # (matching the table plane).
        # If ee_euler is not provided, assume yaw=0.
        yaw = 0.0
        if ee_euler is not None and len(ee_euler) >= 3:
            yaw = ee_euler[2]

        p1, p2 = self.EE_pose_to_pushers_2D([ee_position, [0, 0, yaw]], self.previous_gripper_cmd)
        # That gives z=0, so separately we subtract gripper_width_to_height from ee z to get the correct z height for the tips.
        tip_z = ee_position[2] - self.vertical_offset - self.gripper_width_to_height(self.previous_gripper_cmd)
        p1[2] = tip_z
        p2[2] = tip_z
        yaw_r = np.deg2rad(yaw)
        edge_dx = GRIPPER_FINGER_OFFSET * np.cos(yaw_r+np.pi/2) # Rotate 90 degrees to get the direction perpendicular to the gripper span
        edge_dy = GRIPPER_FINGER_OFFSET * np.sin(yaw_r+np.pi/2)
        p1_adjusted = [p1[0] - edge_dx, p1[1] - edge_dy, p1[2]]
        p2_adjusted = [p2[0] - edge_dx, p2[1] - edge_dy, p2[2]]
        poly = PolygonStamped()
        poly.header.stamp = self.node.get_clock().now().to_msg()
        poly.header.frame_id = "base"
        poly.polygon.points = [
            Point32(x=float(p1_adjusted[0]), y=float(p1_adjusted[1]), z=float(p1_adjusted[2])),
            Point32(x=float(p2_adjusted[0]), y=float(p2_adjusted[1]), z=float(p2_adjusted[2])),
            Point32(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            Point32(x=float(p2[0]), y=float(p2[1]), z=float(p2[2])),
        ]
        self.tip_pub.publish(poly)

    def pointspan_to_acts_safe(self, EE_pose_now, pusher_1_target, 
                                        span_target, yaw_target, duration):
        """
        Improved with retract, raise, move, lower, and advance actions. 
        Fortunately we only need the EE pose at the first instant. 
        """
        print(f"We're currently at {pose_text(EE_pose_now)}")
        moves = []
        # Part 1: Retreat. Find the yaw, and set a waypoint 1 cm backward
        retreat_distance = 0.03
        initial_pos, initial_ori = EE_pose_now
        # At 0 yaw, the retreat direction is towards +90 degrees.
        retreat_angle = initial_ori[2] + 90
        retreat_angle_r = np.deg2rad(retreat_angle)
        retreat_dx = retreat_distance * np.cos(retreat_angle_r)
        retreat_dy = retreat_distance * np.sin(retreat_angle_r)
        retreat_x = initial_pos[0] + retreat_dx
        retreat_y = initial_pos[1] + retreat_dy
        retreat_position = [retreat_x, retreat_y, initial_pos[2]]
        retreat_pose = [retreat_position, initial_ori]

        # Skip if EE starting position is already high
        if initial_pos[2] <= GRIPPER_TABLE[0] + self.vertical_offset:
            moves.append(move(retreat_position, initial_ori, duration))
            print(f"{'RETREAT POSE:':19}{pose_text(retreat_pose)}")

        # Part 2: Raise. Move to a height of 300mm
        raise_position = retreat_position.copy()
        raise_position[2] = 0.3 + self.vertical_offset
        initial_width = self.height_to_gripper_width(initial_pos[2] - self.vertical_offset) * 0.0001
        if initial_width > 0.04:
            intermed_height = self.gripper_width_to_height(400) + self.vertical_offset
            intermed_position = raise_position.copy()
            intermed_position[2] = intermed_height
            intermed_pose = [intermed_position, initial_ori]
            print(f"{'INTERMED 1 POSE:':19}{pose_text(intermed_pose)}")
            
            # Skip if EE starting position is already high
            if initial_pos[2] <= GRIPPER_TABLE[0] + self.vertical_offset:
                moves.extend([move(intermed_position, initial_ori, duration),
                            move(raise_position, initial_ori, 2*duration)])
        else:
            if initial_pos[2] <= GRIPPER_TABLE[0] + self.vertical_offset:
                moves.append(move(raise_position, initial_ori, 2*duration))

        raise_pose = [raise_position, initial_ori]
        # Skip if EE starting position is already high
        if initial_pos[2] <= GRIPPER_TABLE[0] + self.vertical_offset:
            print(f"{'RAISE POSE:':19}{pose_text(raise_pose)}")


        # Part 6: Advance. Move above a final position. 
        # We calculate this first, then back calculate the other steps.
        end_pos, end_ori = self.pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)

        # For the end position we must consider the gripper width. 
        # So, we repeat some of Part 1's retreat. 
        advance_angle = end_ori[2] + 90
        advance_angle_r = np.deg2rad(advance_angle)
        end_dx = GRIPPER_FINGER_OFFSET * np.cos(advance_angle_r)
        end_dy = GRIPPER_FINGER_OFFSET * np.sin(advance_angle_r)
        end_x = end_pos[0] + end_dx
        end_y = end_pos[1] + end_dy
        end_position = [end_x, end_y, end_pos[2]]
        end_moves = [move(end_position, end_ori, duration)] # To be reversed later
        # We assume that the camera side is facing the object, so moving forward means approaching.
        advance_distance = retreat_distance
        advance_dx = advance_distance * np.cos(advance_angle_r)
        advance_dy = advance_distance * np.sin(advance_angle_r)
        advance_x = end_pos[0] + advance_dx
        advance_y = end_pos[1] + advance_dy
        advance_position = [advance_x, advance_y, end_pos[2]]
        end_moves.append(move(advance_position, end_ori, duration))
        
        # Part 5: Lower. Again, reverse Part 2.
        lower_position = advance_position.copy()
        lower_position[2] = 0.3 + self.vertical_offset

        end_width = self.height_to_gripper_width(end_pos[2] - self.vertical_offset) * 0.0001
        if end_width > 0.04:
            intermed_height = self.gripper_width_to_height(400) + self.vertical_offset
            intermed_position = lower_position.copy()
            intermed_position[2] = intermed_height
            end_moves.extend([move(intermed_position, end_ori, duration),
                        move(lower_position, end_ori, 2*duration)])
        else:
            end_moves.append(move(lower_position, end_ori, 2*duration))
        lower_pose = [lower_position, end_ori]
        print(f"{'SLIDE:':19}{pose_text(lower_pose)}")
        advance_pose = [advance_position, end_ori]
        print(f"{'LOWER POSE:':19}{pose_text(advance_pose)}")
        end_pose = [end_pos, end_ori]
        print(f"{'END POSE:':19}{pose_text(end_pose)}")

        # Part 4: Gripper. Size the gripper for the end height. We do this before the EE begins lowering, so that the gripper is already at the correct width when it gets close to the table. This reduces the risk of collision with the table.
        gripper_width_planned = self.height_to_gripper_width(end_pos[2] - self.vertical_offset)
        end_moves.append(gripper_width(gripper_width_planned))
        # self.set_width_for_height(end_pos[2])

        # Part 3: Translate. Simply join the two lists together
        moves.extend(reversed(end_moves))
        return moves
