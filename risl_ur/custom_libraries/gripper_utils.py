import numpy as np
import time
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PolygonStamped, Point32
from risl_ur.custom_libraries.generallibraries import pose_text
from risl_ur.custom_libraries.actionlibrariesmax import move, gripper_width

GRIPPER_TABLE = { # Known, measured values. Gripper width in 0.1mm.
       0: 0.153,
     200: 0.150,
     400: 0.147,
     600: 0.141,
     800: 0.132,
    1000: 0.114,
}

# Measured distance from gripper finger center to tip, normal to span
# AKA gripper fingers are twice as long as this distance. 
GRIPPER_FINGER_OFFSET = 0.010 # m

class GripperHandler:
    """Contains a live gripper width adjuster that widens the gripper as the ee gets close to the table. 
    A closed gripper will protrude further, causing collision."""
    def __init__(self, node: Node, vertical_offset = 0.01):
        self.node = node
        self.gripper_pub = self.node.create_publisher(String, "/gripper_command", 10)
        self.tip_pub = self.node.create_publisher(PolygonStamped, "/gripper/tip_points", 10)
        # self.gripper_query_sub = self.node.create_subscription(Float64, 'gripper_queried_width', self.query_callback, 10)
        self.previous_gripper_cmd = 0
        # self.queried_width_mm = 0.0
        self.action_tolerance_mm = 0.5 # Only move gripper if queried width is more than this amount different
        # self.timer = self.node.create_timer(0.1, self.get_gripper)
        self.vertical_offset = vertical_offset # Virtual table level in mm
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
            # print(f"\nCommanded width {self.previous_gripper_cmd} is not close to actual width {width}")
            # print(f"interpreted height is {interpreted_height}")
            self.publish(str(width))
            self.previous_gripper_cmd = width

        self.last_ee_position = ee_position
        self.last_ee_euler = ee_euler
        self.publish_tip_points(ee_position, ee_euler)

    def publish(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.gripper_pub.publish(msg)

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

    # def query_callback(self, msg: Float64):
    #     self.queried_width_mm = msg.data
    #     print(f"Query Complete! Verdict: {msg.data}")

    # def get_gripper(self):
    #     msg = String()
    #     msg.data = str("query")
    #     self.gripper_pub.publish(msg)

    def gripper_width_to_height(self, gripper_width):
        # Clamp input to valid range
        if gripper_width <= 0:
            return GRIPPER_TABLE[0]
        if gripper_width >= 1000:
            return GRIPPER_TABLE[1000]

        # Sort the keys to find where gripper_width fits
        keys = sorted(GRIPPER_TABLE.keys())
        for i in range(len(keys) - 1):
            low = keys[i]
            high = keys[i + 1]
            if low <= gripper_width <= high:
                # Linear interpolation
                low_val = GRIPPER_TABLE[low]
                high_val = GRIPPER_TABLE[high]
                t = (gripper_width - low) / (high - low)
                return low_val + t * (high_val - low_val)

    def height_to_gripper_width(self, height):
        # Convert GRIPPER_TABLE to (height -> width) tuples for inverse lookup
        items = sorted(((v, k) for k, v in GRIPPER_TABLE.items()), reverse=True)

        # Clamp to valid range
        if height >= items[0][0]:
            return items[0][1]
        if height <= items[-1][0]:
            return items[-1][1]

        # Find surrounding interval
        for i in range(len(items) - 1):
            high_val, low_key = items[i]
            low_val, high_key = items[i + 1]
            if high_val >= height >= low_val:
                t = (height - low_val) / (high_val - low_val)
                return int(high_key + t * (low_key - high_key))

    def EE_pose_to_pushers_2D(self, EE_pose, gripper_width_cmd=None):
        """
        Returns the gripper positions given the current EE pose, assuming the gripper width is set according to height.
        Currently doesn't intelligently distinguish which gripper's which.
        Pusher 1 is the left one when looking down the camera.
        Pusher positions are [x, y, 0]
        """
        position, orientation = EE_pose
        center_x, center_y, height = position
        height -= self.vertical_offset
        yaw = orientation[2]
        yaw_r = np.deg2rad(yaw)
        if gripper_width_cmd is None:
            gripper_width_cmd = self.height_to_gripper_width(height)
        gripper_width = gripper_width_cmd * 0.0001 # Conversion to meters
        half_width = gripper_width / 2

        dx = half_width * np.cos(yaw_r)
        dy = half_width * np.sin(yaw_r)
        pusher1_position = [center_x + dx, center_y + dy, 0.0]
        pusher2_position = [center_x - dx, center_y - dy, 0.0]

        return pusher1_position, pusher2_position

    def pushers_to_EE_pose_2D(self, pusher_1_position, pusher_2_position, given_yaw=0):
        """
        Returns the EE pose such that the gripper can place its two tips at the given XY locations
        pusher_positions are [x1, y1, z1], [x2, y2, z2] meters, where zs are ignored
        returns position, orientation_deg
        Prone to singularity (pushers at the same point)
        """
        pos1_np = np.array(pusher_1_position)
        pos2_np = np.array(pusher_2_position)
        gripper_width = np.linalg.norm(pos1_np-pos2_np)
        # Assert that the two points are within the gripper maximum (110 mm in this case)
        if gripper_width > 0.11:
            return None
        gripper_height = self.gripper_width_to_height(gripper_width*10000) # conversion to their weird measurement
        center = (pos1_np + pos2_np) / 2
        position = [center[0], center[1], gripper_height + self.vertical_offset]
        if gripper_width < 0.01:
            yaw = given_yaw
        else:
            # If gripper width is more than 10 mm, probably okay to measure angle.
            yaw = np.atan2(pos1_np[1]-pos2_np[1], pos1_np[0]-pos2_np[0])
            yaw = np.rad2deg(yaw)
        orientation = [0, 180, yaw]
        return position, orientation

    def pointspan_to_pushers(self, pusher_1, span, yaw_d):
        """Provides a pair of pusher positions given pusher 1's position, span, and the angle"""
        yaw_r = np.deg2rad(yaw_d)
        dx = span * np.cos(yaw_r)
        dy = span * np.sin(yaw_r)
        new_x = pusher_1[0] - dx
        new_y = pusher_1[1] - dy
        pusher_2 = [new_x, new_y, 0.0]
        return pusher_1, pusher_2

    def pointspan_to_EE_pose_2D(self, pusher_1, span, yaw_d):
        """
        Provides a EE pose given pusher 1's position, span, and the angle
        Pusher positions are [x, y, --] meters
        given yaw is in degrees
        span is in meters
        pose is [position, euler orientation degrees]
        """
        yaw_r = np.deg2rad(yaw_d)
        half_span = span/2
        dx = half_span * np.cos(yaw_r)
        dy = half_span * np.sin(yaw_r)
        cen_x = pusher_1[0] - dx
        cen_y = pusher_1[1] - dy
        gripper_height = self.gripper_width_to_height(span*10000) # conversion to their weird measurement
        position = [cen_x, cen_y, gripper_height + self.vertical_offset]
        orientation = [0, 180, yaw_d]
        return position, orientation

    def EE_pose_to_pointspan(self, EE_pose):
        """
        Returns the pointspan given the current EE pose, assuming the gripper width is set according to height.
        Pusher 1 is the left one when looking down the camera.
        Pusher positions are [x, y, 0]
        """
        position, orientation = EE_pose
        center_x, center_y, height = position
        height -= self.vertical_offset
        yaw_d = orientation[2]
        yaw_r = np.deg2rad(yaw_d)
        gripper_width = self.height_to_gripper_width(height) * 0.0001 # Conversion to meters
        half_width = gripper_width / 2

        dx = half_width * np.cos(yaw_r)
        dy = half_width * np.sin(yaw_r)
        pusher1_position = [center_x + dx, center_y + dy, 0.0]

        return pusher1_position, gripper_width, yaw_d

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
        if not EE_pose_now[0][2] > GRIPPER_TABLE[0] + 0.01:
            moves.append(move(retreat_position, initial_ori, duration))
            print(f"{'RETREAT POSE:':19}{pose_text(retreat_pose)}")

        # Part 2: Raise. Move to a height of 300mm
        raise_position = retreat_position
        raise_position[2] = 0.3 + self.vertical_offset
        initial_width = self.height_to_gripper_width(initial_pos[2] - self.vertical_offset) * 0.0001
        if initial_width > 0.04:
            intermed_height = self.gripper_width_to_height(400) + self.vertical_offset
            intermed_position = raise_position.copy()
            intermed_position[2] = intermed_height
            intermed_pose = [intermed_position, initial_ori]
            print(f"{'INTERMED 1 POSE:':19}{pose_text(intermed_pose)}")
            
            # Skip if EE starting position is already high
            if not EE_pose_now[0][2] > GRIPPER_TABLE[0] + 0.01:
                moves.extend([move(intermed_position, initial_ori, duration),
                            move(raise_position, initial_ori, 2*duration)])
        else:
            if not EE_pose_now[0][2] > GRIPPER_TABLE[0] + 0.01:
                moves.append(move(raise_position, initial_ori, 2*duration))

        raise_pose = [raise_position, initial_ori]
        # Skip if EE starting position is already high
        if not EE_pose_now[0][2] > GRIPPER_TABLE[0] + 0.01:
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
        lower_position = advance_position
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

# # Unconverted function
# def pointspan_to_trajectories(EE_pose_now, pusher_1_target, span_target, yaw_target, duration):
#     """
#     Provides a list of joint angle trajectories based on where you want your pushers
#     Pusher positions are [x, y, --] meters
#     given yaw is in degrees
#     span is in meters
#     pose is [position, euler orientation degrees]
#     """
#     print(f"""We're currently at:
# {pose_text(EE_pose_now)}""")
#     pusher_1_now, pusher_2_now = EE_pose_to_pushers_2D(EE_pose_now)
#     print("pushers now")
#     print(pushers_text(pusher_1_now, pusher_2_now))
#     pos1_now_np = np.array(pusher_1_now)
#     pos2_now_np = np.array(pusher_2_now)
#     pusher_1_target, pusher_2_target = pointspan_to_pushers(pusher_1_target, span_target, yaw_target)
#     print("pushers target")
#     print(pushers_text(pusher_1_target, pusher_2_target))
#     pos1_target_np = np.array(pusher_1_target)
#     pos2_target_np = np.array(pusher_2_target)
#     width_now = np.linalg.norm(pos1_now_np - pos2_now_np)
#     width_target = np.linalg.norm(pos1_target_np - pos2_target_np)
#     # Checks if the 40mm width mark is crossed
#     if width_now <= 0.04 <= width_target or width_target <= 0.04 <= width_now:
#         # If it is, we need an intermediate waypoint. 
#         # Otherwise, the controller will increase/decrease height linearly, which can bump the gripper into the table.
#         # This is to have the controller raise/lower more slowly when it matters most.

#         # If pushers move linearly at constant speed, then width will increase linearly
#         # We need to find at what point does width = 0.04.
#         proportion = abs(0.04 - width_now) / abs(width_target - width_now)
#         # Then, find the vector each pusher travels along and find the appropriate point.
#         vec1 = pos1_target_np - pos1_now_np
#         vec2 = pos2_target_np - pos2_now_np
#         prop_vec1 = proportion * vec1
#         prop_vec2 = proportion * vec2
#         intermed_pos1 = pos1_now_np + prop_vec1
#         intermed_pos2 = pos2_now_np + prop_vec2
#         print("pushers intermed")
#         print(pushers_text(intermed_pos1, intermed_pos2))
#         intermed_position, intermed_rpy = pushers_to_EE_pose_2D(intermed_pos1, intermed_pos2)
#         position, rpy = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
#         print(f"""Got coordinates: 
# {pose_text([intermed_position, intermed_rpy])}
#            then:
# {pose_text([position, rpy])}""")
#         return [move(intermed_position, intermed_rpy, duration), move(position, rpy, duration)]
#     else:
#         position, rpy = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
#         print(f"""Got coordinates: 
# {pose_text([position, rpy])}""")
#         return [move(position, rpy, duration)]
