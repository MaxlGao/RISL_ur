"""Shared measured RG2 geometry, referenced to the zero-offset UR tool frame."""

import numpy as np


# Keys: opening in 0.1 mm. Values: tool0 to distal fingertip edge in meters.
GRIPPER_TABLE = {
    0: 0.153,
    200: 0.150,
    400: 0.147,
    600: 0.141,
    800: 0.132,
    1000: 0.114,
}
# FYI: distance undershoots on linear interpolation; quadratic fit uses
# dist = -4.4196e-8 * width^2 + 7.7677e-6 * width + 0.1518, with width in 0.1 mm and dist in m.

# Half the fingertip length perpendicular to the opening direction, in meters.
GRIPPER_FINGER_OFFSET = 0.010


def gripper_width_to_reach(width_tenths_mm):
    """Interpolate axial reach in meters, holding endpoints outside the table."""
    if not np.isfinite(width_tenths_mm):
        raise ValueError('Gripper opening must be finite')
    keys = sorted(GRIPPER_TABLE)
    return float(np.interp(
        width_tenths_mm, keys, [GRIPPER_TABLE[key] for key in keys]))


def gripper_reach_to_width(reach_m):
    """Invert the reach table; return an integer opening in tenths of a mm."""
    if not np.isfinite(reach_m):
        raise ValueError('Gripper reach must be finite')
    items = sorted((reach, width) for width, reach in GRIPPER_TABLE.items())
    return int(np.interp(reach_m, [item[0] for item in items],
                         [item[1] for item in items]))


class GripperGeometry:
    """Planar gripper conversions, independent of ROS.

    Positions, spans and vertical_offset are meters; Euler angles are degrees.
    Width commands are tenths of a millimeter. Pusher 1 lies along positive yaw
    from the tool center. Projected pushers lie on z=0 and ignore roll/pitch.
    """

    def __init__(self, vertical_offset=0.01):
        self.vertical_offset = vertical_offset

    def gripper_width_to_height(self, gripper_width):
        return gripper_width_to_reach(gripper_width)

    def height_to_gripper_width(self, height):
        return gripper_reach_to_width(height)

    def EE_pose_to_pushers_2D(self, EE_pose, gripper_width_cmd=None):
        """
        Returns the gripper positions given the current EE pose, assuming the gripper width is set according to height.
        Pusher 1 lies along positive yaw from the tool center.
        Pusher positions are [x, y, 0]
        """
        position, orientation = EE_pose
        center_x, center_y, height = position
        height -= self.vertical_offset
        yaw = orientation[2]
        yaw_r = np.deg2rad(yaw)
        if gripper_width_cmd is None:
            gripper_width_cmd = self.height_to_gripper_width(height)
        if not np.isfinite(gripper_width_cmd) or not 0 <= gripper_width_cmd <= 1100:
            raise ValueError("Gripper opening must be between 0 and 1100 tenths of mm")
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
        Uses given_yaw for spans below 10 mm, including coincident pushers.
        """
        pos1_np = np.asarray(pusher_1_position, dtype=float)[:2]
        pos2_np = np.asarray(pusher_2_position, dtype=float)[:2]
        if not (np.all(np.isfinite(pos1_np)) and np.all(np.isfinite(pos2_np))):
            raise ValueError("Pusher XY positions must be finite")
        gripper_width = np.linalg.norm(pos1_np-pos2_np)
        # Assert that the two points are within the gripper maximum (110 mm in this case)
        if gripper_width > 0.11 + 1e-12:
            return None
        gripper_height = self.gripper_width_to_height(gripper_width*10000) # meters to tenths of a millimeter
        center = (pos1_np + pos2_np) / 2
        position = [center[0], center[1], gripper_height + self.vertical_offset]
        if gripper_width < 0.01:
            yaw = given_yaw
        else:
            # If gripper width is more than 10 mm, probably okay to measure angle.
            yaw = np.arctan2(pos1_np[1]-pos2_np[1], pos1_np[0]-pos2_np[0])
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
        if not np.isfinite(span) or not 0 <= span <= 0.11:
            raise ValueError("Pusher span must be between 0 and 0.11 meters")
        p1, p2 = self.pointspan_to_pushers(pusher_1, span, yaw_d)
        position, _ = self.pushers_to_EE_pose_2D(p1, p2, yaw_d)
        return position, [0, 180, yaw_d]

    def EE_pose_to_pointspan(self, EE_pose):
        """
        Returns the pointspan given the current EE pose, assuming the gripper width is set according to height.
        Pusher 1 is the left one when looking down the camera.
        Pusher positions are [x, y, 0]
        """
        position, orientation = EE_pose
        width = self.height_to_gripper_width(position[2] - self.vertical_offset)
        pusher_1, _ = self.EE_pose_to_pushers_2D(EE_pose, width)
        return pusher_1, width * 0.0001, orientation[2]
