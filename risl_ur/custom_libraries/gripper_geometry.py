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

# Half the fingertip length perpendicular to the opening direction, in meters.
GRIPPER_FINGER_OFFSET = 0.010


def gripper_width_to_reach(width_tenths_mm):
    """Interpolate axial reach in meters, holding endpoints outside the table."""
    if not np.isfinite(width_tenths_mm):
        raise ValueError('Gripper opening must be finite')
    keys = sorted(GRIPPER_TABLE)
    return float(np.interp(
        width_tenths_mm, keys, [GRIPPER_TABLE[key] for key in keys]))
