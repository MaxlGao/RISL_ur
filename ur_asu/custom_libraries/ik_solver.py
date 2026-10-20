import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from ur_asu.custom_libraries.generallibraries import pose_text
import time

# UR5e DH parameters
dh_params = [
    (0,  0.1625,  0,     np.pi/2),  
    (0,  0,      -0.425,  0),       
    (0,  0,      -0.3922, 0),       
    (0,  0.1333,  0,     np.pi/2),  
    (0,  0.0997,  0,    -np.pi/2),  
    (0,  0.0996,  0,     0)
]

def rpy_to_matrix(rpy):
    """In degrees"""
    return R.from_euler('xyz', rpy, degrees=True).as_matrix()

def matrix_to_rpy(matrix):
    """In degrees"""
    return R.from_matrix(matrix).as_euler('xyz', degrees=True)

def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1]
    ])

def forward_kinematics(dh_params, joint_angles):
    T = np.eye(4)
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
        T = np.dot(T, T_i)
    return T

def Tmat_to_pose(Tmat):
    position = Tmat[:3, 3]
    rot_matrix = Tmat[:3, :3]
    rpy = matrix_to_rpy(rot_matrix)
    return (position, rpy)

def ik_objective(q, target_pose, verbose=False):
    T_fk = forward_kinematics(dh_params, q)
    pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])
    # rot_error = np.linalg.norm(T_fk[:3, :3] - target_pose[:3, :3])
    R_fk = T_fk[:3, :3]
    R_target = target_pose[:3, :3]
    R_err = R_fk.T @ R_target
    angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    # if verbose:
    #     print(f"POS ERROR {pos_error}")
    #     print(f"ROT ERROR {angle_err}")
    return 1.0 * pos_error + 0.2 * angle_err

def compute_ik(position, rpy, q_guess=None, max_tries=5, dx=0.001, q6_sweep=np.linspace(-np.pi, np.pi, 3)):
    target_pose = np.eye(4)
    target_pose[:3, 3] = position
    target_pose[:3, :3] = rpy_to_matrix(rpy)

    if q_guess is None:
        q6 = -(np.mod(rpy[2] + 180, 360) - 180) # Adjust initial guess based on given yaw.
        q_guess = np.radians([90, -80, 90, -90, -90, q6])

    joint_bounds = [(0.0, np.pi)] + [(-np.pi, np.pi)] * 5
    best_q, best_cost = None, np.inf

    # --- quick coarse sweep on q6 ---
    for q6 in q6_sweep:
        q_try = q_guess.copy()
        q_try[5] = q6
        res = minimize(
            ik_objective, q_try, args=(target_pose,),
            method='L-BFGS-B', bounds=joint_bounds,
            options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-5}
            # options={'maxiter': 30}
        )
        cost = ik_objective(res.x, target_pose)
        if cost < best_cost:
            best_q, best_cost = res.x, cost
        if best_cost < 0.01:
            return best_q  # early success; don't bother refining

    # --- refine the best one more deeply ---
    if best_q is not None:
        res = minimize(
            ik_objective, best_q, args=(target_pose,),
            method='L-BFGS-B', bounds=joint_bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            # options={'maxiter': 500}
        )
        best_q, best_cost = res.x, ik_objective(res.x, target_pose)

    if cost > 0.01:
        print(f"\nWARNING: HIGH IK ERROR! Cost for pose {pose_text((position, rpy))} is {cost}")
        actual_tmat = forward_kinematics(dh_params, best_q)
        actual_position, actual_rotation = Tmat_to_pose(actual_tmat)
        print(f"Actual Achieved Pose:                 {pose_text((actual_position, actual_rotation))}")

    return best_q


def compute_jacobian(joint_angles):
    """
    Compute the 6x6 Jacobian matrix at the given joint configuration using DH parameters.
    Returns:
        J: A 6x6 numpy array
    """
    n = len(joint_angles)
    T = np.eye(4)
    origins = [T[:3, 3]]
    z_axes = [T[:3, 2]]

    Ts = []  # Store intermediate transforms

    # Compute transformation matrices up to each joint
    for i in range(n):
        theta, d, a, alpha = dh_params[i]
        T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
        T = T @ T_i
        Ts.append(T.copy())
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    J = np.zeros((6, n))

    o_n = origins[-1]
    for i in range(n):
        z = z_axes[i]
        o_i = origins[i]
        J[:3, i] = np.cross(z, o_n - o_i)   # Linear velocity part
        J[3:, i] = z                        # Angular velocity part

    return J

if __name__ == "__main__":
    # Test some trial poses
    import statistics

    timings = []
    poses = [
        ([-0.356, -0.323, 0.149], [0.0, 180.0, -35.0]), # Has two very different IK solutions, one with q6 near 0 and one with q6 near 180. Good for testing robustness to multiple solutions.
    ]

    trials_per_run = 5
    yaw_values = range(-180, 210, 30)
    total_runs = len(poses) * len(list(yaw_values)) * trials_per_run
    print(f"Running {total_runs} IK tests...")

    run_count = 0
    for yaw in yaw_values:
        for position, rpy in poses:
            for tri in range(trials_per_run):
                rpy_current = rpy.copy()
                rpy_current[2] = yaw
                t0 = time.perf_counter()
                q = compute_ik(position, rpy_current)
                dt = (time.perf_counter() - t0)
                timings.append(dt)
                run_count += 1
                q_deg = np.rad2deg(np.array(q)).round(2)
                print(f"\nCalculated q={q_deg}")
                print(f"\rRun {run_count:3d}/{total_runs} | Last: {dt*1000:5.1f} ms", end="")

    print("\n--- Timing Results ---")
    print(f"Runs: {len(timings)}")
    print(f"Average: {1000*statistics.mean(timings):.2f} ms")
    print(f"Median:  {1000*statistics.median(timings):.2f} ms")
    print(f"Min:     {1000*min(timings):.2f} ms")
    print(f"Max:     {1000*max(timings):.2f} ms")
    print(f"StdDev:  {1000*statistics.stdev(timings):.2f} ms")
    

# Original ik solver

# def compute_ik(position, rpy, q_guess=None, max_tries=5, dx=0.001):
#     if q_guess is None:
#         q6 = -(np.mod(rpy[2] + 180, 360) - 180) # Adjust initial guess based on given yaw.
#         q_guess = np.radians([90, -80, 90, -90, -90, q6])

#     original_position = np.array(position)

#     for i in range(max_tries):
#         # Try small x-shift each iteration
#         perturbed_position = original_position.copy()
#         perturbed_position[0] += i * dx  # only nudging x

#         target_pose_matrix = np.eye(4)
#         target_pose_matrix[:3, 3] = perturbed_position
#         target_pose_matrix[:3, :3] = rpy_to_matrix(rpy)

#         joint_bounds = [
#             (-np.pi, np.pi),
#             (-np.pi, np.pi),
#             (-np.pi, np.pi),
#             (-np.pi, np.pi),
#             (-np.pi, np.pi),
#             (-np.pi, np.pi)
#         ]

#         result = minimize(ik_objective, q_guess, args=(target_pose_matrix,), method='L-BFGS-B', bounds=joint_bounds, options={'maxiter':1000})

#         if result.success:
#             cost = ik_objective(result.x, target_pose_matrix, verbose=True)
#             if cost > 0.01:
#                 print(f"WARNING: HIGH IK ERROR! Cost for pose {pose_text((position, rpy))} is {cost}")
#                 actual_tmat = forward_kinematics(dh_params, result.x)
#                 actual_position, actual_rotation = Tmat_to_pose(actual_tmat)
#                 print(f"Actual Achieved Pose:                 {pose_text((actual_position, actual_rotation))}")

#             return result.x

#     print(f"IK failed after {max_tries} attempts. Tried perturbing from {pose_text((position, rpy))}.")
#     return None
