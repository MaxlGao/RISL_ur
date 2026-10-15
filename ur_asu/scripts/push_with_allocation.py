import time
import os
import numpy as np
import rclpy
import csv
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Bool
from geometry_msgs.msg import PoseStamped, TwistStamped, PointStamped, Vector3Stamped, WrenchStamped
from scipy.spatial.transform import Rotation as R

from ur_asu.custom_libraries.motion_utils import MotionExecutor
from ur_asu.custom_libraries.pusher_utils import PusherHandler
from ur_asu.custom_libraries.gripper_utils import GripperHandler, GRIPPER_TABLE
from ur_asu.custom_libraries.actionlibrariesmax import move, velocity, force, append_new_act, gripper_width

class OverController(Node):
    def __init__(self):
        super().__init__('motion_planner')

        # Switch 1
        # self.sticky = False                # Two friction modes: Sticky or Smooth
        self.sticky = True                # Two friction modes: Sticky or Smooth

        # Switch 2
        # self.target_name = "wrench"        # Options: jenga_6, wrench, allen_key
        # self.target_name = "allen_key"        # Options: jenga_6, wrench, allen_key
        self.target_name = "jenga_6"        # Options: jenga_6, wrench, allen_key

        # Switch 3
        # self.experiment_type = "single_wrench_repeated"   # "sequence", "single_wrench_repeated", or "custom_sequence"
        self.experiment_type = "custom_sequence"   # "sequence", "single_wrench_repeated", or "custom_sequence"
        self.single_wrench_index = 3        # 1-based index into Sequence_1 wrench list (1, 2, or 3)
        self.single_wrench_repeats = 8
        self.custom_sequence_indices = "1-2-3-1-2-3"  # Dash-separated 1-based indices for custom sequence

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # self.linear = True                 # Linear or rotational testing
        # Controller Names
        self.position_controller = 'scaled_joint_trajectory_controller'
        self.velocity_controller = 'forward_velocity_controller'
        self.force_controller = 'force_mode_controller'
        self.passthrough_controller = 'passthrough_trajectory_controller'
        recommended_topics = [
            ('pusher_1_position', '/vision/recommended_pusher_1/position', PointStamped),
            ('pusher_2_position', '/vision/recommended_pusher_2/position', PointStamped),
            ('pusher_1_normal',   '/vision/recommended_pusher_1/normal',   Vector3Stamped),
            ('pusher_2_normal',   '/vision/recommended_pusher_2/normal',   Vector3Stamped)
        ]

        # Subclasses
        self.pusher = PusherHandler(self, recommended_topics)
        if self.sticky:
            self.vert_offset = 0.005 # Vertical offset to add to all gripper heights to prevent scraping the table.
        else:
            self.vert_offset = 0.003 # sticky surface is a bit higher
        self.gripper = GripperHandler(self, vertical_offset = self.vert_offset)
        self.motion = MotionExecutor(
            self, self.joint_names,
            self.position_controller,
            self.velocity_controller,
            self.force_controller,
            self.passthrough_controller,
            self.gripper,
            pusher=self.pusher,
            auto_gripper = True
            )
        
        print(f"Looking for object {self.target_name}")
        self.sub_object_data =                    self.create_subscription(PoseStamped,  f"/vision/object_poses/{self.target_name}", self.object_pose_callback, 10)
        self.sub_recommended_gripper_twist =      self.create_subscription(TwistStamped, f"/vision/recommended_gripper_twist_world", self.gripper_twist_callback, 10)
        self.sub_recommended_gripper_twist_body = self.create_subscription(TwistStamped, "/allocation/recommended_gripper_twist_body", self.gripper_twist_body_callback, 10)
        self.sub_v_hat =                          self.create_subscription(TwistStamped, "/allocation/v_hat", self.v_hat_callback, 10)
        self.sub_w_hat =                          self.create_subscription(WrenchStamped, "/allocation/w_hat", self.w_hat_callback, 10)
        self.sub_object_twist_body =              self.create_subscription(TwistStamped, f"/vision/object_twists_body/{self.target_name}", self.object_twist_body_callback, 10)
        self.sub_object_twist_avg =               self.create_subscription(TwistStamped, f"/vision/object_twists_push_avg/{self.target_name}", self.object_twist_avg_callback, 10)
        self.sub_wrench_echo =                    self.create_subscription(Float32MultiArray, "/allocation/wrench_desired", self.wrench_echo_callback, 10)
        self.sub_mu_alpha =                       self.create_subscription(Float32MultiArray, "/allocation/mu_alpha", self.mu_alpha_callback, 10)
        self.pub_desired_wrench =                 self.create_publisher(Float32MultiArray, "/robot/wrench_desired", 10)
        self.pub_target_name =                    self.create_publisher(String,            "/robot/target_name", 10)
        self.pub_pushing_flag =                   self.create_publisher(Bool,              "/robot/currently_pushing", 10)
        
        # Because we need to be sure the allocation node has the right desired wrench
        self.buffer_advance_talker =      self.create_publisher(String, "buffer_advance_channel", 10)
        self.buffer_advance_listener = self.create_subscription(String, "buffer_advance_channel", self.confirm_recommendation_callback, 10)

        self.latest_target_pose = None
        self.previous_target_pose = None
        self.latest_pose_time = time.time()
        self.pose_timeout = 0.5 # Allowed time to not see the object.
        self.last_pose_time = time.time()
        self.last_twist_time = 0.0

        # Motion Planning
        self.push_duration = 6.5          # seconds per trial
        self.latest_commanded_gripper_twist = np.zeros(6)
        self.latest_commanded_gripper_twist_body = np.zeros(3)
        self.latest_object_twist_body = np.zeros(3)
        self.latest_object_twist_avg = np.zeros(3)
        self.latest_wrench_echo = np.zeros(3)
        # [mu_x, mu_y, mu_rot, alpha_ub_effective, alpha_ub_closed]
        self.latest_mu_alpha = [1., 1., 1., 1., 1.]
        self.latest_v_hat = np.zeros(3)
        self.latest_w_hat = np.zeros(3)
        self.push_records = []
        self.push_summary = []
        self.current_push_id = -1
        self.current_push_label = ""
        self.previous_avg_twist = None
        self.current_push_contact_data = None
        
        # Two regions on the table with different friction coeffs
        if self.sticky:
            adj1 = "sticky"
            self.home_position = [-0.361, -0.310, GRIPPER_TABLE[0] + self.vert_offset + 0.05]
        else:
            adj1 = "smooth"
            self.home_position = [ 0.340, -0.310, GRIPPER_TABLE[0] + self.vert_offset + 0.05]
        self.base_orientation = [0, 180, 0]

        # (Fx, Fy, Tz) in (N, N, Nm)
        # Desired Wrenches in body frame
        if self.target_name == "allen_key":
            base_wrench_sequence = [(0.0, 4.0, 0.0), (0.0, 4.0, -0.2), (4.0, 0.0, -0.05)]
        elif self.target_name == "wrench":
            base_wrench_sequence = [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.2), (3.0, 1.0, 0.0)]
        elif "jenga" in self.target_name:
            # Torques reduced in accordance to smaller body size
            base_wrench_sequence = [(3.0, 1.0, 0.0), (0.0, 4.0, 0.1), (1.0, -4.0, 0.1)]
        else:
            base_wrench_sequence = [(0.0, 4.0, 0.0)]

        self.push_schedule, exp_label = self.build_push_schedule(base_wrench_sequence)
        self.wrench_count = len(self.push_schedule)
        self.log_prefix = f"{adj1}_{self.target_name}_{exp_label}"
        self.timeseries_logfile = f"{self.log_prefix}_timeseries.csv"
        self.summary_logfile = f"{self.log_prefix}_summary.csv"
        # gripper_open_act = gripper_width('1100') # Fully open gripper before moving
        start_duration = 3
        first_move = move(self.home_position, self.base_orientation, start_duration)
        self.acts = {}
        self.executing = False
        self.PSO_Processing = False
        self.recommendation_refresh_confirmed = False
        self.i = 0
        self.execute_single_act(first_move)
        self.run_push_experiment()

    def object_pose_callback(self, msg: PoseStamped):
        xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = (msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w)
        rpy = R.from_quat(quat).as_euler('xyz', degrees=True)
        self.latest_target_pose = (xyz, rpy)
        self.last_pose_time = time.time()

    def gripper_twist_callback(self, msg: TwistStamped):
        """Fast world frame command given so long as the object is visible"""
        xyz = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        rpy = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        self.latest_commanded_gripper_twist = np.array([*xyz, *rpy])
        self.last_twist_time = time.time()

    def gripper_twist_body_callback(self, msg: TwistStamped):
        self.latest_commanded_gripper_twist_body = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
        ])

    def v_hat_callback(self, msg: TwistStamped):
        self.latest_v_hat = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
        ])

    def w_hat_callback(self, msg: WrenchStamped):
        self.latest_w_hat = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.torque.z
        ])
        print(f"Received w_hat: {self.latest_w_hat}")

    def object_twist_body_callback(self, msg: TwistStamped):
        self.latest_object_twist_body = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
        ])

    def object_twist_avg_callback(self, msg: TwistStamped):
        self.latest_object_twist_avg = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z
        ])

    def mu_alpha_callback(self, msg: Float32MultiArray):
        # Expecting [mu_x, mu_y, mu_rot, alpha_ub_effective, alpha_ub_closed]
        if len(msg.data) >= 5:
            self.latest_mu_alpha = [
                float(msg.data[0]), float(msg.data[1]), float(msg.data[2]),
                float(msg.data[3]), float(msg.data[4])
            ]
        elif len(msg.data) >= 4:
            # Backwards-compatible: if only 4 values, treat the last as both effective and closed.
            self.latest_mu_alpha = [
                float(msg.data[0]), float(msg.data[1]), float(msg.data[2]),
                float(msg.data[3]), float(msg.data[3])
            ]

    def publish_wrench(self, wrench, name):
        target_msg = String()
        target_msg.data = name
        self.pub_target_name.publish(target_msg)

        wrench_msg = Float32MultiArray()
        wrench_msg.data = wrench
        self.pub_desired_wrench.publish(wrench_msg)

        self.PSO_Processing = True
    
    def declare_pushing_status(self, status: bool):
        ms = Bool()
        ms.data = status
        self.pub_pushing_flag.publish(ms)

    def wrench_echo_callback(self, msg: Float32MultiArray):
        self.PSO_Processing = False # Hearing a wrench back means PSO's done.
        if len(msg.data) >= 3:
            self.latest_wrench_echo = np.array([msg.data[0], msg.data[1], msg.data[2]], dtype=float)
        self.get_logger().info(f"PSO for w_d = {msg.data.tolist()} Completed!")

    def confirm_recommendation_callback(self, msg: String):
        if "Confirmed" in msg.data:
            self.recommendation_refresh_confirmed = True

    def recommendation_refresh(self):
        msg = String()
        time = self.get_clock().now().to_msg()
        msg.data = f"Buffer Advance Request at t = {time}"
        self.buffer_advance_talker.publish(msg)
        self.recommendation_refresh_confirmed = False
        # Wait until we get confirm from localization module
        while not self.recommendation_refresh_confirmed:
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Clear recommended pushers
        self.pusher.recommend_data = {
            "pusher_1": {"position": None, "normal": None},
            "pusher_2": {"position": None, "normal": None},
        }

        # Wait until pusher recommendations actually come in
        while not self.pusher.pusher_recommendations_exist():
            print("!", end="")
            rclpy.spin_once(self, timeout_sec=0.01)

    def run_push_experiment(self):
        self.get_logger().info(f"Starting planar pushing experiment with {self.wrench_count} pushes")
        try:
            # Send off first wrench, since we're gonna wait for it anyway
            first_wrench = self.push_schedule[0]["wrench"]
            self.publish_wrench(first_wrench, self.target_name)
            self.get_logger().info(f"Waiting for PSO to finish w_d = {first_wrench}")
            while self.PSO_Processing:
                # Wait for PSO to finish for the first wrench
                rclpy.spin_once(self, timeout_sec=0.01)

            for i, push in enumerate(self.push_schedule):
                wrench = push["wrench"]
                self.current_push_id = i
                self.current_push_label = push["label"]
                self.current_push_contact_data = None

                # Changeover to the next displayed wrench/twist/pushers
                # Blocking function. Should prevent prior data being used.
                self.recommendation_refresh()

                # Prepare pushes (retract-lift-move-drop-advance takes 20 to 25 seconds)
                self.move_to_point_positions()

                # Main push segment
                self.start_time = time.time()
                zero_twist = np.array([0.0] * 6)

                self.declare_pushing_status(True)
                while time.time() - self.start_time < self.push_duration:
                    if time.time() - self.last_twist_time > 0.5:
                        twist = zero_twist
                        self.motion.send_cartesian_velocity_step(zero_twist)
                    else:
                        twist = self.latest_commanded_gripper_twist
                        self.motion.send_cartesian_velocity_step(twist)
                    time.sleep(0.05)
                    rclpy.spin_once(self, timeout_sec=0.05)
                    twstring = ", ".join([f"{v: 7.4f}" for v in twist])
                    elapsed = time.time() - self.start_time
                    self.record_push_timeseries(elapsed, wrench)
                    print(f"Pushing w/ twist {twstring}, t={elapsed:4.2f} / {self.push_duration:4.2f}s", end='\r')
                self.motion.send_cartesian_velocity_step(zero_twist)
                self.get_logger().info(f"Pushed with v_g = {twstring}")

                self.declare_pushing_status(False)
                # Allow adaptation callback to process and publish mu/alpha.
                time.sleep(0.2)
                rclpy.spin_once(self, timeout_sec=0.1)
                self.record_push_summary(push, wrench)

                # Get the next wrench going.
                if i < self.wrench_count - 1:
                    next_wrench = self.push_schedule[i + 1]["wrench"]
                    self.publish_wrench(next_wrench, self.target_name)
                    self.get_logger().info(f"Pushing with w_d = {wrench}, PSO working on w_d = {next_wrench}")
                else:
                    self.get_logger().info(f"Pushing with w_d = {wrench} (PSO Done)")

                while self.PSO_Processing:
                    # Wait for PSO to finish for this current wrench before moving onto the next one.
                    rclpy.spin_once(self, timeout_sec=0.01)
        finally:
            self.flush_logs_to_csv()
        self.get_logger().info("Push experiment complete.")
        raise SystemExit

    def build_push_schedule(self, base_wrench_sequence):
        if self.experiment_type == "single_wrench_repeated":
            # Single-wrench repeated mode picks from the same base sequence used for sequence mode.
            seq1 = [tuple(w) for w in base_wrench_sequence]
            if not seq1:
                seq1 = [(0.0, 4.0, 0.0)]
            idx0 = int(self.single_wrench_index) - 1
            if idx0 < 0 or idx0 >= len(seq1):
                self.get_logger().warn(
                    f"single_wrench_index={self.single_wrench_index} out of range; clamping to [1, {len(seq1)}]."
                )
                idx0 = min(max(idx0, 0), len(seq1) - 1)
            selected_wrench = tuple(seq1[idx0])
            schedule = []
            for i in range(self.single_wrench_repeats):
                schedule.append({
                    "wrench": selected_wrench,
                    "label": f"single_repeat_{i + 1}",
                    "repeat_idx": i + 1,
                    "group": "single_wrench_repeated",
                })
            return schedule, f"w{idx0 + 1}"

        elif self.experiment_type == "custom_sequence":
            # Custom sequence mode: specify a series of 1-based indices
            seq1 = [tuple(w) for w in base_wrench_sequence]
            if not seq1:
                seq1 = [(0.0, 4.0, 0.0)]
            indices = [int(idx.strip()) - 1 for idx in self.custom_sequence_indices.split('-')]
            schedule = []
            for i, idx in enumerate(indices):
                if idx < 0 or idx >= len(seq1):
                    self.get_logger().warn(
                        f"Custom sequence index {idx + 1} out of range; clamping to [1, {len(seq1)}]."
                    )
                    idx = min(max(idx, 0), len(seq1) - 1)
                wrench = tuple(seq1[idx])
                schedule.append({
                    "wrench": wrench,
                    "label": f"custom_{i + 1}_w{idx + 1}",
                    "repeat_idx": i + 1,
                    "group": "custom_sequence",
                })
            return schedule, f"custom_{self.custom_sequence_indices.replace('-', '')}"

        schedule = []
        for i, wrench in enumerate(base_wrench_sequence):
            schedule.append({
                "wrench": tuple(wrench),
                "label": f"step_{i + 1}",
                "repeat_idx": 1,
                "group": "sequence",
            })
        return schedule, "default_sequence"

    def record_push_timeseries(self, elapsed_s, wrench):
        now = self.get_clock().now().to_msg()
        row = {
            "sec": now.sec,
            "nanosec": now.nanosec,
            "push_index": self.current_push_id,
            "push_label": self.current_push_label,
            "elapsed_s": elapsed_s,
            "target_name": self.target_name,
            "wrench_fx": float(wrench[0]),
            "wrench_fy": float(wrench[1]),
            "wrench_tz": float(wrench[2]),
            "wrench_echo_fx": float(self.latest_wrench_echo[0]),
            "wrench_echo_fy": float(self.latest_wrench_echo[1]),
            "wrench_echo_tz": float(self.latest_wrench_echo[2]),
            "object_x": np.nan,
            "object_y": np.nan,
            # "object_z": np.nan,
            # "object_roll_deg": np.nan,
            # "object_pitch_deg": np.nan,
            "object_yaw_deg": np.nan,
            "gripper_cmd_twist_world_vx": float(self.latest_commanded_gripper_twist[0]),
            "gripper_cmd_twist_world_vy": float(self.latest_commanded_gripper_twist[1]),
            "gripper_cmd_twist_world_wz": float(self.latest_commanded_gripper_twist[5]),
            "gripper_cmd_twist_body_vx": float(self.latest_commanded_gripper_twist_body[0]),
            "gripper_cmd_twist_body_vy": float(self.latest_commanded_gripper_twist_body[1]),
            "gripper_cmd_twist_body_wz": float(self.latest_commanded_gripper_twist_body[2]),
            "v_tilde_body_vx": float(self.latest_object_twist_body[0]),
            "v_tilde_body_vy": float(self.latest_object_twist_body[1]),
            "v_tilde_body_wz": float(self.latest_object_twist_body[2]),
            "v_hat_body_vx": float(self.latest_v_hat[0]),
            "v_hat_body_vy": float(self.latest_v_hat[1]),
            "v_hat_body_wz": float(self.latest_v_hat[2]),
            "error_twist_body_vx": float(self.latest_v_hat[0] - self.latest_object_twist_body[0]),
            "error_twist_body_vy": float(self.latest_v_hat[1] - self.latest_object_twist_body[1]),
            "error_twist_body_wz": float(self.latest_v_hat[2] - self.latest_object_twist_body[2]),
            "mu_x": float(self.latest_mu_alpha[0]),
            "mu_y": float(self.latest_mu_alpha[1]),
            "mu_rot": float(self.latest_mu_alpha[2]),
            "alpha_ub": float(self.latest_mu_alpha[3]),
            "alpha_ub_closed": float(self.latest_mu_alpha[4]),
        }
        if self.latest_target_pose is not None:
            xyz, rpy = self.latest_target_pose
            row.update({
                "object_x": float(xyz[0]),
                "object_y": float(xyz[1]),
                # "object_z": float(xyz[2]),
                # "object_roll_deg": float(rpy[0]),
                # "object_pitch_deg": float(rpy[1]),
                "object_yaw_deg": float(rpy[2]),
            })
        if elapsed_s > 0.5:
            self.push_records.append(row)

    def record_push_summary(self, push, wrench):
        avg_twist = self.latest_object_twist_avg.copy()
        cmd = self.latest_commanded_gripper_twist_body.copy()
        err = cmd - avg_twist
        delta_prev = np.array([np.nan, np.nan, np.nan], dtype=float)
        if self.previous_avg_twist is not None:
            delta_prev = avg_twist - self.previous_avg_twist
        self.previous_avg_twist = avg_twist.copy()

        row = {
            "push_index": self.current_push_id,
            "push_label": push["label"],
            "repeat_idx": push["repeat_idx"],
            "group": push["group"],
            "target_name": self.target_name,
            "push_duration_s": float(self.push_duration),
            "wrench_fx": float(wrench[0]),
            "wrench_fy": float(wrench[1]),
            "wrench_tz": float(wrench[2]),
            "w_hat_fx": float(self.latest_w_hat[0]),
            "w_hat_fy": float(self.latest_w_hat[1]),
            "w_hat_tz": float(self.latest_w_hat[2]),
            "v_hat_body_vx": float(self.latest_v_hat[0]),
            "v_hat_body_vy": float(self.latest_v_hat[1]),
            "v_hat_body_wz": float(self.latest_v_hat[2]),
            "v_tilde_avg_body_vx": float(avg_twist[0]),
            "v_tilde_avg_body_vy": float(avg_twist[1]),
            "v_tilde_avg_body_wz": float(avg_twist[2]),
            "gripper_cmd_twist_body_vx": float(cmd[0]),
            "gripper_cmd_twist_body_vy": float(cmd[1]),
            "gripper_cmd_twist_body_wz": float(cmd[2]),
            "error_twist_body_vx": float(self.latest_v_hat[0] - avg_twist[0]),
            "error_twist_body_vy": float(self.latest_v_hat[1] - avg_twist[1]),
            "error_twist_body_wz": float(self.latest_v_hat[2] - avg_twist[2]),
            # "delta_avg_twist_from_prev_vx": float(delta_prev[0]),
            # "delta_avg_twist_from_prev_vy": float(delta_prev[1]),
            # "delta_avg_twist_from_prev_wz": float(delta_prev[2]),
            "mu_x": float(self.latest_mu_alpha[0]),
            "mu_y": float(self.latest_mu_alpha[1]),
            "mu_rot": float(self.latest_mu_alpha[2]),
            "alpha_ub": float(self.latest_mu_alpha[3]),
            "alpha_ub_closed": float(self.latest_mu_alpha[4]),
            "contact_p1_x": np.nan,
            "contact_p1_y": np.nan,
            "contact_p2_x": np.nan,
            "contact_p2_y": np.nan,
            "contact_span_m": np.nan,
            "contact_yaw": np.nan,
        }
        if self.current_push_contact_data is not None:
            row.update(self.current_push_contact_data)
        self.push_summary.append(row)

    def flush_logs_to_csv(self):
        if not self.push_records and not self.push_summary:
            self.get_logger().warn("No log rows captured; skipping CSV export.")
            return
        self.write_rows_to_csv(self.timeseries_logfile, self.push_records)
        self.write_rows_to_csv(self.summary_logfile, self.push_summary)
        self.get_logger().info(f"Wrote timeseries data to {os.path.abspath(self.timeseries_logfile)}")
        self.get_logger().info(f"Wrote summary data to {os.path.abspath(self.summary_logfile)}")

    def write_rows_to_csv(self, path, rows):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        if os.path.exists(path):
            base, ext = os.path.splitext(path)
            suffix = 1
            while True:
                candidate = f"{base}{suffix}{ext}"
                if not os.path.exists(candidate):
                    path = candidate
                    break
                suffix += 1
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def execute_single_act(self, act):
        """Execute a single act."""
        act_type = act["type"]
        self.executing = True # Hold any other dependent process

        if act_type == "position":
            joint_angles = act["joint_angles"]
            seconds = act["time_from_start"]
            self.motion.send_joint_angles(joint_angles, seconds)
        elif act_type == "velocity":
            velocity = act["velocity"]
            seconds = act["duration"]
            self.motion.send_cartesian_velocity(velocity, seconds)
        elif act_type == "force":
            force = act["force"]
            seconds = act["duration"]
            # selection vector is x y z rx ry rz boolean to permit movement in the given axis
            selection_vector = act["selection_vector"]
            self.motion.send_cartesian_force_async(force, seconds, selection_vector=selection_vector)
        elif act_type == "gripper":
            cmd = act["cmd"]
            self.gripper.publish(str(cmd))
            self.gripper.previous_gripper_cmd = cmd
        
        self.executing = False  # All done

    def execute_act_in_series(self, act_name):
        """A recursive function like this allows for a dynamically changing acts list"""
        act = self.acts[act_name]
        self.execute_single_act(act)
        if self.i >= len(self.acts) - 1:
            self.get_logger().info("Done with current list")
        else:
            self.i += 1
            act = list(self.acts)[self.i]
            self.execute_act_in_series(act)

    def check_for_new_pose(self):
        """Checks the timestamp on incoming object poses and decides if we've dropped
        localization for too long."""
        if (time.time() - self.last_pose_time) > self.pose_timeout:
            if not self.executing:
                self.get_logger().warn("Lost target — initiating search pattern - STAND CLEAR.")
                self.run_search_pattern()
            return

        if self.latest_target_pose is None:
            # No pose available
            return

    def move_to_point_positions(self, segment_duration=3):
        """
        This is a pointspan generation function that makes a list
        of acts, bringing the grippers safely to their target.
        We assume that there exist actual recommended points and corresponding normal vectors. 
        This will lift the gripper up, move it over and away from the target,
        bring it down slowly, and then move it slowly to just touch the object.
        """ 
        pusher_1, pusher_2, given_yaw = self.pusher.extract_pusher_locations()
        print(f"Pusher Locations: {1000*pusher_1[0]: 6.1f}, {1000*pusher_1[1]: 6.1f} and {1000*pusher_2[0]: 6.1f}, {1000*pusher_2[1]: 6.1f} mm")
        pusher_1_orig = pusher_1.copy()
        pusher_2_orig = pusher_2.copy()
        goal_pos, goal_ori = self.gripper.pushers_to_EE_pose_2D(pusher_1_orig, pusher_2_orig, given_yaw)
        pusher_1, span, yaw = self.gripper.EE_pose_to_pointspan([goal_pos, goal_ori])
        self.current_push_contact_data = {
            "contact_p1_x": float(pusher_1_orig[0]),
            "contact_p1_y": float(pusher_1_orig[1]),
            "contact_p2_x": float(pusher_2_orig[0]),
            "contact_p2_y": float(pusher_2_orig[1]),
            "contact_span_m": float(span),
            "contact_yaw": float(yaw),
        }

        moves = self.gripper.pointspan_to_acts_safe(
            [self.motion.ee_position, self.motion.ee_euler], 
            pusher_1, span, yaw, segment_duration)

        # Disable auto-gripper updates while executing the lowering / slide moves
        # prev_auto = self.motion.auto_gripper
        self.motion.set_auto_gripper(False)

        # flush queue and start
        self.acts = {}
        self.i = 0 
        for mov in moves:
            self.acts = append_new_act(self.acts, mov)
        act = list(self.acts)[self.i]
        self.execute_act_in_series(act)

        # Re-enable auto-gripper (if it was enabled before)
        # self.motion.set_auto_gripper(prev_auto)

    def run_search_pattern(self):
        current_pos = np.array(self.motion.ee_position)
        current_ori = np.array(self.motion.ee_euler)
        target_pos = current_pos + np.array([0, 0, 0.15])
        lift = move(
            (target_pos).tolist(),
            current_ori.tolist(),
            4
        )
        self.execute_single_act(lift)
        self.get_logger().info("Search pattern complete — waiting for detections.")


def main(args=None):
    rclpy.init(args=args)
    node = OverController()

    try:
        rclpy.spin(node)
    except (RuntimeError, SystemExit):
        node.get_logger().info("Shutting down")
    node.get_logger().info("Script complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
