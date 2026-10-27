from functools import partial
from geometry_msgs.msg import PointStamped, Vector3Stamped
import numpy as np

class PusherHandler:
    """Container for all *pusher* related commands, including recommended and actual pusher locations"""
    def __init__(self, node, recommended_topics=None):
        self.node = node
        self.pusher_pub_l = self.node.create_publisher(PointStamped, "/gripper_pusher_left", 10)
        self.pusher_pub_r = self.node.create_publisher(PointStamped, "/gripper_pusher_right", 10)

        self.recommend_data = {
            "pusher_1": {"position": None, "normal": None},
            "pusher_2": {"position": None, "normal": None},
        }
        if recommended_topics is None:
            recommended_topics = [
                (f"pusher_{number}_{attr}",
                 f"/vision/recommended_pusher_{number}/{attr}", msg_type)
                for number in (1, 2)
                for attr, msg_type in (("position", PointStamped), ("normal", Vector3Stamped))
            ]
        self.recommended_topics = recommended_topics
        self.recommended_subscriptions = {}
        for name, topic, msg_type in self.recommended_topics:
            callback_with_name = partial(self.push_recommend_callback, name=name)
            sub = self.node.create_subscription(msg_type, topic, callback_with_name, 10)
            self.recommended_subscriptions[name] = sub

    def push_recommend_callback(self, msg, name):
        # Partial function, since this gets called for EVERY recommend subscriber (there are 4, and they publish one after another)
        # self.get_logger().info(f"Received {name}: {msg}")
        _, number, attr = name.split("_")
        self.recommend_data[f"pusher_{number}"][attr] = msg

    def update(self, ee_position, ee_euler, *, gripper):
        """Publish projected positions using the shared gripper command state."""
        p1, p2 = gripper.EE_pose_to_pushers_2D(
            (ee_position, ee_euler), gripper.previous_gripper_cmd)
        now = self.node.get_clock().now().to_msg()
        for pub, pos in zip([self.pusher_pub_l, self.pusher_pub_r], 
                            [p1, p2]):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = "base"
            msg.point.x, msg.point.y, msg.point.z = pos
            pub.publish(msg)

    def pusher_recommendations_exist(self):
        return all(value is not None
                   for pusher in self.recommend_data.values()
                   for value in pusher.values())

    def extract_pusher_locations(self):
        if not self.pusher_recommendations_exist():
            raise ValueError("Pusher recommendations are incomplete")
        # Find the EE Pose from pushers
        # normals are normal out. We want that to be anti-camera, which is +90deg
        # normals are also unit vector. 
        x, y, z = [
            self.recommend_data["pusher_1"]["normal"].vector.x,
            self.recommend_data["pusher_1"]["normal"].vector.y,
            self.recommend_data["pusher_1"]["normal"].vector.z
        ]
        normal_yaw = np.arctan2(y, x)  # radians
        normal_yaw_d = np.degrees(normal_yaw)
        given_yaw = normal_yaw_d - 90
        pusher_1 = [
            self.recommend_data["pusher_1"]["position"].point.x,
            self.recommend_data["pusher_1"]["position"].point.y,
            self.recommend_data["pusher_1"]["position"].point.z
        ]
        pusher_2 = [
            self.recommend_data["pusher_2"]["position"].point.x,
            self.recommend_data["pusher_2"]["position"].point.y,
            self.recommend_data["pusher_2"]["position"].point.z
        ]
        return pusher_1, pusher_2, given_yaw
