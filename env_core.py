# env_core.py
import time
import random
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu, JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from env_utils import quat_to_rpy, depth_image_norm, wheel_terrain_feats, goal_feats, locomotion_feats
import threading

DT = 0.10
DS = 8

R0 = 0.04
W  = 0.20
M  = 2.7

DIST_NORM = 15.0
V_MAX = 0.4
W_MAX = 1.8

GOAL_X_MIN, GOAL_X_MAX = -8.0, 8.0
GOAL_Y_MIN, GOAL_Y_MAX = -8.0, 8.0

LEFT_JOINTS  = ["left_up_wheel_link_joint", "left_down_wheel_link_joint"]
RIGHT_JOINTS = ["right_up_wheel_link_joint", "right_down_wheel_link_joint"]


class RosNavCore:
    def __init__(self, force_zero_pred=False):
        if not rclpy.ok():
            rclpy.init()

        self.node = Node("env_core_goal_depth")
        self.bridge = CvBridge()

        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

        # KEEP subscription refs + use sensor QoS for high-rate topics
        self.sub_depth = self.node.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, qos_profile_sensor_data
        )
        self.sub_odom = self.node.create_subscription(
            Odometry, "/odom_fused", self.cb_odom, 10
        )
        self.sub_imu = self.node.create_subscription(
            Imu, "/imu", self.cb_imu, qos_profile_sensor_data
        )
        self.sub_js = self.node.create_subscription(
            JointState, "/joint_states", self.cb_js, 10
        )

        self.reset_cli  = self.node.create_client(Empty, "/reset_world")
        self.spawn_cli  = self.node.create_client(SpawnEntity, "/spawn_entity")
        self.delete_cli = self.node.create_client(DeleteEntity, "/delete_entity")
        self.goal_spawned = False
        self.force_zero_pred = force_zero_pred

        from predictor import  WheelRuntimePredictor

        self.pred_wheel = WheelRuntimePredictor("wheel_predictor.pt", device="cpu")
        self.pred_dim = 2  # H slip + 2 wheels


        # state
        self.depth = np.zeros((60, 80), dtype=np.float32)
        self.x = self.y = self.yaw = 0.0
        self.v_meas = self.ang_vel = 0.0
        self.yawrate = 0.0
        self.roll = self.pitch = 0.0
        self.omega_l = self.omega_r = 0.0
        self.jmap = None

        self.goal_x = 0.0
        self.goal_y = 0.0
        self.have_odom = False
        self.have_js = False

        # background spin (ONLY place spinning happens)
        self.exec = MultiThreadedExecutor(num_threads=2)
        self.exec.add_node(self.node)
        self._spin_thread = threading.Thread(target=self.exec.spin, daemon=True)
        self._spin_thread.start()

        self._last_odom_print = 0.0  # throttle

    # ---------------- callbacks ----------------
    def cb_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        d = np.nan_to_num(d, nan=5.0, posinf=5.0, neginf=5.0)
        d = np.clip(d, 0.0, 5.0)
        self.depth = d

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = float(p.x)
        self.y = float(p.y)
        _, _, self.yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_meas = float((vx*vx + vy*vy) ** 0.5)
        self.ang_vel = float(msg.twist.twist.angular.z)
        self.have_odom = True

        # throttle prints (printing every msg can “freeze” your node)
        now = time.time()
        if now - self._last_odom_print > 1.0:
            #print("v_meas:", self.v_meas)
            self._last_odom_print = now

    def cb_imu(self, msg: Imu):
        q = msg.orientation
        self.roll, self.pitch, _ = quat_to_rpy(q.x, q.y, q.z, q.w)
        self.yawrate = float(msg.angular_velocity.z)

    def cb_js(self, msg: JointState):
        if self.jmap is None:
            self.jmap = {n: i for i, n in enumerate(msg.name)}
        L, R = [], []
        for n in LEFT_JOINTS:
            i = self.jmap.get(n)
            if i is not None and i < len(msg.velocity):
                L.append(msg.velocity[i])
        for n in RIGHT_JOINTS:
            i = self.jmap.get(n)
            if i is not None and i < len(msg.velocity):
                R.append(msg.velocity[i])
        if L: self.omega_l = float(np.mean(L))
        if R: self.omega_r = float(np.mean(R))
        self.have_js = True

    # ---------------- helpers ----------------
    def send(self, v, w):
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def _wait_done(self, fut, timeout=2.0):
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < timeout:
            time.sleep(0.01)
        return fut.done()

    def _spawn_goal(self):
        # delete old (fire & forget is ok)
        if self.goal_spawned and self.delete_cli.wait_for_service(timeout_sec=1.0):
            req = DeleteEntity.Request()
            req.name = "goal_marker"
            self.delete_cli.call_async(req)
            self.goal_spawned = False
            time.sleep(0.05)

        pkg_share = get_package_share_directory("ugv_gazebo")
        sdf_path = pkg_share + "/models/goal/model.sdf"
        with open(sdf_path, "r") as f:
            xml = f.read()

        if self.spawn_cli.wait_for_service(timeout_sec=1.0):
            req = SpawnEntity.Request()
            req.name = "goal_marker"
            req.xml = xml
            req.initial_pose.position.x = float(self.goal_x)
            req.initial_pose.position.y = float(self.goal_y)
            req.initial_pose.position.z = 0.15
            self.spawn_cli.call_async(req)
            self.goal_spawned = True
            time.sleep(0.05)

    def reset(self):
        self.send(0.0, 0.0)
        time.sleep(0.1)

        if self.reset_cli.wait_for_service(timeout_sec=1.0):
            fut = self.reset_cli.call_async(Empty.Request())
            self._wait_done(fut, timeout=3.0)

        # clear + wait for fresh callbacks (executor thread will deliver them)
        self.have_odom = False
        self.have_js = False
        t0 = time.time()
        while time.time() - t0 < 3.0 and not (self.have_odom and self.have_js):
            time.sleep(0.01)

        self.goal_x = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        self.goal_y = random.uniform(GOAL_Y_MIN, GOAL_Y_MAX)
        self._spawn_goal()

        self.send(0.0, 0.0)
        time.sleep(.1)

    def obs(self, last_cmd_v, last_cmd_w):
        # tiny yield so background thread runs smoothly
        time.sleep(0.001)

        depth_obs = depth_image_norm(self.depth, 60, 80)

        hdiff, dist = goal_feats(self.x, self.y, self.yaw, self.goal_x, self.goal_y)

        wheel4, slip = wheel_terrain_feats(
            v_meas=self.v_meas, ang_vel=self.ang_vel,
            omega_l=self.omega_l, omega_r=self.omega_r,
            wheel_radius=R0, track_width=W,
            roll=self.roll, pitch=self.pitch,
            robot_mass=M,
        )

        loco6 = locomotion_feats(dist, hdiff, self.roll, self.pitch, self.v_meas, self.ang_vel,
                                DIST_NORM, V_MAX, W_MAX)

        prop6 = loco6.astype(np.float32)

        x8 = np.array([
            float(last_cmd_v), float(last_cmd_w),
            float(self.v_meas), float(self.yawrate),
            float(self.omega_l), float(self.omega_r),
            float(self.roll), float(self.pitch),
        ], dtype=np.float32)

        if self.force_zero_pred:
            pred = np.ones((self.pred_dim,), dtype=np.float32)
        else:
            pred = self.pred_wheel(x8).astype(np.float32)  # (2,) in [-1,1]
        
        obs = {"depth": depth_obs.astype(np.float32), "prop": prop6, "pred": pred.astype(np.float32)}
        info = {"dist": dist, "hdiff": hdiff, "slip": float(slip), "wheel4": wheel4.astype(np.float32)}
        #print(pred)
        return obs, info

    def close(self):
        import rclpy
        try:
            # if you have an executor
            if hasattr(self, "exec_") and self.exec_ is not None:
                self.exec_.shutdown()

            # if you have a node
            if hasattr(self, "node") and self.node is not None:
                self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()

