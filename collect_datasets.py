#!/usr/bin/env python3
import os, math, time, random, glob
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from env_utils import quat_to_rpy  # keep using your existing helper

# ===== minimal knobs =====
DT = 0.10
EPISODES = 400
STEPS_PER_EP = 400

ERR_LO, ERR_HI = 0.2, 1.8
W = 0.20
CMD_V_MAX = 0.4
CMD_W_MAX = 1.8

LEFT_JOINTS  = ["left_up_wheel_link_joint", "left_down_wheel_link_joint"]
RIGHT_JOINTS = ["right_up_wheel_link_joint", "right_down_wheel_link_joint"]

OUT_DIR = "wheel_dataset"
CHUNK_EP = 50  # saves every N episodes into one file
COMPRESS = True
# =========================

class Collector(Node):
    def __init__(self):
        super().__init__("collect_wheel_dataset")
        os.makedirs(OUT_DIR, exist_ok=True)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom_fused", self.cb_odom, 10)
        self.create_subscription(Imu, "/imu", self.cb_imu, 10)
        self.create_subscription(JointState, "/joint_states", self.cb_js, 10)
        self.reset_cli = self.create_client(Empty, "/reset_world")

        # latest sensors
        self.v_meas = 0.0
        self.ang_vel = 0.0
        self.yawrate = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.omega_l = 0.0
        self.omega_r = 0.0
        self.jmap = None

        # episode state
        self.ep = 0
        self.t = 0
        self.errL = 1.0
        self.errR = 1.0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # chunk buffers
        self.chunk_id = 0
        self.chunk_ep = 0
        self.Xc = []
        self.Yc = []

        self.timer = self.create_timer(DT, self.tick)
        self.start_episode()

    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self.v_meas = float(math.hypot(vx, vy))
        self.ang_vel = float(msg.twist.twist.angular.z)

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
                L.append(float(msg.velocity[i]))
        for n in RIGHT_JOINTS:
            i = self.jmap.get(n)
            if i is not None and i < len(msg.velocity):
                R.append(float(msg.velocity[i]))
        if L: self.omega_l = float(np.mean(L))
        if R: self.omega_r = float(np.mean(R))

    def _reset_world(self):
        if self.reset_cli.wait_for_service(timeout_sec=1.0):
            self.reset_cli.call_async(Empty.Request())

    def _send(self, v, w):
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

    def start_episode(self):
        self._send(0.0, 0.0)
        self._reset_world()
        time.sleep(3.0)

        self.errL = random.uniform(ERR_LO, ERR_HI)
        self.errR = random.uniform(ERR_LO, ERR_HI)
        self.t = 0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.get_logger().info(f"EP {self.ep}/{EPISODES} errL={self.errL:.3f} errR={self.errR:.3f}")

    def _flush_chunk(self, final=False):
        if not self.Xc:
            return
        X = np.asarray(self.Xc, dtype=np.float32)
        Y = np.asarray(self.Yc, dtype=np.float32)

        save = np.savez_compressed if COMPRESS else np.savez
        path = os.path.join(OUT_DIR, f"wheel_part_{self.chunk_id:04d}.npz")
        save(path, X=X, Y=Y)

        self.get_logger().info(f"Saved {os.path.basename(path)}  X={X.shape}  Y={Y.shape}")
        self.chunk_id += 1
        self.chunk_ep = 0
        self.Xc.clear()
        self.Yc.clear()

        if final:
            self._send(0.0, 0.0)

    def save_and_exit(self):
        self._flush_chunk(final=True)
        self.get_logger().info("Done.")
        self.timer.cancel()
        rclpy.shutdown()
        raise SystemExit

    def tick(self):
        # random command
        cmd_v = random.uniform(0.0, CMD_V_MAX)
        cmd_w = random.uniform(-CMD_W_MAX, CMD_W_MAX)

        # apply per-episode wheel radius factors (synthetic)
        vL = (cmd_v - 0.5 * W * cmd_w) * self.errL
        vR = (cmd_v + 0.5 * W * cmd_w) * self.errR
        v_sent = 0.5 * (vL + vR)
        w_sent = (vR - vL) / W

        self._send(v_sent, w_sent)
        self.last_cmd_v = float(v_sent)
        self.last_cmd_w = float(w_sent)

        x8 = np.array([
            self.last_cmd_v, self.last_cmd_w,
            float(self.v_meas), float(self.yawrate),
            float(self.omega_l), float(self.omega_r),
            float(self.roll), float(self.pitch),
        ], dtype=np.float32)

        y2 = np.array([self.errL, self.errR], dtype=np.float32)

        self.Xc.append(x8)
        self.Yc.append(y2)

        self.t += 1
        if self.t >= STEPS_PER_EP:
            self.ep += 1
            self.chunk_ep += 1
            if self.chunk_ep >= CHUNK_EP:
                self._flush_chunk(final=False)

            if self.ep >= EPISODES:
                self.save_and_exit()
            else:
                self.start_episode()

def main():
    rclpy.init()
    node = Collector()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass

if __name__ == "__main__":
    main()
