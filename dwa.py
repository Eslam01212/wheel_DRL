#!/usr/bin/env python3
# dwa_100trials.py  (classic DWA-style reactive baseline, 100 trials, logs metrics)
# Topics (same as your env_core): /cmd_vel, /camera/depth/image_raw, /odom_fused, /imu, /joint_states, /reset_world, /spawn_entity, /delete_entity

import time, math, random, threading
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


# ---------------- constants (match your env_core-ish setup)
DT = 0.10
DS = 8

V_MAX = 0.4
W_MAX = 1.8

R0 = 0.04
TRACK_W = 0.20
MASS = 2.7

GOAL_X_MIN, GOAL_X_MAX = 4.0, 9.0
GOAL_Y_MIN, GOAL_Y_MAX = 4.0, 9.0

LEFT_JOINTS  = ["left_up_wheel_link_joint", "left_down_wheel_link_joint"]
RIGHT_JOINTS = ["right_up_wheel_link_joint", "right_down_wheel_link_joint"]

# ---------------- DWA knobs (minimal)
SAFE = 0.2          # stop if closer than this (m)
GOAL_THR = 0.50
MAX_STEPS = 400

# sample window (very small classic DWA)
NV = 7
NW = 11
SIM_T = 1.0
SIM_DT = 0.10

# weights
W_GOAL = 2.0
W_HEAD = 1.0
W_CLEAR = 0.01
W_VEL = 1.0


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_rpy(qx, qy, qz, qw):
    # same math as your env_utils
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = 1.0 if sinp > 1.0 else (-1.0 if sinp < -1.0 else sinp)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def wheel_slip(v_meas, w_meas, omega_l, omega_r, R=R0, W=TRACK_W):
    # same slip ratio idea as your env_utils (symmetric denom, bounded)
    eps = 1e-4
    v_l = v_meas - 0.5 * W * w_meas
    v_r = v_meas + 0.5 * W * w_meas
    vwl = R * omega_l
    vwr = R * omega_r
    if (abs(v_meas) < 0.02) and (abs(omega_l) < 0.5) and (abs(omega_r) < 0.5):
        return 0.0
    denom_l = max(abs(vwl) + abs(v_l), eps)
    denom_r = max(abs(vwr) + abs(v_r), eps)
    sr_l = 2.0 * (vwl - v_l) / denom_l
    sr_r = 2.0 * (vwr - v_r) / denom_r
    sr_l = float(np.clip(sr_l, -1.0, 1.0))
    sr_r = float(np.clip(sr_r, -1.0, 1.0))
    return float(np.clip(max(abs(sr_l), abs(sr_r)), 0.0, 1.0))

def min_depth_forward(depth_60x80):
    # very concise: forward "corridor" in the depth image
    if depth_60x80 is None:
        return 5.0
    d = np.asarray(depth_60x80, dtype=np.float32)
    d = np.nan_to_num(d, nan=5.0, posinf=5.0, neginf=5.0)
    d = np.clip(d, 0.0, 5.0)
    h, w = d.shape
    r0, r1 = int(0.30*h), int(0.85*h)
    c0, c1 = int(0.35*w), int(0.65*w)
    return float(np.min(d[r0:r1, c0:c1]))

def goal_feats(x, y, yaw, gx, gy):
    dx, dy = gx - x, gy - y
    dist = math.hypot(dx, dy)
    goal_ang = math.atan2(dy, dx)
    hdiff = wrap_pi(goal_ang - yaw)
    return float(dist), float(hdiff)


class DwaEnv:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("dwa_baseline")

        self.bridge = CvBridge()
        self.cmd_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)

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

        self.reset_cli = self.node.create_client(Empty, "/reset_world")
        self.spawn_cli = self.node.create_client(SpawnEntity, "/spawn_entity")
        self.delete_cli = self.node.create_client(DeleteEntity, "/delete_entity")

        self.exec = MultiThreadedExecutor(num_threads=2)
        self.exec.add_node(self.node)
        self.thr = threading.Thread(target=self.exec.spin, daemon=True)
        self.thr.start()

        # state
        self.depth = None
        self.x = self.y = self.yaw = 0.0
        self.v_meas = self.w_meas = 0.0
        self.roll = self.pitch = 0.0
        self.omega_l = self.omega_r = 0.0
        self.have_odom = self.have_js = False

        # goal
        self.goal_x = self.goal_y = 0.0
        self.goal_spawned = False

        # last cmd (for DWA window center)
        self.last_v = 0.0
        self.last_w = 0.0
        self.mL = 1.0
        self.mR = 1.0
        self.rng = np.random.RandomState(0)

    def cb_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        d = np.nan_to_num(d, nan=5.0, posinf=5.0, neginf=5.0)
        d = np.clip(d, 0.0, 5.0)
        self.depth = d[::DS, ::DS]  # -> ~60x80

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = float(p.x); self.y = float(p.y)
        _, _, self.yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_meas = float(math.sqrt(vx*vx + vy*vy))
        self.w_meas = float(msg.twist.twist.angular.z)
        self.have_odom = True

    def cb_imu(self, msg: Imu):
        q = msg.orientation
        self.roll, self.pitch, _ = quat_to_rpy(q.x, q.y, q.z, q.w)

    def cb_js(self, msg: JointState):
        if not msg.name:
            return
        if self.have_odom is False:
            # still ok, but we wait for both on reset
            pass
        idx = {n:i for i,n in enumerate(msg.name)}
        def avg(names):
            vals = []
            for n in names:
                if n in idx and idx[n] < len(msg.velocity):
                    vals.append(float(msg.velocity[idx[n]]))
            return float(sum(vals)/len(vals)) if vals else 0.0
        self.omega_l = avg(LEFT_JOINTS)
        self.omega_r = avg(RIGHT_JOINTS)
        self.have_js = True

    def send(self, v, w):
        # mimic wheel-radius mismatch using differential drive relations
        # desired wheel linear speeds
        vL = v - 0.5 * TRACK_W * w
        vR = v + 0.5 * TRACK_W * w

        # mismatch applied (left/right scale)
        vL2 = self.mL * vL
        vR2 = self.mR * vR

        # back to twist (effective)
        v_sent = 0.5 * (vR2 + vL2)
        w_sent = (vR2 - vL2) / max(TRACK_W, 1e-6)

        # clamp to limits
        v_sent = float(np.clip(v_sent, 0.0, V_MAX))
        w_sent = float(np.clip(w_sent, -W_MAX, W_MAX))

        tw = Twist()
        tw.linear.x = v_sent
        tw.angular.z = w_sent
        self.cmd_pub.publish(tw)
        self.last_v, self.last_w = v_sent, w_sent


    def _wait(self, cond, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout and not cond():
            time.sleep(0.01)

    def _spawn_goal(self):
        # delete old
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
            self.reset_cli.call_async(Empty.Request())

        self.have_odom = False
        self.have_js = False
        self._wait(lambda: self.have_odom and self.have_js, timeout=3.0)

        self.goal_x = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        self.goal_y = random.uniform(GOAL_Y_MIN, GOAL_Y_MAX)
        self._spawn_goal()

        self.last_v = 0.0
        self.last_w = 0.0
        self.mL = float(self.rng.uniform(0.2, 1.8))
        self.mR = float(self.rng.uniform(0.2, 1.8))
        time.sleep(SIM_DT*50)

    def obs(self):
        dist, hdiff = goal_feats(self.x, self.y, self.yaw, self.goal_x, self.goal_y)
        clear = min_depth_forward(self.depth) if self.depth is not None else 5.0
        slip = wheel_slip(self.v_meas, self.w_meas, self.omega_l, self.omega_r)
        return dist, hdiff, clear, slip, float(self.roll), float(self.pitch)

    def close(self):
        try: self.exec.shutdown()
        except: pass
        try: self.node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass


def dwa_pick(v0, w0, dist, hdiff, clear):
    # if immediate obstacle, turn in place away from it (ultra-reactive fallback)
    if clear < SAFE:
        return 0.0, (W_MAX * 0.6) * (1.0 if hdiff > 0 else -1.0)

    # very small window around current cmd (classic DWA idea)
    v_min = max(0.0, v0 - 0.20)
    v_max = min(V_MAX, v0 + 0.20)
    w_min = max(-W_MAX, w0 - 0.80)
    w_max = min(W_MAX, w0 + 0.80)

    vs = np.linspace(v_min, v_max, NV, dtype=np.float32)
    ws = np.linspace(w_min, w_max, NW, dtype=np.float32)

    best = (0.0, 0.0)
    best_cost = 1e9

    for v in vs:
        for w in ws:
            # forward simulate (unicycle approx)
            x = 0.0; y = 0.0; yaw = 0.0
            t = 0.0
            while t < SIM_T:
                yaw = wrap_pi(yaw + float(w) * SIM_DT)
                x += float(v) * math.cos(yaw) * SIM_DT
                y += float(v) * math.sin(yaw) * SIM_DT
                t += SIM_DT

            # "local goal" direction is simply current goal direction -> (dist, hdiff)
            # predict end distance roughly by moving along current heading correction
            # (still reactive / minimal)
            dist_end = max(0.0, dist - float(v) * SIM_T)
            head_end = abs(wrap_pi(hdiff - yaw))

            # clearance proxy: current clearance discounted by forward motion
            clear_end = max(0.0, clear - float(v) * SIM_T)

            # hard reject if predicted clearance becomes too small
            if clear_end < SAFE:
                continue

            cost = (
                W_GOAL * dist_end +
                W_HEAD * head_end +
                W_CLEAR * (1.0 / (clear_end + 1e-3)) +
                W_VEL * (V_MAX - float(v))
            )

            if cost < best_cost:
                best_cost = cost
                best = (float(v), float(w))

    # if all rejected, rotate to search
    if best_cost >= 1e8:
        return 0.0, (W_MAX * 0.6) * (1.0 if hdiff > 0 else -1.0)
    return best


def main():
    env = DwaEnv()

    trials = 20
    succ = 0
    int_slips = []
    steps_list = []
    int_rolls = []
    int_pitchs = []
    for ep in range(1, trials + 1):
        env.reset()
        dist0, _, _, _, _, _ = env.obs()

        ep_steps = 0
        ep_int_slip = 0.0
        ep_int_roll = 0.0
        ep_int_pitch = 0.0
        slip_count = 0

        ok = False
        for t in range(MAX_STEPS):
            dist, hdiff, clear, slip, roll, pitch = env.obs()

            # integrate (time integral)
            ep_int_slip  += abs(slip)  * DT
            ep_int_roll  += abs(roll)  * DT
            ep_int_pitch += abs(pitch) * DT

            if dist < GOAL_THR:
                ok = True
                break

            # collision-ish stop condition (reactive baseline)
            """if clear < 0.20:
                ok = False
                break"""

            # slip termination (count steps where slip is "high")
            if slip > 0.5:
                slip_count += 1
            else:
                slip_count = 0

            if slip_count > 20:
                ok = False
                break

            if abs(roll) > math.radians(30) or abs(pitch) > math.radians(30):
                ok = False
                break

            v_cmd, w_cmd = dwa_pick(env.last_v, env.last_w, dist, hdiff, clear)
            env.send(v_cmd, w_cmd)
            time.sleep(DT)
            ep_steps += 1

        env.send(0.0, 0.0)
        time.sleep(0.1)

        succ += 1 if ok else 0
        int_slips.append(ep_int_slip)
        int_rolls.append(ep_int_roll)
        int_pitchs.append(ep_int_pitch)
        steps_list.append(ep_steps)

        print(f"EP {ep:03d}/{trials}  success={int(ok)}  steps={ep_steps:3d}  int_slip={ep_int_slip:.3f}  int_roll={ep_int_roll:.3f}  int_pitch={ep_int_pitch:.3f}  slip_count={slip_count}")

    # final metrics requested
    success_rate  = succ / float(trials)
    avg_int_slip  = float(np.mean(int_slips)) if int_slips else 0.0
    avg_steps     = float(np.mean(steps_list)) if steps_list else 0.0
    avg_int_pitch = float(np.mean(int_pitchs)) if int_pitchs else 0.0
    avg_int_roll  = float(np.mean(int_rolls)) if int_rolls else 0.0

    print("\n==== DWA (reactive) 100-trials summary ====")
    print(f"success_rate   : {success_rate:.3f}")
    print(f"avg_int_slip   : {avg_int_slip:.3f}")
    print(f"avg_steps      : {avg_steps:.3f}")
    print(f"avg_int_pitch  : {avg_int_pitch:.3f}")
    print(f"avg_int_roll   : {avg_int_roll:.3f}")

    env.close()


if __name__ == "__main__":
    main()
