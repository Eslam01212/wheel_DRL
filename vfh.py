#!/usr/bin/env python3
# vfh_plus_100trials.py
# Minimal VFH+ (with optional Vector-Field mode) baseline for your harsh-env comparison.
# Mirrors the same topics + 5 metrics used in your DWA baseline. (See dwa.py) 

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

# ---------------- constants (kept aligned with your DWA baseline)
DT = 0.10
DS = 8

V_MAX = 0.4
W_MAX = 1.8

R0 = 0.04
TRACK_W = 0.20

GOAL_X_MIN, GOAL_X_MAX = 4.0, 9.0
GOAL_Y_MIN, GOAL_Y_MAX = 4.0, 9.0

LEFT_JOINTS  = ["left_up_wheel_link_joint", "left_down_wheel_link_joint"]
RIGHT_JOINTS = ["right_up_wheel_link_joint", "right_down_wheel_link_joint"]

# termination / safety
SAFE = 0.20         # obstacle threshold (m)

# --- hill avoidance (IMU-based)
TILT_WARN = math.radians(10.0)   # start reacting to hills
TILT_HARD = math.radians(18.0)   # strong reaction (rotate-in-place)
GOAL_THR = 0.50
MAX_STEPS = 400

# --- depth -> pseudo-laser parameters
FOV_H = math.radians(70.0)   # approximate horizontal FOV (good enough for baseline)
MIN_RANGE = 0.05
MAX_RANGE = 5.0

# -------- helpers
def wrap_pi(a): return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_rpy(qx, qy, qz, qw):
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

def goal_feats(x, y, yaw, gx, gy):
    dx, dy = gx - x, gy - y
    dist = math.hypot(dx, dy)
    goal_ang = math.atan2(dy, dx)
    hdiff = wrap_pi(goal_ang - yaw)
    return float(dist), float(hdiff)

def depth_to_pseudolaser(depth_60x80):
    """
    Convert downsampled depth (H,W) to ranges per bearing (W,).
    Uses near-bottom band to emphasize obstacles on the ground in front.
    """
    if depth_60x80 is None:
        return None, None

    d = np.asarray(depth_60x80, dtype=np.float32)
    d = np.nan_to_num(d, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=MAX_RANGE)
    d = np.clip(d, MIN_RANGE, MAX_RANGE)

    H, W = d.shape
    r0, r1 = int(0.45 * H), int(0.95 * H)
    band = d[r0:r1, :]                          # (hb, W)
    ranges = np.percentile(band, 20, axis=0)     # robust "near" range per column
    angles = np.linspace(-FOV_H/2, +FOV_H/2, W, dtype=np.float32)
    return ranges.astype(np.float32), angles.astype(np.float32)

# -------- baseline env (same I/O as DWA)
class BaselineEnv:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("vfh_plus_baseline")

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

        self.depth = None
        self.x = self.y = self.yaw = 0.0
        self.v_meas = self.w_meas = 0.0
        self.roll = self.pitch = 0.0
        self.omega_l = self.omega_r = 0.0
        self.have_odom = self.have_js = False

        self.goal_x = self.goal_y = 0.0
        self.goal_spawned = False

        self.last_v = 0.0
        self.last_w = 0.0

        # wheel wear / radius mismatch (same idea as DWA baseline)
        self.mL = 1.0
        self.mR = 1.0
        self.rng = np.random.RandomState(0)

    def cb_depth(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
        d = np.nan_to_num(d, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=MAX_RANGE)
        d = np.clip(d, 0.0, MAX_RANGE)
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
        idx = {n:i for i,n in enumerate(msg.name)}
        def avg(names):
            vals = []
            for n in names:
                i = idx.get(n, None)
                if i is not None and i < len(msg.velocity):
                    vals.append(float(msg.velocity[i]))
            return float(sum(vals)/len(vals)) if vals else 0.0
        self.omega_l = avg(LEFT_JOINTS)
        self.omega_r = avg(RIGHT_JOINTS)
        self.have_js = True

    def send(self, v, w):
        # apply mismatch in wheel linear speeds (same as DWA file)
        vL = v - 0.5 * TRACK_W * w
        vR = v + 0.5 * TRACK_W * w
        vL2 = self.mL * vL
        vR2 = self.mR * vR
        v_sent = 0.5 * (vR2 + vL2)
        w_sent = (vR2 - vL2) / max(TRACK_W, 1e-6)

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
        time.sleep(DT * 50)

    def obs(self):
        dist, hdiff = goal_feats(self.x, self.y, self.yaw, self.goal_x, self.goal_y)
        ranges, angs = depth_to_pseudolaser(self.depth)  # (W,), (W,)
        slip = wheel_slip(self.v_meas, self.w_meas, self.omega_l, self.omega_r)
        return dist, hdiff, ranges, angs, slip, float(self.roll), float(self.pitch)

    def close(self):
        try: self.exec.shutdown()
        except: pass
        try: self.node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass

# -------- controllers
def pick_vfh_plus(last_w, hdiff, ranges, angs):
    """
    Minimal VFH+: pick a steering direction from a smoothed "free-space" histogram,
    then set (v,w) with simple heuristics.
    """
    if ranges is None or angs is None:
        return 0.0, (W_MAX * 0.6) * (1.0 if hdiff > 0 else -1.0)

    # obstacle density (bigger when closer)
    dens = np.clip((SAFE / np.maximum(ranges, 1e-3))**2, 0.0, 5.0)

    # smooth histogram (tiny window)
    k = 7
    kernel = np.ones(k, dtype=np.float32) / float(k)
    dens_s = np.convolve(dens, kernel, mode="same")

    # free bins (low density) + keep some margin
    free = (ranges > (SAFE + 0.05)) & (dens_s < 0.8)

    if not np.any(free):
        # no opening -> rotate towards goal
        return 0.0, (W_MAX * 0.6) * (1.0 if hdiff > 0 else -1.0)

    # candidate angles (subset to keep it fast)
    cand_idx = np.where(free)[0][::2]
    cand_ang = angs[cand_idx]

    # cost: goal alignment + smoothness + (slight) clearance preference
    k_goal = 2.0
    k_smooth = 0.5
    k_clear = 0.2

    # estimate clearance along candidate (use its range)
    cand_r = ranges[cand_idx]
    steer_prev = float(np.clip(last_w / max(W_MAX, 1e-6), -1.0, 1.0)) * (FOV_H/2)

    best_i = 0
    best_cost = 1e9
    for i, a in enumerate(cand_ang):
        cost = (
            k_goal * abs(wrap_pi(float(hdiff) - float(a))) +
            k_smooth * abs(wrap_pi(float(a) - float(steer_prev))) +
            k_clear * (1.0 / (float(cand_r[i]) + 1e-3))
        )
        if cost < best_cost:
            best_cost = cost
            best_i = i

    steer = float(cand_ang[best_i])

    # map steer -> w, and clearance -> v
    w_cmd = float(np.clip(2.2 * steer, -W_MAX, W_MAX))
    # pick forward speed based on forward-looking minimum
    r_fwd = float(np.min(ranges[len(ranges)//2 - 5: len(ranges)//2 + 5]))
    v_cmd = float(np.clip(0.8 * (r_fwd - SAFE), 0.0, V_MAX))
    # if goal is very sideways, slow down (classic VFH+ behavior)
    v_cmd *= float(np.clip(1.0 - abs(hdiff)/math.radians(90.0), 0.2, 1.0))
    return v_cmd, w_cmd

def pick_vector_field(hdiff, ranges, angs):
    """
    Tiny vector-field baseline: attractive to goal + repulsive from close obstacles.
    """
    if ranges is None or angs is None:
        return 0.0, (W_MAX * 0.6) * (1.0 if hdiff > 0 else -1.0)

    # attractive (unit) in robot frame
    ax = math.cos(hdiff); ay = math.sin(hdiff)

    # repulsive sum
    rep_range = 1.2
    rx = 0.0; ry = 0.0
    for r, a in zip(ranges[::2], angs[::2]):
        r = float(r); a = float(a)
        if r < rep_range:
            mag = (1.0 / max(r, 1e-3) - 1.0 / rep_range)
            # push away from obstacle direction
            rx -= mag * math.cos(a)
            ry -= mag * math.sin(a)

    vx = ax + 0.8 * rx
    vy = ay + 0.8 * ry
    steer = math.atan2(vy, max(vx, 1e-3))

    # if obstacle very close in front -> hard turn
    r_fwd = float(np.min(ranges[len(ranges)//2 - 5: len(ranges)//2 + 5]))
    if r_fwd < SAFE:
        return 0.0, (W_MAX * 0.7) * (1.0 if hdiff > 0 else -1.0)

    w_cmd = float(np.clip(2.0 * steer, -W_MAX, W_MAX))
    v_cmd = float(np.clip(0.9 * (r_fwd - SAFE), 0.0, V_MAX))
    v_cmd *= float(np.clip(1.0 - abs(steer)/math.radians(90.0), 0.2, 1.0))
    return v_cmd, w_cmd

# -------- main loop (same 5 metrics)
def main():
    # choose baseline: "vfh" (default) or "vf"
    MODE = "vfh"  # change to "vf" for vector-field baseline
    TRIALS = 100

    env = BaselineEnv()

    succ = 0
    int_slips, steps_list, int_rolls, int_pitchs = [], [], [], []

    for ep in range(1, TRIALS + 1):
        env.reset()

        ep_steps = 0
        ep_int_slip = 0.0
        ep_int_roll = 0.0
        ep_int_pitch = 0.0
        slip_count = 0
        ok = False

        for _ in range(MAX_STEPS):
            dist, hdiff, ranges, angs, slip, roll, pitch = env.obs()

            ep_int_slip  += abs(slip)  * DT
            ep_int_roll  += abs(roll)  * DT
            ep_int_pitch += abs(pitch) * DT

            if dist < GOAL_THR:
                ok = True
                break

            # slip termination (same as DWA baseline)
            if slip > 0.5: slip_count += 1
            else: slip_count = 0
            if slip_count > 20:
                ok = False
                break

            # roll/pitch safety (same as DWA baseline)
            if abs(roll) > math.radians(30) or abs(pitch) > math.radians(30):
                ok = False
                break

            if MODE == "vf":
                v_cmd, w_cmd = pick_vector_field(hdiff, ranges, angs)
            else:
                v_cmd, w_cmd = pick_vfh_plus(env.last_w, hdiff, ranges, angs)

            # --- NEW: avoid hills more using roll/pitch (still terminates at 30deg like DWA)
            tilt = max(abs(roll), abs(pitch))
            if tilt > TILT_WARN:
                if tilt >= TILT_HARD:
                    v_cmd = 0.0
                    w_cmd = (W_MAX * 0.6) * (1.0 if hdiff > 0 else -1.0)
                else:
                    s = 1.0 - (tilt - TILT_WARN) / max((TILT_HARD - TILT_WARN), 1e-6)
                    s = float(np.clip(s, 0.0, 1.0))
                    v_cmd = float(v_cmd) * s
                    w_cmd = float(np.clip(float(w_cmd) * (1.0 + 0.7 * (1.0 - s)), -W_MAX, W_MAX))

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

        print(f"EP {ep:03d}/{TRIALS}  mode={MODE}  success={int(ok)}  steps={ep_steps:3d}  int_slip={ep_int_slip:.3f}  int_roll={ep_int_roll:.3f}  int_pitch={ep_int_pitch:.3f}  slip_count={slip_count}")

    success_rate  = succ / float(TRIALS)
    avg_int_slip  = float(np.mean(int_slips)) if int_slips else 0.0
    avg_steps     = float(np.mean(steps_list)) if steps_list else 0.0
    avg_int_pitch = float(np.mean(int_pitchs)) if int_pitchs else 0.0
    avg_int_roll  = float(np.mean(int_rolls)) if int_rolls else 0.0

    print("\n==== VFH+ / Vector-Field baseline summary ====")
    print(f"mode           : {MODE}")
    print(f"success_rate   : {success_rate:.3f}")
    print(f"avg_int_slip   : {avg_int_slip:.3f}")
    print(f"avg_steps      : {avg_steps:.3f}")
    print(f"avg_int_pitch  : {avg_int_pitch:.3f}")
    print(f"avg_int_roll   : {avg_int_roll:.3f}")

    env.close()

if __name__ == "__main__":
    main()
