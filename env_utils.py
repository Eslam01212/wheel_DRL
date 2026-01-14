# env_utils.py
import math
import numpy as np
import cv2


def quat_to_rpy(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = 1.0 if sinp > 1.0 else (-1.0 if sinp < -1.0 else sinp)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def depth_image_norm(depth_m, out_h=60, out_w=80, dmax=5.0):
    d = np.asarray(depth_m, dtype=np.float32)
    d = np.nan_to_num(d, nan=dmax, posinf=dmax, neginf=dmax)
    d = np.clip(d, 0.0, dmax)
    d = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    d = (d / dmax) * 2.0 - 1.0               # [-1,1]
    return d[None, :, :].astype(np.float32)  # (1,H,W)


def goal_feats(x, y, yaw, gx, gy):
    dx = gx - x
    dy = gy - y
    dist = math.hypot(dx, dy)
    goal_ang = math.atan2(dy, dx)
    hdiff = (goal_ang - yaw + math.pi) % (2 * math.pi) - math.pi
    return float(hdiff), float(dist)


def locomotion_feats(dist, hdiff, roll, pitch, v_meas, ang_vel, dist_norm, v_max, w_max):
    d_norm01 = min(dist / max(dist_norm, 1e-6), 1.0)
    d_norm = 2.0 * d_norm01 - 1.0
    psi_norm = hdiff / math.pi
    phi_norm = roll / math.pi
    theta_norm = pitch / math.pi
    v_norm = float(np.clip(v_meas / max(v_max, 1e-3), -1.0, 1.0))
    w_norm = float(np.clip(ang_vel / max(w_max, 1e-3), -1.0, 1.0))
    #print(np.array([d_norm, psi_norm, phi_norm, theta_norm, v_norm, w_norm], dtype=np.float32))
    return np.array([d_norm, psi_norm, phi_norm, theta_norm, v_norm, w_norm], dtype=np.float32)


def wheel_terrain_feats(v_meas, ang_vel, omega_l, omega_r,
                        wheel_radius, track_width,
                        roll, pitch, robot_mass):
    """
    Return (wheel4, slip) where:
      wheel4 = [SR_l, SR_r, 0, 0] in [-1,1]
      slip   = max(|SR_l|,|SR_r|) in [0,1]

    SR uses a symmetric denominator to behave better when v or R*omega is small.
    """
    eps = 1e-4
    R = float(wheel_radius)
    W = float(track_width)

    # left/right longitudinal speed at wheel centers (diff-drive kinematics)
    v_l = float(v_meas - 0.5 * W * ang_vel)
    v_r = float(v_meas + 0.5 * W * ang_vel)

    # wheel peripheral speeds
    vwl = float(R * omega_l)
    vwr = float(R * omega_r)

    # deadband: if basically stopped, report zero slip (avoids reset transients)
    if (abs(v_meas) < 0.02) and (abs(omega_l) < 0.5) and (abs(omega_r) < 0.5):
        wheel4 = np.zeros(4, dtype=np.float32)
        return wheel4, 0.0

    # symmetric slip ratio (bounded, less sensitive to near-zero terms)
    denom_l = max(abs(vwl) + abs(v_l), eps)
    denom_r = max(abs(vwr) + abs(v_r), eps)

    SR_l = 2.0 * (vwl - v_l) / denom_l
    SR_r = 2.0 * (vwr - v_r) / denom_r

    SR_l = float(np.clip(SR_l, -1.0, 1.0))
    SR_r = float(np.clip(SR_r, -1.0, 1.0))
    slip = float(np.clip(max(abs(SR_l), abs(SR_r)), 0.0, 1.0))

    wheel4 = np.array([SR_l, SR_r, 0.0, 0.0], dtype=np.float32)
    return wheel4, slip


def compute_reward_no_scan(hdiff, prev_dist, dist, slip, roll, pitch,
                           step_i, slip_count, max_steps):

    beta_d = 100.0
    beta_s = -1
    beta_i = -10.0
    beta_cs = -0.1

    r_d = (prev_dist - dist) if step_i > 1 else 0.0
    rs  = slip  if slip  > .6 else 0.0
    r_i_roll  = abs(roll)  if abs(roll)  > math.radians(5) else 0.0
    r_i_pitch = abs(pitch) if abs(pitch) > math.radians(5) else 0.0
    r_i = (r_i_roll + r_i_pitch)   # or sum
    r_cs = abs(hdiff)
    r = beta_d * r_d + beta_s * rs + beta_i * r_i + beta_cs * r_cs 
    #print(f"{r},,,,r_d={beta_d * r_d:.3f}, rs={beta_s * rs:.3f}, r_i={beta_i * r_i:.5f}, r_cs={beta_cs * r_cs:.2f}")

    done = False
    trunc = False

    if dist < 0.5:
        done = True
        r += 50.0
        print(f"dist={dist:.2f}")

    # non-geometric hazard: high slip (more than step of sipping)
    if slip_count > 20 :
        done = True
        r += -10.0
        print(f"High slip_count: {slip_count:.2f}")

    if abs(roll) > math.radians(30) or abs(pitch) > math.radians(30):
        done = True
        r += -10.0
        print(f"roll={roll:.5f} or pitch={pitch:.5f}")

    if step_i >= max_steps:
        trunc = True
        print(f"step_i={step_i:.2f}")

    return float(r), done, trunc

