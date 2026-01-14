# env_teacher.py
import time, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env_core import RosNavCore, DT, V_MAX, W_MAX
from env_utils import compute_reward_no_scan


MAX_STEPS = 400
SLIP_THR = 0.6
ERR_LO, ERR_HI = 0.20, 1.8  # synthetic radius factors per episode
W  = 0.20   # track width [m]

class RosNavEnv(gym.Env):
    def __init__(self, force_zero_pred=False):
        super().__init__()
        self.core = RosNavCore(
            force_zero_pred=force_zero_pred,
        )

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(-1.0, 1.0, shape=(1, 60, 80), dtype=np.float32),
            "prop":  spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
            "pred":  spaces.Box(-1.0, 1.0, shape=(self.core.pred_dim,), dtype=np.float32),
        })

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.step_i = 0
        self.prev_dist = 0.1
        self.slip_count = 0
        self.last_v = 0.0
        self.last_w = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset()

        self.step_i = 0
        self.slip_count = 0
        self.last_v, self.last_w = 0.0, 0.0
        self.errL = random.uniform(ERR_LO, ERR_HI)
        self.errR = random.uniform(ERR_LO, ERR_HI)
        time.sleep(DT*30)
        obs, info = self.core.obs(self.last_v, self.last_w)
        self.prev_dist = float(info["dist"])
        return obs, {}

    def step(self, action):

        v = (-0.0) + (V_MAX - (-0.0)) * (float(action[0]) + 1.0) / 2.0
        w = float(action[1]) * W_MAX
        self.last_v, self.last_w = v, w

        vL = v - 0.5 * W * w
        vR = v + 0.5 * W * w
        vL *= self.errL
        vR *= self.errR
        v_sent = 0.5 * (vL + vR)
        w_sent = (vR - vL) / W

        self.core.send(v_sent, w_sent)
        time.sleep(DT)

        obs, info = self.core.obs(self.last_v, self.last_w)

        dist = float(info["dist"])
        hdiff = float(info["hdiff"])
        slip = float(info["slip"])
        roll = float(self.core.roll)
        pitch = float(self.core.pitch)

        # slip counter (simple)
        if slip > SLIP_THR:
            self.slip_count += 1
        else:
            self.slip_count = 0

        self.step_i += 1

        r, done, trunc = compute_reward_no_scan(
            hdiff=hdiff,
            prev_dist=self.prev_dist,
            dist=dist,
            slip=slip,
            roll=roll,
            pitch=pitch,
            step_i=self.step_i,
            slip_count=self.slip_count,
            max_steps=MAX_STEPS,
        )

        self.prev_dist = dist

        return obs, r, done, trunc, {"dist": dist, "slip": slip, "slip_count": self.slip_count}
        
    def close(self):
        try:
            self.core.close()
        finally:
            try: super().close()
            except Exception: pass
