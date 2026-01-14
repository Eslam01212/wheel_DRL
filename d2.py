#!/usr/bin/env python3
import os
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from env_teacher import RosNavEnv


class SaveEveryN(BaseCallback):
    def __init__(self, n=10_000, path="./models"):
        super().__init__()
        self.n = int(n)
        self.path = path

    def _on_step(self) -> bool:
        if self.n > 0 and (self.num_timesteps % self.n == 0):
            self.model.save(os.path.join(self.path, f"ppo_step_{self.num_timesteps}.zip"))
        return True


class SaveBestEpisode(BaseCallback):
    """Saves ./models/ppo_teacher.zip when an episode achieves a new best return.
    Requires Monitor wrapper to populate info['episode'].
    """
    def __init__(self, path="./models/ppo_teacher.zip"):
        super().__init__()
        self.path = path
        self.best = -np.inf

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True

        for done, info in zip(dones, infos):
            if done and isinstance(info, dict) and "episode" in info:
                r = float(info["episode"]["r"])
                if r > self.best:
                    self.best = r
                    self.model.save(self.path)  # overwrite best checkpoint
        return True

    
class DepthCNN(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 3, stride=1, padding=1), nn.ReLU()
        )
    def forward(self, x): return self.net(x)


class AttnPool(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Parameter(th.zeros(1, 1, c))
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.scale = c ** -0.5
    def forward(self, feat):
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        q = self.q.expand(B, -1, -1)
        attn = (q @ self.k(x).transpose(1, 2)) * self.scale
        w = th.softmax(attn, dim=-1)
        return (w @ self.v(x)).squeeze(1)


class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim)
        prop_dim = observation_space["prop"].shape[0]
        pred_dim = observation_space["pred"].shape[0]

        self.cnn = DepthCNN(64)
        self.pool = AttnPool(64)
        self.prop = nn.Sequential(nn.Linear(prop_dim, 64), nn.ReLU())
        self.pred = nn.Sequential(nn.Linear(pred_dim, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(64 + 64 + 64, features_dim), nn.ReLU())

    def forward(self, obs):
        zD = self.pool(self.cnn(obs["depth"]))
        zP = self.prop(obs["prop"])
        zR = self.pred(obs["pred"])
        return self.fuse(th.cat([zD, zP, zR], dim=1))


def main():
    force_zero_pred=False
    env = RosNavEnv(force_zero_pred)
    """model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=Extractor,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        device="cuda" if th.cuda.is_available() else "cpu",
        verbose=0,
        tensorboard_log="./tb",
    )"""
    model = SAC(
        "MultiInputPolicy",
        env,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=0,
        tensorboard_log="./tb",
    )

    #model = PPO.load("./models/ppo_teacher", env=env)   # <- continue from here

    callback = CallbackList([
        SaveEveryN(5_000, "./models"),
        SaveBestEpisode("./models/ppo_teacher.zip"),
    ])
    
    model.learn(
        total_timesteps=80_000,
        callback=callback,
        tb_log_name="ppo",
    )
    if force_zero_pred:
        model.save("./models/ppo_no_predictor.zip")
    else:
        model.save("./models/ppo_with_predictor.zip")

    env.close()



if __name__ == "__main__":
    main()

