#!/usr/bin/env python3
import numpy as np
from stable_baselines3 import SAC

from env_teacher import RosNavEnv
from env_core import DT

SUCCESS_DIST = 0.5


def eval_one(model_path: str, force_zero_pred: bool, n_eps: int = 20):
    env = RosNavEnv(force_zero_pred=force_zero_pred)
    model = SAC.load(model_path, env=env, device="cpu")

    succ = 0
    slip_I, steps_L, pitch_I, roll_I = [], [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()
        done = trunc = False
        sI = pI = rI = 0.0
        steps = 0
        info = {}

        while not (done or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(act)

            steps += 1
            sI += float(info.get("slip", 0.0)) * DT
            pI += abs(float(getattr(env.core, "pitch", 0.0))) * DT
            rI += abs(float(getattr(env.core, "roll", 0.0))) * DT

        if bool(done) and (float(info.get("dist", 1e9)) < SUCCESS_DIST):
            succ += 1
            slip_I.append(sI)
            steps_L.append(steps)
            pitch_I.append(pI)
            roll_I.append(rI)

    env.close()

    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return succ / float(n_eps), mean(slip_I), mean(steps_L), mean(pitch_I), mean(roll_I), succ


def main():
    n_eps = 20
    print(f"DT={DT}  SUCCESS_DIST={SUCCESS_DIST}  n_eps={n_eps}")

    sr0, sI0, st0, pI0, rI0, ns0 = eval_one("./compare/ppo_no_predictor.zip",  True,  n_eps)
    sr1, sI1, st1, pI1, rI1, ns1 = eval_one("./compare/ppo_with_predictor.zip", False, n_eps)

    print(f"\nNo predictor ({ns0}/{n_eps} success)")
    print("  success_rate  =", sr0)
    print("  avg_int_slip  =", sI0)
    print("  avg_steps     =", st0)
    print("  avg_int_pitch =", pI0)
    print("  avg_int_roll  =", rI0)

    print(f"\nWith predictor ({ns1}/{n_eps} success)")
    print("  success_rate  =", sr1)
    print("  avg_int_slip  =", sI1)
    print("  avg_steps     =", st1)
    print("  avg_int_pitch =", pI1)
    print("  avg_int_roll  =", rI1)


if __name__ == "__main__":
    main()
