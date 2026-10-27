# -*- coding: utf-8 -*-
"""行为克隆热启动：用启发式航线生成示范数据，监督预训练 PPO 策略网络。

启发式（合法航线骨架）：
  1) 爬升至巡航高度（130 或 155，随机化）
  2) 水平朝目标飞
  3) 距目标 < 下降点（450 或 350，随机化）且高于 130 → 降一步
  4) 130m 平飞直到进入到达半径

ε=0.15 随机动作扰动保证数据覆盖。产出 ppo_bc.zip 供 --init-model 微调。

用法：python pretrain_bc.py --episodes 200
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import EnvConfig  # noqa: E402
from evtol_env import EVTOLEnv  # noqa: E402


def make_env(cfg: EnvConfig):
    return EVTOLEnv(
        map_file=cfg.map_file, stride=cfg.stride, max_steps=cfg.max_steps,
        goal_radius_h=cfg.goal_radius_h, goal_radius_v=cfg.goal_radius_v,
        check_step=cfg.check_step, obs_patch=cfg.obs_patch, obs_radius=cfg.obs_radius,
        cruise_speed=cfg.cruise_speed, use_energy=cfg.use_energy, use_wind=cfg.use_wind,
        use_flight_profile=cfg.use_flight_profile, height_hard_limit=cfg.height_hard_limit,
        height_penalty=cfg.height_penalty, wind_speed=cfg.wind_speed,
        wind_direction=cfg.wind_direction, turbulence=cfg.turbulence,
        shield=cfg.shield, blocked_penalty=cfg.blocked_penalty,
        start_altitude=cfg.start_altitude, seed=0,
    )


def heuristic(env, cruise_alt, descend_at):
    up = int(np.where((env._DIRS == np.array([0, 0, 1])).all(axis=1))[0][0])
    down = int(np.where((env._DIRS == np.array([0, 0, -1])).all(axis=1))[0][0])
    hd = float(np.hypot(env.goal[0] - env.pos[0], env.goal[1] - env.pos[1]))
    z = env.pos[2]
    if z < cruise_alt and hd > descend_at:
        return up
    if hd < descend_at and z > cruise_alt - 25:
        return down
    d = np.array([env.goal[0] - env.pos[0], env.goal[1] - env.pos[1], 0.0])
    return int(np.argmax(env._DIRS_UNIT @ (d / (np.linalg.norm(d) + 1e-9))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--save-path", default="ppo_bc.zip")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    cfg = EnvConfig()
    # 示范生成用软罚：ε 扰动的一次越带不应致死（硬终止下 196/200 被随机动作杀死）
    cfg.height_hard_limit = float('inf')
    env = make_env(cfg)
    rng = np.random.default_rng(0)

    obs_list, act_list = [], []
    outcomes = {}
    for ep in range(args.episodes):
        cruise_alt = float(rng.choice([130.0, 155.0]))
        descend_at = float(rng.choice([450.0, 350.0]))
        obs, _ = env.reset(seed=ep)
        while True:
            a = heuristic(env, cruise_alt, descend_at)
            if rng.random() < args.epsilon:
                a = int(rng.integers(env.N_ACTIONS))
            obs_list.append(obs.copy())
            act_list.append(a)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                key = info.get("reason", "?")
                if info.get("blocked"):
                    key = "blocked"
                outcomes[key] = outcomes.get(key, 0) + 1
                break
        if (ep + 1) % 50 == 0:
            print(f"  示范 {ep+1}/{args.episodes}，累计样本 {len(obs_list)}")

    print("示范结局分布:", outcomes)

    # ---- 监督预训练策略网络 ----
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.999,
        gae_lambda=0.95, ent_coef=0.01, verbose=0, seed=0,
    )
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    X = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.as_tensor(np.array(act_list), dtype=torch.long)
    n = len(X)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot, hit = 0.0, 0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            dist = policy.get_distribution(X[idx])
            loss = -dist.log_prob(Y[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            hit += int((dist.distribution.logits.argmax(dim=-1) == Y[idx]).sum())
        print(f"  BC epoch {epoch+1}/{args.epochs}: loss={tot/n:.4f} acc={hit/n:.3f}")

    model.save(os.path.join(_HERE, args.save_path))
    print(f"BC 预训练模型已保存: {args.save_path}（用 --init-model 接 PPO 微调）")


if __name__ == "__main__":
    main()
