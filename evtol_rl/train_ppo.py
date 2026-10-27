# -*- coding: utf-8 -*-
"""
PPO 训练脚本 —— 第 2 周用
=========================================================================
依赖 gymnasium + stable-baselines3：
    pip install gymnasium stable-baselines3 tensorboard

用法：
    cd ~/evtol/evtol_rl
    python train_ppo.py --timesteps 1000000
    tensorboard --logdir runs/          # 另开终端看曲线
    python train_ppo.py --eval-only     # 只评估已保存的模型

⚠️ 第 2 周末必须拿到一条像样的训练曲线（reward vs steps）+ 成功率数字。
   面试官只认这个，不认"我在做"。
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import EnvConfig, TrainConfig  # noqa: E402
from evtol_env import EVTOLEnv  # noqa: E402


def make_env(cfg: EnvConfig, seed: int = 0):
    def _thunk():
        return EVTOLEnv(
            map_file=cfg.map_file,
            stride=cfg.stride,
            max_steps=cfg.max_steps,
            goal_radius_h=cfg.goal_radius_h,
            goal_radius_v=cfg.goal_radius_v,
            check_step=cfg.check_step,
            obs_patch=cfg.obs_patch,
            obs_radius=cfg.obs_radius,
            cruise_speed=cfg.cruise_speed,
            use_energy=cfg.use_energy,
            use_wind=cfg.use_wind,
            use_flight_profile=cfg.use_flight_profile,
            height_hard_limit=cfg.height_hard_limit,
            height_penalty=cfg.height_penalty,
            wind_speed=cfg.wind_speed,
            wind_direction=cfg.wind_direction,
            turbulence=cfg.turbulence,
            seed=seed,
        )
    return _thunk


def evaluate(model, env_cfg: EnvConfig, episodes: int = 50, seed: int = 123):
    """跑 N 个 episode，统计成功率与平均能耗。"""
    env = env_cfg and make_env(env_cfg, seed=seed)()
    stats = {"success": 0, "collision": 0, "timeout": 0,
             "energy_kwh": [], "steps": [], "distance": []}

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        energy = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            energy += info.get("energy_j", 0.0)
            if term or trunc:
                reason = info["reason"]
                stats[reason] = stats.get(reason, 0) + 1
                stats["energy_kwh"].append(energy / 3.6e6)
                stats["steps"].append(info["step"])
                stats["distance"].append(info["distance"])
                break

    n = max(episodes, 1)
    return {
        "success_rate": stats.get("arrived", 0) / n,
        "collision_rate": stats.get("collision", 0) / n,
        "timeout_rate": stats.get("timeout", 0) / n,
        "avg_energy_kwh": float(np.mean(stats["energy_kwh"])) if stats["energy_kwh"] else 0.0,
        "avg_steps": float(np.mean(stats["steps"])) if stats["steps"] else 0.0,
        "avg_final_distance": float(np.mean(stats["distance"])) if stats["distance"] else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=TrainConfig.total_timesteps)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--map", default=EnvConfig.map_file)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--save-path", default=TrainConfig.save_path)
    args = ap.parse_args()

    try:
        import gymnasium  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as exc:
        print("缺少依赖。请先安装：\n    pip install gymnasium stable-baselines3 tensorboard")
        print(f"（原始错误：{exc}）")
        return 1

    env_cfg = EnvConfig(map_file=args.map)
    train_cfg = TrainConfig(total_timesteps=args.timesteps, seed=args.seed,
                            save_path=args.save_path)

    print("=" * 62)
    print("PPO 训练 eVTOL 航迹决策策略")
    print(f"  地图      : {env_cfg.map_file}")
    print(f"  步长/步数 : {env_cfg.stride} m / {env_cfg.max_steps} 步  (dt={env_cfg.dt:.3f} s)")
    print(f"  总步数    : {train_cfg.total_timesteps:,}")
    print("=" * 62)

    if args.eval_only:
        model = PPO.load(train_cfg.save_path)
        res = evaluate(model, env_cfg, episodes=train_cfg.eval_episodes)
        print("\n评估结果：")
        for k, v in res.items():
            print(f"  {k:18s}: {v:.4f}")
        return 0

    env = Monitor(make_env(env_cfg, seed=train_cfg.seed)())

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        ent_coef=train_cfg.ent_coef,
        verbose=1,
        tensorboard_log=train_cfg.log_dir,
        seed=train_cfg.seed,
    )

    ckpt = CheckpointCallback(
        save_freq=50_000, save_path=os.path.join(_HERE, "checkpoints"),
        name_prefix="ppo_evtol",
    )
    model.learn(total_timesteps=train_cfg.total_timesteps, callback=ckpt,
                progress_bar=False)

    model.save(os.path.join(_HERE, train_cfg.save_path))
    print(f"\n模型已保存: {os.path.join(_HERE, train_cfg.save_path)}")

    res = evaluate(model, env_cfg, episodes=train_cfg.eval_episodes)
    print("\n评估结果：")
    for k, v in res.items():
        print(f"  {k:18s}: {v:.4f}")

    print("\n下一步：把这条曲线和上面的成功率数字写进 README（面试要看）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
