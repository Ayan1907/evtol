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
            r_arrive=cfg.r_arrive,
            r_collide=cfg.r_collide,
            r_step=cfg.r_step,
            k_progress=cfg.k_progress,
            pbrs_gamma=cfg.pbrs_gamma,
            w_energy=cfg.w_energy,
            w_margin=cfg.w_margin,
            margin_band=cfg.margin_band,
            e_ref_kwh=cfg.e_ref_kwh,
            start_altitude=cfg.start_altitude,
            shield=cfg.shield,
            blocked_penalty=cfg.blocked_penalty,
            scenario_jitter_m=cfg.scenario_jitter_m,
            randomize_wind=cfg.randomize_wind,
            wind_speed_range=cfg.wind_speed_range,
            seed=seed,
        )
    return _thunk


def evaluate(model, env_cfg: EnvConfig, episodes: int = 50, seed: int = 123):
    """跑 N 个 episode，统计成功率与平均能耗。"""
    env = env_cfg and make_env(env_cfg, seed=seed)()
    stats = {"success": 0, "collision": 0, "timeout": 0, "blocked": 0,
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
            if info.get("blocked"):
                stats["blocked"] += 1
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
    ap.add_argument("--w-energy", type=float, default=None, help="能耗项权重（覆盖 config）")
    ap.add_argument("--w-margin", type=float, default=None, help="安全裕度项权重（覆盖 config）")
    ap.add_argument("--eval-episodes", type=int, default=None, help="评估 episode 数（覆盖 config）")
    ap.add_argument("--ent-coef", type=float, default=None, help="熵系数（覆盖 config）")
    ap.add_argument("--gamma", type=float, default=None, help="折扣 γ（同时同步 env 的 pbrs_gamma）")
    ap.add_argument("--height-hard-limit", type=float, default=None, help="高度硬终止阈值 m（inf=关闭硬终止，仅软罚）")
    ap.add_argument("--start-altitude", type=float, default=None, help="课程学习：起点高度覆盖")
    ap.add_argument("--no-shield", action="store_true", help="关闭动作屏蔽（碰撞即终止）")
    ap.add_argument("--lr", type=float, default=None, help="学习率（覆盖 config，微调用 5e-5）")
    ap.add_argument("--ckpt-prefix", default="ppo_evtol", help="checkpoint 文件名前缀")
    ap.add_argument("--stride", type=float, default=None, help="分辨率课程：每步位移 m")
    ap.add_argument("--max-steps", type=int, default=None, help="分辨率课程：episode 步数上限")
    ap.add_argument("--init-model", default=None, help="从已有模型继续训练")
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
    if args.w_energy is not None:
        env_cfg.w_energy = args.w_energy
    if args.w_margin is not None:
        env_cfg.w_margin = args.w_margin
    train_cfg = TrainConfig(total_timesteps=args.timesteps, seed=args.seed,
                            save_path=args.save_path)
    if args.eval_episodes is not None:
        train_cfg.eval_episodes = args.eval_episodes
    if args.ent_coef is not None:
        train_cfg.ent_coef = args.ent_coef
    if args.gamma is not None:
        train_cfg.gamma = args.gamma
        env_cfg.pbrs_gamma = args.gamma
    if args.height_hard_limit is not None:
        env_cfg.height_hard_limit = args.height_hard_limit
    if args.start_altitude is not None:
        env_cfg.start_altitude = args.start_altitude
    if args.no_shield:
        env_cfg.shield = False
    if args.stride is not None:
        env_cfg.stride = args.stride
    if args.max_steps is not None:
        env_cfg.max_steps = args.max_steps
    if args.lr is not None:
        train_cfg.learning_rate = args.lr

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

    # tensorboard 可选：未安装则退化为无 TB 日志（不阻塞训练）
    try:
        import tensorboard  # noqa: F401
        tb_dir = train_cfg.log_dir
    except ImportError:
        tb_dir = None
        print("（未安装 tensorboard，跳过 TB 日志；训练曲线用 checkpoint 评估替代）")

    if args.init_model:
        model = PPO.load(args.init_model, env=env)
        print(f"从 {args.init_model} 继续训练（课程学习微调）")
    else:
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
            tensorboard_log=tb_dir,
            seed=train_cfg.seed,
        )

    ckpt = CheckpointCallback(
        save_freq=50_000, save_path=os.path.join(_HERE, "checkpoints"),
        name_prefix=args.ckpt_prefix,
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
