# -*- coding: utf-8 -*-
"""手动评估 ppo_v2：逐集打印，定位挂点。"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EnvConfig
from evtol_env import EVTOLEnv
from stable_baselines3 import PPO

cfg = EnvConfig()
cfg.height_hard_limit = float("inf")
print("创建环境...", flush=True)
env = EVTOLEnv(map_file=cfg.map_file, stride=cfg.stride, max_steps=cfg.max_steps,
               goal_radius_h=cfg.goal_radius_h, goal_radius_v=cfg.goal_radius_v,
               check_step=cfg.check_step, obs_patch=cfg.obs_patch, obs_radius=cfg.obs_radius,
               cruise_speed=cfg.cruise_speed, use_energy=cfg.use_energy, use_wind=cfg.use_wind,
               use_flight_profile=cfg.use_flight_profile, height_hard_limit=cfg.height_hard_limit,
               height_penalty=cfg.height_penalty, wind_speed=cfg.wind_speed,
               wind_direction=cfg.wind_direction, turbulence=cfg.turbulence,
               shield=cfg.shield, blocked_penalty=cfg.blocked_penalty,
               start_altitude=cfg.start_altitude, seed=123)
print("加载模型...", flush=True)
model = PPO.load(r"\\wsl.localhost\Ubuntu\home\ayan1907\evtol\evtol_rl\ppo_bc.zip")
print("评估 BC 模型 10 集...", flush=True)
res = {}
for ep in range(10):
    obs, _ = env.reset(seed=123 + ep)
    energy, blocked, t0 = 0.0, 0, time.time()
    while True:
        a, _ = model.predict(obs, deterministic=False)  # 随机采样绕开 shield 确定性死循环伪影
        obs, r, term, trunc, info = env.step(int(a))
        energy += info.get("energy_j", 0.0)
        blocked += info.get("blocked", False)
        if term or trunc:
            key = info.get("reason", "?")
            if info.get("blocked"): key = "blocked"
            res[key] = res.get(key, 0) + 1
            print(f"  ep{ep}: {key:16s} {info['step']:3d}步 {info['distance']:7.1f}m "
                  f"能耗{energy/3.6e6:.3f}kWh 阻挡{blocked} 用时{time.time()-t0:.1f}s", flush=True)
            break
print("结局分布:", res)