# baseline_compare.py —— 第 3 周启用
# =========================================================================
# 把 RL 策略跑出来的航迹，用**原项目自己的评估器**打分，
# 与现成的三个基线放在同一张表里。
#
# 为什么这一步重要：
#   用原项目的 PathEvaluator 打分 = 用同一把尺子量。
#   自己另写一套评价指标会被面试官问"你的指标跟原项目为什么不一样"。
#
# 现成的基线数字（来自 V4 报告表 4，可直接引用）：
#   标准 3D A*      3.40 kWh / 0.27 s / 合规 0%
#   低碳 A*         4.139 kWh / 合规 100%
#   NSGA-II         17.19 s / 8 个 Pareto 解 / Hypervolume 2.10e8
#   RL（待测）      ? kWh / ? ms
#
# ⚠️ 本文件是接口骨架，第 3 周首次运行前请先确认两件事：
#     1. PathEvaluator 的构造签名 (env, energy_model, wind_field) 是否与 v7 一致
#     2. 路径格式要求：坐标列表 [(x, y, z), ...]，还是 LowCarbonNode 列表
# =========================================================================

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_evaluator(map_file: str):
    """构造原项目 v7 的 PathEvaluator（与 NSGA-II 用的是同一个）。"""
    from evtol_path_planning_v3_clean import load_environment_from_json
    from evtol_v7.evtol_path_planning_v7 import PathEvaluator
    from evtol_high_fidelity_energy import create_energy_model
    from evtol_path_planning_v4 import WindField

    env, start, goal, meta = load_environment_from_json(map_file)
    energy = create_energy_model("EH216-S")
    wind = WindField(env=env, base_wind_speed=5.0,
                     base_wind_direction=315.0, turbulence_intensity=0.1)

    evaluator = PathEvaluator(env, energy, wind)
    return evaluator, env, start, goal


def score(evaluator, path):
    """返回 [energy, time, noise, safety, length]（全部 MINIMIZE）。"""
    obj = evaluator.evaluate([tuple(map(float, p)) for p in path])
    return np.asarray(obj, dtype=float)


def rollout_rl(map_file: str, model_path: str, seed: int = 0):
    """用训练好的 PPO 模型跑一条航迹。

    返回 (path, info)。**必须看 info['reason']**：
    若 RL 因 "collision" 或 "height_violation" 提前终止，
    拿到的是一条断掉的航迹，此时能耗/长度指标没有可比性，不要往表里填。
    """
    from stable_baselines3 import PPO
    from evtol_env import EVTOLEnv

    env = EVTOLEnv(map_file=map_file, seed=seed)
    model = PPO.load(model_path)

    obs, _ = env.reset(seed=seed)
    info: dict = {}
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        if term or trunc:
            break
    return env.trajectory_array, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map_standard.json")
    ap.add_argument("--model", default="ppo_evtol.zip")
    ap.add_argument("--timing", action="store_true",
                    help="顺便测 RL 策略的单次推理耗时（对应在线重规划这个卖点）")
    args = ap.parse_args()

    map_path = args.map if os.path.isabs(args.map) else os.path.join(_ROOT, args.map)
    evaluator, env, start, goal = build_evaluator(map_path)

    path, info = rollout_rl(map_path, os.path.join(_HERE, args.model))
    obj = score(evaluator, path)

    print("=" * 62)
    print("RL 策略航迹 —— 用原项目 PathEvaluator 打分")
    print(f"  终止原因 : {info.get('reason', '?')}"
          f"   ← 非 arrived 时下面的指标不可比，别填表")
    print(f"  路径点数 : {len(path)}")
    print(f"  能耗     : {obj[0]:.4f} kWh")
    print(f"  时间     : {obj[1]:.2f} s")
    print(f"  噪声     : {obj[2]:.4f}")
    print(f"  安全     : {obj[3]:.4f}")
    print(f"  长度     : {obj[4]:.1f} m")
    print("=" * 62)

    if args.timing:
        import time
        from stable_baselines3 import PPO
        from evtol_env import EVTOLEnv
        env2 = EVTOLEnv(map_file=map_path)
        model = PPO.load(os.path.join(_HERE, args.model))
        obs, _ = env2.reset(seed=0)
        t0 = time.perf_counter()
        for _ in range(200):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env2.step(int(a))
            if term or trunc:
                obs, _ = env2.reset(seed=0)
        dt = (time.perf_counter() - t0) / 200
        print(f"单步推理耗时: {dt * 1000:.3f} ms  "
              f"（对比 NSGA-II 重算一次的 17.19 s）")

    print("\n把它填进对比表：")
    print("  | 方法      | 能耗(kWh) | 规划耗时 | 合规 |")
    print("  |-----------|-----------|----------|------|")
    print("  | 标准 3D A*| 3.40      | 0.27 s   | 0%   |")
    print("  | 低碳 A*   | 4.139     | —        | 100% |")
    print("  | NSGA-II   | 前沿      | 17.19 s  | 前沿 |")
    print(f"  | RL(本项)  | {obj[0]:.3f}     | ?        | ?    |")


if __name__ == "__main__":
    main()
