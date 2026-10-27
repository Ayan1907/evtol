# -*- coding: utf-8 -*-
"""map 基线对照：随机策略 / 不爬升直飞 / 爬升到指定高度后直飞。

用法（Windows 或 WSL 均可）：
    python run_baselines.py                      # 默认 map_dense.json
    python run_baselines.py --map map_dense.json --alts 130,155,180 --seeds 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from evtol_env import EVTOLEnv


def run_random(env: EVTOLEnv, seed: int):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_r, n_viol, energy = 0.0, 0, 0.0
    while True:
        obs, r, term, trunc, info = env.step(int(rng.integers(env.N_ACTIONS)))
        total_r += r
        energy += info.get("energy_j", 0.0)
        n_viol += info.get("height_violation_m", 0.0) > 0.0
        if term or trunc:
            break
    return info, total_r, n_viol, energy


def run_straight_no_climb(env: EVTOLEnv, seed: int):
    """不爬升：保持起始高度，水平朝目标直飞（对照组，预期违规终止）。"""
    return _fly(env, seed, cruise_alt=None)


def run_climb_straight(env: EVTOLEnv, seed: int, cruise_alt: float):
    """先爬升到 cruise_alt，再保持高度水平朝目标飞。"""
    return _fly(env, seed, cruise_alt=cruise_alt)


def _fly(env: EVTOLEnv, seed: int, cruise_alt):
    obs, _ = env.reset(seed=seed)
    up_action = int(np.where((env._DIRS == np.array([0, 0, 1])).all(axis=1))[0][0])
    total_r, n_viol, energy = 0.0, 0, 0.0
    while True:
        if cruise_alt is not None and env.pos[2] < cruise_alt:
            a = up_action
        else:
            d = env.goal - env.pos
            d = np.array([d[0], d[1], 0.0])
            a = int(np.argmax(env._DIRS_UNIT @ (d / (np.linalg.norm(d) + 1e-9))))
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        energy += info.get("energy_j", 0.0)
        n_viol += info.get("height_violation_m", 0.0) > 0.0
        if term or trunc:
            break
    return info, total_r, n_viol, energy


def fmt(tag: str, info, total_r, n_viol, energy):
    e = f"{energy / 3.6e6:7.3f}" if energy else "    —  "
    return (f"{tag:34s} {info['step']:4d} 步 | {info['reason']:18s} | "
            f"剩 {info['distance']:7.1f} m | 违规步 {n_viol:3d} | "
            f"能耗 {e} kWh | R {total_r:8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/maps/map_dense.json")
    ap.add_argument("--alts", default="130,155,180")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    env = EVTOLEnv(map_file=args.map, verbose=True)
    if env.energy_error:
        print(f"[warn] 能耗模型不可用（{env.energy_error}），能耗列将为 —")

    print("\n===== 基线 1：随机策略 =====")
    for s in range(args.seeds):
        print(fmt(f"random seed={s}", *run_random(env, s)))

    print("\n===== 基线 2：不爬升直飞（起点高度平飞）=====")
    for s in range(args.seeds):
        print(fmt(f"no-climb seed={s}", *run_straight_no_climb(env, s)))

    print("\n===== 基线 3：爬升到指定高度后直飞 =====")
    alts = [float(x) for x in args.alts.split(",")]
    for alt in alts:
        for s in range(args.seeds):
            print(fmt(f"climb→{alt:.0f} m seed={s}",
                      *run_climb_straight(env, s, alt)))


if __name__ == "__main__":
    sys.exit(main())