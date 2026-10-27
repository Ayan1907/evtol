#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能优化算法对比：NSGA-II（进化） vs MOEA/D（分解）

同一问题实例 / 同一评估函数 / 同一评价预算（population_size × (n_generations+1) 次评估），
Pareto 前沿用 **统一参考点的蒙特卡洛 HV** 作为同一把尺子计算，保证跨算法可比。

面试讲法（对应 JD 关键词"智能优化/进化算法/启发式"）：
  "把 v7 的多目标航迹优化做成算法可切换架构，在同一评价预算下对比 NSGA-II 与
   MOEA/D；用统一参考点 HV 保证可比性，暴露了库内 MOEA/D 实现的 Tchebycheff
   未归一化缺陷。"

用法：
  python test_optimizer_compare.py --gens 30 --pop 30     # 原基准口径
  python test_optimizer_compare.py --gens 5  --pop 12     # 快速冒烟
"""

import sys
import os
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from evtol_path_planning_v7 import EVTOLV7, FlightProfile
from evtol_path_planning_v3_clean import load_environment_from_json
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport
from evtol_path_planning_v4 import DynamicEnvironment, WindField
from multi_objective_optimizer import NSGAIIOptimizer


def build_planner():
    print("=" * 60)
    print("【对比实验】构建问题实例（两种算法共用）")
    print("=" * 60)
    map_file = os.path.join(_ROOT, 'test_map_standard.json')
    if not os.path.exists(map_file):
        map_file = os.path.join(_ROOT, 'test_map_v3_compatible.json')
    print(f"  加载地图: {os.path.basename(map_file)}")
    env, start_coords, goal_coords, _ = load_environment_from_json(map_file)

    standard = EVTOLStandard()
    start_vp = Vertiport(x=start_coords[0], y=start_coords[1], z=start_coords[2], standard=standard)
    goal_vp = Vertiport(x=goal_coords[0], y=goal_coords[1], z=goal_coords[2], standard=standard)

    wind_field = WindField(env=env, base_wind_speed=5.0,
                           base_wind_direction=315.0, turbulence_intensity=0.1)
    dynamic_env = DynamicEnvironment(env)
    dynamic_env.wind_field = wind_field

    flight_profile = FlightProfile(cruise_min=120.0, cruise_max=180.0,
                                   climb_distance_threshold=400.0,
                                   approach_distance_threshold=400.0)

    planner = EVTOLV7(env, standard, start_vp, goal_vp, dynamic_env,
                      flight_profile=flight_profile,
                      use_improved_wind=True, use_high_fidelity_energy=True)
    print("  ✓ V7规划器创建成功\n")
    return planner


def obj_matrix(front):
    rows = [list(map(float, ind.objectives)) for ind in front.individuals
            if ind.objectives is not None and len(ind.objectives) > 0]
    return np.array(rows) if rows else np.zeros((0, 5))


def common_hv(matrix_a, matrix_b, ref, n_samples=10000):
    """统一参考点 ref 下，分别计算两个前沿的蒙特卡洛 HV（同一把尺子）。"""
    calc = NSGAIIOptimizer(n_objectives=ref.size)
    calc.set_fixed_reference_point(ref)
    out = {}
    for name, m in (("nsga2", matrix_a), ("moead", matrix_b)):
        if m.shape[0] == 0:
            out[name] = 0.0
        else:
            # 只用非支配解算 HV（沿用 NSGA 内部的 MC 实现）
            out[name] = float(calc._hv_monte_carlo(m, np.asarray(ref, dtype=float), n_samples))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--samples", type=int, default=10000, help="HV 蒙特卡洛采样数")
    args = ap.parse_args()

    planner = build_planner()

    results = {"config": {"n_generations": args.gens, "population_size": args.pop,
                          "budget_evals": args.pop * (args.gens + 1)},
               "algorithms": {}}
    fronts = {}
    times = {}

    for algo in ("nsga2", "moead"):
        print("\n" + "#" * 60)
        print(f"# 运行 {algo.upper()}")
        print("#" * 60)
        t0 = time.time()
        front = planner.optimize_multi_objective(n_generations=args.gens,
                                                 population_size=args.pop,
                                                 algorithm=algo)
        dt = time.time() - t0
        m = obj_matrix(front)
        fronts[algo] = m
        times[algo] = dt
        print(f"\n  >>> {algo}: {dt:.2f}s, Pareto 解 {m.shape[0]} 个")

    # ---- 统一尺子 HV ----
    both = [m for m in fronts.values() if m.shape[0] > 0]
    if both:
        combined = np.vstack(both)
        ref = combined.max(axis=0) * 1.2
        hv = common_hv(fronts.get("nsga2", np.zeros((0, 5))),
                       fronts.get("moead", np.zeros((0, 5))), ref, args.samples)
    else:
        ref, hv = None, {"nsga2": 0.0, "moead": 0.0}

    obj_names = ["energy(kWh)", "time(s)", "noise(dB)", "safety", "length(m)"]
    print("\n" + "=" * 60)
    print("【对比结果】(统一参考点 HV，同一评价预算)")
    print("=" * 60)
    print(f"  评价预算: {args.pop}×({args.gens}+1) = {args.pop * (args.gens + 1)} 次评估/算法")
    print(f"  统一参考点: {np.round(ref, 3).tolist() if ref is not None else '—'}")
    for algo in ("nsga2", "moead"):
        m = fronts[algo]
        print(f"\n  [{algo}] 用时 {times[algo]:.2f}s | Pareto {m.shape[0]} 个 | HV(统一尺) {hv[algo]:.4e}")
        if m.shape[0]:
            best = m.min(axis=0)
            for n, b in zip(obj_names, best):
                print(f"      最优 {n:12s}: {b:.4f}")

    # 结论提示（诚实边界）
    print("\n  [诚实边界] 库内 MOEA/D 的 Tchebycheff 分解使用固定参考点 ref=2 且未做目标归一化，")
    print("  在本问题 5 个目标量纲悬殊（length~2800, safety~1）的场合会退化；")
    print("  若 MOEA/D 显著劣于 NSGA-II，这是实现缺陷而非分解式方法本身不行——")
    print("  '对比并发现实现缺陷'本身就是这个实验的价值之一。")

    results["algorithms"] = {
        algo: {"time_s": round(times[algo], 3),
               "n_pareto": int(fronts[algo].shape[0]),
               "hv_common_ref": hv[algo],
               "obj_best": [round(float(x), 4) for x in (fronts[algo].min(axis=0) if fronts[algo].shape[0] else [])]}
        for algo in ("nsga2", "moead")}
    results["reference_point"] = [round(float(x), 4) for x in ref] if ref is not None else None

    out_dir = os.path.join(_HERE, "evtol_v7_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "optimizer_compare_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()