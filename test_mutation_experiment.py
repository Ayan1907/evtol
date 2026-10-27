#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变异概率调整实验 - NSGA-II参数优化
=====================================

任务一: 测试不同变异概率对优化结果的影响

变异概率范围: 0.1, 0.15, 0.2, 0.3
评估指标:
1. 最终帕累托解数量
2. 最终超体积(HV)
3. 最终间距(Spacing)
4. 收敛速度(HV达到最终值90%所需的代数)
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

# 导入必要模块
from evtol_v6_plus import EVTOLV6Plus, FlightProfile
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment
from multi_objective_optimizer import NSGAIIOptimizer, Objective, ObjectiveType, Individual, ParetoFront


def setup_environment():
    """设置测试环境"""
    print("=" * 70)
    print("变异概率调整实验")
    print("=" * 70)
    
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    standard = EVTOLStandard()
    start_vp = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vp = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    dynamic_env = DynamicEnvironment(env)
    wind_field = WindField(
        env, base_wind_speed=8.0, base_wind_direction=135.0,
        turbulence_intensity=0.15, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    
    return env, standard, start_vp, goal_vp, dynamic_env, start_pos, goal_pos


def run_mutation_experiment(mutation_rates: List[float], 
                          env, standard, start_vp, goal_vp, dynamic_env,
                          n_generations: int = 30, 
                          population_size: int = 30,
                          n_keypoints: int = 10) -> Dict:
    """
    运行变异概率实验
    
    参数:
        mutation_rates: 要测试的变异概率列表
        其他: 环境参数
        
    返回:
        实验结果字典
    """
    print("\n" + "=" * 70)
    print("开始变异概率实验")
    print("=" * 70)
    print(f"  变异概率: {mutation_rates}")
    print(f"  进化代数: {n_generations}")
    print(f"  种群大小: {population_size}")
    print()
    
    # 创建飞行剖面
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    # 创建规划器
    planner = EVTOLV6Plus(
        env, standard, start_vp, goal_vp, dynamic_env,
        flight_profile=flight_profile,
        use_high_fidelity_energy=True
    )
    
    # 获取环境信息
    start_pos = start_vp.get_TLOF_center()
    goal_pos = goal_vp.get_TLOF_center()
    
    # 评估器
    evaluator = planner.evaluator
    
    # 定义优化目标
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    results = {}
    
    for pm in mutation_rates:
        print(f"\n{'='*50}")
        print(f"测试变异概率: {pm}")
        print(f"{'='*50}")
        
        # 生成初始路径
        initial_paths = _generate_initial_paths(planner, start_pos, goal_pos, n_paths=5)
        
        # 创建优化器
        optimizer = NSGAIIOptimizer(
            n_objectives=5,
            population_size=population_size,
            n_generations=n_generations,
            crossover_prob=0.9,
            mutation_prob=pm,  # 变异概率
            eta_crossover=20,
            eta_mutation=20,
            random_seed=42,
            n_keypoints=n_keypoints
        )
        optimizer.set_objectives(objectives)
        optimizer.set_environment({
            'start': start_pos,
            'goal': goal_pos,
            'bounds': ((0, 2500), (0, 2500), (30, 300))
        })
        optimizer.set_initialization_strategy('diverse')
        
        if initial_paths:
            optimizer.set_initial_paths(initial_paths)
        
        # 评估函数
        def eval_func(genes):
            path = _genes_to_path(genes, start_pos, goal_pos)
            return evaluator.evaluate(path)
        
        # 运行优化
        start_time = time.time()
        result = optimizer.optimize(eval_func)
        opt_time = time.time() - start_time
        
        # 计算收敛速度
        hv_history = result.convergence_history['hypervolume']
        final_hv = hv_history[-1] if hv_history else 0
        target_hv = final_hv * 0.9
        
        convergence_gen = 0
        for i, hv in enumerate(hv_history):
            if hv >= target_hv:
                convergence_gen = i + 1
                break
        
        # 记录结果
        results[f"pm_{pm}"] = {
            'mutation_prob': pm,
            'n_pareto_solutions': len(result.pareto_front.individuals),
            'hypervolume': result.pareto_front.hypervolume,
            'spacing': result.pareto_front.spacing,
            'convergence_generation': convergence_gen,
            'final_hv': final_hv,
            'hv_history': hv_history,
            'spacing_history': result.convergence_history['spacing'],
            'pareto_size_history': result.convergence_history['pareto_size'],
            'optimization_time': opt_time,
            'n_evaluations': result.n_function_evaluations,
        }
        
        print(f"\n  结果汇总:")
        print(f"    帕累托解数量: {len(result.pareto_front.individuals)}")
        print(f"    超体积(HV): {result.pareto_front.hypervolume:.2f}")
        print(f"    间距(Spacing): {result.pareto_front.spacing:.4f}")
        print(f"    收敛代数(90%HV): {convergence_gen}")
        print(f"    优化时间: {opt_time:.2f}s")
    
    return results


def _generate_initial_paths(planner, start_pos, goal_pos, n_paths: int = 5) -> List[List]:
    """生成多条不同偏好的初始路径"""
    initial_paths = []
    base_planner = planner._base_planner
    
    # 保存原始设置
    original_weights = base_planner.weights.copy()
    original_verbose = getattr(base_planner, 'verbose', True)
    base_planner.verbose = False
    
    try:
        # 策略1: 平衡策略
        _set_weights(base_planner, omega1=0.7, omega2=0.3, alpha1=0.25, alpha2=0.2, alpha3=0.2, alpha4=0.2, alpha5=0.15)
        path = base_planner.find_path()
        if path and len(path) >= 2:
            coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else (p[0], p[1], p[2]) for p in path]
            initial_paths.append(coords)
        
        # 策略2: 时间优先
        _set_weights(base_planner, omega1=0.9, omega2=0.1, alpha1=0.1, alpha2=0.4, alpha3=0.2, alpha4=0.2, alpha5=0.1)
        path = base_planner.find_path()
        if path and len(path) >= 2:
            coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else (p[0], p[1], p[2]) for p in path]
            initial_paths.append(coords)
        
        # 策略3: 能耗优先
        _set_weights(base_planner, omega1=0.5, omega2=0.5, alpha1=0.5, alpha2=0.1, alpha3=0.2, alpha4=0.1, alpha5=0.1)
        path = base_planner.find_path()
        if path and len(path) >= 2:
            coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else (p[0], p[1], p[2]) for p in path]
            initial_paths.append(coords)
        
        # 策略4: 安全优先
        _set_weights(base_planner, omega1=0.6, omega2=0.4, alpha1=0.2, alpha2=0.2, alpha3=0.1, alpha4=0.4, alpha5=0.1)
        path = base_planner.find_path()
        if path and len(path) >= 2:
            coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else (p[0], p[1], p[2]) for p in path]
            initial_paths.append(coords)
        
        # 策略5: 噪声优先
        _set_weights(base_planner, omega1=0.5, omega2=0.5, alpha1=0.2, alpha2=0.2, alpha3=0.4, alpha4=0.1, alpha5=0.1)
        path = base_planner.find_path()
        if path and len(path) >= 2:
            coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else (p[0], p[1], p[2]) for p in path]
            initial_paths.append(coords)
            
    finally:
        base_planner.weights = original_weights
        base_planner.verbose = original_verbose
    
    return initial_paths[:n_paths]


def _set_weights(planner, **weights):
    """设置规划器权重"""
    for key, value in weights.items():
        if key in planner.weights:
            planner.weights[key] = value


def _genes_to_path(genes: List, start_pos, goal_pos) -> List:
    """将基因解码为完整路径"""
    path = [start_pos]
    for gene in genes:
        if hasattr(gene, 'x'):
            path.append((gene.x, gene.y, gene.z))
        elif isinstance(gene, (list, tuple)) and len(gene) >= 3:
            path.append((gene[0], gene[1], gene[2]))
        else:
            path.append(gene)
    path.append(goal_pos)
    return path


def print_comparison_table(results: Dict):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("变异概率对比结果")
    print("=" * 80)
    print(f"{'变异概率':^12} | {'帕累托解':^10} | {'超体积(HV)':^14} | {'间距(Spacing)':^14} | {'收敛代数':^10} | {'优化时间':^10}")
    print("-" * 80)
    
    # 找出最优HV
    best_hv = max(r['hypervolume'] for r in results.values())
    
    for key, r in sorted(results.items(), key=lambda x: x[1]['mutation_prob']):
        hv_str = f"{r['hypervolume']:.2f}"
        if r['hypervolume'] == best_hv:
            hv_str += " *"
        print(f"{r['mutation_prob']:^12.2f} | {r['n_pareto_solutions']:^10d} | {hv_str:^14} | {r['spacing']:^14.4f} | {r['convergence_generation']:^10d} | {r['optimization_time']:^10.2f}s")
    
    print("-" * 80)
    print("* 表示该变异概率下HV最优")


def recommend_best_mutation(results: Dict) -> Tuple[float, str]:
    """推荐最佳变异概率"""
    print("\n" + "=" * 80)
    print("最佳变异概率推荐分析")
    print("=" * 80)
    
    # 计算综合评分
    scores = {}
    
    for key, r in results.items():
        # 归一化各指标
        hv_score = r['hypervolume'] / max(res['hypervolume'] for res in results.values())
        size_score = r['n_pareto_solutions'] / max(res['n_pareto_solutions'] for res in results.values())
        spacing_score = 1 / (1 + r['spacing'])  # spacing越小越好
        conv_score = 1 / (1 + r['convergence_generation'])  # 收敛越快越好
        
        # 综合评分 (加权)
        total_score = 0.4 * hv_score + 0.25 * size_score + 0.2 * spacing_score + 0.15 * conv_score
        scores[r['mutation_prob']] = total_score
    
    # 找出最佳
    best_pm = max(scores, key=scores.get)
    
    print(f"\n各指标权重: HV(40%), 帕累托解数量(25%), 间距(20%), 收敛速度(15%)")
    print("\n综合评分:")
    for pm, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <-- 推荐" if pm == best_pm else ""
        print(f"  pm={pm:.2f}: {score:.4f}{marker}")
    
    # 分析推荐理由
    best_result = results[f"pm_{best_pm}"]
    print(f"\n推荐理由:")
    print(f"  - 超体积(HV): {best_result['hypervolume']:.2f}")
    print(f"  - 帕累托解数量: {best_result['n_pareto_solutions']}")
    print(f"  - 收敛代数: {best_result['convergence_generation']}")
    
    return best_pm, f"综合评分最高({scores[best_pm]:.4f})"


def save_experiment_results(results: Dict, output_path: str):
    """保存实验结果"""
    # 转换numpy数组为列表
    save_data = {}
    for key, r in results.items():
        save_data[key] = {
            'mutation_prob': float(r['mutation_prob']),
            'n_pareto_solutions': int(r['n_pareto_solutions']),
            'hypervolume': float(r['hypervolume']),
            'spacing': float(r['spacing']),
            'convergence_generation': int(r['convergence_generation']),
            'final_hv': float(r['final_hv']),
            'hv_history': [float(v) for v in r['hv_history']],
            'spacing_history': [float(v) for v in r['spacing_history']],
            'pareto_size_history': [int(v) for v in r['pareto_size_history']],
            'optimization_time': float(r['optimization_time']),
            'n_evaluations': int(r['n_evaluations']),
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)


def plot_mutation_comparison(results: Dict, output_dir: str):
    """绘制变异概率对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('变异概率对NSGA-II性能的影响', fontsize=14, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    # 图1: 超体积变化曲线
    ax = axes[0, 0]
    for i, (key, r) in enumerate(sorted(results.items(), key=lambda x: x[1]['mutation_prob'])):
        generations = range(1, len(r['hv_history']) + 1)
        ax.plot(generations, r['hv_history'], color=colors[i % len(colors)], 
                marker=markers[i % len(markers)], markersize=4, label=f'pm={r["mutation_prob"]:.2f}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Hypervolume (HV)')
    ax.set_title('超体积随代数变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图2: 帕累托解数量柱状图
    ax = axes[0, 1]
    pms = [r['mutation_prob'] for r in sorted(results.values(), key=lambda x: x['mutation_prob'])]
    sizes = [r['n_pareto_solutions'] for r in sorted(results.values(), key=lambda x: x['mutation_prob'])]
    bars = ax.bar([str(pm) for pm in pms], sizes, color=colors[:len(pms)], alpha=0.8)
    ax.set_xlabel('Mutation Probability')
    ax.set_ylabel('Number of Pareto Solutions')
    ax.set_title('帕累托解数量对比')
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
               str(size), ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3: 最终指标对比
    ax = axes[1, 0]
    x = np.arange(len(pms))
    width = 0.35
    
    # 归一化HV
    max_hv = max(r['hypervolume'] for r in results.values())
    hv_norm = [r['hypervolume'] / max_hv for r in sorted(results.values(), key=lambda x: x['mutation_prob'])]
    spacing_norm = [1 / (1 + r['spacing']) for r in sorted(results.values(), key=lambda x: x['mutation_prob'])]  # 反转spacing
    
    ax.bar(x - width/2, hv_norm, width, label='HV (normalized)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, spacing_norm, width, label='1/(1+Spacing)', color='coral', alpha=0.8)
    ax.set_xlabel('Mutation Probability')
    ax.set_ylabel('Normalized Score')
    ax.set_title('归一化指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels([str(pm) for pm in pms])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图4: 收敛速度
    ax = axes[1, 1]
    conv_gens = [r['convergence_generation'] for r in sorted(results.values(), key=lambda x: x['mutation_prob'])]
    opt_times = [r['optimization_time'] for r in sorted(results.values(), key=lambda x: x['mutation_prob'])]
    
    ax2 = ax.twinx()
    bars = ax.bar([str(pm) for pm in pms], conv_gens, color='lightgreen', alpha=0.7, label='收敛代数')
    ax2.plot([str(pm) for pm in pms], opt_times, 'ro-', markersize=8, linewidth=2, label='优化时间')
    
    ax.set_xlabel('Mutation Probability')
    ax.set_ylabel('Convergence Generation', color='green')
    ax2.set_ylabel('Optimization Time (s)', color='red')
    ax.set_title('收敛速度对比')
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = f'{output_dir}/mutation_comparison.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n变异概率对比图已保存: {output}")
    
    return output


if __name__ == '__main__':
    # 设置环境
    env, standard, start_vp, goal_vp, dynamic_env, start_pos, goal_pos = setup_environment()
    
    # 变异概率列表
    mutation_rates = [0.1, 0.15, 0.2, 0.3]
    
    # 运行实验
    results = run_mutation_experiment(
        mutation_rates,
        env, standard, start_vp, goal_vp, dynamic_env,
        n_generations=30,
        population_size=30
    )
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 推荐最佳变异概率
    best_pm, reason = recommend_best_mutation(results)
    
    # 保存结果
    output_dir = '/home/ayan1907/evtol/evtol_v6_plus_results'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    save_experiment_results(results, f'{output_dir}/mutation_experiment_results.json')
    
    # 绘制对比图
    plot_mutation_comparison(results, output_dir)
    
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
