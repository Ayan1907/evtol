#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II早熟收敛诊断脚本
=========================

诊断内容:
1. 可视化诊断 - 帕累托解分布
2. 约束分析 - 测试约束宽松化效果
3. 参数敏感性扫描 - 变异率对比

使用方法: python test_nsga2_diagnosis.py
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from datetime import datetime

# 导入多目标优化器
from multi_objective_optimizer import (
    NSGAIIOptimizer, MOEADOptimizer,
    Objective, ObjectiveType, ParetoDecisionMaker
)

# 导入评估器
sys.path.insert(0, '/home/ayan1907/evtol')
from evtol_v6_plus import EVTOLV6Plus, FlightProfile
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment


# ============================================================================
# 简化的路径评估函数
# ============================================================================

def decode_and_evaluate_genes(genes, start_pos, goal_pos, env, strict_mode=True):
    """
    将基因(关键点)解码为完整路径并评估
    
    参数:
        genes: 关键点列表 [(x,y,z), ...]
        start_pos: 起点
        goal_pos: 终点
        env: 环境对象
        strict_mode: 是否使用严格约束
    
    返回:
        objectives: [energy, time, noise, safety, length]
    """
    # 解码基因为完整路径
    path = [start_pos]
    for gene in genes:
        path.append(gene)
    path.append(goal_pos)
    
    # 计算路径长度
    total_length = 0
    min_height = float('inf')
    max_height = 0
    height_sum = 0
    
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        dz = path[i+1][2] - path[i][2]
        total_length += math.sqrt(dx**2 + dy**2 + dz**2)
        min_height = min(min_height, path[i+1][2])
        max_height = max(max_height, path[i+1][2])
        height_sum += path[i+1][2]
    
    avg_height = height_sum / max(1, len(path) - 1)
    
    # 计算各项指标
    # 1. 时间 (与长度成正比)
    flight_time = total_length / 50.0  # 假设平均速度50m/s
    
    # 2. 简化的能耗模型 (与长度和高度变化相关)
    vertical_work = abs(max_height - min_height) * 9.8 * 2000 / 1000  # kJ
    horizontal_work = total_length * 0.5  # kJ
    energy = (vertical_work + horizontal_work) / 3600  # kWh
    
    # 3. 噪声 (与高度和距离相关)
    noise = 100 / max(avg_height, 50) * (1 + total_length / 5000)
    
    # 4. 安全评分 (基于高度和障碍物碰撞风险)
    # 检查是否飞越建筑物
    collision_risk = 0.0
    for i in range(len(path) - 1):
        px, py = (path[i][0] + path[i+1][0]) / 2, (path[i][1] + path[i+1][1]) / 2
        for b in env.buildings:
            if (b['x'] <= px <= b['x'] + b['length'] and 
                b['y'] <= py <= b['y'] + b['width']):
                collision_risk += 0.1
    
    # 高度安全 (太低或太高都不好)
    height_risk = 0.0
    if min_height < 50:
        height_risk += (50 - min_height) / 50
    if max_height > 250:
        height_risk += (max_height - 250) / 50
    
    safety = collision_risk * 0.6 + height_risk * 0.4 + 0.01
    
    # 5. 路径长度
    length = total_length
    
    # 约束处理
    if strict_mode:
        # 硬约束违反惩罚
        violation_penalty = 0
        if min_height < 30:
            violation_penalty += (30 - min_height) * 0.5
        if collision_risk > 0:
            violation_penalty += collision_risk * 100
        
        if violation_penalty > 0:
            energy += violation_penalty
            safety += violation_penalty * 0.1
    
    return np.array([energy, flight_time, noise, safety, length])


# ============================================================================
# 诊断1: 可视化诊断
# ============================================================================

def diagnostic_visualization():
    """诊断1: 帕累托解分布可视化"""
    print("\n" + "="*70)
    print("诊断1: 可视化诊断")
    print("="*70)
    
    # 加载环境
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
    
    # 创建评估函数 (接收genes作为输入)
    def eval_func(genes):
        return decode_and_evaluate_genes(genes, start_pos, goal_pos, env, strict_mode=True)
    
    # 定义目标
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    # 运行NSGA-II优化 (记录每代历史)
    print("\n运行NSGA-II优化 (30代)...")
    optimizer = NSGAIIOptimizer(
        n_objectives=5,
        population_size=30,
        n_generations=30,
        crossover_prob=0.9,
        mutation_prob=0.1,
        random_seed=42
    )
    optimizer.set_objectives(objectives)
    optimizer.set_environment({
        'start': start_pos,
        'goal': goal_pos,
        'bounds': ((0, 2500), (0, 2500), (30, 300))
    })
    
    # 收集每代数据
    generation_data = []
    
    def callback(gen, pareto_front, history):
        if pareto_front.individuals:
            objs = np.array([ind.objectives for ind in pareto_front.individuals])
            generation_data.append({
                'gen': gen,
                'energy': objs[:, 0].tolist(),
                'time': objs[:, 1].tolist(),
                'noise': objs[:, 2].tolist(),
                'safety': objs[:, 3].tolist(),
                'length': objs[:, 4].tolist(),
                'hv': pareto_front.hypervolume,
                'size': len(pareto_front.individuals)
            })
    
    result = optimizer.optimize(eval_func, callback=callback)
    
    # ===== 图A: 帕累托解分布 =====
    print("\n生成图A: 帕累托解分布...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 最终帕累托解
    final_objs = np.array([ind.objectives for ind in result.pareto_front.individuals])
    
    # 图A1: 能耗 vs 时间
    ax = axes[0, 0]
    ax.scatter(final_objs[:, 0], final_objs[:, 1], c='steelblue', s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('能耗 (kWh)', fontsize=11)
    ax.set_ylabel('飞行时间 (s)', fontsize=11)
    ax.set_title('图A1: 能耗 vs 飞行时间', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注序号
    for i, (e, t) in enumerate(zip(final_objs[:, 0], final_objs[:, 1])):
        ax.annotate(f'{i+1}', (e, t), textcoords="offset points", xytext=(5,5), fontsize=9)
    
    # 图A2: 安全评分 vs 航程
    ax = axes[0, 1]
    ax.scatter(final_objs[:, 3], final_objs[:, 4], c='coral', s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('安全评分', fontsize=11)
    ax.set_ylabel('航程 (m)', fontsize=11)
    ax.set_title('图A2: 安全评分 vs 航程', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 图A3: 所有目标的相关性热力图
    ax = axes[1, 0]
    if len(final_objs) > 1:
        corr_matrix = np.corrcoef(final_objs.T)
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['能耗', '时间', '噪声', '安全', '长度'], fontsize=9)
        ax.set_yticklabels(['能耗', '时间', '噪声', '安全', '长度'], fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title('图A3: 目标相关性矩阵', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, '帕累托解不足\n无法计算相关性', ha='center', va='center', fontsize=14)
        ax.set_title('图A3: 目标相关性矩阵', fontsize=12, fontweight='bold')
    
    # 图A4: 帕累托解数量演变
    ax = axes[1, 1]
    gens = [d['gen'] for d in generation_data]
    sizes = [d['size'] for d in generation_data]
    ax.plot(gens, sizes, 'b-o', linewidth=2, markersize=5)
    ax.set_xlabel('代数', fontsize=11)
    ax.set_ylabel('帕累托解数量', fontsize=11)
    ax.set_title('图A4: 帕累托解数量演变', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(sizes) * 1.2 if sizes else 10)
    
    plt.suptitle('NSGA-II早熟收敛诊断 - 图A: 帕累托解分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output = '/home/ayan1907/evtol/evtol_v6_plus_results/nsga2_diagnosis_A.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output}")
    
    # ===== 图B: 进化动画(静态展示) =====
    print("\n生成图B: 进化过程快照...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    snapshot_gens = [0, 4, 9, 14, 19, 24, 29]  # 选择7个时间点
    for idx, gen in enumerate(snapshot_gens[:7]):
        ax = axes[idx]
        if gen < len(generation_data):
            data = generation_data[gen]
            ax.scatter(data['energy'], data['time'], c='steelblue', s=60, alpha=0.7)
            ax.set_xlabel('能耗', fontsize=9)
            ax.set_ylabel('时间', fontsize=9)
            ax.set_title(f'第{gen+1}代 (n={data["size"]})', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title(f'第{gen+1}代', fontsize=10)
    
    plt.suptitle('图B: 进化动画(静态) - 能耗 vs 时间平面', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output = '/home/ayan1907/evtol/evtol_v6_plus_results/nsga2_diagnosis_B.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output}")
    
    # 保存诊断数据
    diagnosis_data = {
        'generation_data': generation_data,
        'final_pareto': {
            'objectives': final_objs.tolist() if len(final_objs) > 0 else [],
            'hv': float(result.pareto_front.hypervolume),
            'spacing': float(result.pareto_front.spacing),
            'size': len(result.pareto_front.individuals)
        }
    }
    
    return diagnosis_data


# ============================================================================
# 诊断2: 约束分析
# ============================================================================

def diagnostic_constraint_analysis():
    """诊断2: 约束宽松化效果测试"""
    print("\n" + "="*70)
    print("诊断2: 约束分析 - 测试约束宽松化效果")
    print("="*70)
    
    # 加载环境
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
    
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    planner = EVTOLV6Plus(
        env, standard, start_vp, goal_vp, dynamic_env,
        flight_profile=flight_profile,
        use_high_fidelity_energy=True
    )
    
    # 创建评估函数
    def eval_func_strict(genes):
        return decode_and_evaluate_genes(genes, start_pos, goal_pos, env, strict_mode=True)
    
    def eval_func_relaxed(genes):
        return decode_and_evaluate_genes(genes, start_pos, goal_pos, env, strict_mode=False)
    
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    results = {}
    
    # 测试严格vs宽松模式
    for mode, label, eval_fn in [('strict', '严格约束', eval_func_strict), 
                                   ('relaxed', '宽松约束(软约束)', eval_func_relaxed)]:
        print(f"\n  运行{label}优化 (15代)...")
        
        optimizer = NSGAIIOptimizer(
            n_objectives=5,
            population_size=30,
            n_generations=15,
            crossover_prob=0.9,
            mutation_prob=0.1,
            random_seed=42
        )
        optimizer.set_objectives(objectives)
        optimizer.set_environment({
            'start': start_pos,
            'goal': goal_pos,
            'bounds': ((0, 2500), (0, 2500), (30, 300))
        })
        
        result = optimizer.optimize(eval_fn)
        
        results[mode] = {
            'size': len(result.pareto_front.individuals),
            'hv': float(result.pareto_front.hypervolume),
            'spacing': float(result.pareto_front.spacing),
            'objectives': [ind.objectives for ind in result.pareto_front.individuals]
        }
        
        print(f"    帕累托解数量: {results[mode]['size']}")
        print(f"    超体积(HV): {results[mode]['hv']:.2f}")
        print(f"    间距(Spacing): {results[mode]['spacing']:.2f}")
    
    # 可视化对比
    print("\n生成约束分析图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 帕累托解分布对比
    ax = axes[0]
    strict_objs = np.array(results['strict']['objectives'])
    relaxed_objs = np.array(results['relaxed']['objectives'])
    
    if len(strict_objs) > 0:
        ax.scatter(strict_objs[:, 0], strict_objs[:, 1], c='blue', s=100, alpha=0.7, 
                   label=f"严格约束 (n={len(strict_objs)})", edgecolors='black')
    if len(relaxed_objs) > 0:
        ax.scatter(relaxed_objs[:, 0], relaxed_objs[:, 1], c='red', s=100, alpha=0.5, 
                   marker='x', label=f"宽松约束 (n={len(relaxed_objs)})")
    
    ax.set_xlabel('能耗 (kWh)', fontsize=11)
    ax.set_ylabel('飞行时间 (s)', fontsize=11)
    ax.set_title('约束宽松化效果: 能耗 vs 时间', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右图: 指标对比柱状图
    ax = axes[1]
    modes = ['严格约束', '宽松约束']
    sizes = [results['strict']['size'], results['relaxed']['size']]
    colors = ['blue', 'red']
    
    bars = ax.bar(modes, sizes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('帕累托解数量', fontsize=11)
    ax.set_title('约束宽松化效果: 解数量对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('诊断2: 约束分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output = '/home/ayan1907/evtol/evtol_v6_plus_results/nsga2_diagnosis_constraint.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output}")
    
    return results


# ============================================================================
# 诊断3: 参数敏感性扫描
# ============================================================================

def diagnostic_parameter_sensitivity():
    """诊断3: 参数敏感性扫描"""
    print("\n" + "="*70)
    print("诊断3: 参数敏感性扫描 - 变异率对比")
    print("="*70)
    
    # 加载环境
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
    
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    # 创建评估函数
    def eval_func(genes):
        return decode_and_evaluate_genes(genes, start_pos, goal_pos, env, strict_mode=True)
    
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    mutation_rates = [0.1, 0.3, 0.5]
    results = {}
    
    # 变异率测试
    print("\n测试不同变异率 (5代)...")
    for pm in mutation_rates:
        print(f"\n  变异率: {pm}")
        
        optimizer = NSGAIIOptimizer(
            n_objectives=5,
            population_size=30,
            n_generations=5,
            crossover_prob=0.9,
            mutation_prob=pm,
            random_seed=42
        )
        optimizer.set_objectives(objectives)
        optimizer.set_environment({
            'start': start_pos,
            'goal': goal_pos,
            'bounds': ((0, 2500), (0, 2500), (30, 300))
        })
        
        result = optimizer.optimize(eval_func)
        
        # 收集第1代的初始解数量
        initial_pareto_size = len(result.history[0].individuals) if result.history else 0
        
        results[pm] = {
            'initial_pareto': initial_pareto_size,
            'final_pareto': len(result.pareto_front.individuals),
            'hv': float(result.pareto_front.hypervolume),
            'spacing': float(result.pareto_front.spacing),
            'objectives': [ind.objectives for ind in result.pareto_front.individuals]
        }
        
        print(f"    初始帕累托解数量: {results[pm]['initial_pareto']}")
        print(f"    最终帕累托解数量: {results[pm]['final_pareto']}")
    
    # 算法对比
    print("\n对比NSGA-II vs MOEA/D (5代)...")
    print("\n  MOEA/D...")
    moead_optimizer = MOEADOptimizer(
        n_objectives=5,
        population_size=30,
        n_generations=5,
        random_seed=42
    )
    moead_optimizer.set_objectives(objectives)
    moead_optimizer.set_environment({
        'start': start_pos,
        'goal': goal_pos,
        'bounds': ((0, 2500), (0, 2500), (30, 300))
    })
    
    moead_result = moead_optimizer.optimize(eval_func)
    results['MOEAD'] = {
        'initial_pareto': len(moead_result.history[0].individuals) if moead_result.history else 0,
        'final_pareto': len(moead_result.pareto_front.individuals),
        'hv': float(moead_result.pareto_front.hypervolume),
        'objectives': [ind.objectives for ind in moead_result.pareto_front.individuals]
    }
    print(f"    初始帕累托解数量: {results['MOEAD']['initial_pareto']}")
    print(f"    最终帕累托解数量: {results['MOEAD']['final_pareto']}")
    
    # 可视化
    print("\n生成参数敏感性图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 变异率vs初始解数量
    ax = axes[0]
    pm_labels = [f'pm={pm}' for pm in mutation_rates]
    initial_sizes = [results[pm]['initial_pareto'] for pm in mutation_rates]
    final_sizes = [results[pm]['final_pareto'] for pm in mutation_rates]
    
    x = np.arange(len(mutation_rates))
    width = 0.35
    ax.bar(x - width/2, initial_sizes, width, label='初始帕累托解', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, final_sizes, width, label='最终帕累托解', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pm_labels)
    ax.set_ylabel('帕累托解数量')
    ax.set_title('变异率敏感性: 解数量', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 右图: 算法对比
    ax = axes[1]
    alg_labels = ['NSGA-II\n(pm=0.1)', 'MOEA/D']
    alg_final_sizes = [results[0.1]['final_pareto'], results['MOEAD']['final_pareto']]
    colors = ['steelblue', 'green']
    bars = ax.bar(alg_labels, alg_final_sizes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('最终帕累托解数量')
    ax.set_title('算法对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, alg_final_sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('诊断3: 参数敏感性扫描', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output = '/home/ayan1907/evtol/evtol_v6_plus_results/nsga2_diagnosis_params.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output}")
    
    return results


# ============================================================================
# 主诊断报告
# ============================================================================

def generate_diagnosis_report(d1_data, d2_data, d3_data):
    """生成诊断报告"""
    print("\n" + "="*70)
    print("="*70)
    print("NSGA-II早熟收敛诊断报告")
    print("="*70)
    
    print("\n" + "-"*70)
    print("【关键发现】")
    print("-"*70)
    
    # 发现1: 可视化诊断
    final_pareto = d1_data['final_pareto']
    print(f"\n1. 帕累托解分布:")
    print(f"   - 最终帕累托解数量: {final_pareto['size']}")
    print(f"   - 超体积(HV): {final_pareto['hv']:.2f}")
    print(f"   - 间距(Spacing): {final_pareto['spacing']:.2f}")
    
    if final_pareto['spacing'] > 100000:
        print(f"   ⚠️ 间距过大({final_pareto['spacing']:.0f})，解分布极不均匀!")
    
    # 发现2: 约束分析
    print(f"\n2. 约束影响:")
    strict_size = d2_data['strict']['size']
    relaxed_size = d2_data['relaxed']['size']
    print(f"   - 严格约束: {strict_size}个解")
    print(f"   - 宽松约束: {relaxed_size}个解")
    
    if strict_size < relaxed_size:
        improvement = (relaxed_size - strict_size) / max(strict_size, 1) * 100
        print(f"   ⚠️ 约束过严导致解空间受限，宽松化可提升{improvement:.0f}%!")
    
    # 发现3: 参数敏感性
    print(f"\n3. 变异率敏感性:")
    for pm in [0.1, 0.3, 0.5]:
        print(f"   - pm={pm}: 初始{results_d3[pm]['initial_pareto']}个 → 最终{results_d3[pm]['final_pareto']}个")
    
    print("\n" + "-"*70)
    print("【具体建议】")
    print("-"*70)
    
    print("\n🔧 最高优先级修改:")
    
    # 分析最可能的原因
    causes = []
    
    if final_pareto['spacing'] > 100000:
        causes.append("解分布极不均匀，可能是目标空间存在强相关性或约束形成狭长可行域")
    
    if strict_size < relaxed_size and (relaxed_size - strict_size) / max(strict_size, 1) > 0.5:
        causes.append("约束过严，建议将部分硬约束改为带惩罚的软约束")
    
    if d3_data[0.1]['initial_pareto'] < 5:
        causes.append("初始种群多样性不足，建议增大种群规模或使用更好的初始化策略")
    
    if len(causes) == 0:
        causes.append("需要进一步分析，可能是目标函数设计问题")
    
    for i, cause in enumerate(causes, 1):
        print(f"  {i}. {cause}")
    
    print("\n📋 建议的修改方案:")
    print("  1. 将部分硬约束改为软约束（加惩罚项），增加可行解数量")
    print("  2. 增大种群规模到50-100，或增大变异率到0.3-0.5")
    print("  3. 使用更好的初始种群策略（如先运行A*生成多样化初始解）")
    
    print("\n" + "="*70)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("NSGA-II 早熟收敛诊断")
    print("="*70)
    print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    import os
    os.makedirs('/home/ayan1907/evtol/evtol_v6_plus_results', exist_ok=True)
    
    # 执行诊断
    d1_data = diagnostic_visualization()
    d2_data = diagnostic_constraint_analysis()
    d3_data = diagnostic_parameter_sensitivity()
    
    # 保存数据
    diagnosis_results = {
        'visualization': d1_data,
        'constraint_analysis': d2_data,
        'parameter_sensitivity': d3_data
    }
    
    with open('/home/ayan1907/evtol/evtol_v6_plus_results/nsga2_diagnosis_results.json', 'w', encoding='utf-8') as f:
        json.dump(diagnosis_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 生成报告
    results_d3 = d3_data  # 用于报告生成
    generate_diagnosis_report(d1_data, d2_data, d3_data)
    
    print("\n诊断完成! 输出文件:")
    print("  - nsga2_diagnosis_A.png (帕累托解分布)")
    print("  - nsga2_diagnosis_B.png (进化过程)")
    print("  - nsga2_diagnosis_constraint.png (约束分析)")
    print("  - nsga2_diagnosis_params.png (参数敏感性)")
    print("  - nsga2_diagnosis_results.json (详细数据)")
