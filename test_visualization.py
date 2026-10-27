#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试脚本 - 变异概率实验 + 增强可视化
=========================================

【修改说明】解决变异概率失效和早熟收敛问题

1. 变异概率调整实验 (增强版)
   - 测试不同的 eta_mutation 值 [10, 20, 30]
   - 新增自适应停滞检测和处理机制
   - 新增多样性指标监控

2. 增强可视化分析
   - 进化过程监控图
   - 帕累托前沿分析图
   - 解集分布热力图
   - 目标间权衡关系图

3. 主要修改点:
   - 种群规模从30提升至60
   - 初始路径从5条减少至1条最优路径
   - 新增自适应停滞处理
   - eta_mutation 可配置

使用方法: python test_visualization.py
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from datetime import datetime
import zhplot

# 导入自定义模块
from evtol_v6_plus import EVTOLV6Plus, FlightProfile
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment
from multi_objective_optimizer import NSGAIIOptimizer, Objective, ObjectiveType, Individual, ParetoFront
from enhanced_visualizer import generate_comprehensive_report


def setup_environment():
    """设置测试环境"""
    print("=" * 70)
    print("eVTOL NSGA-II 参数优化与可视化分析")
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


def generate_initial_paths(planner, start_pos, goal_pos, n_paths: int = 5) -> list:
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


def genes_to_path(genes: list, start_pos, goal_pos) -> list:
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


def run_mutation_experiment(mutation_rates, env, standard, start_vp, goal_vp, dynamic_env,
                           n_generations=30, population_size=60, eta_mutation_values=[20]):
    """
    【增强版】运行变异概率实验
    
    【修改说明】
    - 种群规模从30提升至60
    - 新增 eta_mutation 参数测试
    - 新增多样性指标收集
    - 新增自适应停滞检测输出
    """
    print("\n" + "=" * 70)
    print("任务一: 变异概率调整实验 (增强版)")
    print("=" * 70)
    print(f"  种群规模: {population_size} (增强版)")
    print(f"  eta_mutation 测试范围: {eta_mutation_values}")
    
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
    
    evaluator = planner.evaluator
    
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    results = {}
    
    for pm in mutation_rates:
        for eta_m in eta_mutation_values:
            config_name = f"pm{pm}_eta{eta_m}"
            print(f"\n{'='*50}")
            print(f"配置: 变异概率={pm}, eta_mutation={eta_m}")
            print(f"{'='*50}")
            
            # 生成初始路径 (增强版只使用1条最优路径)
            initial_paths = generate_initial_paths(planner, start_vp.get_TLOF_center(), goal_vp.get_TLOF_center())
            
            # 创建优化器 (增强版)
            optimizer = NSGAIIOptimizer(
                n_objectives=5,
                population_size=population_size,  # 【增强】从30提升至60
                n_generations=n_generations,
                crossover_prob=0.9,
                mutation_prob=pm,
                eta_crossover=20,
                eta_mutation=eta_m,  # 【新增】可配置
                random_seed=42,
                n_keypoints=10
            )
            
            # 【新增】启用自适应停滞处理
            optimizer.stagnation_threshold = 5
            optimizer.hv_improvement_min = 1e-3
            optimizer.adaptive_mutation_prob = 0.3
            optimizer.adaptive_reset_ratio = 0.2
            
            optimizer.set_objectives(objectives)
            optimizer.set_environment({
                'start': start_vp.get_TLOF_center(),
                'goal': goal_vp.get_TLOF_center(),
                'bounds': ((0, 2500), (0, 2500), (30, 300))
            })
            optimizer.set_initialization_strategy('diverse')  # 【增强】使用增强版初始化
            
            if initial_paths:
                optimizer.set_initial_paths(initial_paths[:1])  # 【修改】只使用1条最优路径
            
            def eval_func(genes):
                path = genes_to_path(genes, start_vp.get_TLOF_center(), goal_vp.get_TLOF_center())
                return evaluator.evaluate(path)
            
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
            
            # 【新增】计算多样性统计
            diversity_history = result.convergence_history.get('diversity', [0] * len(hv_history))
            mutation_count_history = result.convergence_history.get('mutation_count', [0] * len(hv_history))
            
            # 【新增】检测自适应触发次数
            adaptive_trigger_count = 0
            for i in range(1, len(hv_history)):
                if hv_history[i] == hv_history[i-1]:
                    adaptive_trigger_count += 1
            
            results[config_name] = {
                'mutation_prob': pm,
                'eta_mutation': eta_m,
                'n_pareto_solutions': len(result.pareto_front.individuals),
                'hypervolume': result.pareto_front.hypervolume,
                'spacing': result.pareto_front.spacing,
                'convergence_generation': convergence_gen,
                'final_hv': final_hv,
                'hv_history': hv_history,
                'spacing_history': result.convergence_history['spacing'],
                'pareto_size_history': result.convergence_history['pareto_size'],
                'diversity_history': diversity_history,
                'mutation_count_history': mutation_count_history,
                'optimization_time': opt_time,
                'n_evaluations': result.n_function_evaluations,
                'optimization_result': result,
                'adaptive_trigger_count': adaptive_trigger_count,
            }
            
            # 【新增】打印增强信息
            avg_diversity = np.mean(diversity_history) if diversity_history else 0
            avg_mutation_effect = np.mean(mutation_count_history) if mutation_count_history else 0
            print(f"\n  【增强指标】")
            print(f"    平均多样性: {avg_diversity:.4f}")
            print(f"    平均变异生效数: {avg_mutation_effect:.1f} / {population_size}")
            print(f"    自适应触发次数: {adaptive_trigger_count}")
            print(f"\n  结果: 帕累托解={len(result.pareto_front.individuals)}, HV={result.pareto_front.hypervolume:.2f}")
    
    return results


def analyze_and_print_results(results):
    """【增强版】分析并打印实验结果"""
    print("\n" + "=" * 80)
    print("变异概率对比结果 (增强版)")
    print("=" * 80)
    print(f"{'配置':^20} | {'帕累托解':^10} | {'超体积(HV)':^14} | {'间距':^10} | {'多样性':^10} | {'自适应':^8}")
    print("-" * 100)
    
    best_hv = max(r['hypervolume'] for r in results.values())
    
    # 按配置排序
    sorted_results = sorted(results.items(), key=lambda x: (x[1]['mutation_prob'], x[1]['eta_mutation']))
    
    for key, r in sorted_results:
        config_name = f"pm{r['mutation_prob']}_eta{r['eta_mutation']}"
        hv_str = f"{r['hypervolume']:.2f}"
        if r['hypervolume'] == best_hv:
            hv_str += " *"
        
        # 计算平均多样性
        avg_div = np.mean(r.get('diversity_history', [0])) if r.get('diversity_history') else 0
        
        print(f"{config_name:^20} | {r['n_pareto_solutions']:^10d} | {hv_str:^14} | "
              f"{r['spacing']:^10.4f} | {avg_div:^10.4f} | {r.get('adaptive_trigger_count', 0):^8d}")
    
    print("-" * 100)
    print("* 表示该配置下HV最优\n")
    
    # 【新增】新增指标评分
    print("=" * 80)
    print("【增强版】最佳配置推荐分析")
    print("=" * 80)
    
    scores = {}
    for key, r in results.items():
        hv_score = r['hypervolume'] / best_hv
        size_score = r['n_pareto_solutions'] / max(res['n_pareto_solutions'] for res in results.values())
        spacing_score = 1 / (1 + r['spacing'])
        conv_score = 1 / (1 + r['convergence_generation'])
        
        # 【新增】多样性评分
        diversity_history = r.get('diversity_history', [0])
        avg_div = np.mean(diversity_history) if diversity_history else 0
        diversity_score = avg_div
        
        # 【新增】自适应稳定性评分
        adaptive_triggers = r.get('adaptive_trigger_count', 0)
        stability_score = 1 / (1 + adaptive_triggers)
        
        # 综合评分: HV(30%) + 解数量(15%) + 间距(10%) + 多样性(20%) + 稳定性(15%) + 收敛(10%)
        total_score = (0.30 * hv_score + 0.15 * size_score + 0.10 * spacing_score + 
                      0.20 * diversity_score + 0.15 * stability_score + 0.10 * conv_score)
        scores[key] = total_score
    
    best_key = max(scores, key=scores.get)
    best_result = results[best_key]
    
    print(f"\n各指标权重: HV(30%), 解数量(15%), 间距(10%), 多样性(20%), 稳定性(15%), 收敛(10%)")
    print("\n综合评分:")
    for key, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <-- 推荐" if key == best_key else ""
        r = results[key]
        print(f"  {key}: {score:.4f} (HV={r['hypervolume']:.2f}, 多样性={np.mean(r.get('diversity_history', [0])):.4f}){marker}")
    
    print(f"\n【增强版推荐】最佳配置: {best_key}")
    print(f"  - 变异概率: {best_result['mutation_prob']}")
    print(f"  - eta_mutation: {best_result['eta_mutation']}")
    print(f"  - 超体积(HV): {best_result['hypervolume']:.2f}")
    print(f"  - 帕累托解数量: {best_result['n_pareto_solutions']}")
    print(f"  - 间距: {best_result['spacing']:.4f}")
    print(f"  - 平均多样性: {np.mean(best_result.get('diversity_history', [0])):.4f}")
    print(f"  - 自适应触发次数: {best_result.get('adaptive_trigger_count', 0)}")
    
    return best_key, best_result


def plot_mutation_comparison(results, output_dir):
    """【增强版】绘制变异概率对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Enhanced Mutation & Eta_mutation Impact on NSGA-II Performance', fontsize=14, fontweight='bold')
    
    # 颜色配置
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 按配置排序
    sorted_results = sorted(results.items(), key=lambda x: (x[1]['mutation_prob'], x[1]['eta_mutation']))
    
    # 图1: 超体积变化曲线
    ax = axes[0, 0]
    for i, (key, r) in enumerate(sorted_results):
        config_name = f"pm{r['mutation_prob']}_eta{r['eta_mutation']}"
        generations = range(1, len(r['hv_history']) + 1)
        ax.plot(generations, r['hv_history'], color=colors[i], 
                marker=markers[i % len(markers)], markersize=3, label=config_name, linewidth=1.5)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_ylabel('Hypervolume (HV)', fontsize=10)
    ax.set_title('Hypervolume Evolution', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 图2: 帕累托解数量
    ax = axes[0, 1]
    config_names = [f"pm{r['mutation_prob']}_eta{r['eta_mutation']}" for _, r in sorted_results]
    sizes = [r['n_pareto_solutions'] for _, r in sorted_results]
    bars = ax.bar(range(len(config_names)), sizes, color=colors, alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=10)
    ax.set_ylabel('Number of Pareto Solutions', fontsize=10)
    ax.set_title('Pareto Solution Count', fontsize=11)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
               str(size), ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3: 【新增】多样性变化曲线
    ax = axes[0, 2]
    for i, (key, r) in enumerate(sorted_results):
        config_name = f"pm{r['mutation_prob']}_eta{r['eta_mutation']}"
        diversity = r.get('diversity_history', [0])
        if diversity:
            generations = range(1, len(diversity) + 1)
            ax.plot(generations, diversity, color=colors[i], 
                    marker=markers[i % len(markers)], markersize=3, label=config_name, linewidth=1.5)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_ylabel('Genotypic Diversity', fontsize=10)
    ax.set_title('Population Diversity Evolution', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 图4: 归一化指标对比
    ax = axes[1, 0]
    x = np.arange(len(sorted_results))
    width = 0.35
    
    max_hv = max(r['hypervolume'] for r in results.values())
    hv_norm = [r['hypervolume'] / max_hv for _, r in sorted_results]
    diversity_norm = [np.mean(r.get('diversity_history', [0])) for _, r in sorted_results]
    max_div = max(diversity_norm) if max(diversity_norm) > 0 else 1
    diversity_norm = [d / max_div for d in diversity_norm]
    
    ax.bar(x - width/2, hv_norm, width, label='HV (norm)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, diversity_norm, width, label='Diversity (norm)', color='forestgreen', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=10)
    ax.set_ylabel('Normalized Score', fontsize=10)
    ax.set_title('HV vs Diversity Comparison', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图5: 【新增】变异生效统计
    ax = axes[1, 1]
    mutation_counts = [np.mean(r.get('mutation_count_history', [0])) for _, r in sorted_results]
    bars = ax.bar(range(len(config_names)), mutation_counts, color='coral', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=10)
    ax.set_ylabel('Avg Mutation Effects', fontsize=10)
    ax.set_title('Mutation Effectiveness (per generation)', fontsize=11)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
    for bar, count in zip(bars, mutation_counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{count:.1f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图6: 【新增】自适应触发统计
    ax = axes[1, 2]
    adaptive_counts = [r.get('adaptive_trigger_count', 0) for _, r in sorted_results]
    bars = ax.bar(range(len(config_names)), adaptive_counts, color='purple', alpha=0.7)
    ax.set_xlabel('Configuration', fontsize=10)
    ax.set_ylabel('Adaptive Triggers', fontsize=10)
    ax.set_title('Adaptive Stagnation Handling Count', fontsize=11)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8)
    for bar, count in zip(bars, adaptive_counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
               str(count), ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = f'{output_dir}/mutation_comparison.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n【增强版】变异概率对比图已保存: {output}")
    
    return output


def save_results(results, best_key, output_path):
    """【增强版】保存实验结果"""
    save_data = {}
    for key, r in results.items():
        save_data[key] = {
            'mutation_prob': float(r['mutation_prob']),
            'eta_mutation': float(r['eta_mutation']),
            'n_pareto_solutions': int(r['n_pareto_solutions']),
            'hypervolume': float(r['hypervolume']),
            'spacing': float(r['spacing']),
            'convergence_generation': int(r['convergence_generation']),
            'final_hv': float(r['final_hv']),
            'hv_history': [float(v) for v in r['hv_history']],
            'spacing_history': [float(v) for v in r['spacing_history']],
            'pareto_size_history': [int(v) for v in r['pareto_size_history']],
            'diversity_history': [float(v) for v in r.get('diversity_history', [])],
            'mutation_count_history': [int(v) for v in r.get('mutation_count_history', [])],
            'adaptive_trigger_count': int(r.get('adaptive_trigger_count', 0)),
            'optimization_time': float(r['optimization_time']),
            'n_evaluations': int(r['n_evaluations']),
        }
    
    save_data['best_config'] = best_key
    save_data['best_mutation_prob'] = float(results[best_key]['mutation_prob'])
    save_data['best_eta_mutation'] = float(results[best_key]['eta_mutation'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)


def main():
    """主函数 - 增强版"""
    import os
    
    # 设置环境
    env, standard, start_vp, goal_vp, dynamic_env, start_pos, goal_pos = setup_environment()
    
    output_dir = '/home/ayan1907/evtol/evtol_v6_plus_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 【修改】增强版参数配置
    # 变异概率列表
    mutation_rates = [0.1, 0.15, 0.2, 0.3]
    # eta_mutation 测试范围
    eta_mutation_values = [10, 20, 30]
    # 种群规模提升至60
    population_size = 60
    # 进化代数
    n_generations = 50
    
    # ========== 任务一: 变异概率实验 (增强版) ==========
    print("\n" + "=" * 70)
    print("开始变异概率调整实验 (增强版 - 解决早熟收敛问题)")
    print("=" * 70)
    print(f"  变异概率测试: {mutation_rates}")
    print(f"  eta_mutation 测试: {eta_mutation_values}")
    print(f"  种群规模: {population_size} (从30提升)")
    print(f"  进化代数: {n_generations}")
    
    results = run_mutation_experiment(
        mutation_rates,
        env, standard, start_vp, goal_vp, dynamic_env,
        n_generations=n_generations,
        population_size=population_size,
        eta_mutation_values=eta_mutation_values
    )
    
    # 分析结果
    best_key, best_result = analyze_and_print_results(results)
    
    # 保存结果
    save_results(results, best_key, f'{output_dir}/mutation_experiment_results.json')
    
    # 绘制变异概率对比图
    plot_mutation_comparison(results, output_dir)
    
    # ========== 任务二: 增强可视化 ==========
    print("\n" + "=" * 70)
    print("生成增强可视化报告 (任务二)")
    print("=" * 70)
    
    # 使用最佳配置的结果进行可视化
    optimization_result = best_result['optimization_result']
    
    # 生成完整可视化报告
    visualization_outputs = generate_comprehensive_report(
        optimization_result,
        output_dir=output_dir,
        bounds=((0, 2500), (0, 2500), (30, 300))
    )
    
    # ========== 最终汇总 ==========
    print("\n" + "=" * 70)
    print("实验完成 - 最终汇总 (增强版)")
    print("=" * 70)
    
    print(f"\n【最佳配置】")
    print(f"  变异概率: pm = {best_result['mutation_prob']}")
    print(f"  eta_mutation: {best_result['eta_mutation']}")
    
    print(f"\n【帕累托前沿质量】")
    print(f"  帕累托解数量: {len(optimization_result.pareto_front.individuals)}")
    print(f"  超体积(HV): {optimization_result.pareto_front.hypervolume:.4f}")
    print(f"  间距(Spacing): {optimization_result.pareto_front.spacing:.4f}")
    
    print(f"\n【增强指标】")
    avg_diversity = np.mean(best_result.get('diversity_history', [0]))
    avg_mutation = np.mean(best_result.get('mutation_count_history', [0]))
    print(f"  平均多样性: {avg_diversity:.4f}")
    print(f"  平均变异生效: {avg_mutation:.1f} / {population_size}")
    print(f"  自适应触发: {best_result.get('adaptive_trigger_count', 0)} 次")
    
    print(f"\n【生成的文件】")
    print(f"  1. mutation_experiment_results.json - 实验数据")
    print(f"  2. mutation_comparison.png - 【增强版】变异概率对比图")
    print(f"  3. evolution_monitoring.png - 进化过程监控图")
    print(f"  4. pareto_front_analysis.png - 帕累托前沿分析图")
    print(f"  5. solution_distribution_heatmap.png - 解集分布热力图")
    print(f"  6. tradeoff_curves.png - 目标间权衡关系图")
    
    print(f"\n所有结果保存在: {output_dir}")
    
    print("\n" + "=" * 70)
    print("【增强版】进一步调整建议")
    print("=" * 70)
    print(f"""
1. 基于增强版实验结果，推荐配置:
   - 变异概率 pm = {best_result['mutation_prob']}
   - eta_mutation = {best_result['eta_mutation']}
   - 种群规模 = {population_size}

2. 【解决早熟收敛问题】:
   - 自适应停滞处理已启用，连续5代无提升会自动调整
   - 初始路径从5条减少至1条，增强种群多样性
   - 多样性指标已实时监控

3. 如果多样性仍然较低:
   - 进一步降低初始路径依赖
   - 增加 _generate_highly_diverse_genes 中的扰动幅度
   - 考虑增加种群规模至80-100

4. 其他可调参数:
   - stagnation_threshold: 停滞检测阈值 (当前5)
   - adaptive_mutation_prob: 停滞时变异概率 (当前0.3)
   - adaptive_reset_ratio: 停滞时重置比例 (当前0.2)
""")


if __name__ == '__main__':
    main()
