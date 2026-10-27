#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6版本核心功能对比测试
展示多目标优化与帕累托最优概念

使用方法: python test_v6_comparison.py
"""

import sys
import time
import json
import random
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zhplot

# 导入多目标优化模块
from multi_objective_optimizer import (
    NSGAIIOptimizer,
    MOEADOptimizer,
    ParetoDecisionMaker,
    Objective,
    ObjectiveType,
)


def evaluate_random_path(genes):
    """评估随机生成的航迹（模拟）"""
    if not genes or len(genes) < 3:
        return [1e10] * 5
    
    # 模拟能耗模型（基于路径长度和高度变化）
    energy = len(genes) * 0.15 + random.uniform(0, 0.5)
    
    # 模拟飞行时间
    time_f = len(genes) * 2.5 + random.uniform(0, 5)
    
    # 模拟噪声（与高度相关）
    avg_z = np.mean([g[2] for g in genes if isinstance(g, (list, tuple)) and len(g) >= 3])
    noise = max(0.1, 1.0 - avg_z / 300) + random.uniform(-0.1, 0.1)
    
    # 模拟碰撞风险
    collision = random.uniform(0.01, 0.15)
    
    # 计算路径长度
    length = 0
    for i in range(len(genes) - 1):
        if isinstance(genes[i], (list, tuple)) and isinstance(genes[i+1], (list, tuple)):
            length += np.sqrt(sum((a-b)**2 for a, b in zip(genes[i], genes[i+1])))
    
    return [energy, time_f, noise, collision, max(length, 100)]


def evaluate_baseline():
    """评估"传统单目标优化"（基线）"""
    print("\n" + "="*60)
    print("评估1: 传统单目标优化 (基线)")
    print("="*60)
    print("  模拟传统A*算法：仅优化路径长度")
    
    n_runs = 3
    results = []
    
    for i in range(n_runs):
        random.seed(i * 100)
        np.random.seed(i * 100)
        
        # 模拟A*路径（只优化长度）
        path = []
        for j in range(10):
            x = 250 + j * 200 + random.uniform(-50, 50)
            y = 250 + j * 200 + random.uniform(-50, 50)
            z = 100 + random.uniform(-20, 20)
            path.append([x, y, z])
        
        objectives = evaluate_random_path(path)
        
        results.append({
            'energy': objectives[0],
            'time': objectives[1],
            'noise': objectives[2],
            'collision': objectives[3],
            'length': objectives[4],
            'success': True
        })
        
        print(f"  运行{i+1}: 长度={objectives[4]:.1f}m, 能耗={objectives[0]:.3f}kWh, 噪声={objectives[2]:.3f}")
    
    return results


def evaluate_multi_objective(n_pop=50, n_gen=100):
    """评估NSGA-II多目标优化"""
    print("\n" + "="*60)
    print(f"评估2: NSGA-II多目标优化 (种群={n_pop}, 代数={n_gen})")
    print("="*60)
    print("  优化目标: 能耗、时间、噪声、碰撞风险、路径长度")
    
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='collision', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    optimizer = NSGAIIOptimizer(
        n_objectives=5, 
        population_size=n_pop, 
        n_generations=n_gen,
        random_seed=42
    )
    optimizer.set_objectives(objectives)
    optimizer.set_environment({
        'start': (250, 250, 100),
        'goal': (2250, 2250, 100),
        'bounds': ((0, 2500), (0, 2500), (30, 200))
    })
    
    start_time = time.time()
    result = optimizer.optimize(evaluate_random_path)
    opt_time = time.time() - start_time
    
    # 提取帕累托前沿
    pareto_solutions = []
    for ind in result.pareto_front.individuals:
        pareto_solutions.append({
            'energy': ind.objectives[0],
            'time': ind.objectives[1],
            'noise': ind.objectives[2],
            'collision': ind.objectives[3],
            'length': ind.objectives[4]
        })
    
    # TOPSIS决策
    decision_maker = ParetoDecisionMaker(objectives)
    best_topsis = decision_maker.select_by_topsis(result.pareto_front)
    
    print(f"\n  优化完成!")
    print(f"  帕累托解数量: {len(pareto_solutions)}")
    print(f"  优化时间: {opt_time:.2f}s")
    print(f"  超体积指标(HV): {result.pareto_front.hypervolume:.2f}")
    print(f"  间距指标(Spacing): {result.pareto_front.spacing:.4f}")
    
    if best_topsis:
        print(f"\n  TOPSIS推荐方案:")
        print(f"    能耗: {best_topsis.objectives[0]:.3f} kWh")
        print(f"    时间: {best_topsis.objectives[1]:.1f} s")
        print(f"    噪声: {best_topsis.objectives[2]:.3f}")
        print(f"    碰撞风险: {best_topsis.objectives[3]:.4f}")
        print(f"    路径长度: {best_topsis.objectives[4]:.1f} m")
        
        best_solution = {
            'energy': best_topsis.objectives[0],
            'time': best_topsis.objectives[1],
            'noise': best_topsis.objectives[2],
            'collision': best_topsis.objectives[3],
            'length': best_topsis.objectives[4]
        }
    else:
        best_solution = pareto_solutions[0] if pareto_solutions else None
    
    return {
        'n_solutions': len(pareto_solutions),
        'hypervolume': result.pareto_front.hypervolume,
        'spacing': result.pareto_front.spacing,
        'optimization_time': opt_time,
        'topsis_best': best_solution,
        'pareto_front': pareto_solutions,
        'pareto_front_obj': result.pareto_front
    }


def evaluate_moea_d(n_pop=50, n_gen=100):
    """评估MOEA/D算法"""
    print("\n" + "="*60)
    print(f"评估3: MOEA/D多目标优化 (种群={n_pop}, 代数={n_gen})")
    print("="*60)
    
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='collision', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
    ]
    
    optimizer = MOEADOptimizer(
        n_objectives=5,
        population_size=n_pop,
        n_generations=n_gen,
        random_seed=42
    )
    optimizer.set_objectives(objectives)
    optimizer.set_environment({
        'start': (250, 250, 100),
        'goal': (2250, 2250, 100),
        'bounds': ((0, 2500), (0, 2500), (30, 200))
    })
    
    start_time = time.time()
    result = optimizer.optimize(evaluate_random_path)
    opt_time = time.time() - start_time
    
    print(f"\n  优化完成!")
    print(f"  非支配解数量: {len(result.pareto_front.individuals)}")
    print(f"  优化时间: {opt_time:.2f}s")
    print(f"  超体积指标(HV): {result.pareto_front.hypervolume:.2f}")
    
    return result


def calculate_improvement(baseline, nsga2_result, moea_d_result):
    """计算改进效果"""
    print("\n" + "="*60)
    print("改进效果分析")
    print("="*60)
    
    improvements = {}
    
    if baseline and nsga2_result.get('topsis_best'):
        # 能耗对比
        base_energy = np.mean([b['energy'] for b in baseline if b.get('success')])
        mo_energy = nsga2_result['topsis_best']['energy']
        improvements['energy_change'] = ((mo_energy - base_energy) / max(base_energy, 0.001)) * 100
        
        # 噪声对比
        base_noise = np.mean([b['noise'] for b in baseline if b.get('success')])
        mo_noise = nsga2_result['topsis_best']['noise']
        improvements['noise_change'] = ((mo_noise - base_noise) / max(base_noise, 0.001)) * 100
        
        # 碰撞风险对比
        base_collision = np.mean([b['collision'] for b in baseline if b.get('success')])
        mo_collision = nsga2_result['topsis_best']['collision']
        improvements['collision_change'] = ((mo_collision - base_collision) / max(base_collision, 0.001)) * 100
        
        print(f"  【NSGA-II vs 传统单目标】")
        print(f"    能耗变化: {improvements['energy_change']:+.1f}%")
        print(f"    噪声变化: {improvements['noise_change']:+.1f}%")
        print(f"    碰撞风险变化: {improvements['collision_change']:+.1f}%")
    
    # 算法对比
    if nsga2_result and moea_d_result:
        print(f"\n  【算法对比: NSGA-II vs MOEA/D】")
        print(f"    NSGA-II 超体积: {nsga2_result['hypervolume']:.2f}")
        print(f"    MOEA/D 超体积: {moea_d_result.pareto_front.hypervolume:.2f}")
        
        if moea_d_result.pareto_front.hypervolume > nsga2_result['hypervolume']:
            print(f"    结论: MOEA/D在当前问题上表现更好")
        else:
            print(f"    结论: NSGA-II在当前问题上表现更好")
    
    return improvements


def visualize_comparison(nsga2_result, moea_d_result, output_dir):
    """可视化对比结果"""
    
    # 获取帕累托解的目标值
    nsga2_pareto = nsga2_result['pareto_front_obj'].individuals
    moea_pareto = moea_d_result.pareto_front.individuals
    
    # 安全获取目标值
    def safe_mean(individuals, idx):
        values = []
        for ind in individuals:
            if hasattr(ind, 'objectives') and ind.objectives is not None:
                try:
                    val = ind.objectives[idx]
                    if val is not None and not np.isnan(val):
                        values.append(val)
                except:
                    pass
        return np.mean(values) if values else 0
    
    objectives_names = ['能耗', '时间', '噪声', '碰撞风险', '长度']
    nsga2_means = [safe_mean(nsga2_pareto, i) for i in range(5)]
    moea_means = [safe_mean(moea_pareto, i) for i in range(5)]
    
    print(f"\n  [调试] NSGA-II目标均值: {nsga2_means}")
    print(f"  [调试] MOEA/D目标均值: {moea_means}")
    
    # =====================================================
    # 1. 雷达图 - 多目标综合对比（推荐）
    # =====================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
    plt.suptitle('多目标优化算法对比 - 雷达图视图', fontsize=14, fontweight='bold')
    
    # 子图1: 绝对值雷达图
    ax_radar1 = axes1[0]
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 归一化到0-1范围（用于雷达图）
    max_vals = [max(nsga2_means[i], moea_means[i]) for i in range(5)]
    nsga2_norm = [nsga2_means[i] / max_vals[i] for i in range(5)]
    moea_norm = [moea_means[i] / max_vals[i] for i in range(5)]
    nsga2_norm += nsga2_norm[:1]
    moea_norm += moea_norm[:1]
    
    ax_radar1.plot(angles, nsga2_norm, 'o-', linewidth=2, label='NSGA-II', color='steelblue')
    ax_radar1.fill(angles, nsga2_norm, alpha=0.25, color='steelblue')
    ax_radar1.plot(angles, moea_norm, 's-', linewidth=2, label='MOEA/D', color='coral')
    ax_radar1.fill(angles, moea_norm, alpha=0.25, color='coral')
    ax_radar1.set_xticks(angles[:-1])
    ax_radar1.set_xticklabels(objectives_names, fontsize=10)
    ax_radar1.set_title('归一化对比\n(越小越好)', fontsize=11, pad=20)
    ax_radar1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # 子图2: 反转雷达图（用于最小化目标，越靠外越差）
    ax_radar2 = axes1[1]
    # 反转变量：面积代表"劣于最优解的程度"
    inv_nsga2 = [1 - v for v in nsga2_norm[:-1]]
    inv_moea = [1 - v for v in moea_norm[:-1]]
    inv_nsga2 += inv_nsga2[:1]
    inv_moea += inv_moea[:1]
    
    ax_radar2.plot(angles, inv_nsga2, 'o-', linewidth=2, label='NSGA-II', color='steelblue')
    ax_radar2.fill(angles, inv_nsga2, alpha=0.25, color='steelblue')
    ax_radar2.plot(angles, inv_moea, 's-', linewidth=2, label='MOEA/D', color='coral')
    ax_radar2.fill(angles, inv_moea, alpha=0.25, color='coral')
    ax_radar2.set_xticks(angles[:-1])
    ax_radar2.set_xticklabels(objectives_names, fontsize=10)
    ax_radar2.set_title('劣质程度对比\n(越小=越优)', fontsize=11, pad=20)
    ax_radar2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    output_path1 = f'{output_dir}/radar_comparison.png'
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  雷达图已保存: {output_path1}")
    
    # =====================================================
    # 2. 帕累托前沿投影散点图
    # =====================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle('帕累托前沿投影对比', fontsize=14, fontweight='bold')
    
    objective_pairs = [
        (0, 1, '能耗(kWh)', '时间(s)'),
        (0, 4, '能耗(kWh)', '长度(m)'),
        (2, 3, '噪声', '碰撞风险'),
        (1, 4, '时间(s)', '长度(m)'),
    ]
    
    colors = {'nsga2': 'steelblue', 'moea_d': 'coral'}
    
    for ax, (i, j, xi, yj) in zip(axes2.flat, objective_pairs):
        x_nsga2 = [ind.objectives[i] for ind in nsga2_pareto if hasattr(ind, 'objectives') and len(ind.objectives) > max(i,j)]
        y_nsga2 = [ind.objectives[j] for ind in nsga2_pareto if hasattr(ind, 'objectives') and len(ind.objectives) > max(i,j)]
        x_moea = [ind.objectives[i] for ind in moea_pareto if hasattr(ind, 'objectives') and len(ind.objectives) > max(i,j)]
        y_moea = [ind.objectives[j] for ind in moea_pareto if hasattr(ind, 'objectives') and len(ind.objectives) > max(i,j)]
        
        min_len = min(len(x_nsga2), len(y_nsga2), len(x_moea), len(y_moea))
        if min_len > 0:
            ax.scatter(x_nsga2[:min_len], y_nsga2[:min_len], c=colors['nsga2'], s=60, alpha=0.7, 
                       edgecolors='white', linewidth=1, label='NSGA-II')
            ax.scatter(x_moea[:min_len], y_moea[:min_len], c=colors['moea_d'], s=60, alpha=0.7,
                       marker='s', edgecolors='white', linewidth=1, label='MOEA/D')
        
        ax.set_xlabel(xi, fontsize=10)
        ax.set_ylabel(yj, fontsize=10)
        ax.set_title(f'{xi} vs {yj}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    output_path2 = f'{output_dir}/scatter_comparison.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  散点对比图已保存: {output_path2}")
    
    # =====================================================
    # 3. 性能提升热力图
    # =====================================================
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    
    # 计算NSGA-II相对于MOEA/D的提升百分比
    improvements = []
    for i in range(5):
        if moea_means[i] > 0:
            imp = (moea_means[i] - nsga2_means[i]) / moea_means[i] * 100
        else:
            imp = 0
        improvements.append(imp)
    
    # 创建热力图数据
    improvement_data = np.array([[imp] for imp in improvements])
    
    # 绘制热力图
    im = ax3.imshow(improvement_data, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=100)
    
    # 添加数值标签
    for i, imp in enumerate(improvements):
        color = 'white' if abs(imp) > 50 else 'black'
        ax3.text(0, i, f'{imp:+.1f}%', ha='center', va='center', fontsize=12, color=color, fontweight='bold')
    
    ax3.set_yticks(range(5))
    ax3.set_yticklabels(objectives_names)
    ax3.set_xticks([0])
    ax3.set_xticklabels(['NSGA-II vs MOEA/D'])
    ax3.set_title('性能提升热力图 (绿色=NSGA-II更优, 红色=MOEA/D更优)', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('提升百分比 (%)')
    
    plt.tight_layout()
    output_path3 = f'{output_dir}/improvement_heatmap.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  热力图已保存: {output_path3}")
    
    # =====================================================
    # 4. 综合仪表盘
    # =====================================================
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
    plt.suptitle('多目标优化综合评估仪表盘', fontsize=14, fontweight='bold')
    
    # 4.1 超体积对比柱状图
    ax41 = axes4[0, 0]
    hv_labels = ['NSGA-II', 'MOEA/D']
    hv_values = [nsga2_result['hypervolume'], moea_d_result.pareto_front.hypervolume]
    colors_hv = ['steelblue', 'coral']
    bars = ax41.bar(hv_labels, hv_values, color=colors_hv, alpha=0.8)
    ax41.set_ylabel('超体积 (HV)')
    ax41.set_title('帕累托前沿质量 (HV越大越好)')
    for bar, val in zip(bars, hv_values):
        ax41.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax41.grid(True, alpha=0.3, axis='y')
    
    # 4.2 解数量对比
    ax42 = axes4[0, 1]
    n_solutions = [nsga2_result['n_solutions'], len(moea_pareto)]
    bars = ax42.bar(hv_labels, n_solutions, color=colors_hv, alpha=0.8)
    ax42.set_ylabel('帕累托解数量')
    ax42.set_title('解集多样性')
    for bar, val in zip(bars, n_solutions):
        ax42.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax42.grid(True, alpha=0.3, axis='y')
    
    # 4.3 能耗对比
    ax43 = axes4[0, 2]
    bars = ax43.bar(hv_labels, [nsga2_means[0], moea_means[0]], color=colors_hv, alpha=0.8)
    ax43.set_ylabel('能耗 (kWh)')
    ax43.set_title('平均能耗对比')
    for bar, val in zip(bars, [nsga2_means[0], moea_means[0]]):
        ax43.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.2f}', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax43.grid(True, alpha=0.3, axis='y')
    
    # 4.4 时间对比
    ax44 = axes4[1, 0]
    bars = ax44.bar(hv_labels, [nsga2_means[1], moea_means[1]], color=colors_hv, alpha=0.8)
    ax44.set_ylabel('时间 (s)')
    ax44.set_title('平均飞行时间对比')
    for bar, val in zip(bars, [nsga2_means[1], moea_means[1]]):
        ax44.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax44.grid(True, alpha=0.3, axis='y')
    
    # 4.5 碰撞风险对比
    ax45 = axes4[1, 1]
    bars = ax45.bar(hv_labels, [nsga2_means[3], moea_means[3]], color=colors_hv, alpha=0.8)
    ax45.set_ylabel('碰撞风险')
    ax45.set_title('平均碰撞风险对比')
    for bar, val in zip(bars, [nsga2_means[3], moea_means[3]]):
        ax45.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax45.grid(True, alpha=0.3, axis='y')
    
    # 4.6 长度对比
    ax46 = axes4[1, 2]
    bars = ax46.bar(hv_labels, [nsga2_means[4], moea_means[4]], color=colors_hv, alpha=0.8)
    ax46.set_ylabel('路径长度 (m)')
    ax46.set_title('平均路径长度对比')
    for bar, val in zip(bars, [nsga2_means[4], moea_means[4]]):
        ax46.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}', 
                  ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax46.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path4 = f'{output_dir}/dashboard.png'
    plt.savefig(output_path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  综合仪表盘已保存: {output_path4}")
    
    print(f"\n  算法对比图已保存: {output_path2}")


def main():
    print("="*60)
    print("eVTOL V6版本多目标优化核心功能测试")
    print("="*60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n本测试展示多目标优化算法（NSGA-II, MOEA/D）的核心能力")
    print("对比传统单目标优化与帕累托最优解集的区别")
    
    # 评估1: 基线（传统方法）
    baseline_results = evaluate_baseline()
    
    # 评估2: NSGA-II
    nsga2_results = evaluate_multi_objective(n_pop=50, n_gen=100)
    
    # 评估3: MOEA/D
    moea_d_results = evaluate_moea_d(n_pop=50, n_gen=100)
    
    # 计算改进效果
    improvements = calculate_improvement(baseline_results, nsga2_results, moea_d_results)
    
    # 可视化
    print("\n" + "="*60)
    print("生成可视化")
    print("="*60)
    output_dir = '/home/ayan1907/evtol/evtol_v6_comparison_results'
    import os
    os.makedirs(output_dir, exist_ok=True)
    visualize_comparison(nsga2_results, moea_d_results, output_dir)
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'method': '传统单目标优化 (A*)',
            'n_runs': len(baseline_results),
            'avg_energy': float(np.mean([b['energy'] for b in baseline_results])),
            'avg_noise': float(np.mean([b['noise'] for b in baseline_results])),
            'avg_collision': float(np.mean([b['collision'] for b in baseline_results])),
        },
        'nsga2': {
            'method': 'NSGA-II',
            'n_pareto_solutions': nsga2_results['n_solutions'],
            'hypervolume': nsga2_results['hypervolume'],
            'spacing': nsga2_results['spacing'],
            'optimization_time': nsga2_results['optimization_time'],
            'topsis_best': nsga2_results['topsis_best'],
            'pareto_front': nsga2_results['pareto_front_obj']
        },
        'moea_d': {
            'method': 'MOEA/D',
            'n_solutions': len(moea_d_results.pareto_front.individuals),
            'hypervolume': moea_d_results.pareto_front.hypervolume,
        },
        'improvements': improvements
    }
    
    report_path = f'{output_dir}/core_comparison_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    # 打印总结
    print("\n" + "="*60)
    print("测试完成总结")
    print("="*60)
    print(f"  帕累托前沿质量:")
    print(f"    NSGA-II 解数量: {nsga2_results['n_solutions']}, 超体积: {nsga2_results['hypervolume']:.2f}")
    print(f"    MOEA/D 解数量: {len(moea_d_results.pareto_front.individuals)}, 超体积: {moea_d_results.pareto_front.hypervolume:.2f}")
    print(f"\n  报告已保存至: {report_path}")
    print(f"\n  可视化结果:")
    print(f"    - {output_dir}/algorithm_comparison.png")
    print(f"    - {output_dir}/pareto_distribution.png")
    
    print("\n" + "="*60)
    print("核心结论")
    print("="*60)
    print("""
  1. 多目标优化 vs 单目标优化:
     - 传统A*仅优化路径长度，忽略能耗、噪声、安全性
     - NSGA-II/MOEA/D同时优化5个目标，提供多样化权衡方案
     
  2. 帕累托最优:
     - 帕累托前沿上的解互不支配
     - 用户可根据偏好选择最终方案
     
  3. 决策支持:
     - TOPSIS方法自动推荐最优折衷方案
     - 模糊决策支持不确定偏好表达
     
  4. 工程价值:
     - 支持复杂城市环境下的多目标航迹规划
     - 为eVTOL绿色安全运行提供优化决策支持
""")


if __name__ == "__main__":
    main()
