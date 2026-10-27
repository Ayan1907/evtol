#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6+ 整合版测试脚本

展示V6+版本如何整合:
1. 高保真能耗子模型
2. 蒙特卡洛不确定性分析
3. 帕累托多目标优化

使用方法: python test_v6_plus.py
"""

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import zhplot

# 导入V6+整合版
from evtol_v6_plus import EVTOLV6Plus, FlightProfile
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment

# 导入V6可视化器
from v6_visualizer import V6Visualizer


def setup_environment():
    """设置测试环境"""
    print("=" * 70)
    print("V6+ 整合版测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建城市环境
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    print(f"\n起点: {start_pos}")
    print(f"终点: {goal_pos}")
    
    # 创建标准
    standard = EVTOLStandard()
    start_vp = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vp = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    # 创建动态环境
    dynamic_env = DynamicEnvironment(env)
    wind_field = WindField(
        env, base_wind_speed=8.0, base_wind_direction=135.0,
        turbulence_intensity=0.15, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    
    return env, standard, start_vp, goal_vp, dynamic_env


def test_basic_planning(env, standard, start_vp, goal_vp, dynamic_env):
    """测试1: 标准规划"""
    print("\n" + "=" * 70)
    print("测试1: 标准路径规划 (V6基础功能)")
    print("=" * 70)
    
    # 创建合适的飞行剖面
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,  # 降低巡航高度以适应小地图
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    # 创建V6+规划器
    planner = EVTOLV6Plus(
        env, standard, start_vp, goal_vp, dynamic_env,
        flight_profile=flight_profile,
        use_high_fidelity_energy=True
    )
    
    # 规划路径
    print("\n规划路径...")
    path = planner.plan()
    
    if path:
        print(f"  找到路径! 节点数: {len(path)}")
        
        # 评估路径
        metrics = planner.evaluate_path(path)
        print(f"\n  路径指标:")
        print(f"    能耗: {metrics['energy']:.3f} kWh")
        print(f"    时间: {metrics['time']:.1f} s")
        print(f"    噪声: {metrics['noise']:.3f}")
        print(f"    安全: {metrics['safety']:.4f}")
        print(f"    长度: {metrics['length']:.1f} m")
        
        return path, metrics
    else:
        print("  路径规划失败!")
        return None, None


def test_multi_objective(env, standard, start_vp, goal_vp, dynamic_env):
    """测试2: 多目标优化"""
    print("\n" + "=" * 70)
    print("测试2: 帕累托多目标优化")
    print("=" * 70)
    
    # 创建合适的飞行剖面
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
    
    # 多目标优化
    print("\n运行多目标优化 (30代, 种群20)...")
    pareto_front = planner.optimize_multi_objective(n_generations=30, population_size=20)
    
    print(f"\n  帕累托解数量: {len(pareto_front.individuals)}")
    print(f"  超体积(HV): {pareto_front.hypervolume:.2f}")
    print(f"  间距(Spacing): {pareto_front.spacing:.4f}")
    
    # 选择最优解
    best = planner.select_best_solution(pareto_front, method='topsis')
    if best:
        print(f"\n  TOPSIS最优解:")
        print(f"    能耗: {best.objectives[0]:.3f} kWh")
        print(f"    时间: {best.objectives[1]:.1f} s")
        print(f"    噪声: {best.objectives[2]:.3f}")
        print(f"    安全: {best.objectives[3]:.4f}")
        print(f"    长度: {best.objectives[4]:.1f} m")
    
    return pareto_front, best


def test_uncertainty(env, standard, start_vp, goal_vp, dynamic_env):
    """测试3: 不确定性分析"""
    print("\n" + "=" * 70)
    print("测试3: 蒙特卡洛不确定性分析")
    print("=" * 70)
    
    # 创建合适的飞行剖面
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
    
    # 运行不确定性分析
    print("\n运行不确定性分析 (200次采样)...")
    results = planner.analyze_uncertainty(
        n_samples=200,
        include_wind=True,
        include_mass=True,
        include_temp=True
    )
    
    print(f"\n  采样成功率: {results['n_successful']}/{results['n_samples']}")
    
    print(f"\n  能耗统计:")
    print(f"    均值: {results['energy']['mean']:.3f} kWh")
    print(f"    标准差: {results['energy']['std']:.3f} kWh")
    print(f"    范围: [{results['energy']['min']:.3f}, {results['energy']['max']:.3f}]")
    print(f"    5%-95%: [{results['energy']['p5']:.3f}, {results['energy']['p95']:.3f}]")
    
    print(f"\n  风险评估:")
    print(f"    能耗超标概率: {results['risk_assessment']['energy_exceed_prob']*100:.1f}%")
    print(f"    时间超标概率: {results['risk_assessment']['time_exceed_prob']*100:.1f}%")
    print(f"    能耗变异系数: {results['risk_assessment']['energy_coefficient_variation']*100:.1f}%")
    
    return results


def test_full_analysis(env, standard, start_vp, goal_vp, dynamic_env):
    """测试4: 完整分析报告"""
    print("\n" + "=" * 70)
    print("测试4: V6+ 完整分析")
    print("=" * 70)
    
    # 创建合适的飞行剖面
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
    
    # 完整分析
    report = planner.get_full_analysis(
        n_samples=200,
        n_generations=30
    )
    
    return report


def visualize_results(results, output_dir, env=None, flight_profile=None, dynamic_env=None):
    """可视化结果"""
    print("\n" + "=" * 70)
    print("生成可视化")
    print("=" * 70)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取路径信息
    path = results.get('basic_path')
    start_pos = (100, 100, 30)  # 默认起点
    goal_pos = (2400, 2400, 30)  # 默认终点
    if env:
        try:
            map_data = env.load_from_json('test_map_v3_compatible.json')
            start_pos = tuple(map_data['start'])
            goal_pos = tuple(map_data['goal'])
        except:
            pass
    
    # 将LowCarbonNode对象列表转换为坐标元组列表
    if path:
        path_coords = []
        for p in path:
            if hasattr(p, 'x'):
                path_coords.append((p.x, p.y, p.z))
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                path_coords.append((p[0], p[1], p[2]))
        path = path_coords
    
    # ========== V6可视化器完整输出 ==========
    if path and env:
        print("\n  [1/9] 生成路径俯视图...")
        V6Visualizer.plot_path_with_buildings(
            env, path, start_pos, goal_pos, flight_profile,
            filename=f'{output_dir}/v6_plus_01_path_topview.png'
        )
        
        print("  [2/9] 生成3D路径剖面图...")
        V6Visualizer.plot_3d_path_profile(
            env, path, start_pos, goal_pos, flight_profile,
            filename=f'{output_dir}/v6_plus_02_3d_path.png'
        )
        
        print("  [3/9] 生成高度剖面图...")
        V6Visualizer.plot_height_profile(
            path, start_pos, goal_pos, flight_profile,
            filename=f'{output_dir}/v6_plus_03_height_profile.png'
        )
        
        print("  [4/9] 生成飞行阶段统计图...")
        if flight_profile:
            profile_data = calculate_profile_stats(path, start_pos, goal_pos, flight_profile)
            V6Visualizer.plot_phase_statistics(
                profile_data,
                filename=f'{output_dir}/v6_plus_04_phase_stats.png'
            )
        
        print("  [5/9] 生成风场2D图...")
        if dynamic_env and hasattr(dynamic_env, 'wind_field'):
            V6Visualizer.plot_wind_field_2d(
                env, dynamic_env.wind_field, path,
                filename=f'{output_dir}/v6_plus_05_wind_field.png'
            )
        
        print("  [6/9] 生成动态障碍物2D图...")
        if dynamic_env:
            V6Visualizer.plot_dynamic_obstacles_2d(
                env, dynamic_env, path, time_sec=0.0,
                filename=f'{output_dir}/v6_plus_06_dynamic_obstacles.png'
            )
        
        print("  [7/9] 生成动态障碍物3D图...")
        if dynamic_env:
            V6Visualizer.plot_dynamic_obstacles_3d(
                env, dynamic_env, path, time_sec=0.0,
                filename=f'{output_dir}/v6_plus_07_dynamic_obstacles_3d.png'
            )
        
        print("  [8/9] 生成风场代价热力图...")
        if dynamic_env and hasattr(dynamic_env, 'wind_field'):
            V6Visualizer.plot_wind_cost_heatmap(
                env, dynamic_env.wind_field, path,
                filename=f'{output_dir}/v6_plus_08_wind_cost.png'
            )
        
        print("  [9/9] 生成路径风场3D图...")
        if dynamic_env and hasattr(dynamic_env, 'wind_field'):
            V6Visualizer.plot_path_with_wind_3d(
                env, dynamic_env.wind_field, path, start_pos, goal_pos,
                filename=f'{output_dir}/v6_plus_09_path_wind_3d.png'
            )
    
    # ========== 帕累托优化可视化 ==========
    print("\n  [A] 生成帕累托前沿可视化...")
    if 'pareto_front' in results and results['pareto_front']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle('V6+ 帕累托多目标优化结果', fontsize=14, fontweight='bold')
        
        pareto = results['pareto_front'].individuals
        pareto_arr = np.array([ind.objectives for ind in pareto])
        
        pairs = [(0, 1), (0, 2), (1, 3), (2, 4)]
        labels = [('能耗', '时间'), ('能耗', '噪声'), ('时间', '安全'), ('噪声', '长度')]
        
        for ax, (i, j), (xi, yj) in zip(axes.flat, pairs, labels):
            ax.scatter(pareto_arr[:, i], pareto_arr[:, j], c='steelblue', s=60, alpha=0.7)
            
            if 'best_solution' in results and results['best_solution']:
                best = results['best_solution']
                ax.scatter(best.objectives[i], best.objectives[j], 
                          c='red', s=150, marker='*', label='TOPSIS最优')
            
            ax.set_xlabel(xi, fontsize=10)
            ax.set_ylabel(yj, fontsize=10)
            ax.set_title(f'{xi} vs {yj}', fontsize=11)
            ax.grid(True, alpha=0.3)
            if 'best_solution' in results and results['best_solution']:
                ax.legend()
        
        plt.tight_layout()
        output = f'{output_dir}/v6_plus_pareto.png'
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    帕累托前沿图: {output}")
    
    # ========== 不确定性分析可视化 ==========
    print("\n  [B] 生成不确定性分析可视化...")
    if 'uncertainty_analysis' in results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plt.suptitle('V6+ 不确定性分析', fontsize=14, fontweight='bold')
        
        ua = results['uncertainty_analysis']
        
        # 能耗分布
        ax = axes[0]
        energy = ua['energy']
        x = np.linspace(energy['min'], energy['max'], 100)
        from scipy import stats
        y = stats.norm.pdf(x, energy['mean'], energy['std'])
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axvline(energy['mean'], color='r', linestyle='--', label=f"均值: {energy['mean']:.2f}")
        ax.axvline(energy['p5'], color='g', linestyle=':', label=f"5%: {energy['p5']:.2f}")
        ax.axvline(energy['p95'], color='g', linestyle=':', label=f"95%: {energy['p95']:.2f}")
        ax.set_xlabel('能耗 (kWh)')
        ax.set_ylabel('概率密度')
        ax.set_title('能耗分布')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 时间分布
        ax = axes[1]
        time_data = ua['time']
        x = np.linspace(time_data['min'], time_data['max'], 100)
        y = stats.norm.pdf(x, time_data['mean'], time_data['std'])
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axvline(time_data['mean'], color='r', linestyle='--', label=f"均值: {time_data['mean']:.1f}")
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('概率密度')
        ax.set_title('飞行时间分布')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 风险评估
        ax = axes[2]
        risks = ua['risk_assessment']
        categories = ['能耗超标', '时间超标']
        values = [risks['energy_exceed_prob'] * 100, risks['time_exceed_prob'] * 100]
        colors = ['green' if v < 10 else 'orange' if v < 30 else 'red' for v in values]
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('风险概率 (%)')
        ax.set_title('风险评估')
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output = f'{output_dir}/v6_plus_uncertainty.png'
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    不确定性分析图: {output}")
    
    # ========== 综合评估图 ==========
    print("\n  [C] 生成综合评估图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle('V6+ 综合性能评估', fontsize=14, fontweight='bold')
    
    # 左图: 帕累托解分布
    ax = axes[0]
    if 'pareto_front' in results and results['pareto_front']:
        pareto = results['pareto_front'].individuals
        objectives = ['能耗', '时间', '噪声', '安全', '长度']
        means = [np.mean([ind.objectives[i] for ind in pareto]) for i in range(5)]
        
        ax.bar(objectives, means, color='steelblue', alpha=0.8)
        ax.set_ylabel('目标值')
        ax.set_title('帕累托解均值')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 右图: 不确定性
    ax = axes[1]
    if 'uncertainty_analysis' in results:
        ua = results['uncertainty_analysis']
        categories = ['能耗', '时间', '噪声']
        means = [ua['energy']['mean'], ua['time']['mean'], ua['noise']['mean']]
        stds = [ua['energy']['std'], ua['time']['std'], ua['noise']['std']]
        
        x = np.arange(len(categories))
        ax.bar(x, means, yerr=stds, color='steelblue', alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('值 (±σ)')
        ax.set_title('不确定性分析 (带误差棒)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output = f'{output_dir}/v6_plus_summary.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    综合评估图: {output}")
    
    print(f"\n  可视化完成! 共生成 {9 + 3} 张图片")


def calculate_profile_stats(path, start_pos, goal_pos, flight_profile):
    """计算飞行阶段统计信息"""
    profile_data = {
        'climb': {'count': 0, 'min_height': 1000, 'max_height': 0, 'avg_height': 0, 'total_dist': 0},
        'cruise': {'count': 0, 'min_height': 1000, 'max_height': 0, 'avg_height': 0, 'total_dist': 0},
        'approach': {'count': 0, 'min_height': 1000, 'max_height': 0, 'avg_height': 0, 'total_dist': 0},
    }
    
    heights = {'climb': [], 'cruise': [], 'approach': []}
    
    for p in path:
        phase = flight_profile.get_flight_phase((p[0], p[1], p[2]), start_pos, goal_pos)
        profile_data[phase]['count'] += 1
        heights[phase].append(p[2])
    
    for phase in ['climb', 'cruise', 'approach']:
        if heights[phase]:
            profile_data[phase]['min_height'] = min(heights[phase])
            profile_data[phase]['max_height'] = max(heights[phase])
            profile_data[phase]['avg_height'] = sum(heights[phase]) / len(heights[phase])
    
    return profile_data


def main():
    """主函数"""
    # 设置环境
    env, standard, start_vp, goal_vp, dynamic_env = setup_environment()
    
    output_dir = '/home/ayan1907/evtol/evtol_v6_plus_results'
    
    # 存储结果
    results = {}
    
    # 测试1: 标准规划
    path, metrics = test_basic_planning(env, standard, start_vp, goal_vp, dynamic_env)
    results['basic_path'] = path
    results['basic_metrics'] = metrics
    
    # 测试2: 多目标优化
    pareto_front, best = test_multi_objective(env, standard, start_vp, goal_vp, dynamic_env)
    results['pareto_front'] = pareto_front
    results['best_solution'] = best
    
    # 测试3: 不确定性分析
    uncertainty = test_uncertainty(env, standard, start_vp, goal_vp, dynamic_env)
    results['uncertainty_analysis'] = uncertainty
    
    # 测试4: 完整分析
    report = test_full_analysis(env, standard, start_vp, goal_vp, dynamic_env)
    
    # 可视化
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    visualize_results(results, output_dir, env, flight_profile, dynamic_env)
    
    # 保存报告
    import json
    report_path = f'{output_dir}/v6_plus_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': results.get('basic_metrics'),
            'pareto_summary': {
                'n_solutions': len(pareto_front.individuals) if pareto_front else 0,
                'hypervolume': float(pareto_front.hypervolume) if pareto_front else 0,
            },
            'uncertainty_summary': {
                'energy_mean': float(uncertainty['energy']['mean']) if uncertainty else 0,
                'energy_std': float(uncertainty['energy']['std']) if uncertainty else 0,
                'risk': float(uncertainty['risk_assessment']['energy_exceed_prob']) if uncertainty else 0,
            }
        }
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {report_path}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
