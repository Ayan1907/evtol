#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6蒙特卡洛不确定性分析集成测试

功能:
1. 使用V6规划器生成路径
2. 运行蒙特卡洛不确定性分析
3. 生成鲁棒性评估报告
4. 可视化不确定性分布
"""

import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import List, Tuple

# 导入V6模块
from evtol_path_planning_v6 import LowCarbonAStarV6, FlightProfile, smooth_path_nodes
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment, DynamicObstacle

# 导入能耗模型
from evtol_high_fidelity_energy import create_energy_model

# 导入不确定性分析模块
from monte_carlo_uncertainty_analysis import (
    MonteCarloUncertaintyAnalyzer,
    UncertaintyConfig,
    print_uncertainty_report,
    analyze_path_robustness,
    WindUncertaintyModel,
    BATTERY_UNCERTAINTY_PARAMS,
    POWER_SYSTEM_UNCERTAINTY,
)


def run_uncertainty_analysis():
    """
    完整的不确定性分析流程
    """
    print("=" * 70)
    print("V6蒙特卡洛不确定性分析集成测试")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载地图和创建环境
    # =========================================================================
    print("\n[1/6] 加载地图...")
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")
    
    # =========================================================================
    # 2. 创建标准和环境
    # =========================================================================
    print("\n[2/6] 创建标准和环境...")
    standard = EVTOLStandard()
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    dynamic_env = DynamicEnvironment(env)
    
    wind_field = WindField(
        env, base_wind_speed=8.0, base_wind_direction=135.0,
        turbulence_intensity=0.15, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    
    # 添加动态障碍物
    dynamic_env.add_dynamic_obstacle(
        DynamicObstacle(
            id=1, type='drone', x=1000, y=1000, z=150,
            vx=5.0, vy=5.0, vz=0.0, size=10.0,
            valid_time_start=0.0, valid_time_end=300.0
        )
    )
    
    # =========================================================================
    # 3. 创建飞行剖面和规划器
    # =========================================================================
    print("\n[3/6] 创建规划器...")
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    planner = LowCarbonAStarV6(
        env, standard, start_vertiport, goal_vertiport, dynamic_env, flight_profile,
        omega1=0.7, omega2=0.3,
        alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25,
        alpha5=0.25,
        standard_weight=0.5,
        use_improved_wind=True
    )
    
    # =========================================================================
    # 4. 搜索路径
    # =========================================================================
    print("\n[4/6] 搜索路径...")
    path = planner.find_path(max_iterations=150000, current_time=0.0)
    
    if not path:
        print("路径搜索失败!")
        return None
    
    print(f"\n找到路径: {len(path)} 个节点")
    
    # 平滑路径
    path = smooth_path_nodes(path, start_pos, goal_pos, flight_profile, planner)
    print(f"平滑后: {len(path)} 个节点")
    
    # =========================================================================
    # 5. 计算基线能耗
    # =========================================================================
    print("\n[5/6] 计算基线能耗...")
    energy_model = create_energy_model('EH216-S')
    
    # 转换路径格式
    path_coords = [(n.x, n.y, n.z) for n in path]
    
    # 计算基线能耗
    baseline_result = energy_model.calculate_path_energy(
        path_coords, wind_field=wind_field, dt=1.0
    )
    baseline_energy = baseline_result['total_energy_kwh']
    
    print(f"  基线总能耗: {baseline_energy:.3f} kWh")
    print(f"  总距离: {baseline_result['total_distance_m']:.1f} m")
    print(f"  单位能耗: {baseline_result['energy_per_km_kwh']:.3f} kWh/km")
    
    # =========================================================================
    # 6. 运行蒙特卡洛不确定性分析
    # =========================================================================
    print("\n[6/6] 运行蒙特卡洛不确定性分析...")
    
    # 配置不确定性分析
    config = UncertaintyConfig(
        n_samples=500,           # 样本数量
        sampling_method='lhs',  # 拉丁超立方采样
        confidence_level=0.95,
        failure_threshold=0.05,  # 能耗超过基线5%视为失效
        enable_wind=True,
        enable_battery=True,
        enable_power=True,
        enable_localization=False,  # 本次不启用定位不确定性
        random_seed=42
    )
    
    # 创建分析器
    analyzer = MonteCarloUncertaintyAnalyzer(energy_model, config)
    
    # 运行分析
    result = analyzer.run_analysis(path_coords, baseline_energy, wind_field)
    
    # 打印报告
    print_uncertainty_report(result, baseline_energy)
    
    # =========================================================================
    # 7. 保存结果
    # =========================================================================
    output_dir = "evtol_v6_uncertainty_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON结果
    result_data = {
        'baseline_energy_kwh': baseline_energy,
        'uncertainty_config': {
            'n_samples': config.n_samples,
            'sampling_method': config.sampling_method,
            'confidence_level': config.confidence_level,
            'failure_threshold': config.failure_threshold,
        },
        'energy_statistics': {
            'mean': result.energy_mean,
            'std': result.energy_std,
            'min': result.energy_min,
            'max': result.energy_max,
            'median': result.energy_median,
            'p5': result.energy_p5,
            'p95': result.energy_p95,
        },
        'risk_metrics': {
            'failure_probability': result.failure_probability,
            'energy_exceedance_prob': result.energy_exceedance_prob,
            'robustness_index': result.robustness_index,
            'cvar_95': result.conditional_value_at_risk,
        },
        'sensitivity': {
            'wind': result.sensitivity_to_wind,
            'battery': result.sensitivity_to_battery,
            'power': result.sensitivity_to_power,
        },
        'path_info': {
            'num_nodes': len(path),
            'total_distance_m': baseline_result['total_distance_m'],
        }
    }
    
    result_file = os.path.join(output_dir, 'uncertainty_analysis_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")
    
    # =========================================================================
    # 8. 可视化
    # =========================================================================
    print("\n生成不确定性可视化...")
    
    # 能耗分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图1: 能耗分布直方图
    ax1 = axes[0, 0]
    if result.samples and 'energies' in result.samples:
        energies = result.samples['energies']
        ax1.hist(energies, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        ax1.axvline(baseline_energy, color='red', linestyle='--', linewidth=2, label=f'基线: {baseline_energy:.2f} kWh')
        ax1.axvline(result.energy_mean, color='green', linestyle='-', linewidth=2, label=f'期望: {result.energy_mean:.2f} kWh')
        ax1.axvline(result.energy_p95, color='orange', linestyle=':', linewidth=2, label=f'95%分位: {result.energy_p95:.2f} kWh')
        ax1.set_xlabel('能耗 (kWh)')
        ax1.set_ylabel('概率密度')
        ax1.set_title('能耗不确定性分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 子图2: 敏感性分析柱状图
    ax2 = axes[0, 1]
    sensitivities = [result.sensitivity_to_wind, result.sensitivity_to_battery, result.sensitivity_to_power]
    labels = ['风场', '电池', '动力系统']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax2.bar(labels, sensitivities, color=colors, edgecolor='white')
    ax2.set_ylabel('敏感性指数')
    ax2.set_title('不确定性源敏感性分析')
    for bar, val in zip(bars, sensitivities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 子图3: 风险指标
    ax3 = axes[1, 0]
    risk_metrics = ['失效概率', '超标概率', '鲁棒性指数']
    values = [result.failure_probability * 100, 
              result.energy_exceedance_prob * 100,
              result.robustness_index * 100]
    colors_risk = ['#FF6B6B', '#FFE66D', '#4ECDC4']
    bars3 = ax3.bar(risk_metrics, values, color=colors_risk, edgecolor='white')
    ax3.set_ylabel('百分比 (%)')
    ax3.set_title('风险指标评估')
    ax3.set_ylim(0, 100)
    for bar, val in zip(bars3, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 能耗分位数箱线图
    ax4 = axes[1, 1]
    box_data = [[result.energy_min, result.energy_p5, result.energy_median, 
                result.energy_p95, result.energy_max]]
    bp = ax4.boxplot(box_data, vert=False, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    ax4.axvline(baseline_energy, color='red', linestyle='--', linewidth=2, label='基线能耗')
    ax4.set_xlabel('能耗 (kWh)')
    ax4.set_title('能耗分位数箱线图')
    ax4.set_yticks([1])
    ax4.set_yticklabels(['能耗分布'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {os.path.join(output_dir, 'uncertainty_analysis.png')}")
    
    # =========================================================================
    # 9. 综合分析报告
    # =========================================================================
    print("\n" + "=" * 70)
    print("V6蒙特卡洛不确定性分析总结")
    print("=" * 70)
    
    print(f"""
【分析概况】
  • 路径长度: {baseline_result['total_distance_m']:.1f} m
  • 路径节点: {len(path)} 个
  • Monte Carlo样本: {config.n_samples} 个
  • 采样方法: {config.sampling_method}

【能耗评估】
  • 基线能耗: {baseline_energy:.3f} kWh
  • 期望能耗: {result.energy_mean:.3f} kWh ({result.energy_mean/baseline_energy*100:.1f}%)
  • 能耗波动: ±{result.energy_std:.3f} kWh ({result.energy_std/baseline_energy*100:.1f}%)

【风险评估】
  • 失效概率: {result.failure_probability:.2%}
  • 鲁棒性指数: {result.robustness_index:.2%}
  • CVaR(95%): {result.conditional_value_at_risk:.3f} kWh

【敏感性排序】
  1. 风场不确定性: {result.sensitivity_to_wind:.3f}
  2. 电池不确定性: {result.sensitivity_to_battery:.3f}
  3. 动力系统不确定性: {result.sensitivity_to_power:.3f}

【建议】
""")
    
    # 根据敏感性分析给出建议
    max_sensitivity = max(result.sensitivity_to_wind, 
                         result.sensitivity_to_battery,
                         result.sensitivity_to_power)
    
    if max_sensitivity == result.sensitivity_to_wind:
        print("  → 主要不确定性来自风场，建议增强风场预测精度或增加风场代价权重")
    elif max_sensitivity == result.sensitivity_to_battery:
        print("  → 主要不确定性来自电池，建议加强电池状态监测或增加备用电量")
    else:
        print("  → 主要不确定性来自动力系统，建议定期校准电机效率参数")
    
    if result.failure_probability < 0.05:
        print("  → 路径鲁棒性良好，失败概率较低，可以执行飞行任务")
    elif result.failure_probability < 0.15:
        print("  → 路径存在一定风险，建议增加电池余量或优化航迹")
    else:
        print("  → 路径风险较高，建议重新规划或增加安全裕度")
    
    print("=" * 70)
    
    return result


def compare_sampling_methods():
    """
    比较不同采样方法的效果
    """
    print("\n" + "=" * 70)
    print("采样方法对比测试")
    print("=" * 70)
    
    # 创建简单测试路径
    path = [(0, 0, 50), (500, 0, 150), (1000, 0, 150), (1500, 500, 150), (2000, 1000, 50)]
    
    energy_model = create_energy_model('EH216-S')
    baseline_result = energy_model.calculate_path_energy(path, dt=1.0)
    baseline_energy = baseline_result['total_energy_kwh']
    
    methods = ['mc', 'lhs']
    results = {}
    
    for method in methods:
        config = UncertaintyConfig(
            n_samples=200,
            sampling_method=method,
            random_seed=42
        )
        
        analyzer = MonteCarloUncertaintyAnalyzer(energy_model, config)
        result = analyzer.run_analysis(path, baseline_energy)
        
        results[method] = {
            'mean': result.energy_mean,
            'std': result.energy_std,
            'p95': result.energy_p95,
        }
        
        print(f"\n{method.upper()}采样结果:")
        print(f"  期望: {result.energy_mean:.3f} kWh")
        print(f"  标准差: {result.energy_std:.3f} kWh")
        print(f"  95%分位: {result.energy_p95:.3f} kWh")
    
    print("\n" + "-" * 40)
    print("采样效率对比:")
    print(f"  LHS相比朴素MC收敛更快，方差更小")
    print(f"  推荐使用LHS进行不确定性分析")


if __name__ == "__main__":
    # 运行不确定性分析
    result = run_uncertainty_analysis()
    
    # 可选: 比较采样方法
    # compare_sampling_methods()
