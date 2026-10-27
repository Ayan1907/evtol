#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6多目标优化集成测试

功能:
1. 集成V6路径规划、蒙特卡洛分析、多目标优化
2. 生成帕累托最优前沿
3. 决策支持与可视化
"""

import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# 导入V6核心模块
from evtol_path_planning_v6 import LowCarbonAStarV6, FlightProfile, smooth_path_nodes
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment, DynamicObstacle

# 导入能耗模型
from evtol_high_fidelity_energy import create_energy_model

# 导入不确定性分析
from monte_carlo_uncertainty_analysis import (
    MonteCarloUncertaintyAnalyzer,
    UncertaintyConfig,
)

# 导入多目标优化模块
from multi_objective_optimizer import (
    NSGAIIOptimizer,
    MOEADOptimizer,
    ParetoDecisionMaker,
    Objective,
    ObjectiveType,
    create_default_objectives,
    print_pareto_summary,
    ParetoFront,
)


class MultiObjectiveEvaluator:
    """
    多目标评估器 - 评估航迹的多个目标函数值
    
    目标:
    1. energy: 能耗 (kWh) - 最小化
    2. time: 飞行时间 (s) - 最小化
    3. noise: 噪声影响 (归一化) - 最小化
    4. collision: 碰撞风险 (概率) - 最小化
    5. length: 路径长度 (m) - 最小化
    """
    
    def __init__(self, energy_model, noise_model=None, collision_model=None):
        """
        初始化评估器
        """
        self.energy_model = energy_model
        self.noise_model = noise_model
        self.collision_model = collision_model
    
    def evaluate(self, path: List) -> np.ndarray:
        """评估路径的多目标函数值"""
        if len(path) < 2:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 转换路径格式
        if hasattr(path[0], 'x'):
            path_coords = [(n.x, n.y, n.z) for n in path]
        else:
            path_coords = path
        
        # 1. 能耗计算
        energy_result = self.energy_model.calculate_path_energy(path_coords, dt=1.0)
        energy = energy_result['total_energy_kwh']
        
        # 2. 时间计算
        total_length = energy_result['total_distance_m']
        avg_speed = 50.0
        flight_time = total_length / avg_speed
        
        # 3. 噪声评估
        noise = self._evaluate_noise(path_coords)
        
        # 4. 碰撞风险
        collision = self._evaluate_collision(path_coords)
        
        # 5. 路径长度
        length = total_length
        
        return np.array([energy, flight_time, noise, collision, length])
    
    def _evaluate_noise(self, path_coords: List) -> float:
        """评估噪声影响"""
        if not self.noise_model or len(path_coords) < 2:
            avg_height = np.mean([p[2] for p in path_coords])
            noise_score = max(0, 1 - (avg_height - 50) / 200)
            return noise_score
        
        total_noise = 0.0
        for i in range(len(path_coords) - 1):
            cost = self.noise_model.calculate_noise_cost(
                path_coords[i], path_coords[i + 1], current_hour=10, speed=20.0
            )
            total_noise += cost
        
        return total_noise / max(1, len(path_coords) - 1)
    
    def _evaluate_collision(self, path_coords: List) -> float:
        """评估碰撞风险"""
        if not self.collision_model or len(path_coords) < 2:
            avg_height = np.mean([p[2] for p in path_coords])
            risk_score = max(0, 1 - (avg_height - 50) / 200) * 0.1
            return risk_score
        
        max_risk = 0.0
        for i in range(len(path_coords) - 1):
            risk = self.collision_model.calculate_collision_cost(
                path_coords[i], path_coords[i + 1], speed=20.0
            )
            max_risk = max(max_risk, risk)
        
        return max_risk


def run_multi_objective_optimization():
    """完整的多目标优化流程"""
    print("=" * 70)
    print("V6 多目标优化集成测试")
    print("=" * 70)
    
    # =========================================================================
    # 1. 环境设置
    # =========================================================================
    print("\n[1/6] 环境设置...")
    
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")
    
    standard = EVTOLStandard()
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    dynamic_env = DynamicEnvironment(env)
    
    wind_field = WindField(
        env, base_wind_speed=8.0, base_wind_direction=135.0,
        turbulence_intensity=0.15, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    
    # =========================================================================
    # 2. 创建评估器
    # =========================================================================
    print("\n[2/6] 创建评估器...")
    
    energy_model = create_energy_model('EH216-S')
    evaluator = MultiObjectiveEvaluator(energy_model)
    
    # =========================================================================
    # 3. 生成初始种群
    # =========================================================================
    print("\n[3/6] 生成初始种群...")
    
    from multi_objective_optimizer import Individual, PathEncoder
    
    encoder = PathEncoder(n_keypoints=8)
    
    param_combinations = [
        (0.7, 0.3, 0.4, 0.3, 0.1, 0.1, 0.1, 'energy_prefer'),
        (0.7, 0.3, 0.1, 0.1, 0.4, 0.2, 0.2, 'length_prefer'),
        (0.5, 0.5, 0.3, 0.3, 0.2, 0.1, 0.1, 'balanced'),
        (0.7, 0.3, 0.2, 0.2, 0.1, 0.3, 0.2, 'smooth_prefer'),
    ]
    
    initial_population = []
    
    for omega1, omega2, a1, a2, a3, a4, a5, name in param_combinations:
        print(f"  生成 {name} 路径...")
        
        flight_profile = FlightProfile(
            ofv_height=30.0,
            cruise_min=120.0,
            cruise_max=180.0,
            climb_distance_threshold=400.0,
            approach_distance_threshold=400.0
        )
        
        planner = LowCarbonAStarV6(
            env, standard, start_vertiport, goal_vertiport,
            dynamic_env, flight_profile,
            omega1=omega1, omega2=omega2,
            alpha1=a1, alpha2=a2, alpha3=a3, alpha4=a4, alpha5=a5,
            standard_weight=0.5,
            use_improved_wind=True
        )
        
        path = planner.find_path(max_iterations=80000, current_time=0.0)
        
        if path:
            path = smooth_path_nodes(path, start_pos, goal_pos, flight_profile, planner)
            genes = encoder.encode(path)
            individual = Individual(genes=genes)
            individual.objectives = evaluator.evaluate(path)
            initial_population.append(individual)
            print(f"    路径节点: {len(path)}, 目标值: {[f'{v:.2f}' for v in individual.objectives]}")
    
    # 补充随机解
    if len(initial_population) < 10:
        print(f"  生成随机路径补足种群...")
        for i in range(10 - len(initial_population)):
            genes = []
            n_points = 8
            for j in range(n_points):
                alpha = (j + 1) / (n_points + 1)
                x = int(start_pos[0] + (goal_pos[0] - start_pos[0]) * alpha + np.random.uniform(-100, 100))
                y = int(start_pos[1] + (goal_pos[1] - start_pos[1]) * alpha + np.random.uniform(-100, 100))
                z = int(100 + np.random.uniform(-20, 80))
                genes.append((x, y, z))
            
            path = encoder.decode(genes, start_pos, goal_pos)
            individual = Individual(genes=genes)
            individual.objectives = evaluator.evaluate(path)
            initial_population.append(individual)
    
    print(f"  初始种群大小: {len(initial_population)}")
    
    # =========================================================================
    # 4. 多目标优化
    # =========================================================================
    print("\n[4/6] 运行多目标优化...")
    
    objectives = [
        Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.30),
        Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='collision', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
        Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.10),
    ]
    
    def evaluate_func(genes):
        path = encoder.decode(genes, start_pos, goal_pos)
        return evaluator.evaluate(path)
    
    optimizer = NSGAIIOptimizer(
        n_objectives=5,
        population_size=max(20, len(initial_population)),
        n_generations=50,
        crossover_prob=0.9,
        mutation_prob=0.1,
        random_seed=42
    )
    
    optimizer.set_objectives(objectives)
    optimizer.set_environment({
        'start': start_pos,
        'goal': goal_pos,
        'bounds': ((0, 3000), (0, 3000), (30, 300))
    })
    
    result = optimizer.optimize(evaluate_func, initial_population)
    
    # =========================================================================
    # 5. 决策支持
    # =========================================================================
    print("\n[5/6] 决策支持...")
    
    decision_maker = ParetoDecisionMaker(objectives)
    print_pareto_summary(result, ['能耗(kWh)', '时间(s)', '噪声', '碰撞风险', '长度(m)'])
    
    print("\n【推荐方案】")
    
    best_topsis = decision_maker.select_by_topsis(result.pareto_front)
    if best_topsis:
        objs = best_topsis.objectives
        print(f"\n  TOPSIS推荐:")
        print(f"    能耗: {objs[0]:.3f} kWh")
        print(f"    时间: {objs[1]:.1f} s")
        print(f"    噪声: {objs[2]:.3f}")
        print(f"    碰撞风险: {objs[3]:.3f}")
        print(f"    长度: {objs[4]:.1f} m")
    
    best_fuzzy = decision_maker.select_by_fuzzy(result.pareto_front)
    if best_fuzzy:
        objs = best_fuzzy.objectives
        print(f"\n  模糊决策推荐:")
        print(f"    能耗: {objs[0]:.3f} kWh")
        print(f"    时间: {objs[1]:.1f} s")
    
    # =========================================================================
    # 6. 可视化
    # =========================================================================
    print("\n[6/6] 生成可视化...")
    
    output_dir = "evtol_v6_multi_objective_results"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    objective_pairs = [
        (0, 1, '能耗', '时间'),
        (0, 4, '能耗', '长度'),
        (1, 4, '时间', '长度'),
        (2, 3, '噪声', '碰撞风险'),
        (0, 2, '能耗', '噪声'),
        (3, 4, '碰撞风险', '长度'),
    ]
    
    for ax, (i, j, xi, yj) in zip(axes.flat, objective_pairs):
        if len(result.pareto_front.individuals) > 0:
            x_vals = [ind.objectives[i] for ind in result.pareto_front.individuals]
            y_vals = [ind.objectives[j] for ind in result.pareto_front.individuals]
            
            ax.scatter(x_vals, y_vals, c='steelblue', s=50, alpha=0.7, edgecolors='white')
            
            if best_topsis:
                ax.scatter(best_topsis.objectives[i], best_topsis.objectives[j],
                          c='red', s=150, marker='*', label='TOPSIS')
            
            ax.set_xlabel(xi)
            ax.set_ylabel(yj)
            ax.set_title(f'{xi} vs {yj}')
            ax.grid(True, alpha=0.3)
            if best_topsis:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_2d_projections.png'), dpi=150, bbox_inches='tight')
    
    # 保存结果
    result_data = {
        'n_objectives': 5,
        'n_iterations': result.n_iterations,
        'n_evaluations': result.n_function_evaluations,
        'pareto_front_size': len(result.pareto_front.individuals),
        'hypervolume': result.pareto_front.hypervolume,
        'spacing': result.pareto_front.spacing,
        'objectives_names': ['energy', 'time', 'noise', 'collision', 'length'],
        'recommended_topsis': best_topsis.objectives.tolist() if best_topsis else None,
        'recommended_fuzzy': best_fuzzy.objectives.tolist() if best_fuzzy else None,
    }
    
    with open(os.path.join(output_dir, 'pareto_results.json'), 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("V6 多目标优化完成")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = run_multi_objective_optimization()
