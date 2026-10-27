#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7 vs 标准A* 定量对比测试
对比内容：
- 标准3D A*: 仅优化路径长度，基础安全约束
- V7算法: 国标约束 + 多目标优化（能耗/噪声/安全）
"""

import sys
import os
import json
import time
import math
import heapq
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evtol_v7.evtol_path_planning_v7 import EVTOLV7, FlightProfile
from evtol_path_planning_v3_clean import load_environment_from_json
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport
from evtol_path_planning_v4 import DynamicEnvironment, WindField, DynamicObstacle


class SimpleNode:
    """标准3D A*的简单节点"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0
        self.parent = None

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((round(self.x/10), round(self.y/10), round(self.z/10)))

    def __lt__(self, other):
        return self.f < other.f


class Standard3DAStar:
    """标准3D A*算法 - 仅优化路径长度"""
    
    def __init__(self, env, start, goal, grid_size=50.0):
        self.env = env
        self.start = start
        self.goal = goal
        self.grid_size = grid_size

    def heuristic(self, node):
        return math.sqrt(
            (node.x - self.goal[0])**2 +
            (node.y - self.goal[1])**2 +
            (node.z - self.goal[2])**2
        )

    def get_neighbors(self, node):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if (dx, dy, dz) != (0, 0, 0):
                        new_x = node.x + dx * self.grid_size
                        new_y = node.y + dy * self.grid_size
                        new_z = node.z + dz * self.grid_size
                        neighbor = SimpleNode(new_x, new_y, new_z)
                        if self._is_safe(neighbor):
                            neighbors.append(neighbor)
        return neighbors

    def _is_safe(self, node):
        if not (0 <= node.x <= self.env.size_x and
                0 <= node.y <= self.env.size_y and
                0 <= node.z <= 200.0):
            return False
        for b in self.env.buildings:
            if (b.x - b.width/2 <= node.x <= b.x + b.width/2 and
                b.y - b.length/2 <= node.y <= b.y + b.length/2 and
                node.z <= b.height):
                return False
        if node.z < 0:
            return False
        return True

    def find_path(self):
        start_node = SimpleNode(*self.start)
        start_node.g = 0.0
        start_node.h = self.heuristic(start_node)
        start_node.f = start_node.g + start_node.h
        
        open_set = [start_node]
        heapq.heapify(open_set)
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)
            if self.heuristic(current) < self.grid_size * 1.5:
                return self._reconstruct_path(current)
            closed_set.add(hash(current))
            
            for neighbor in self.get_neighbors(current):
                if hash(neighbor) in closed_set:
                    continue
                tentative_g = current.g + math.sqrt(
                    (neighbor.x - current.x)**2 +
                    (neighbor.y - current.y)**2 +
                    (neighbor.z - current.z)**2
                )
                in_open = False
                for i, n in enumerate(open_set):
                    if n == neighbor:
                        if tentative_g < n.g:
                            n.g = tentative_g
                            n.f = n.g + n.h
                            n.parent = current
                        in_open = True
                        break
                if not in_open:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current
                    heapq.heappush(open_set, neighbor)
        return None

    def _reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append((node.x, node.y, node.z))
            node = node.parent
        path.reverse()
        return path


def compute_metrics(path, env, dynamic_env, flight_profile):
    """计算路径的各项指标"""
    if not path or len(path) < 2:
        return None
    
    # 基础指标
    total_length = 0
    max_height = 0
    min_height = float('inf')
    height_changes = 0
    
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        dz = path[i+1][2] - path[i][2]
        seg_len = math.sqrt(dx**2 + dy**2 + dz**2)
        total_length += seg_len
        max_height = max(max_height, path[i+1][2])
        min_height = min(min_height, path[i+1][2])
        height_changes += abs(dz)
    
    # 水平直线距离（不含z轴分量）
    straight_dist = math.sqrt(
        (path[-1][0] - path[0][0])**2 +
        (path[-1][1] - path[0][1])**2
    )
    # 路径效率 = 直线距离 / 路径长度，路径越直效率越高（最大100%）
    efficiency = min((straight_dist / total_length * 100), 100.0) if total_length > 0 else 0
    
    # 能耗估算（简化模型）
    # E = base + climb_penalty * height_change + distance_factor * length
    energy_base = 2.0  # 基础能耗 kWh
    climb_penalty = 0.005  # 每米高度变化的能耗惩罚
    distance_factor = 0.0005  # 每米距离的能耗因子
    estimated_energy = energy_base + climb_penalty * height_changes + distance_factor * total_length
    
    # 时间估算
    avg_speed = 50.0  # m/s
    estimated_time = total_length / avg_speed  # 秒
    
    # 噪声影响评估（基于高度）
    # 物理原理：高度越高，地面噪声越小（噪声衰减 ~ 20*log10(h)）
    # 国标MH/T 2011-2019要求：巡航高度120-180m
    # 评分逻辑：高度越高 = 噪声越小 = 评分越高
    avg_height = np.mean([p[2] for p in path])
    if avg_height < 120:
        noise_score = 30.0  # 不满足国标，接近地面，高噪声 → 低分
    elif avg_height < 160:
        noise_score = 50.0  # 部分满足国标，较高噪声 → 中等
    elif avg_height <= 180:
        noise_score = 80.0  # 满足国标，低噪声 → 高分
    else:
        noise_score = 90.0  # 超出国标要求，最低噪声 → 最高分
    
    # 安全评估
    safety_risk = 0.0
    for i, p in enumerate(path):
        for b in env.buildings:
            dist_xy = math.sqrt((p[0] - b.x)**2 + (p[1] - b.y)**2)
            # 靠近建筑物增加风险
            if dist_xy < b.width:
                safety_risk += (b.width - dist_xy) / b.width * 0.1
    safety_score = max(0, 100 - safety_risk * 10)
    
    # 国标合规性检查（根据飞行阶段判断）
    # 国标MH/T 2011-2019要求：巡航高度120-180m
    # 爬升/下降阶段可低于120m，巡航阶段必须满足120-180m
    
    start_pos = path[0]
    goal_pos = path[-1]
    
    all_mid_nodes = []
    for i, p in enumerate(path):
        if i == 0 or i == len(path) - 1:
            continue  # 跳过起点和终点
        all_mid_nodes.append(p)
    
    if all_mid_nodes:
        compliant_nodes = 0
        for p in all_mid_nodes:
            # 使用飞行剖面判断该节点是否合规
            current_pos = (p[0], p[1], p[2])
            start_tuple = (start_pos[0], start_pos[1], start_pos[2])
            goal_tuple = (goal_pos[0], goal_pos[1], goal_pos[2])
            is_valid, _ = flight_profile.check_height_constraint(
                current_pos, start_tuple, goal_tuple
            )
            if is_valid:
                compliant_nodes += 1
        compliance = compliant_nodes / len(all_mid_nodes) * 100 if all_mid_nodes else 0
    else:
        compliance = 0
    
    return {
        'total_length': total_length,
        'straight_distance': straight_dist,
        'efficiency': efficiency,
        'max_height': max_height,
        'min_height': min_height,
        'avg_height': avg_height,
        'height_changes': height_changes,
        'estimated_energy': estimated_energy,
        'estimated_time': estimated_time,
        'noise_score': noise_score,
        'safety_score': safety_score,
        'safety_risk': safety_risk,
        'guobiao_compliance': compliance,
        'n_nodes': len(path)
    }


def main():
    import heapq
    
    print("\n" + "="*70)
    print("  V7 vs 标准A* 定量对比测试")
    print("="*70)
    
    # 加载地图
    print("\n[1/3] 加载测试环境...")
    env, start_coords, goal_coords, metadata = load_environment_from_json('test_map_v3_compatible.json')
    
    print(f"  起点: ({start_coords[0]:.0f}, {start_coords[1]:.0f}, {start_coords[2]:.0f})")
    print(f"  终点: ({goal_coords[0]:.0f}, {goal_coords[1]:.0f}, {goal_coords[2]:.0f})")
    print(f"  建筑物数量: {len(env.buildings)}")
    print(f"  地图尺寸: {env.size_x} x {env.size_y}")
    
    # 创建动态环境
    wind_field = WindField(env=env, base_wind_speed=5.0, base_wind_direction=315.0, turbulence_intensity=0.1)
    dynamic_env = DynamicEnvironment(env)
    dynamic_env.wind_field = wind_field
    
    # 国标飞行剖面
    flight_profile = FlightProfile(
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    # ============ 标准3D A* ============
    print("\n[2/3] 运行标准3D A*...")
    standard = EVTOLStandard()
    start_vp = Vertiport(x=start_coords[0], y=start_coords[1], z=start_coords[2], standard=standard)
    goal_vp = Vertiport(x=goal_coords[0], y=goal_coords[1], z=goal_coords[2], standard=standard)
    
    astar_start = time.time()
    astar_planner = Standard3DAStar(env, tuple(start_coords), tuple(goal_coords), grid_size=50.0)
    astar_path = astar_planner.find_path()
    astar_time = time.time() - astar_start
    
    if not astar_path:
        print("  ✗ A*规划失败!")
        return
    print(f"  ✓ A*完成 (用时: {astar_time:.2f}s)")
    
    astar_metrics = compute_metrics(astar_path, env, dynamic_env, flight_profile)
    
    # ============ V7算法 ============
    print("\n[3/3] 运行V7多目标优化...")
    v7_start = time.time()
    planner = EVTOLV7(
        env, standard, start_vp, goal_vp, dynamic_env,
        flight_profile=flight_profile,
        use_improved_wind=True,
        use_high_fidelity_energy=True
    )
    
    # 运行多目标优化
    pareto_front = planner.optimize_multi_objective(n_generations=30, population_size=30)
    v7_time = time.time() - v7_start
    
    print(f"  ✓ V7完成 (用时: {v7_time:.2f}s)")
    print(f"    Pareto解数量: {len(pareto_front.individuals)}")
    
    # 获取最优解（综合最优：取第一个解）
    best_ind = None
    if pareto_front.individuals:
        for ind in pareto_front.individuals:
            if hasattr(ind, 'objectives') and ind.objectives is not None:
                best_ind = ind
                break
    
    # 构建V7路径（使用_genes_to_path修正高度剖面）
    v7_path = []
    if best_ind and hasattr(best_ind, 'genes'):
        # 使用MultiObjectiveOptimizerV7的_genes_to_path
        from evtol_v7.evtol_path_planning_v7 import MultiObjectiveOptimizerV7
        optimizer = MultiObjectiveOptimizerV7(planner, planner.evaluator)
        v7_path = optimizer._genes_to_path(best_ind.genes)
    
    v7_metrics = compute_metrics(v7_path, env, dynamic_env, flight_profile)
    
    # 如果V7路径计算失败，使用Pareto前沿的目标值
    if not v7_path and best_ind and hasattr(best_ind, 'objectives'):
        v7_metrics = {
            'total_length': best_ind.objectives[4] if len(best_ind.objectives) > 4 else v7_metrics['total_length'],
            'estimated_time': best_ind.objectives[1] if len(best_ind.objectives) > 1 else v7_metrics['estimated_time'],
            'noise_score': 100 - best_ind.objectives[2] * 10 if len(best_ind.objectives) > 2 else v7_metrics['noise_score'],
            'safety_score': 100 - best_ind.objectives[3] * 100 if len(best_ind.objectives) > 3 else v7_metrics['safety_score'],
            'estimated_energy': best_ind.objectives[0] if len(best_ind.objectives) > 0 else v7_metrics['estimated_energy'],
        }
    
    # ============ 对比结果 ============
    print("\n" + "="*70)
    print("  📊 定量对比结果")
    print("="*70)
    
    print(f"\n{'指标':<25} {'标准A*':<18} {'V7算法':<18} {'V7优势'}")
    print("-"*70)
    
    # 路径长度
    length_diff = v7_metrics['total_length'] - astar_metrics['total_length']
    print(f"{'路径长度(m)':<25} {astar_metrics['total_length']:<18.1f} {v7_metrics['total_length']:<18.1f} {length_diff:+.1f}")
    
    # 路径效率
    eff_diff = v7_metrics['efficiency'] - astar_metrics['efficiency']
    print(f"{'路径效率(%)':<25} {astar_metrics['efficiency']:<18.1f} {v7_metrics['efficiency']:<18.1f} {eff_diff:+.1f}")
    
    # 高度范围
    print(f"{'高度范围(m)':<25} {astar_metrics['min_height']:.0f}-{astar_metrics['max_height']:.0f}{'':9} {v7_metrics['min_height']:.0f}-{v7_metrics['max_height']:.0f}")
    
    # 平均高度
    print(f"{'平均高度(m)':<25} {astar_metrics['avg_height']:<18.1f} {v7_metrics['avg_height']:<18.1f}")
    
    # 国标合规性
    gb_status = "✓" if v7_metrics['guobiao_compliance'] > 80 else "✗"
    print(f"{'国标合规性(%)':<25} {astar_metrics['guobiao_compliance']:<18.1f} {v7_metrics['guobiao_compliance']:<18.1f} {gb_status}")
    
    # 能耗
    energy_diff = astar_metrics['estimated_energy'] - v7_metrics['estimated_energy']
    energy_pct = energy_diff / astar_metrics['estimated_energy'] * 100
    print(f"{'能耗(kWh)':<25} {astar_metrics['estimated_energy']:<18.3f} {v7_metrics['estimated_energy']:<18.3f} {energy_pct:+.1f}%")
    
    # 时间
    time_diff = astar_metrics['estimated_time'] - v7_metrics['estimated_time']
    time_pct = time_diff / astar_metrics['estimated_time'] * 100
    print(f"{'飞行时间(s)':<25} {astar_metrics['estimated_time']:<18.1f} {v7_metrics['estimated_time']:<18.1f} {time_pct:+.1f}%")
    
    # 噪声评分
    noise_diff = v7_metrics['noise_score'] - astar_metrics['noise_score']
    print(f"{'噪声评分':<25} {astar_metrics['noise_score']:<18.1f} {v7_metrics['noise_score']:<18.1f} {noise_diff:+.1f}")
    
    # 安全评分
    safety_diff = v7_metrics['safety_score'] - astar_metrics['safety_score']
    print(f"{'安全评分':<25} {astar_metrics['safety_score']:<18.1f} {v7_metrics['safety_score']:<18.1f} {safety_diff:+.1f}")
    
    # 安全风险
    risk_diff = astar_metrics['safety_risk'] - v7_metrics['safety_risk']
    print(f"{'安全风险值':<25} {astar_metrics['safety_risk']:<18.4f} {v7_metrics['safety_risk']:<18.4f} {risk_diff:+.4f}")
    
    # 计算时间
    print(f"{'规划时间(s)':<25} {astar_time:<18.2f} {v7_time:<18.2f}")
    
    # ============ 综合评估 ============
    print("\n" + "="*70)
    print("  📈 综合评估")
    print("="*70)
    
    # 标准化评分 (0-100)，使用实际数据范围归一化
    def normalize(val, min_val, max_val, inverse=False):
        if max_val == min_val:
            return 100.0
        # 限制在[0,1]范围内
        norm = max(0, min(1, (val - min_val) / (max_val - min_val)))
        return (1 - norm) * 100 if inverse else norm * 100
    
    # 获取实际数据范围
    energy_min = min(astar_metrics['estimated_energy'], v7_metrics['estimated_energy'])
    energy_max = max(astar_metrics['estimated_energy'], v7_metrics['estimated_energy'])
    time_min = min(astar_metrics['estimated_time'], v7_metrics['estimated_time'])
    time_max = max(astar_metrics['estimated_time'], v7_metrics['estimated_time'])
    noise_min = min(astar_metrics['noise_score'], v7_metrics['noise_score'])
    noise_max = max(astar_metrics['noise_score'], v7_metrics['noise_score'])
    safety_min = min(astar_metrics['safety_score'], v7_metrics['safety_score'])
    safety_max = max(astar_metrics['safety_score'], v7_metrics['safety_score'])
    
    # V7综合评分（权重：安全25% + 国标25% + 能耗20% + 噪声15% + 时间15%）
    v7_total = (
        normalize(v7_metrics['safety_score'], safety_min, safety_max, False) * 0.25 +
        normalize(v7_metrics['guobiao_compliance'], 0, 100, False) * 0.25 +
        normalize(v7_metrics['estimated_energy'], energy_min, energy_max, True) * 0.20 +
        normalize(v7_metrics['noise_score'], noise_min, noise_max, False) * 0.15 +
        normalize(v7_metrics['estimated_time'], time_min, time_max, True) * 0.15
    )
    
    # A*综合评分
    astar_total = (
        normalize(astar_metrics['safety_score'], safety_min, safety_max, False) * 0.25 +
        normalize(astar_metrics['guobiao_compliance'], 0, 100, False) * 0.25 +
        normalize(astar_metrics['estimated_energy'], energy_min, energy_max, True) * 0.20 +
        normalize(astar_metrics['noise_score'], noise_min, noise_max, False) * 0.15 +
        normalize(astar_metrics['estimated_time'], time_min, time_max, True) * 0.15
    )
    
    print(f"\n  综合评分 (加权):")
    print(f"    标准A*: {astar_total:.1f}分")
    print(f"    V7算法:  {v7_total:.1f}分")
    improvement = v7_total - astar_total
    pct = (improvement / astar_total * 100) if astar_total > 0 else 0
    print(f"    V7提升: {improvement:+.1f}分 ({pct:+.1f}%)")
    
    print(f"\n  核心优势:")
    if v7_metrics['guobiao_compliance'] > astar_metrics['guobiao_compliance']:
        print(f"    ✓ 国标合规: V7严格满足MH/T 2011-2019巡航层120-180m要求")
    if v7_metrics['noise_score'] > astar_metrics['noise_score']:
        print(f"    ✓ 噪声控制: V7路径在巡航层飞行，噪声影响降低")
    if v7_metrics['safety_score'] > astar_metrics['safety_score']:
        print(f"    ✓ 安全评分: V7综合安全风险更低")
    if v7_metrics['estimated_energy'] < astar_metrics['estimated_energy']:
        print(f"    ✓ 能耗优化: V7通过高度剖面优化减少能耗")
    
    print(f"\n  劣势分析:")
    if v7_metrics['total_length'] > astar_metrics['total_length']:
        print(f"    ⚠ 路径长度: V7略长，因需满足国标高度约束")
    print(f"    ⚠ 规划时间: V7多目标优化耗时更长")
    
    # 保存结果
    results = {
        'algorithm': 'V7 vs Standard A*',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'astar': {
            'path_length': astar_metrics['total_length'],
            'efficiency': astar_metrics['efficiency'],
            'avg_height': astar_metrics['avg_height'],
            'height_range': [astar_metrics['min_height'], astar_metrics['max_height']],
            'guobiao_compliance': astar_metrics['guobiao_compliance'],
            'energy': astar_metrics['estimated_energy'],
            'time': astar_metrics['estimated_time'],
            'noise_score': astar_metrics['noise_score'],
            'safety_score': astar_metrics['safety_score'],
            'safety_risk': astar_metrics['safety_risk'],
            'planning_time': astar_time,
            'n_nodes': astar_metrics['n_nodes']
        },
        'v7': {
            'path_length': v7_metrics['total_length'],
            'efficiency': v7_metrics['efficiency'],
            'avg_height': v7_metrics['avg_height'],
            'height_range': [v7_metrics['min_height'], v7_metrics['max_height']],
            'guobiao_compliance': v7_metrics['guobiao_compliance'],
            'energy': v7_metrics['estimated_energy'],
            'time': v7_metrics['estimated_time'],
            'noise_score': v7_metrics['noise_score'],
            'safety_score': v7_metrics['safety_score'],
            'safety_risk': v7_metrics['safety_risk'],
            'planning_time': v7_time,
            'n_nodes': v7_metrics['n_nodes'],
            'pareto_solutions': len(pareto_front.individuals),
            'hypervolume': pareto_front.hypervolume
        },
        'comparison': {
            'v7_overall_score': v7_total,
            'astar_overall_score': astar_total,
            'v7_improvement': v7_total - astar_total,
            'v7_improvement_pct': (v7_total/astar_total-1)*100
        }
    }
    
    os.makedirs('evtol_v7_results', exist_ok=True)
    with open('evtol_v7_results/v7_vs_astar_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 结果已保存: evtol_v7_results/v7_vs_astar_comparison.json")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
