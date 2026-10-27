#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
sys.path.insert(0, '.')

from evtol_v7.evtol_path_planning_v7 import EVTOLV7, FlightProfile, MultiObjectiveOptimizerV7
from evtol_path_planning_v3_clean import load_environment_from_json
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport
from evtol_path_planning_v4 import DynamicEnvironment, WindField

# 加载V6的地图
print('加载地图: test_map_v3_compatible.json')
env, start_coords, goal_coords, metadata = load_environment_from_json('test_map_v3_compatible.json')
print(f'地图尺寸: {env.size_x} x {env.size_y}')
print(f'建筑物数量: {len(env.buildings)}')
print(f'起点: {start_coords}')
print(f'终点: {goal_coords}')

# 创建V7
standard = EVTOLStandard()
start_vp = Vertiport(x=start_coords[0], y=start_coords[1], z=start_coords[2], standard=standard)
goal_vp = Vertiport(x=goal_coords[0], y=goal_coords[1], z=goal_coords[2], standard=standard)

wind_field = WindField(env=env, base_wind_speed=5.0, base_wind_direction=315.0, turbulence_intensity=0.1)
dynamic_env = DynamicEnvironment(env)
dynamic_env.wind_field = wind_field

planner = EVTOLV7(env, standard, start_vp, goal_vp, dynamic_env)

# 运行多目标优化
print('\n开始V7多目标优化...')
result = planner.optimize_multi_objective(n_generations=30, population_size=30)

# 获取最优解 - 使用MultiObjectiveOptimizerV7的_genes_to_path
if result.individuals:
    best = result.individuals[0]
    
    # 创建临时优化器来解码路径
    optimizer = MultiObjectiveOptimizerV7(planner, planner.evaluator)
    path = optimizer._genes_to_path(best.genes)
    
    print(f'\n============ V7测试结果 (V6地图) ============')
    print(f'路径节点数: {len(path)}')
    print(f'X范围: {path[0][0]:.0f} -> {path[-1][0]:.0f}')
    heights = [p[2] for p in path]
    print(f'高度范围: {min(heights):.0f} - {max(heights):.0f}m')
    print(f'平均高度: {sum(heights)/len(heights):.1f}m')
    
    # 计算路径长度
    total_length = 0
    for i in range(len(path)-1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        dz = path[i+1][2] - path[i][2]
        total_length += np.sqrt(dx**2 + dy**2 + dz**2)
    print(f'总路径长度: {total_length:.1f}m')
    
    # 保存路径供可视化
    path_data = {
        'nodes': [[float(p[0]), float(p[1]), float(p[2])] for p in path],
        'n_nodes': len(path),
        'height_range': [float(min(heights)), float(max(heights))],
        'avg_height': float(sum(heights)/len(heights)),
        'path_length': float(total_length)
    }
    with open('evtol_v7_results/v7_path_data.json', 'w') as f:
        json.dump(path_data, f, indent=2)
    print('\n路径数据已保存: evtol_v7_results/v7_path_data.json')
