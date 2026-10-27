#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6噪声约束可视化测试脚本
生成噪声传播热力图和沿路径的噪声变化曲线图
"""

import sys
sys.path.append('/home/ayan1907/evtol')

from evtol_path_planning_v6 import LowCarbonAStarV6, CityEnvironment, ImprovedWindField, FlightProfile
from evtol_path_planning_v3_standard import Vertiport
from noise_model import NoiseModel
from evtol_noise_visualizer import NoiseVisualizer
import json
import zhplot


def load_environment_and_path():
    """加载环境和路径数据"""
    
    # 1. 创建环境并从JSON加载
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_v3_compatible.json')
    print(f"✓ 地图加载完成: {len(env.buildings)} 栋建筑物")
    print(f"  地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    
    # 2. 加载路径结果
    with open('evtol_v6_results/v6_main_result.json', 'r') as f:
        result_data = json.load(f)
    
    path = result_data['path']
    start_pos = tuple(result_data['start'])
    goal_pos = tuple(result_data['goal'])
    
    print(f"✓ 路径加载完成: {len(path)} 个节点")
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")
    print(f"  总距离: {result_data['stats']['total_distance']:.2f}m")
    
    return env, path, start_pos, goal_pos


def setup_flight_profile(start_pos, goal_pos):
    """设置飞行剖面"""
    # 创建飞行剖面
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    print(f"✓ 飞行剖面设置完成")
    print(f"  巡航层: {flight_profile.cruise_min}m - {flight_profile.cruise_max}m")
    print(f"  OFV高度: {flight_profile.ofv_height}m")
    
    return flight_profile


def setup_noise_model(env):
    """设置噪声模型"""
    noise_model = NoiseModel(
        env=env,
        source_level=85.0,      # 源头噪声级 85dB @ 100m
        reference_distance=100.0
    )
    
    print(f"✓ 噪声模型设置完成")
    print(f"  源头噪声级: {noise_model.source.source_level} dB(A)")
    print(f"  参考距离: {noise_model.source.reference_distance} m")
    print(f"  干扰阈值: {noise_model.disturbance_threshold} dB(A)")
    print(f"  严重阈值: {noise_model.severe_threshold} dB(A)")
    
    return noise_model


def generate_noise_visualizations():
    """生成噪声可视化图表"""
    
    print("=" * 70)
    print("V6噪声约束可视化测试")
    print("=" * 70)
    print()
    
    # 1. 加载环境和路径
    env, path, start_pos, goal_pos = load_environment_and_path()
    print()
    
    # 2. 设置飞行剖面
    flight_profile = setup_flight_profile(start_pos, goal_pos)
    print()
    
    # 3. 设置噪声模型
    noise_model = setup_noise_model(env)
    print()
    
    # 4. 创建噪声可视化器
    print("-" * 70)
    print("生成噪声可视化图表...")
    print("-" * 70)
    
    visualizer = NoiseVisualizer(env, noise_model)
    
    # 生成噪声传播热力图
    print("\n1. 生成噪声传播热力图...")
    visualizer.plot_noise_heatmap(
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        filename='evtol_v6_results/v6_noise_heatmap.png',
        grid_spacing=20,
        time_hour=10  # 白天时段（10:00）
    )
    
    # 生成沿路径的噪声变化曲线图
    print("\n2. 生成沿路径噪声变化曲线图...")
    visualizer.plot_noise_along_path(
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        flight_profile=flight_profile,
        filename='evtol_v6_results/v6_noise_along_path.png',
        current_hour=10,  # 白天时段（10:00）
        speed=20.0  # 飞行速度 20m/s
    )
    
    print()
    print("=" * 70)
    print("✓ 噪声可视化完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. evtol_v6_results/v6_noise_heatmap.png - 噪声传播热力图")
    print("  2. evtol_v6_results/v6_noise_along_path.png - 沿路径噪声变化曲线图")
    print()


if __name__ == "__main__":
    generate_noise_visualizations()
