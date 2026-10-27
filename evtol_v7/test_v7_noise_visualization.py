#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7噪声可视化测试脚本
生成噪声传播热力图和沿路径的噪声变化曲线图
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from evtol_path_planning_v7 import EVTOLV7, FlightProfile
from evtol_path_planning_v3_clean import load_environment_from_json
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport
from evtol_path_planning_v4 import DynamicEnvironment, WindField
from noise_model import NoiseModel
from v7_visualizer import visualize_noise


def load_environment_and_path():
    """加载环境和路径数据"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_file = os.path.join(project_root, 'test_map_v3_compatible.json')
    
    # 加载地图
    env, start_coords, goal_coords, metadata = load_environment_from_json(map_file)
    print(f"✓ 地图加载完成: {len(env.buildings)} 栋建筑物")
    print(f"  地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    
    # 尝试加载V7结果
    result_file = os.path.join(project_root, 'evtol_v7_results/v7_main_result.json')
    
    # 如果V7路径存在且高度在合理范围，使用V7路径
    use_v7 = False
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        path = result_data['path']
        heights = [p[2] for p in path]
        if max(heights) <= env.size_z:
            use_v7 = True
            print(f"  ✓ 使用V7自身规划的路径")
    
    if not use_v7:
        # 回退到V6结果
        result_file = os.path.join(project_root, 'evtol_v6_results/v6_main_result.json')
        print("  使用V6路径结果进行噪声可视化演示")
    
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        path = result_data['path']
        start_pos = tuple(result_data['start'])
        goal_pos = tuple(result_data['goal'])
        
        # 检查路径高度
        heights = [p[2] for p in path]
        print(f"✓ 路径加载完成: {len(path)} 个节点")
        print(f"  起点: {start_pos}, 终点: {goal_pos}")
        print(f"  高度范围: {min(heights):.0f}m - {max(heights):.0f}m")
        
        return env, path, start_pos, goal_pos
    except FileNotFoundError:
        print("  警告: 未找到任何路径结果")
        return env, None, None, None


def run_path_planning(env):
    """运行路径规划"""
    print("\n运行V7路径规划...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_file = os.path.join(project_root, 'test_map_v3_compatible.json')
    
    # 加载地图获取起点终点
    _, start_coords, goal_coords, _ = load_environment_from_json(map_file)
    
    # 创建Vertiport
    standard = EVTOLStandard()
    start_vp = Vertiport(x=start_coords[0], y=start_coords[1], z=start_coords[2], standard=standard)
    goal_vp = Vertiport(x=goal_coords[0], y=goal_coords[1], z=goal_coords[2], standard=standard)
    
    # 创建动态环境
    wind_field = WindField(
        env=env,
        base_wind_speed=5.0,
        base_wind_direction=315.0,
        turbulence_intensity=0.1
    )
    dynamic_env = DynamicEnvironment(env)
    dynamic_env.wind_field = wind_field
    
    # 创建V7规划器
    planner = EVTOLV7(
        env=env,
        standard=standard,
        start_vertiport=start_vp,
        goal_vertiport=goal_vp,
        dynamic_env=dynamic_env,
        use_high_fidelity_energy=False  # 简化模式
    )
    
    # 先运行标准规划获取路径
    print("  执行A*路径规划...")
    path = planner.plan(max_iterations=200000, current_time=0.0)
    
    if path and len(path) >= 2:
        print(f"✓ 路径规划成功! 节点数: {len(path)}")
        
        # 转换路径格式
        path_coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else p for p in path]
        start_pos = planner.start_vp.get_TLOF_center()
        goal_pos = planner.goal_vp.get_TLOF_center()
        
        return path_coords, start_pos, goal_pos
    else:
        print("  路径规划失败!")
        return None, None, None


def generate_noise_visualizations():
    """生成噪声可视化图表"""
    
    print("=" * 70)
    print("V7噪声约束可视化测试")
    print("=" * 70)
    print()
    
    # 1. 加载环境和路径
    env, path, start_pos, goal_pos = load_environment_and_path()
    
    # 如果没有路径，运行规划
    if path is None:
        path, start_pos, goal_pos = run_path_planning(env)
    
    if path is None:
        print("  错误: 无法获取路径，跳过噪声可视化")
        return
    
    print()
    
    # 2. 设置飞行剖面
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=500.0,
        approach_distance_threshold=500.0,
        climb_surface_slope=0.1,
        approach_surface_slope=0.083
    )
    
    print(f"✓ 飞行剖面设置完成")
    print(f"  巡航层: {flight_profile.cruise_min}m - {flight_profile.cruise_max}m")
    print()
    
    # 3. 设置噪声模型（使用更合理的70dB，符合实际eVTOL噪声水平）
    # 根据实际eVTOL: Joby S4~65dB, Volocopter~60dB, EHang~70dB @ 100m
    noise_model = NoiseModel(
        env=env,
        source_level=70.0,      # 源头噪声级 70dB @ 100m（更符合实际）
        reference_distance=100.0
    )
    
    print(f"✓ 噪声模型设置完成")
    print(f"  源头噪声级: {noise_model.source.source_level} dB(A)")
    print(f"  参考距离: {noise_model.source.reference_distance} m")
    print(f"  干扰阈值: {noise_model.disturbance_threshold} dB(A)")
    print(f"  严重阈值: {noise_model.severe_threshold} dB(A)")
    print()
    
    # 4. 生成噪声可视化
    print("-" * 70)
    print("生成噪声可视化图表...")
    print("-" * 70)
    
    visualize_noise(
        env=env,
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        noise_model=noise_model,
        flight_profile=flight_profile,
        output_dir='evtol_v7_results'
    )
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    generate_noise_visualizations()
