#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6噪声约束可视化测试（修正版）
使用更合理的噪声参数
"""
import json
import zhplot
import numpy as np
from evtol_path_planning_v6 import CityEnvironment, FlightProfile
from evtol_path_planning_v3_standard import Vertiport
from noise_model import NoiseModel
from evtol_noise_visualizer import NoiseVisualizer


def load_environment_and_path():
    """加载环境和路径数据"""
    
    # 1. 创建环境并从JSON加载
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    env.load_from_json('test_map_v3_compatible.json')
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
    """
    设置噪声模型（使用更合理的参数）
    
    根据实际eVTOL噪声水平调整：
    - Joby S4: 约65dB @ 100m
    - Volocopter VoloCity: 约60dB @ 100m  
    - EHang 216: 约70dB @ 100m
    
    使用70dB @ 100m作为合理参考值
    """
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
    
    return noise_model


def main():
    print("=" * 70)
    print("V6噪声约束可视化测试（修正版 - 使用合理噪声参数）")
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
    
    # 4. 验证噪声水平
    print("验证噪声计算...")
    noise_levels = []
    heights = []
    for point in path:
        x, y, z = point
        noise_level = noise_model.calculate_noise_level(x, y, z, speed=20.0)
        noise_levels.append(noise_level)
        heights.append(z)
    
    noise_levels = np.array(noise_levels)
    print(f"  沿路径噪声统计:")
    print(f"    平均噪声: {np.mean(noise_levels):.2f} dB")
    print(f"    峰值噪声: {np.max(noise_levels):.2f} dB")
    print(f"    最小噪声: {np.min(noise_levels):.2f} dB")
    print(f"    超过60dB的点: {np.sum(noise_levels > 60)}")
    print(f"    超过75dB的点: {np.sum(noise_levels > 75)}")
    print()
    
    # 5. 生成可视化图表
    print("-" * 70)
    print("生成噪声可视化图表...")
    print("-" * 70)
    print()
    
    visualizer = NoiseVisualizer(env, noise_model)
    
    # 生成热力图
    print("1. 生成噪声传播热力图...")
    visualizer.plot_noise_heatmap(
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        filename='evtol_v6_results/v6_noise_heatmap_corrected.png',
        grid_spacing=20,
        time_hour=10
    )
    
    # 生成沿路径噪声变化曲线
    print()
    print("2. 生成沿路径噪声变化曲线图...")
    visualizer.plot_noise_along_path(
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        flight_profile=flight_profile,
        filename='evtol_v6_results/v6_noise_along_path_corrected.png',
        current_hour=10,
        speed=20.0
    )
    
    print()
    print("=" * 70)
    print("✓ 噪声可视化完成!")
    print("=" * 70)
    print()
    print("生成的文件:")
    print("  1. evtol_v6_results/v6_noise_heatmap_corrected.png - 噪声传播热力图（修正版）")
    print("  2. evtol_v6_results/v6_noise_along_path_corrected.png - 沿路径噪声变化曲线图（修正版）")


if __name__ == "__main__":
    main()
