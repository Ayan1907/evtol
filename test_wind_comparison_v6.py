#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风场对轨迹影响对比分析
生成三张图：
1. 无风场 - 路径偏向最短距离
2. 标准风场 - 路径考虑风向
3. 改进风场 - 路径考虑空间相关风场
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from evtol_path_planning_v6 import create_v6_planner, smooth_path_nodes
from v6_visualizer import V6Visualizer


def create_planner_with_config(map_file, wind_speed, wind_direction,
                                 cruise_min, cruise_max, climb_distance, approach_distance,
                                 use_improved_wind, alpha5):
    """
    创建配置的规划器
    """
    from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
    from evtol_path_planning_v4 import DynamicEnvironment, WindField
    from evtol_path_planning_v6 import LowCarbonAStarV6, FlightProfile

    # 1. 加载地图
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json(map_file)
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])

    # 2. 创建标准参数
    standard = EVTOLStandard()
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)

    # 3. 创建动态环境
    dynamic_env = DynamicEnvironment(env)

    # 添加风场
    if use_improved_wind:
        # 使用改进风场
        from improved_wind_field import ImprovedWindField
        wind_field = ImprovedWindField(
            base_wind_speed=wind_speed,
            base_wind_direction=wind_direction,
            gust_amplitude=3.0,
            gust_frequency=0.05,
            wind_variation_amplitude=15.0,
            spatial_scale=0.005,
            building_wake_factor=0.30,
            building_wake_length=300.0,
            building_channel_factor=2.00
        )
        wind_field.set_environment(env)
    else:
        # 使用标准风场
        wind_field = WindField(
            env,
            base_wind_speed=wind_speed,
            base_wind_direction=wind_direction,
            turbulence_intensity=0.1,
            time_variation=True
        )

    dynamic_env.set_wind_field(wind_field)

    # 4. 创建飞行剖面配置
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=cruise_min,
        cruise_max=cruise_max,
        climb_distance_threshold=climb_distance,
        approach_distance_threshold=approach_distance
    )

    # 5. 创建规划器
    planner = LowCarbonAStarV6(
        env, standard, start_vertiport, goal_vertiport,
        dynamic_env, flight_profile,
        use_improved_wind=use_improved_wind,
        alpha5=alpha5
    )

    return planner, flight_profile, start_pos, goal_pos


def run_wind_comparison():
    """
    运行风场对比实验，生成三张对比图
    """
    # 测试配置
    map_file = 'test_map_standard.json'
    start_pos = (250, 250, 80)
    goal_pos = (2250, 2250, 100)
    cruise_min = 120.0
    cruise_max = 180.0
    climb_distance = 400.0
    approach_distance = 400.0

    # 风场配置
    wind_speed = 8.0
    wind_direction = 315.0  # 西北风

    results = {}

    print("=" * 60)
    print("风场对轨迹影响对比分析")
    print("=" * 60)

    # 实验1: 无风场 (alpha5 = 0)
    print("\n[1/3] 无风场测试...")
    planner1, flight_profile1, start_pos1, goal_pos1 = create_planner_with_config(
        map_file=map_file,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        cruise_min=cruise_min,
        cruise_max=cruise_max,
        climb_distance=climb_distance,
        approach_distance=approach_distance,
        use_improved_wind=False,
        alpha5=0.0  # 无风场影响
    )

    path1 = planner1.find_path(max_iterations=200000, current_time=0.0)
    if path1:
        print(f"  ✓ 找到路径，节点数: {len(path1)}")
        # 平滑路径
        path1 = smooth_path_nodes(path1, start_pos1, goal_pos1, flight_profile1, planner1)
        print(f"  ✓ 平滑后节点数: {len(path1)}")
        results['no_wind'] = path1
    else:
        print("  ✗ 未找到路径")
        return

    # 实验2: 标准风场 (alpha5 = 0.15)
    print("\n[2/3] 标准风场测试...")
    planner2, flight_profile2, start_pos2, goal_pos2 = create_planner_with_config(
        map_file=map_file,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        cruise_min=cruise_min,
        cruise_max=cruise_max,
        climb_distance=climb_distance,
        approach_distance=approach_distance,
        use_improved_wind=False,  # 使用标准风场
        alpha5=0.15  # 标准风场权重
    )

    path2 = planner2.find_path(max_iterations=200000, current_time=0.0)
    if path2:
        print(f"  ✓ 找到路径，节点数: {len(path2)}")
        # 平滑路径
        path2 = smooth_path_nodes(path2, start_pos2, goal_pos2, flight_profile2, planner2)
        print(f"  ✓ 平滑后节点数: {len(path2)}")
        results['standard_wind'] = path2
    else:
        print("  ✗ 未找到路径")
        return

    # 实验3: 改进风场 (alpha5 = 0.25)
    print("\n[3/3] 改进风场测试...")
    planner3, flight_profile3, start_pos3, goal_pos3 = create_planner_with_config(
        map_file=map_file,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        cruise_min=cruise_min,
        cruise_max=cruise_max,
        climb_distance=climb_distance,
        approach_distance=approach_distance,
        use_improved_wind=True,  # 使用改进风场
        alpha5=0.25  # 改进风场权重
    )

    path3 = planner3.find_path(max_iterations=200000, current_time=0.0)
    if path3:
        print(f"  ✓ 找到路径，节点数: {len(path3)}")
        # 平滑路径
        path3 = smooth_path_nodes(path3, start_pos3, goal_pos3, flight_profile3, planner3)
        print(f"  ✓ 平滑后节点数: {len(path3)}")
        results['improved_wind'] = path3
    else:
        print("  ✗ 未找到路径")
        return

    print("\n" + "=" * 60)
    print("生成对比图...")
    print("=" * 60)

    # 生成对比图
    plot_wind_comparison(
        planner1.env,
        path1,
        path2,
        path3,
        start_pos,
        goal_pos
    )

    # 保存结果数据
    save_comparison_data(results, start_pos, goal_pos)

    print("\n✓ 对比图已保存到: evtol_v6_results/v6_wind_comparison.png")
    print("✓ 对比数据已保存到: evtol_v6_results/v6_wind_comparison.json")


def plot_wind_comparison(env, path_no_wind, path_standard, path_improved, start, goal):
    """
    绘制风场对比图 - 三张图并排显示
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 获取建筑物
    buildings = env.buildings if hasattr(env, 'buildings') else []

    # 图1: 无风场
    ax1 = axes[0]
    ax1.set_title('无风场 (α5=0)', fontsize=14, fontweight='bold', pad=10)
    plot_path_on_ax(ax1, buildings, path_no_wind, start, goal, '无风场路径')

    # 图2: 标准风场
    ax2 = axes[1]
    ax2.set_title('标准风场 (α5=0.15)', fontsize=14, fontweight='bold', pad=10)
    plot_path_on_ax(ax2, buildings, path_standard, start, goal, '标准风场路径')

    # 图3: 改进风场
    ax3 = axes[2]
    ax3.set_title('改进风场 (α5=0.25)', fontsize=14, fontweight='bold', pad=10)
    plot_path_on_ax(ax3, buildings, path_improved, start, goal, '改进风场路径')

    # 添加风向指示
    for ax in axes:
        add_wind_direction_indicator(ax, 315.0)  # 西北风

    plt.tight_layout()

    # 保存图片
    output_dir = Path('evtol_v6_results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'v6_wind_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_path_on_ax(ax, buildings, path, start, goal, label):
    """
    在子图上绘制路径
    """
    # 绘制建筑物
    for building in buildings:
        x_center = building['x']
        y_center = building['y']
        width = building['length']
        depth = building['width']

        rect = plt.Rectangle(
            (x_center - width/2, y_center - depth/2),
            width, depth,
            facecolor='gray', edgecolor='black', alpha=0.3
        )
        ax.add_patch(rect)

    # 绘制路径
    if path:
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label=label, alpha=0.8)

        # 绘制起点和终点
        ax.scatter([start[0]], [start[1]], c='green', s=100, marker='s',
                   label='起点', edgecolors='black', linewidths=2, zorder=10)
        ax.scatter([goal[0]], [goal[1]], c='red', s=100, marker='*',
                   label='终点', edgecolors='black', linewidths=2, zorder=10)

        # 添加路径信息
        total_dist = sum(np.sqrt(
            (path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2
        ) for i in range(len(path)-1))
        avg_height = np.mean([node.z for node in path])

        info_text = f'距离: {total_dist:.1f}m\n平均高度: {avg_height:.1f}m'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 2500)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def add_wind_direction_indicator(ax, wind_direction):
    """
    添加风向指示器
    """
    # 在右上角添加风向箭头
    center_x, center_y = 2300, 2300
    arrow_length = 150

    # 风向角度转换为弧度
    angle_rad = np.radians(wind_direction)

    # 箭头终点
    end_x = center_x + arrow_length * np.cos(angle_rad)
    end_y = center_y + arrow_length * np.sin(angle_rad)

    # 绘制箭头
    ax.arrow(center_x, center_y,
             end_x - center_x, end_y - center_y,
             head_width=30, head_length=40,
             fc='red', ec='red', linewidth=2, zorder=5)

    # 添加文字说明
    ax.text(center_x, center_y - 200,
            f'风向\n{wind_direction}°',
            ha='center', va='top',
            fontsize=9, color='red',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))


def save_comparison_data(results, start, goal):
    """
    保存对比数据到JSON文件
    """
    comparison_data = {
        'test_name': 'wind_field_comparison',
        'start': start,
        'goal': goal,
        'wind_direction': 315.0,
        'results': {}
    }

    for wind_type, path in results.items():
        if path:
            # 计算路径统计信息
            total_dist = sum(np.sqrt(
                (path[i+1].x - path[i].x)**2 + (path[i+1].y - path[i].y)**2
            ) for i in range(len(path)-1))

            heights = [node.z for node in path]
            avg_height = np.mean(heights)
            min_height = np.min(heights)
            max_height = np.max(heights)

            # 计算风场代价
            wind_costs = [node.g for node in path if hasattr(node, 'g')]
            total_wind_cost = sum(wind_costs) if wind_costs else 0

            comparison_data['results'][wind_type] = {
                'num_nodes': len(path),
                'total_distance': float(round(total_dist, 2)),
                'avg_height': float(round(avg_height, 2)),
                'height_range': [float(round(min_height, 2)), float(round(max_height, 2))],
                'total_cost': float(round(total_wind_cost, 2)),
                'path_points': [(float(node.x), float(node.y), float(node.z)) for node in path]
            }

    # 保存JSON文件
    output_dir = Path('evtol_v6_results')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'v6_wind_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    run_wind_comparison()
