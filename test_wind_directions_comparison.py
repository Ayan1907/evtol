#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风场热力图对比 - 三种不同风向
使用V5.1的改进风场模型
"""

import matplotlib
matplotlib.use('Agg')
import zhplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
import math
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import DynamicEnvironment
from evtol_path_planning_v6 import LowCarbonAStarV6, FlightProfile, smooth_path_nodes
from improved_wind_field import ImprovedWindField

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def create_wind_field_with_direction(env, wind_speed, wind_direction):
    """
    创建指定风向的改进风场
    """
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
    return wind_field


def plot_wind_cost_heatmap_single(ax, env, wind_field, wind_direction, title, path_dir=None, planned_path=None):
    """
    在单个子图上绘制风场代价热力图
    """
    # 生成网格（20米间隔，加快速度）
    x_range = np.arange(0, int(env.size_x) + 20, 20)
    y_range = np.arange(0, int(env.size_y) + 20, 20)

    # 默认路径方向：从左下到右上（东北方向）
    if path_dir is None:
        path_dir = (1, 1, 0)

    # 计算风场代价
    cost_grid = np.zeros((len(x_range), len(y_range)), dtype=float)
    cost_min = float('inf')
    cost_max = float('-inf')

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            cost = wind_field.calculate_wind_cost(
                x, y, 120, path_dir, 0.0
            )
            cost_grid[i, j] = cost
            cost_min = min(cost_min, cost)
            cost_max = max(cost_max, cost)

    # 绘制热力图
    colors_list = ['#2ecc71', '#f1c40f', '#e74c3c']  # 绿 -> 黄 -> 红
    custom_cmap = LinearSegmentedColormap.from_list('wind_cost', colors_list, N=100)
    im = ax.imshow(cost_grid, origin='lower', cmap=custom_cmap,
                  extent=[0, env.size_y, 0, env.size_x],
                  vmin=cost_min, vmax=cost_max, alpha=0.7, zorder=1)

    # 绘制建筑物
    for building in env.buildings:
        color = building.get('color', 'gray')
        if color == 'gray':
            btype = building.get('btype', 1)
            color_map = {1: 'blue', 2: 'green', 3: 'red'}
            color = color_map.get(btype, 'gray')

        rect = Rectangle(
            (building['y'], building['x']),
            building['width'],
            building['length'],
            facecolor=color, edgecolor='black', alpha=0.8, linewidth=1, zorder=10
        )
        ax.add_patch(rect)

    # 绘制路径（如果有）
    if planned_path and len(planned_path) > 0:
        path_x = [p.x for p in planned_path]
        path_y = [p.y for p in planned_path]
        ax.plot(path_y, path_x, 'white', linewidth=3, alpha=0.9, zorder=30, label='规划路径')
        # 绘制路径点
        ax.scatter(path_y, path_x, c='white', s=30, alpha=0.7, zorder=31, edgecolors='black', linewidth=0.5)

    # 添加风向指示
    add_wind_direction_indicator(ax, wind_direction)

    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Y (m)', fontsize=10)
    ax.set_ylabel('X (m)', fontsize=10)
    ax.set_xlim(0, env.size_y)
    ax.set_ylim(0, env.size_x)
    ax.grid(True, alpha=0.2)

    return im, cost_min, cost_max


def add_wind_direction_indicator(ax, wind_direction):
    """
    添加风向指示器
    """
    center_x, center_y = 2300, 200  # 恢复原位置
    arrow_length = 100  # 恢复原长度

    angle_rad = np.radians(wind_direction)

    # 箭头终点
    end_x = center_x + arrow_length * np.cos(angle_rad)
    end_y = center_y + arrow_length * np.sin(angle_rad)

    # 绘制箭头
    ax.arrow(center_x, center_y,
             end_x - center_x, end_y - center_y,
             head_width=30, head_length=40,  # 恢复原大小
             fc='darkred', ec='darkred', linewidth=3, zorder=50)

    # 添加文字说明
    wind_name = {
        0: '东风',
        90: '南风',
        180: '西风',
        270: '北风',
        45: '东北风',
        135: '东南风',
        225: '西南风',
        315: '西北风'
    }.get(wind_direction, f'{wind_direction}°')

    ax.text(center_x, center_y - 150,  # 恢复原位置
            f'{wind_name}\n{wind_direction}°',
            ha='center', va='top',
            fontsize=10, color='darkred',  # 恢复原字体
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='darkred'))


def plan_path_with_wind(env, wind_speed, wind_direction, use_improved_wind=True):
    """
    使用指定风场进行路径规划
    """
    # 1. 创建标准参数
    standard = EVTOLStandard()
    start_pos = (250, 250, 80)
    goal_pos = (2250, 2250, 100)
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)

    # 2. 创建动态环境
    dynamic_env = DynamicEnvironment(env)

    # 3. 添加风场
    if use_improved_wind:
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
        from evtol_path_planning_v4 import WindField
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
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )

    # 5. 创建规划器
    planner = LowCarbonAStarV6(
        env, standard, start_vertiport, goal_vertiport,
        dynamic_env, flight_profile,
        use_improved_wind=use_improved_wind,
        alpha5=0.25
    )

    # 6. 规划路径
    print("  开始路径规划...")
    path = planner.find_path(max_iterations=200000, current_time=0.0)

    # 7. 平滑路径
    if path and len(path) > 0:
        print("  路径平滑优化...")
        path_before = len(path)
        path = smooth_path_nodes(path, start_pos, goal_pos, flight_profile, planner)
        print(f"  平滑完成: {path_before} -> {len(path)} 节点")

    return path


def plot_three_wind_directions_comparison():
    """
    绘制三种不同风向的风场热力图对比
    """
    # 加载地图
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_standard.json')

    # 三种风向
    wind_directions = [
        (315, '西北风 (315°)', '从西北吹向东南'),
        (90, '南风 (90°)', '从南吹向北'),
        (225, '西南风 (225°)', '从西南吹向东北')
    ]

    # 路径方向（东北方向）
    path_dir = (1, 1, 0)

    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    results = {}

    for idx, (wind_dir, title, desc) in enumerate(wind_directions):
        print(f"\n[{idx+1}/3] 生成 {title} 热力图...")

        # 创建风场
        wind_field = create_wind_field_with_direction(env, wind_speed=8.0, wind_direction=wind_dir)

        # 规划路径
        planned_path = plan_path_with_wind(env, wind_speed=8.0, wind_direction=wind_dir, use_improved_wind=True)
        print(f"  路径包含 {len(planned_path) if planned_path else 0} 个节点")

        # 绘制热力图
        ax = axes[idx]
        full_title = f'{title}\n{desc}'
        im, cost_min, cost_max = plot_wind_cost_heatmap_single(
            ax, env, wind_field, wind_dir, full_title, path_dir, planned_path
        )

        results[wind_dir] = {
            'title': title,
            'description': desc,
            'cost_min': cost_min,
            'cost_max': cost_max,
            'path_length': len(planned_path) if planned_path else 0
        }

        print(f"  代价范围: [{cost_min:.4f}, {cost_max:.4f}]")

    # 添加统一颜色条
    # 使用第一个热力图的im来创建颜色条，保持在右边不与图像重叠
    cbar_ax = fig.add_axes([1.00, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('风场代价 (绿色=低/顺风, 黄色=中/侧风, 红色=高/逆风)', fontsize=10)

    # 添加图例到颜色条右边
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='低代价 (顺风)'),
        Patch(facecolor='#f1c40f', alpha=0.7, label='中等代价 (侧风)'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='高代价 (逆风)'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='住宅建筑'),
        Patch(facecolor='green', edgecolor='black', alpha=0.8, label='公共建筑'),
        Patch(facecolor='red', edgecolor='black', alpha=0.8, label='政府建筑'),
        plt.Line2D([0], [0], color='white', linewidth=3, alpha=0.9, label='规划路径'),
    ]
    fig.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.04, 0.5), fontsize=9)  # 图例在颜色条右边

    plt.suptitle('不同风向的风场代价热力图对比 (改进风场模型)', fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])  # 调整布局

    # 保存图片
    output_dir = Path('evtol_v6_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'v6_wind_directions_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 热力图已保存到: {output_file}")

    # 保存数据
    save_wind_directions_comparison_data(results, path_dir)

    return results


def save_wind_directions_comparison_data(results, path_dir):
    """
    保存风向对比数据
    """
    comparison_data = {
        'test_name': 'wind_directions_comparison',
        'description': '三种不同风向的风场代价对比',
        'path_direction': path_dir,
        'wind_speed': 8.0,
        'wind_field_type': 'ImprovedWindField',
        'results': {
            str(wd): {
                'title': data['title'],
                'description': data['description'],
                'cost_min': data['cost_min'],
                'cost_max': data['cost_max'],
                'path_length': data['path_length']
            }
            for wd, data in results.items()
        }
    }

    # 保存JSON文件
    output_dir = Path('evtol_v6_results')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'v6_wind_directions_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 数据已保存到: {output_file}")


if __name__ == '__main__':
    print("=" * 70)
    print("风场热力图对比 - 三种不同风向")
    print("=" * 70)

    results = plot_three_wind_directions_comparison()

    print("\n" + "=" * 70)
    print("对比结果:")
    print("=" * 70)
    for wind_dir, data in results.items():
        print(f"\n{data['title']} ({data['description']}):")
        print(f"  代价范围: [{data['cost_min']:.4f}, {data['cost_max']:.4f}]")
        print(f"  代价变化: {data['cost_max'] - data['cost_min']:.4f}")
        print(f"  路径节点数: {data['path_length']}")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
