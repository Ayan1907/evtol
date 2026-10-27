#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6版本高清热力图生成脚本
生成300DPI的高清噪声和风场代价热力图
"""

import sys
sys.path.append('/home/ayan1907/evtol')

from evtol_path_planning_v6 import LowCarbonAStarV6, CityEnvironment, ImprovedWindField, FlightProfile
from evtol_noise_visualizer import NoiseVisualizer
from noise_model import NoiseModel
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
import math


def load_environment():
    """加载环境"""
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    env.load_from_json('test_map_v3_compatible.json')
    print(f"✓ 地图加载完成: {len(env.buildings)} 栋建筑物")
    print(f"  地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    return env


def load_path():
    """加载路径"""
    with open('evtol_v6_results/v6_main_result.json', 'r') as f:
        result_data = json.load(f)
    path = result_data['path']
    start_pos = tuple(result_data['start'])
    goal_pos = tuple(result_data['goal'])
    print(f"✓ 路径加载完成: {len(path)} 个节点")
    return path, start_pos, goal_pos


def plot_noise_heatmap_hd(env, path, noise_model, start_pos, goal_pos, 
                           filename='evtol_v6_results/v6_noise_heatmap_hd.png',
                           grid_spacing=10, time_hour=10, dpi=300):
    """绘制高清噪声热力图"""
    print(f"\n生成高清噪声热力图 (DPI={dpi}, 网格间距={grid_spacing}m)...")
    
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # 生成噪声地图
    grid_x = np.arange(0, int(env.size_x) + grid_spacing, grid_spacing)
    grid_y = np.arange(0, int(env.size_y) + grid_spacing, grid_spacing)
    print(f"  网格点数量: {len(grid_x)} x {len(grid_y)} = {len(grid_x)*len(grid_y)}")
    
    # 初始化噪声地图
    noise_map = np.full((len(grid_x), len(grid_y)), -np.inf, dtype=np.float32)
    
    # 对路径上的每个点计算噪声贡献
    for i, path_point in enumerate(path):
        x, y, z = path_point
        sound_level = noise_model._calculate_sound_level(z, y, x)
        impact_radius = noise_model._calculate_impact_radius(sound_level)
        
        if impact_radius > 0:
            for ix, gx in enumerate(grid_x):
                for iy, gy in enumerate(grid_y):
                    dist = math.sqrt((gx - x)**2 + (gy - y)**2)
                    if dist <= impact_radius:
                        if dist > 0:
                            distance_loss = 20 * math.log10(dist / z)
                        else:
                            distance_loss = 0
                        point_noise = sound_level - distance_loss
                        noise_map[ix, iy] = max(noise_map[ix, iy], point_noise)
    
    # 处理无效值
    noise_map[noise_map == -np.inf] = 40.0
    noise_map = np.clip(noise_map, 40, 90)
    
    print(f"  噪声范围: [{noise_map.min():.2f}, {noise_map.max():.2f}] dB")
    
    # 绘制热力图
    colors_list = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000']
    custom_cmap = LinearSegmentedColormap.from_list('noise', colors_list, N=100)
    
    im = ax.imshow(noise_map, origin='lower', cmap=custom_cmap,
                  extent=[0, env.size_y, 0, env.size_x],
                  vmin=50, vmax=90, alpha=0.7, zorder=1)
    
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
            facecolor=color, edgecolor='black', alpha=0.8, linewidth=1.5, zorder=10
        )
        ax.add_patch(rect)
    
    # 绘制路径
    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]
        ax.plot(x_coords, y_coords, color='blue',
               linewidth=5, label='飞行路径', alpha=1.0, zorder=100)
        
        # 标记起点和终点
        ax.scatter([path[0][1]], [path[0][0]], color='green', s=400,
                  marker='o', label='起点', zorder=101, edgecolors='white', linewidth=3)
        ax.scatter([path[-1][1]], [path[-1][0]], color='red', s=400,
                  marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('噪声级 dB(A)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    # 图例
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.7, label='低噪声 (50-60dB)'),
        Patch(facecolor='#f1c40f', alpha=0.7, label='中等噪声 (60-70dB)'),
        Patch(facecolor='#e67e22', alpha=0.7, label='高噪声 (70-80dB)'),
        Patch(facecolor='#8b0000', alpha=0.7, label='严重噪声 (>80dB)'),
        Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='住宅建筑'),
        Patch(facecolor='green', edgecolor='black', alpha=0.8, label='公共建筑'),
        Patch(facecolor='red', edgecolor='black', alpha=0.8, label='政府建筑'),
        plt.Line2D([0], [0], color='blue', linewidth=5, label='飞行路径')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             bbox_to_anchor=(1.35, 1), framealpha=0.95)
    
    ax.set_xlim(0, env.size_y)
    ax.set_ylim(0, env.size_x)
    ax.set_xlabel('Y (米)', fontsize=16)
    ax.set_ylabel('X (米)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(f'eVTOL噪声传播热力图 (时段: {time_hour}:00)\n'
                f'绿色: 低噪声 | 黄色: 中等 | 橙红: 高噪声 | 深红: 严重',
                fontsize=18, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 高清噪声热力图已保存: {filename}")
    plt.close()


def plot_wind_cost_heatmap_hd(env, path, wind_field, 
                               filename='evtol_v6_results/v6_wind_cost_heatmap_hd.png',
                               height=100, time_sec=0.0, dpi=300):
    """绘制高清风场代价热力图"""
    print(f"\n生成高清风场代价热力图 (DPI={dpi})...")
    
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # 生成网格
    x_range = np.arange(0, int(env.size_x) + 10, 10)
    y_range = np.arange(0, int(env.size_y) + 10, 10)
    print(f"  网格点数量: {len(x_range)} x {len(y_range)}")
    
    # 计算路径方向
    if path and len(path) > 1:
        start_x, start_y, _ = path[0]
        end_x, end_y, _ = path[-1]
        dx_path = end_x - start_x
        dy_path = end_y - start_y
        path_length = math.sqrt(dx_path**2 + dy_path**2)
        if path_length > 0:
            path_dir = (dx_path / path_length, dy_path / path_length, 0)
        else:
            path_dir = (1, 0, 0)
    else:
        path_dir = (1, 1, 0)
    
    # 计算风场代价
    cost_grid = np.zeros((len(x_range), len(y_range)), dtype=float)
    cost_min = float('inf')
    cost_max = float('-inf')
    
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            cost = wind_field.calculate_wind_cost(x, y, height, path_dir, time_sec)
            cost_grid[i, j] = cost
            cost_min = min(cost_min, cost)
            cost_max = max(cost_max, cost)
    
    print(f"  代价范围: [{cost_min:.4f}, {cost_max:.4f}]")
    
    # 绘制热力图
    colors_list = ['#2ecc71', '#3498db', '#e74c3c']
    custom_cmap = LinearSegmentedColormap.from_list('wind_cost', colors_list, N=100)
    
    im = ax.imshow(cost_grid, origin='lower', cmap=custom_cmap,
                  extent=[0, env.size_y, 0, env.size_x],
                  vmin=cost_min, vmax=cost_max, alpha=0.6, zorder=1)
    
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
            facecolor=color, edgecolor='black', alpha=1.0, linewidth=1.5, zorder=10
        )
        ax.add_patch(rect)
    
    # 绘制路径
    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]
        ax.plot(x_coords, y_coords, color='blue',
               linewidth=5, label='飞行路径', alpha=1.0, zorder=100)
        
        ax.scatter([path[0][1]], [path[0][0]], color='green', s=400,
                  marker='o', label='起点', zorder=101, edgecolors='white', linewidth=3)
        ax.scatter([path[-1][1]], [path[-1][0]], color='red', s=400,
                  marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('风场代价 (绿色=低, 蓝色=中, 红色=高)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # 图例
    legend_elements = [
        Patch(facecolor='lime', alpha=0.6, label='低代价 (顺风)'),
        Patch(facecolor='#3498db', alpha=0.6, label='中等代价 (侧风)'),
        Patch(facecolor='red', alpha=0.6, label='高代价 (逆风)'),
        Patch(facecolor='blue', edgecolor='black', alpha=1.0, label='住宅建筑'),
        Patch(facecolor='green', edgecolor='black', alpha=1.0, label='公共建筑'),
        Patch(facecolor='red', edgecolor='black', alpha=1.0, label='政府建筑'),
        plt.Line2D([0], [0], color='blue', linewidth=5, label='飞行路径')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             bbox_to_anchor=(1.35, 1), framealpha=0.95)
    
    ax.set_xlim(0, env.size_y)
    ax.set_ylim(0, env.size_x)
    ax.set_xlabel('Y (米)', fontsize=16)
    ax.set_ylabel('X (米)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(f'eVTOL风场代价热力图 (z={height}m, t={time_sec:.1f}s)\n'
                f'绿色: 顺风 (好) | 蓝色: 侧风 (可) | 红色: 逆风 (差)',
                fontsize=18, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 高清风场代价热力图已保存: {filename}")
    plt.close()


def main():
    print("=" * 70)
    print("V6版本高清热力图生成")
    print("=" * 70)
    print()
    
    # 加载环境
    env = load_environment()
    print()
    
    # 加载路径
    path, start_pos, goal_pos = load_path()
    print()
    
    # 创建噪声模型
    noise_model = NoiseModel(
        env=env,
        source_level=85.0,
        reference_distance=100.0
    )
    print("✓ 噪声模型创建完成")
    
    # 创建风场
    wind_field = ImprovedWindField(
        base_wind_speed=8.0,
        base_wind_direction=135.0,
        turbulence_intensity=0.15
    )
    wind_field.set_environment(env)
    print("✓ 风场创建完成")
    print()
    
    # 生成高清噪声热力图
    plot_noise_heatmap_hd(
        env, path, noise_model, start_pos, goal_pos,
        filename='evtol_v6_results/v6_noise_heatmap_hd.png',
        grid_spacing=10, time_hour=10, dpi=300
    )
    
    # 生成高清风场代价热力图
    plot_wind_cost_heatmap_hd(
        env, path, wind_field,
        filename='evtol_v6_results/v6_wind_cost_heatmap_hd.png',
        height=100, time_sec=0.0, dpi=300
    )
    
    print()
    print("=" * 70)
    print("✓ 所有高清热力图生成完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. evtol_v6_results/v6_noise_heatmap_hd.png - 噪声传播热力图 (300DPI)")
    print("  2. evtol_v6_results/v6_wind_cost_heatmap_hd.png - 风场代价热力图 (300DPI)")


if __name__ == "__main__":
    main()
