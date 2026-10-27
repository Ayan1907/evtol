#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声约束可视化模块
生成噪声传播热力图和沿路径的噪声变化曲线图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import math
import json
from typing import List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


class NoiseVisualizer:
    """噪声约束可视化类"""
    
    def __init__(self, env, noise_model):
        """
        初始化噪声可视化器
        
        参数:
            env: 城市环境对象
            noise_model: 噪声模型对象
        """
        self.env = env
        self.noise_model = noise_model
    
    def plot_noise_heatmap(self, path: List, start_pos: tuple, goal_pos: tuple,
                           filename='v6_noise_heatmap.png', grid_spacing=20,
                           time_hour=10):
        """
        绘制噪声传播热力图（显示路径上的最大噪声影响）
        
        包含:
        - 噪声影响热力图（显示每个网格点的最大噪声级）
        - 建筑物（按类型分色）
        - 飞行路径（蓝色线条）
        - 起降场位置标注
        - 噪声阈值标注（60dB/75dB等高线）
        """
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # 生成噪声地图
        grid_x = np.arange(0, int(self.env.size_x) + grid_spacing, grid_spacing)
        grid_y = np.arange(0, int(self.env.size_y) + grid_spacing, grid_spacing)
        
        # 初始化噪声地图为负无穷，后续取最大值
        noise_map = np.full((len(grid_x), len(grid_y)), -np.inf, dtype=np.float32)
        
        # 对路径上的每个点计算噪声贡献，取最大值而非累加
        for i, path_point in enumerate(path):
            x, y, z = path_point
            
            # 计算该点正下方地面的噪声级
            sound_level = self.noise_model._calculate_sound_level(z, y, x)
            
            # 计算影响半径
            impact_radius = self.noise_model._calculate_impact_radius(sound_level)
            
            if impact_radius > 0:
                # 找到影响范围内的网格点
                for ix, gx in enumerate(grid_x):
                    for iy, gy in enumerate(grid_y):
                        dist = math.sqrt((gx - x)**2 + (gy - y)**2)
                        
                        if dist <= impact_radius:
                            # 距离衰减（20*log10(d/ref)）
                            if dist > 0:
                                distance_loss = 20 * math.log10(dist / z)  # 使用高度作为参考距离
                            else:
                                distance_loss = 0
                            
                            # 该点的噪声级
                            point_noise = sound_level - distance_loss
                            
                            # 取最大值（而非累加）
                            noise_map[ix, iy] = max(noise_map[ix, iy], point_noise)
        
        # 将-inf替换为最小显示值
        noise_map[noise_map == -np.inf] = 40.0
        noise_map = np.clip(noise_map, 40, 90)  # 限制在40-90dB范围
        
        print(f"  噪声范围: [{noise_map.min():.2f}, {noise_map.max():.2f}] dB")
        
        # 绘制热力图（绿色=低，黄色=中，红色=高）
        # 颜色范围：50dB(绿) -> 60dB(黄) -> 70dB(橙) -> 80dB(红) -> 90dB(深红)
        colors_list = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000']
        custom_cmap = LinearSegmentedColormap.from_list('noise', colors_list, N=100)
        
        im = ax.imshow(noise_map, origin='lower', cmap=custom_cmap,
                      extent=[0, self.env.size_y, 0, self.env.size_x],
                      vmin=50, vmax=90, alpha=0.7, zorder=1)
        
        # 绘制建筑物
        for building in self.env.buildings:
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
        
        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            ax.plot(x_coords, y_coords, color='blue',
                   linewidth=4, label='飞行路径', alpha=1.0, zorder=100)
            
            # 标记起点和终点
            ax.scatter([path[0][1]], [path[0][0]], color='green', s=300,
                      marker='o', label='起点', zorder=101, edgecolors='white', linewidth=3)
            ax.scatter([path[-1][1]], [path[-1][0]], color='red', s=300,
                      marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('噪声级 dB(A)', fontsize=14)
        
        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', alpha=0.7, label='低噪声 (50-60dB)'),
            Patch(facecolor='#f1c40f', alpha=0.7, label='中等噪声 (60-70dB)'),
            Patch(facecolor='#e67e22', alpha=0.7, label='高噪声 (70-80dB)'),
            Patch(facecolor='#8b0000', alpha=0.7, label='严重噪声 (>80dB)'),
            Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='住宅建筑'),
            Patch(facecolor='green', edgecolor='black', alpha=0.8, label='公共建筑'),
            Patch(facecolor='red', edgecolor='black', alpha=0.8, label='政府建筑'),
            plt.Line2D([0], [0], color='blue', linewidth=4, label='飞行路径')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 bbox_to_anchor=(1.32, 1))
        
        ax.set_xlim(0, self.env.size_y)
        ax.set_ylim(0, self.env.size_x)
        ax.set_xlabel('Y (米)', fontsize=14)
        ax.set_ylabel('X (米)', fontsize=14)
        ax.set_title(f'噪声传播热力图 (时段: {time_hour}:00)\n'
                    f'绿色: 低噪声 | 黄色: 中等噪声 | 橙红: 高噪声 | 深红: 严重噪声',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  噪声传播热力图已保存: {filename}")
        plt.close()
    
    def plot_noise_along_path(self, path: List, start_pos: tuple, goal_pos: tuple,
                              flight_profile=None, filename='v6_noise_along_path.png',
                              current_hour=10, speed=20.0):
        """
        绘制沿路径的噪声变化曲线图
        
        包含:
        - 噪声级变化曲线（沿路径）
        - 噪声阈值线（60dB干扰线、75dB严重干扰线）
        - 高度变化曲线（副轴）
        - 飞行阶段标注
        - 超标区域标注
        """
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # 计算水平距离
        distances = np.cumsum([0] + [
            np.sqrt((path[i][0]-path[i-1][0])**2 +
                   (path[i][1]-path[i-1][1])**2)
            for i in range(1, len(path))
        ])
        
        # 计算每个路径点的噪声级
        noise_levels = []
        heights = [p[2] for p in path]
        
        for point in path:
            x, y, z = point
            noise_level = self.noise_model.calculate_noise_level(x, y, z, speed)
            noise_levels.append(noise_level)
        
        noise_levels = np.array(noise_levels)
        
        # 绘制噪声曲线
        line1 = ax1.plot(distances, noise_levels, color='red', linewidth=2.5,
                        label='噪声级', alpha=0.9, zorder=10)
        
        # 绘制阈值线
        ax1.axhline(y=60, color='orange', linestyle='--', linewidth=2,
                   label='干扰阈值 (60dB)', alpha=0.8)
        ax1.axhline(y=75, color='darkred', linestyle='--', linewidth=2,
                   label='严重干扰阈值 (75dB)', alpha=0.8)
        
        # 标注超标区域
        violation_indices = np.where(noise_levels > 60)[0]
        if len(violation_indices) > 0:
            for idx in violation_indices:
                if idx == 0 or noise_levels[idx-1] <= 60:  # 开始点
                    ax1.scatter(distances[idx], noise_levels[idx],
                              color='orange', s=100, marker='x', zorder=15,
                              label='超60dB' if idx == violation_indices[0] else '')
        
        severe_violation_indices = np.where(noise_levels > 75)[0]
        if len(severe_violation_indices) > 0:
            for idx in severe_violation_indices:
                if idx == 0 or noise_levels[idx-1] <= 75:
                    ax1.scatter(distances[idx], noise_levels[idx],
                              color='darkred', s=100, marker='X', zorder=15,
                              label='超75dB' if idx == severe_violation_indices[0] else '')
        
        ax1.set_xlabel('水平距离 (米)', fontsize=13)
        ax1.set_ylabel('噪声级 dB(A)', fontsize=13, color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.set_ylim(50, 90)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11)
        
        # 创建副轴绘制高度
        ax2 = ax1.twinx()
        line2 = ax2.plot(distances, heights, color='blue', linewidth=2.5,
                        label='高度', alpha=0.7, linestyle='--', zorder=5)
        ax2.set_ylabel('高度 (米)', fontsize=13, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(50, 200)
        
        # 绘制巡航层范围
        if flight_profile:
            ax2.fill_between(distances,
                          [flight_profile.cruise_min]*len(distances),
                          [flight_profile.cruise_max]*len(distances),
                          color='green', alpha=0.1, label='巡航层范围')
        
        # 标注飞行阶段
        if flight_profile:
            phase_colors = {'climb': 'green', 'cruise': 'blue', 'approach': 'red'}
            current_phase = None
            phase_start = 0
            
            for i, p in enumerate(path):
                phase = flight_profile.get_flight_phase(
                    (p[0], p[1], p[2]), start_pos, goal_pos
                )
                if phase != current_phase:
                    if current_phase is not None and i > phase_start:
                        ax1.axvspan(distances[phase_start], distances[i-1],
                                  alpha=0.1, color=phase_colors[current_phase])
                    current_phase = phase
                    phase_start = i
            
            # 最后一个阶段
            if current_phase is not None:
                ax1.axvspan(distances[phase_start], distances[-1],
                          alpha=0.1, color=phase_colors[current_phase])
            
            # 添加阶段标签
            from matplotlib.patches import Patch
            phase_patches = [
                Patch(facecolor='green', alpha=0.15, label='起升阶段'),
                Patch(facecolor='blue', alpha=0.15, label='巡航阶段'),
                Patch(facecolor='red', alpha=0.15, label='下降阶段'),
                Patch(facecolor='green', alpha=0.1, label='巡航层范围')
            ]
            ax1.legend(handles=line1 + line2 + phase_patches,
                      loc='upper right', fontsize=10)
        
        # 统计信息文本框
        avg_noise = np.mean(noise_levels)
        max_noise = np.max(noise_levels)
        min_noise = np.min(noise_levels)
        exceed_60 = np.sum(noise_levels > 60)
        exceed_75 = np.sum(noise_levels > 75)
        
        stats_text = (
            f'统计信息:\n'
            f'平均噪声: {avg_noise:.2f} dB\n'
            f'峰值噪声: {max_noise:.2f} dB\n'
            f'最小噪声: {min_noise:.2f} dB\n'
            f'超60dB点数: {exceed_60}/{len(path)}\n'
            f'超75dB点数: {exceed_75}/{len(path)}'
        )
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_title(f'沿路径噪声变化曲线 (时段: {current_hour}:00)\n'
                      f'高度与噪声级对比',
                      fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  沿路径噪声变化曲线图已保存: {filename}")
        plt.close()
