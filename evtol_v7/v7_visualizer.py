#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7版本可视化系统 - HV修复增强版
基于V6可视化,增强HV收敛曲线可视化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import json
import zhplot
from typing import List, Dict


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


class V7Visualizer:
    """V7版本可视化类 - 增强HV收敛曲线"""
    
    # ==================== V6国标合规可视化 ====================
    
    @staticmethod
    def plot_path_with_buildings(env, path: List, start_pos: tuple, goal_pos: tuple,
                                flight_profile=None, filename='evtol_v7_results/v7_path_with_buildings.png'):
        """绘制路径与建筑物（俯视图）"""
        fig, ax = plt.subplots(figsize=(16, 16))

        # 绘制建筑物
        for building in env.buildings:
            color = getattr(building, 'color', 'gray')
            if color == 'gray':
                btype = getattr(building, 'btype', 1)
                color_map = {1: 'blue', 2: 'green', 3: 'red'}
                color = color_map.get(btype, 'gray')

            rect = Rectangle(
                (building.y, building.x),
                building.width,
                building.length,
                facecolor=color, edgecolor='black', alpha=0.5, linewidth=0.8
            )
            ax.add_patch(rect)

            if building.height >= 100:
                ax.text(building.y + building.width/2,
                       building.x + building.length/2,
                       f"{int(building.height)}m",
                       ha='center', va='center', fontsize=7, color='white',
                       fontweight='bold', alpha=0.8)

        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            ax.plot(x_coords, y_coords, color='blue',
                   linewidth=3, label='V7路径', alpha=0.9, zorder=100)

            ax.scatter([start_pos[1]], [start_pos[0]], color='green', s=300,
                      marker='o', label='起点', zorder=101, edgecolors='white', linewidth=3)
            ax.scatter([goal_pos[1]], [goal_pos[0]], color='red', s=300,
                      marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)

            ofv_radius = flight_profile.ofv_height if flight_profile else 30.0
            ofv1 = plt.Circle((start_pos[1], start_pos[0]), ofv_radius,
                             color='green', alpha=0.2, label='OFV保护区(起点)')
            ofv2 = plt.Circle((goal_pos[1], goal_pos[0]), ofv_radius,
                             color='green', alpha=0.2)
            ax.add_patch(ofv1)
            ax.add_patch(ofv2)

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (m)', fontsize=14)
        ax.set_ylabel('X (m)', fontsize=14)
        ax.set_title('V7路径规划俯视图\n(符合GB 38031-2022国标)',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  俯视图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_3d_path_profile(env, path: List, start_pos: tuple, goal_pos: tuple,
                             flight_profile=None, filename='evtol_v7_results/v7_3d_path_profile.png'):
        """绘制3D路径剖面"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制建筑物线框
        for building in env.buildings:
            x = building.y
            y = building.x
            z = 0
            width = building.width
            length = building.length
            height = building.height

            verts = [
                [x, y, z], [x+width, y, z], [x+width, y+length, z], [x, y+length, z],
                [x, y, z]
            ]
            ax.plot([v[0] for v in verts], [v[1] for v in verts], [v[2] for v in verts],
                   'gray', linewidth=0.5, alpha=0.5)

            for h in [height/4, height/2, height*3/4]:
                verts_mid = [
                    [x, y, h], [x+width, y, h], [x+width, y+length, h], [x, y+length, h], [x, y, h]
                ]
                ax.plot([v[0] for v in verts_mid], [v[1] for v in verts_mid], [v[2] for v in verts_mid],
                       'gray', linewidth=0.3, alpha=0.3)

            ax.plot([x, x], [y, y], [0, height], 'gray', linewidth=0.5, alpha=0.5)
            ax.plot([x+width, x+width], [y, y], [0, height], 'gray', linewidth=0.5, alpha=0.5)
            ax.plot([x, x], [y+length, y+length], [0, height], 'gray', linewidth=0.5, alpha=0.5)
            ax.plot([x+width, x+width], [y+length, y+length], [0, height], 'gray', linewidth=0.5, alpha=0.5)

        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            z_coords = [p[2] for p in path]
            ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=3, label='路径')
            ax.scatter([start_pos[1]], [start_pos[0]], [start_pos[2]],
                      color='green', s=200, marker='o', label='起点')
            ax.scatter([goal_pos[1]], [goal_pos[0]], [goal_pos[2]],
                      color='red', s=200, marker='s', label='终点')

        # 绘制巡航层
        if flight_profile:
            ax.axhline(y=flight_profile.cruise_min, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=flight_profile.cruise_max, color='orange', linestyle='--', alpha=0.5)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('V7 3D路径剖面', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  3D剖面图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_height_profile(path: List, start_pos: tuple, goal_pos: tuple,
                           flight_profile=None, filename='evtol_v7_results/v7_height_profile.png'):
        """绘制高度剖面图"""
        fig, ax = plt.subplots(figsize=(16, 6))

        if path:
            distances = [0]
            heights = [path[0][2]]
            
            total_dist = 0
            for i in range(len(path) - 1):
                dx = path[i+1][0] - path[i][0]
                dy = path[i+1][1] - path[i][1]
                dz = path[i+1][2] - path[i][2]
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                total_dist += dist
                distances.append(total_dist)
                heights.append(path[i+1][2])

            ax.plot(distances, heights, 'b-', linewidth=2, label='路径高度')
            ax.scatter([0], [start_pos[2]], color='green', s=100, marker='o', label='起点')
            ax.scatter([distances[-1]], [goal_pos[2]], color='red', s=100, marker='s', label='终点')

            # 标记飞行阶段
            if flight_profile:
                phase_colors = {'climb': 'orange', 'cruise': 'green', 'approach': 'purple'}
                for i, p in enumerate(path):
                    phase = flight_profile.get_flight_phase(p, start_pos, goal_pos)
                    ax.scatter([distances[i]], [heights[i]], 
                              color=phase_colors.get(phase, 'gray'), s=30, alpha=0.5)

        ax.set_xlabel('累计距离 (m)', fontsize=12)
        ax.set_ylabel('高度 (m)', fontsize=12)
        ax.set_title('V7 高度剖面图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  高度剖面图已保存: {filename}")
        plt.close()

    # ==================== V7新增: HV收敛曲线可视化 ====================
    
    @staticmethod
    def plot_hv_convergence(hv_history: List[float], 
                           diversity_history: List[float] = None,
                           filename: str = 'evtol_v7_results/v7_hv_convergence.png'):
        """
        V7新增: 绘制HV收敛曲线
        
        显示:
        1. HV值随代数变化
        2. 多样性指标变化（可选）
        3. 固定参考点信息
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        generations = list(range(1, len(hv_history) + 1))
        
        # 子图1: HV收敛曲线
        ax1.plot(generations, hv_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.fill_between(generations, 0, hv_history, alpha=0.3)
        ax1.set_ylabel('超体积 (Hypervolume)', fontsize=12)
        ax1.set_title('V7 HV收敛曲线\n(使用固定参考点，跨代数可比较)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=hv_history[-1], color='red', linestyle='--', alpha=0.5, 
                   label=f'最终HV: {hv_history[-1]:.2f}')
        ax1.legend(loc='lower right')
        
        # 子图2: 多样性曲线
        if diversity_history:
            ax2.plot(generations, diversity_history, 'g-', linewidth=2, marker='s', markersize=4)
            ax2.fill_between(generations, 0, diversity_history, alpha=0.3, color='green')
            ax2.set_ylabel('多样性指标', fontsize=12)
            ax2.set_title('种群多样性变化', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            # 显示HV改进率
            hv_improvement = []
            for i in range(1, len(hv_history)):
                if hv_history[i-1] > 0:
                    imp = (hv_history[i] - hv_history[i-1]) / hv_history[i-1] * 100
                else:
                    imp = 0
                hv_improvement.append(imp)
            
            ax2.bar(list(range(2, len(hv_history) + 1)), hv_improvement, 
                   color=['green' if x > 0 else 'red' for x in hv_improvement], alpha=0.7)
            ax2.set_ylabel('HV改进率 (%)', fontsize=12)
            ax2.set_title('代际HV改进率', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('进化代数', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  HV收敛曲线已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_pareto_front_2d(pareto_front, objectives: List[str] = None,
                            filename: str = 'evtol_v7_results/v7_pareto_front_2d.png'):
        """
        V7: 绘制2D帕累托前沿切片
        
        objectives: 选择任意两个目标进行可视化
        """
        if objectives is None:
            objectives = ['energy', 'time', 'noise', 'safety', 'length']
        
        if len(objectives) < 2:
            print("  警告: 需要至少2个目标进行2D可视化")
            return
        
        n_objectives = len(pareto_front.individuals[0].objectives) if pareto_front.individuals else 0
        
        # 选择前4个目标进行2x2可视化
        obj_pairs = [
            (0, 1), (0, 2), (1, 2), (2, 3)
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        obj_names = ['能耗(kWh)', '时间(s)', '噪声', '安全性', '路径(m)']
        
        for idx, (i, j) in enumerate(obj_pairs):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            # 提取目标值（过滤无效个体）
            x_vals = [ind.objectives[i] for ind in pareto_front.individuals 
                     if hasattr(ind, 'objectives') and ind.objectives is not None and len(ind.objectives) > i]
            y_vals = [ind.objectives[j] for ind in pareto_front.individuals 
                     if hasattr(ind, 'objectives') and ind.objectives is not None and len(ind.objectives) > j]
            
            if x_vals and y_vals:
                ax.scatter(x_vals, y_vals, c='blue', s=50, alpha=0.7, label='Pareto解')
            
            ax.set_xlabel(obj_names[i], fontsize=10)
            ax.set_ylabel(obj_names[j], fontsize=10)
            ax.set_title(f'{obj_names[i]} vs {obj_names[j]}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('V7 帕累托前沿 (2D切片视图)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  帕累托前沿2D视图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_pareto_front_3d(pareto_front,
                            filename: str = 'evtol_v7_results/v7_pareto_front_3d.png'):
        """V7: 绘制3D帕累托前沿（能耗-时间-噪声）"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        obj_names = ['能耗', '时间', '噪声', '安全性', '路径']
        
        # 提取前3个目标（过滤无效个体）
        valid_inds = [ind for ind in pareto_front.individuals 
                     if hasattr(ind, 'objectives') and ind.objectives is not None and len(ind.objectives) >= 3]
        x_vals = [ind.objectives[0] for ind in valid_inds]  # energy
        y_vals = [ind.objectives[1] for ind in valid_inds]  # time
        z_vals = [ind.objectives[2] for ind in valid_inds]  # noise
        
        if x_vals:
            # 绘制帕累托前沿
            ax.scatter(x_vals, y_vals, z_vals, c='blue', s=80, alpha=0.7, label='Pareto解')
            
            # 绘制投影到xy平面
            for i in range(len(x_vals)):
                ax.plot([x_vals[i], x_vals[i]], [y_vals[i], y_vals[i]], [0, z_vals[i]], 
                       'gray', linewidth=0.3, alpha=0.3)
        
        ax.set_xlabel(f'{obj_names[0]} (kWh)', fontsize=11)
        ax.set_ylabel(f'{obj_names[1]} (s)', fontsize=11)
        ax.set_zlabel(f'{obj_names[2]}', fontsize=11)
        ax.set_title('V7 帕累托前沿 (3D视图: 能耗-时间-噪声)', fontsize=13, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  帕累托前沿3D视图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_objective_radar(objectives_values: Dict[str, float],
                             filename: str = 'evtol_v7_results/v7_objective_radar.png'):
        """
        V7: 绘制目标值雷达图
        
        用于比较不同解的目标值
        """
        categories = ['能耗', '时间', '噪声', '安全性', '路径']
        values = [
            objectives_values.get('energy', 0),
            objectives_values.get('time', 0),
            objectives_values.get('noise', 0),
            objectives_values.get('safety', 0),
            objectives_values.get('length', 0)
        ]
        
        # 归一化到0-1范围
        max_values = [10, 200, 1, 1, 5000]  # 预估最大值
        normalized = [v / m for v, m in zip(values, max_values)]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized = normalized + [normalized[0]]
        angles = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, normalized, color='blue', alpha=0.25)
        ax.plot(angles, normalized, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_title('V7 目标值雷达图\n(归一化显示)', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  目标值雷达图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_summary_dashboard(convergence_history: Dict,
                              pareto_front,
                              hv_reference_point: List[float],
                              filename: str = 'evtol_v7_results/v7_summary_dashboard.png'):
        """
        V7新增: 综合仪表盘
        
        包含:
        1. HV收敛曲线
        2. Pareto前沿分布
        3. HV信息摘要
        """
        fig = plt.figure(figsize=(18, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. HV收敛曲线 (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        hv_history = convergence_history.get('hypervolume', [])
        if hv_history:
            generations = list(range(1, len(hv_history) + 1))
            ax1.plot(generations, hv_history, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.fill_between(generations, 0, hv_history, alpha=0.3)
            ax1.set_title('HV收敛曲线', fontsize=12, fontweight='bold')
            ax1.set_xlabel('代数')
            ax1.set_ylabel('超体积')
            ax1.grid(True, alpha=0.3)
        
        # 2. HV信息摘要 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        summary_text = f"""【V7 HV信息】

最终HV: {hv_history[-1]:.2f}

固定参考点:
  能耗: {hv_reference_point[0]:.2f}
  时间: {hv_reference_point[1]:.2f}
  噪声: {hv_reference_point[2]:.2f}
  安全: {hv_reference_point[3]:.2f}
  路径: {hv_reference_point[4]:.2f}

帕累托解数: {len(pareto_front.individuals)}
间距指标: {pareto_front.spacing:.4f}
"""
        ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 3. Pareto前沿2D (中左)
        ax3 = fig.add_subplot(gs[1, :2])
        if pareto_front.individuals:
            x_vals = [ind.objectives[0] for ind in pareto_front.individuals]
            y_vals = [ind.objectives[1] for ind in pareto_front.individuals]
            ax3.scatter(x_vals, y_vals, c='blue', s=50, alpha=0.7)
            ax3.set_xlabel('能耗 (kWh)')
            ax3.set_ylabel('时间 (s)')
            ax3.set_title('Pareto前沿: 能耗 vs 时间', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. 目标分布柱状图 (中右)
        ax4 = fig.add_subplot(gs[1, 2])
        if pareto_front.individuals:
            avg_values = np.mean([ind.objectives for ind in pareto_front.individuals], axis=0)
            obj_names = ['能耗', '时间', '噪声', '安全', '路径']
            colors = plt.cm.Set3(np.linspace(0, 1, 5))
            ax4.bar(obj_names[:len(avg_values)], avg_values, color=colors)
            ax4.set_title('目标平均值', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. 多样性历史 (下)
        ax5 = fig.add_subplot(gs[2, :])
        diversity_history = convergence_history.get('diversity', [])
        if diversity_history:
            generations = list(range(1, len(diversity_history) + 1))
            ax5.plot(generations, diversity_history, 'g-', linewidth=2, marker='s', markersize=4)
            ax5.set_title('种群多样性变化', fontsize=12, fontweight='bold')
            ax5.set_xlabel('代数')
            ax5.set_ylabel('多样性指标')
            ax5.grid(True, alpha=0.3)
        else:
            # 显示变异计数
            mutation_history = convergence_history.get('mutation_count', [])
            if mutation_history:
                generations = list(range(1, len(mutation_history) + 1))
                ax5.bar(generations, mutation_history, color='orange', alpha=0.7)
                ax5.set_title('变异操作次数', fontsize=12, fontweight='bold')
                ax5.set_xlabel('代数')
                ax5.set_ylabel('变异次数')
                ax5.grid(True, alpha=0.3)
        
        fig.suptitle('V7 多目标优化综合仪表盘', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  综合仪表盘已保存: {filename}")
        plt.close()

    # ==================== V7噪声可视化 ====================
    
    @staticmethod
    def plot_noise_heatmap(env, path: List, start_pos: tuple, goal_pos: tuple,
                          noise_model,
                          filename: str = 'evtol_v7_results/v7_noise_heatmap.png',
                          grid_spacing: int = 10, time_hour: int = 10):
        """
        V7: 绘制噪声传播热力图
        
        显示:
        - 噪声影响热力图（绿色=低，黄色=中，橙红=高）
        - 建筑物（按类型分色）
        - 飞行路径
        - 起降场位置
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 生成噪声地图 - 使用更精细的网格
        grid_x = np.arange(0, int(env.size_x) + grid_spacing, grid_spacing)
        grid_y = np.arange(0, int(env.size_y) + grid_spacing, grid_spacing)
        
        # 初始化噪声地图为负无穷，后续取最大值
        noise_map = np.full((len(grid_x), len(grid_y)), -np.inf, dtype=np.float32)
        
        # 对路径上的每个点计算噪声贡献（每2个点采样一次，减少计算量）
        path_sample = path[::2] if len(path) > 10 else path
        
        # 获取源头噪声级和参考距离
        source_level = noise_model.source.source_level
        ref_distance = noise_model.source.reference_distance
        
        for path_point in path_sample:
            x, y, z = path_point
            
            # 计算该点正下方地面的声级
            sound_level = noise_model._calculate_sound_level(z, y, x)
            
            # 计算影响半径
            impact_radius = noise_model._calculate_impact_radius(sound_level)
            
            if impact_radius > 0:
                # 找到影响范围内的网格点
                for ix, gx in enumerate(grid_x):
                    for iy, gy in enumerate(grid_y):
                        # 使用斜距计算（正确的物理模型）
                        dist_xy = math.sqrt((gx - x)**2 + (gy - y)**2)  # 水平距离
                        slant_dist = math.sqrt(dist_xy**2 + z**2)       # 斜距
                        
                        if slant_dist <= impact_radius + z:  # 扩大范围以覆盖影响半径
                            # 正确的距离衰减计算（使用斜距）
                            if slant_dist > 0:
                                distance_loss = 20 * math.log10(slant_dist / z)
                            else:
                                distance_loss = 0
                            
                            # 该点的噪声级
                            point_noise = sound_level - distance_loss
                            
                            # 取最大值
                            noise_map[ix, iy] = max(noise_map[ix, iy], point_noise)
        
        # 将-inf替换为最小显示值
        noise_map[noise_map == -np.inf] = 40.0
        noise_map = np.clip(noise_map, 40, 90)
        
        # 使用插值平滑热力图
        from scipy.ndimage import gaussian_filter
        noise_map_smooth = gaussian_filter(noise_map, sigma=1.5)
        
        print(f"  噪声范围: [{noise_map.min():.2f}, {noise_map.max():.2f}] dB")
        
        # 绘制热力图 - 使用插值平滑（更美观的配色方案）
        colors_list = ['#2ecc71', '#58d68d', '#f9e79f', '#f5b041', '#e74c3c', '#943126']
        custom_cmap = LinearSegmentedColormap.from_list('noise', colors_list, N=256)
        
        im = ax.imshow(noise_map_smooth, origin='lower', cmap=custom_cmap,
                      extent=[0, env.size_y, 0, env.size_x],
                      vmin=50, vmax=90, alpha=0.75, zorder=1, interpolation='bilinear')
        
        # 绘制建筑物
        for building in env.buildings:
            color = getattr(building, 'color', 'gray')
            if color == 'gray':
                btype = getattr(building, 'btype', 1)
                color_map = {1: 'blue', 2: 'green', 3: 'red'}
                color = color_map.get(btype, 'gray')
            
            rect = Rectangle(
                (building.y, building.x),
                building.width,
                building.length,
                facecolor=color, edgecolor='black', alpha=0.8, linewidth=1, zorder=10
            )
            ax.add_patch(rect)
        
        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            ax.plot(x_coords, y_coords, color='white',
                   linewidth=4, label='飞行路径', alpha=1.0, zorder=100)
            
            # 标记起点和终点
            ax.scatter([start_pos[1]], [start_pos[0]], color='lime', s=300,
                      marker='o', label='起点', zorder=101, edgecolors='black', linewidth=3)
            ax.scatter([goal_pos[1]], [goal_pos[0]], color='red', s=300,
                      marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('噪声级 dB(A)', fontsize=14)
        
        # 添加图例
        legend_elements = [
            Patch(facecolor='#27ae60', alpha=0.7, label='低噪声 (50-60dB)'),
            Patch(facecolor='#f1c40f', alpha=0.7, label='中等噪声 (60-70dB)'),
            Patch(facecolor='#e67e22', alpha=0.7, label='高噪声 (70-80dB)'),
            Patch(facecolor='#8b0000', alpha=0.7, label='严重噪声 (>80dB)'),
            Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='住宅建筑'),
            Patch(facecolor='green', edgecolor='black', alpha=0.8, label='公共建筑'),
            Patch(facecolor='red', edgecolor='black', alpha=0.8, label='政府建筑'),
            plt.Line2D([0], [0], color='white', linewidth=4, label='飞行路径')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 bbox_to_anchor=(1.32, 1))
        
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (米)', fontsize=14)
        ax.set_ylabel('X (米)', fontsize=14)
        ax.set_title(f'V7 噪声传播热力图 (时段: {time_hour}:00)\n'
                    f'绿色: 低噪声 | 黄色: 中等 | 橙红: 高噪声 | 深红: 严重',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  噪声传播热力图已保存: {filename}")
        plt.close()
    
    @staticmethod
    def plot_noise_along_path(path: List, start_pos: tuple, goal_pos: tuple,
                            noise_model, flight_profile=None,
                            filename: str = 'evtol_v7_results/v7_noise_along_path.png',
                            current_hour: int = 10, speed: float = 20.0):
        """
        V7: 绘制沿路径的噪声变化曲线图
        
        显示:
        - 噪声级变化曲线
        - 噪声阈值线（60dB干扰线、75dB严重干扰线）
        - 高度变化曲线
        - 飞行阶段标注
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # 计算水平距离
        distances = [0]
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(distances[-1] + dist)
        
        # 计算每个路径点的噪声级
        noise_levels = []
        heights = [p[2] for p in path]
        
        for point in path:
            x, y, z = point
            noise_level = noise_model.calculate_noise_level(x, y, z, speed)
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
            ax1.scatter([distances[violation_indices[0]]], 
                       [noise_levels[violation_indices[0]]],
                       color='orange', s=100, marker='x', zorder=15,
                       label='超60dB')
        
        severe_violation_indices = np.where(noise_levels > 75)[0]
        if len(severe_violation_indices) > 0:
            ax1.scatter([distances[severe_violation_indices[0]]], 
                        [noise_levels[severe_violation_indices[0]]],
                        color='darkred', s=100, marker='X', zorder=15,
                        label='超75dB')
        
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
            for i, p in enumerate(path):
                phase = flight_profile.get_flight_phase(p, start_pos, goal_pos)
                ax1.scatter([distances[i]], [heights[i]], 
                           color=phase_colors.get(phase, 'gray'), s=30, alpha=0.5)
            
            from matplotlib.patches import Patch as MplPatch
            phase_patches = [
                MplPatch(facecolor='green', alpha=0.15, label='起升阶段'),
                MplPatch(facecolor='blue', alpha=0.15, label='巡航阶段'),
                MplPatch(facecolor='red', alpha=0.15, label='下降阶段'),
                MplPatch(facecolor='green', alpha=0.1, label='巡航层范围')
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
        
        ax1.set_title(f'V7 沿路径噪声变化曲线 (时段: {current_hour}:00)\n'
                      f'高度与噪声级对比',
                      fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  沿路径噪声变化曲线图已保存: {filename}")
        plt.close()


# =============================================================================
# 便捷函数
# =============================================================================

def create_v7_visualizer() -> V7Visualizer:
    """创建V7可视化器实例"""
    return V7Visualizer()


def visualize_optimization_results(result, output_dir: str = 'evtol_v7_results'):
    """
    可视化优化结果
    
    参数:
        result: 优化结果对象
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    vis = V7Visualizer()
    
    # 绘制HV收敛曲线
    if hasattr(result, 'convergence_history'):
        vis.plot_hv_convergence(
            result.convergence_history.get('hypervolume', []),
            result.convergence_history.get('diversity', []),
            filename=f'{output_dir}/v7_hv_convergence.png'
        )
    
    # 绘制帕累托前沿
    if hasattr(result, 'pareto_front'):
        vis.plot_pareto_front_2d(
            result.pareto_front,
            filename=f'{output_dir}/v7_pareto_front_2d.png'
        )
        vis.plot_pareto_front_3d(
            result.pareto_front,
            filename=f'{output_dir}/v7_pareto_front_3d.png'
        )


def visualize_noise(env, path, start_pos, goal_pos, noise_model, 
                     flight_profile=None, output_dir: str = 'evtol_v7_results'):
    """
    V7噪声可视化便捷函数
    
    参数:
        env: 环境对象
        path: 路径列表
        start_pos: 起点位置
        goal_pos: 终点位置
        noise_model: 噪声模型
        flight_profile: 飞行剖面
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    vis = V7Visualizer()
    
    # 1. 噪声传播热力图
    vis.plot_noise_heatmap(
        env=env,
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        noise_model=noise_model,
        filename=f'{output_dir}/v7_noise_heatmap.png',
        grid_spacing=20,
        time_hour=10
    )
    
    # 2. 沿路径噪声变化曲线
    vis.plot_noise_along_path(
        path=path,
        start_pos=start_pos,
        goal_pos=goal_pos,
        noise_model=noise_model,
        flight_profile=flight_profile,
        filename=f'{output_dir}/v7_noise_along_path.png',
        current_hour=10,
        speed=20.0
    )
    
    print(f"\nV7噪声可视化完成!")
    print(f"生成的文件:")
    print(f"  1. {output_dir}/v7_noise_heatmap.png - 噪声传播热力图")
    print(f"  2. {output_dir}/v7_noise_along_path.png - 沿路径噪声变化曲线")


if __name__ == '__main__':
    print("V7 可视化系统加载成功!")
    print("\nV7新增可视化:")
    print("  1. HV收敛曲线 - 显示固定参考点信息")
    print("  2. Pareto前沿2D/3D视图")
    print("  3. 目标值雷达图")
    print("  4. 综合仪表盘")
