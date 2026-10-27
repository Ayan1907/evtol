#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V6版本可视化系统 - 国标合规修复
基于V5可视化,完全相同
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import json
import zhplot
from typing import List, Dict


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


class V6Visualizer:
    """V6版本可视化类 - 与V5完全相同"""
    
    # ==================== V5国标合规可视化 ====================
    
    @staticmethod
    def plot_path_with_buildings(env, path: List, start_pos: tuple, goal_pos: tuple,
                                flight_profile=None, filename='v5_path_with_buildings.png'):
        """
        绘制V5路径与建筑物（俯视图）

        包含:
        - 建筑物（按类型分色：蓝色=居民楼，绿色=公共建筑，红色=政府建筑）
        - 建筑物高度标注
        - 路径（蓝色线条）
        - 飞行阶段标注（起升/巡航/下降）
        - OFV区域（绿色圆圈）
        - 巡航层范围（橙色虚线）
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        # 绘制建筑物（使用建筑原本的颜色）
        for building in env.buildings:
            # 使用建筑原本的颜色，如果没有则根据类型设置
            color = building.get('color', 'gray')
            if color == 'gray':
                # 如果没有颜色字段，根据类型设置
                btype = building.get('btype', 1)
                color_map = {1: 'blue', 2: 'green', 3: 'red'}
                color = color_map.get(btype, 'gray')

            rect = Rectangle(
                (building['y'], building['x']),
                building['width'],
                building['length'],
                facecolor=color, edgecolor='black', alpha=0.5, linewidth=0.8
            )
            ax.add_patch(rect)

            # 标注建筑高度（只标注高度>100的建筑）
            if building['height'] >= 100:
                ax.text(building['y'] + building['width']/2,
                       building['x'] + building['length']/2,
                       f"{int(building['height'])}m",
                       ha='center', va='center', fontsize=7, color='white',
                       fontweight='bold', alpha=0.8)

        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]  # y坐标
            y_coords = [p[0] for p in path]  # x坐标
            ax.plot(x_coords, y_coords, color='blue',
                   linewidth=3, label='V5路径', alpha=0.9, zorder=100)

            # 标记起点和终点
            ax.scatter([start_pos[1]], [start_pos[0]], color='green', s=300,
                      marker='o', label='起点', zorder=101, edgecolors='white', linewidth=3)
            ax.scatter([goal_pos[1]], [goal_pos[0]], color='red', s=300,
                      marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)

            # 绘制OFV区域（起降场30m保护区）
            ofv_radius = flight_profile.ofv_height if flight_profile else 30.0
            ofv1 = plt.Circle((start_pos[1], start_pos[0]), ofv_radius,
                             color='green', alpha=0.2, label='OFV保护区(起点)')
            ofv2 = plt.Circle((goal_pos[1], goal_pos[0]), ofv_radius,
                             color='green', alpha=0.2)
            ax.add_patch(ofv1)
            ax.add_patch(ofv2)

        # 标注飞行阶段
        if flight_profile and path:
            total_nodes = len(path)
            
            # 找到阶段转换点
            climb_idx = 0
            cruise_idx = 0
            approach_idx = 0
            
            for i, p in enumerate(path):
                phase = flight_profile.get_flight_phase(
                    (p[0], p[1], p[2]), start_pos, goal_pos
                )
                if phase == 'climb':
                    climb_idx = i
                elif phase == 'cruise':
                    cruise_idx = i
                elif phase == 'approach' and approach_idx == 0:
                    approach_idx = i
                    break

            # 标注阶段转换点
            if climb_idx > 0:
                ax.scatter([path[climb_idx][1]], [path[climb_idx][0]],
                         color='orange', s=200, marker='^',
                         label='起升→巡航', zorder=102,
                         edgecolors='black', linewidth=2)
            if approach_idx > 0:
                ax.scatter([path[approach_idx][1]], [path[approach_idx][0]],
                         color='purple', s=200, marker='v',
                         label='巡航→下降', zorder=102,
                         edgecolors='black', linewidth=2)

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (米)', fontsize=14)
        ax.set_ylabel('X (米)', fontsize=14)
        ax.set_title('V5路径规划俯视图\n(符合GB 38031-2022国标)',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  俯视图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_3d_path_profile(env, path: List, start_pos: tuple, goal_pos: tuple,
                             flight_profile=None, filename='v5_3d_path_profile.png'):
        """
        绘制3D路径剖面

        包含:
        - 3D路径轨迹
        - 建筑物（线框）
        - 飞行阶段颜色编码
        - 巡航层平面（透明橙色）
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制建筑物（线框，全部显示）
        for building in env.buildings:
            x, y, z = building['y'], building['x'], 0
            dx, dy, dz = building['width'], building['length'], building['height']

            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            ax.plot(xx, yy, [z]*5, color='gray', linewidth=0.5, alpha=0.4)
            ax.plot(xx, yy, [z+dz]*5, color='gray', linewidth=0.5, alpha=0.4)
            for i in range(5):
                ax.plot([xx[i], xx[i]], [yy[i], yy[i]], [z, z+dz],
                       color='gray', linewidth=0.5, alpha=0.3)

        # 绘制巡航层平面
        if flight_profile:
            xx, yy = np.meshgrid(np.linspace(0, env.size_y, 20),
                                 np.linspace(0, env.size_x, 20))
            zz_cruise_min = np.full_like(xx, flight_profile.cruise_min)
            ax.plot_surface(xx, yy, zz_cruise_min, alpha=0.15, color='orange',
                          label='巡航层最小高度')

        # 绘制路径（按阶段着色）
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            z_coords = [p[2] for p in path]

            # 起升阶段（绿色）
            if flight_profile:
                climb_end = 0
                for i, p in enumerate(path):
                    phase = flight_profile.get_flight_phase(
                        (p[0], p[1], p[2]), start_pos, goal_pos
                    )
                    if phase != 'climb':
                        climb_end = i
                        break
                if climb_end > 1:
                    ax.plot(x_coords[:climb_end], y_coords[:climb_end],
                           z_coords[:climb_end], color='green',
                           linewidth=3, label='起升阶段', alpha=0.9)

                # 巡航阶段（蓝色）
                approach_start = len(path)
                for i in range(len(path)-1, -1, -1):
                    phase = flight_profile.get_flight_phase(
                        (path[i][0], path[i][1], path[i][2]), start_pos, goal_pos
                    )
                    if phase == 'approach':
                        approach_start = i
                    elif phase == 'cruise':
                        break
                if approach_start > climb_end:
                    ax.plot(x_coords[climb_end:approach_start],
                           y_coords[climb_end:approach_start],
                           z_coords[climb_end:approach_start], color='blue',
                           linewidth=3, label='巡航阶段', alpha=0.9)

                # 下降阶段（红色）
                if approach_start < len(path):
                    ax.plot(x_coords[approach_start:], y_coords[approach_start:],
                           z_coords[approach_start:], color='red',
                           linewidth=3, label='下降阶段', alpha=0.9)
            else:
                ax.plot(x_coords, y_coords, z_coords, color='blue',
                       linewidth=3, label='路径', alpha=0.9)

            # 标记起点和终点
            ax.scatter([start_pos[1]], [start_pos[0]], [start_pos[2]],
                     color='green', s=200, marker='o', label='起点', zorder=100)
            ax.scatter([goal_pos[1]], [goal_pos[0]], [goal_pos[2]],
                     color='red', s=200, marker='s', label='终点', zorder=100)

        ax.set_xlabel('Y (米)', fontsize=12)
        ax.set_ylabel('X (米)', fontsize=12)
        ax.set_zlabel('高度 (米)', fontsize=12)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, min(env.size_z, 250))
        ax.set_title('V5 3D路径剖面\n(GB 38031-2022飞行阶段)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  3D路径剖面图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_height_profile(path: List, start_pos: tuple, goal_pos: tuple,
                          flight_profile=None, filename='v5_height_profile.png'):
        """
        绘制高度剖面图（高度 vs 水平距离）

        包含:
        - 高度变化曲线
        - 飞行阶段标注
        - 巡航层范围（橙色区域）
        - OFV高度线（绿色虚线）
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # 计算水平距离
        distances = np.cumsum([0] + [
            np.sqrt((path[i][0]-path[i-1][0])**2 +
                   (path[i][1]-path[i-1][1])**2)
            for i in range(1, len(path))
        ])
        heights = [p[2] for p in path]

        # 绘制路径
        ax.plot(distances, heights, color='blue', linewidth=2.5,
               label='V5路径', alpha=0.9, zorder=10)

        # 绘制巡航层范围
        if flight_profile:
            ax.fill_between(distances, [flight_profile.cruise_min]*len(distances),
                          [flight_profile.cruise_max]*len(distances),
                          color='orange', alpha=0.2, label='巡航层范围')

            # 绘制OFV高度
            ax.axhline(y=flight_profile.ofv_height, color='green',
                      linestyle='--', linewidth=2, label=f'OFV高度 ({flight_profile.ofv_height}米)')

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
                    # 标注上一个阶段
                    if current_phase is not None and i > phase_start:
                        ax.fill_between([distances[phase_start], distances[i-1]],
                                      [min(heights[phase_start:i]), min(heights[phase_start:i])],
                                      [max(heights[phase_start:i]), max(heights[phase_start:i])],
                                      color=phase_colors[current_phase], alpha=0.15)
                    current_phase = phase
                    phase_start = i

            # 最后一个阶段
            if current_phase is not None:
                ax.fill_between([distances[phase_start], distances[-1]],
                              [min(heights[phase_start:]), min(heights[phase_start:])],
                              [max(heights[phase_start:]), max(heights[phase_start:])],
                              color=phase_colors[current_phase], alpha=0.15)

            # 创建图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.2, label='起升阶段'),
                Patch(facecolor='blue', alpha=0.2, label='巡航阶段'),
                Patch(facecolor='red', alpha=0.2, label='下降阶段'),
                Patch(facecolor='orange', alpha=0.2, label='巡航层范围'),
                plt.Line2D([0], [0], color='green', linestyle='--',
                           linewidth=2, label='OFV高度')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        ax.set_xlabel('水平距离 (米)', fontsize=13)
        ax.set_ylabel('高度 (米)', fontsize=13)
        ax.set_title('V5高度剖面图\n(GB 38031-2022飞行阶段)',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  高度剖面图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_phase_statistics(profile_data: Dict, filename='v5_phase_statistics.png'):
        """
        绘制飞行阶段统计图

        包含:
        - 各阶段节点数量（柱状图）
        - 各阶段高度范围（误差条）
        - 路径效率
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        phases = ['起升', '巡航', '下降']
        counts = [profile_data['climb']['count'],
                 profile_data['cruise']['count'],
                 profile_data['approach']['count']]

        # 节点数量柱状图
        colors = ['green', 'blue', 'red']
        bars = ax1.bar(phases, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('节点数量', fontsize=12)
        ax1.set_title('飞行阶段节点分布', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 高度范围图
        heights_min = [profile_data['climb']['min_height'],
                      profile_data['cruise']['min_height'],
                      profile_data['approach']['min_height']]
        heights_max = [profile_data['climb']['max_height'],
                      profile_data['cruise']['max_height'],
                      profile_data['approach']['max_height']]
        heights_avg = [profile_data['climb']['avg_height'],
                      profile_data['cruise']['avg_height'],
                      profile_data['approach']['avg_height']]

        x_pos = np.arange(len(phases))
        ax2.errorbar(x_pos, heights_avg,
                    yerr=[np.array(heights_avg) - np.array(heights_min),
                          np.array(heights_max) - np.array(heights_avg)],
                    fmt='o', color='black', capsize=10, capthick=2,
                    markersize=10, linewidth=2)
        ax2.scatter(x_pos, heights_avg, s=200, color=colors, zorder=5,
                   edgecolors='black', linewidth=2)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(phases)
        ax2.set_ylabel('高度 (米)', fontsize=12)
        ax2.set_title('飞行阶段高度范围', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  阶段统计图已保存: {filename}")
        plt.close()
    
    # ==================== V4动态环境可视化 ====================
    
    @staticmethod
    def plot_wind_field_2d(env, wind_field, path=None, filename='v5_wind_field_2d.png',
                          grid_spacing=100, time_sec=0.0):
        """
        绘制2D风场图（继承V4）

        参数:
            env: 城市环境
            wind_field: 风场对象
            filename: 输出文件名
            path: 飞行路径（可选）
            grid_spacing: 网格间距（米）
            time_sec: 时间（秒）
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        # 绘制建筑物（使用建筑原本的颜色）
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
                facecolor=color, edgecolor='black', alpha=0.6, linewidth=0.8
            )
            ax.add_patch(rect)

        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            ax.plot(x_coords, y_coords, color='blue',
                   linewidth=3, label='飞行路径', alpha=0.9, zorder=100)

            # 标记起点和终点
            ax.scatter([path[0][1]], [path[0][0]], color='green', s=200,
                      marker='o', label='起点', zorder=101, edgecolors='white', linewidth=2)
            ax.scatter([path[-1][1]], [path[-1][0]], color='red', s=200,
                      marker='s', label='终点', zorder=101, edgecolors='white', linewidth=2)

        # 生成风场网格
        x_range = range(0, int(env.size_x), grid_spacing)
        y_range = range(0, int(env.size_y), grid_spacing)

        vx_grid = []
        vy_grid = []
        x_grid = []
        y_grid = []

        for x in x_range:
            for y in y_range:
                if env.is_free(x, y, 100):  # 100m高度
                    vx, vy, vz = wind_field.get_wind_at(x, y, 100, time_sec)
                    vx_grid.append(vx)
                    vy_grid.append(vy)
                    x_grid.append(y)
                    y_grid.append(x)

        # 绘制风向向量
        if len(vx_grid) > 0:
            wind_speeds = np.sqrt(np.array(vx_grid)**2 + np.array(vy_grid)**2)
            max_speed = np.max(wind_speeds) if len(wind_speeds) > 0 else 1.0

            for i in range(len(x_grid)):
                arrow_len = (wind_speeds[i] / max_speed) * 30
                color = plt.cm.jet(wind_speeds[i] / 15.0)
                ax.arrow(x_grid[i], y_grid[i],
                        vx_grid[i]/wind_speeds[i] * arrow_len if wind_speeds[i] > 0 else 0,
                        vy_grid[i]/wind_speeds[i] * arrow_len if wind_speeds[i] > 0 else 0,
                        head_width=4, head_length=6, fc=color,
                        ec='black', alpha=0.8, linewidth=0.5)

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=15))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('风速 (m/s)', fontsize=13)

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (米)', fontsize=13)
        ax.set_ylabel('X (米)', fontsize=13)
        ax.set_title(f'城市风场 (z=100m, t={time_sec:.1f}s)', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  风场图已保存: {filename}")
        plt.close()
    
    @staticmethod
    def plot_dynamic_obstacles_2d(env, dynamic_env, path: List,
                                   time_sec: float, filename='v5_dynamic_obstacles_2d.png'):
        """
        绘制2D动态障碍物图（俯视图）

        包含:
        - 建筑物（按类型分色）
        - 建筑物高度标注
        - 路径（蓝色线条）
        - 动态障碍物（彩色圆圈 + 速度向量）
        - 障碍物信息标注（ID、类型、速度）
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        # 绘制建筑物（使用建筑原本的颜色）
        for building in env.buildings:
            # 使用建筑原本的颜色，如果没有则根据类型设置
            color = building.get('color', 'gray')
            if color == 'gray':
                btype = building.get('btype', 1)
                color_map = {1: 'blue', 2: 'green', 3: 'red'}
                color = color_map.get(btype, 'gray')

            rect = Rectangle(
                (building['y'], building['x']),
                building['width'],
                building['length'],
                facecolor=color, edgecolor='black', alpha=0.5, linewidth=0.8
            )
            ax.add_patch(rect)

            # 标注建筑高度（只标注高度>100的建筑）
            if building['height'] >= 100:
                ax.text(building['y'] + building['width']/2,
                       building['x'] + building['length']/2,
                       f"{int(building['height'])}m",
                       ha='center', va='center', fontsize=7, color='white',
                       fontweight='bold', alpha=0.8)

        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            ax.plot(x_coords, y_coords, color='blue',
                   linewidth=3, label='路径', alpha=0.9, zorder=100)

            # 标记起点和终点
            ax.scatter([path[0][1]], [path[0][0]], color='green', s=300,
                      marker='o', label='起点', zorder=101, edgecolors='white', linewidth=3)
            ax.scatter([path[-1][1]], [path[-1][0]], color='red', s=300,
                      marker='s', label='终点', zorder=101, edgecolors='white', linewidth=3)

        # 绘制动态障碍物（所有障碍物，忽略时间约束）
        colors_map = {
            'drone': 'red',
            'helicopter': 'orange',
            'bird_flock': 'purple',
            'temporary_no_fly': 'darkred'
        }
        labels_map = {
            'drone': '无人机',
            'helicopter': '直升机',
            'bird_flock': '鸟群',
            'temporary_no_fly': '禁飞区'
        }

        # 收集实际存在的障碍物类型
        actual_types = set(obstacle.type for obstacle in dynamic_env.dynamic_obstacles)

        for obstacle in dynamic_env.dynamic_obstacles:
            # 预测位置
            ox, oy, oz = dynamic_env.predict_obstacle_position(obstacle, time_sec)
            color = colors_map.get(obstacle.type, 'red')
            label = labels_map.get(obstacle.type, '未知')

            # 绘制障碍物圆圈
            circle = Circle((oy, ox), obstacle.size * 2,  # 放大显示
                          facecolor=color, edgecolor='black', alpha=0.6, linewidth=2)
            ax.add_patch(circle)

            # 绘制速度向量箭头
            speed = math.sqrt(obstacle.vx**2 + obstacle.vy**2)
            if speed > 0:
                arrow = FancyArrowPatch(
                    (oy, ox),
                    (oy + obstacle.vx * 10, ox + obstacle.vy * 10),  # 放大10倍
                    arrowstyle='-|>', mutation_scale=20,
                    color='black', linewidth=2, alpha=0.8
                )
                ax.add_patch(arrow)

            # 标注障碍物信息
            info_text = f"ID:{obstacle.id}\n{label}\n速度:{speed:.1f}m/s"
            ax.annotate(info_text, xy=(oy, ox), xytext=(oy + obstacle.size*3, ox + obstacle.size*3),
                       fontsize=9, color='black', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=color),
                       arrowprops=dict(arrowstyle='->', color=color, linewidth=1.5))

        # 创建图例（只显示实际存在的类型）
        legend_elements = [
            Patch(facecolor=colors_map[t], alpha=0.6, label=labels_map[t])
            for t in actual_types
        ]
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, label='速度向量'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (米)', fontsize=13)
        ax.set_ylabel('X (米)', fontsize=13)
        ax.set_title(f'动态障碍物俯视图 (t={time_sec:.1f}s)\n所有障碍物 + 速度向量',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  动态障碍物2D俯视图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_dynamic_obstacles_3d(env, dynamic_env, path: List,
                                  time_sec: float, filename='v5_dynamic_obstacles_3d.png'):
        """
        绘制3D动态障碍物图
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制建筑物（线框，全部显示）
        for building in env.buildings:
            x, y, z = building['y'], building['x'], 0
            dx, dy, dz = building['width'], building['length'], building['height']

            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            ax.plot(xx, yy, [z]*5, color='gray', linewidth=0.5, alpha=0.4)
            ax.plot(xx, yy, [z+dz]*5, color='gray', linewidth=0.5, alpha=0.4)
            for i in range(5):
                ax.plot([xx[i], xx[i]], [yy[i], yy[i]], [z, z+dz],
                       color='gray', linewidth=0.5, alpha=0.3)

        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            z_coords = [p[2] for p in path]
            ax.plot(x_coords, y_coords, z_coords, color='blue',
                   linewidth=2, label='路径', alpha=0.8, zorder=100)

        # 绘制动态障碍物（全部显示，忽略时间约束）
        colors_map = {
            'drone': 'red',
            'helicopter': 'orange',
            'bird_flock': 'purple',
            'temporary_no_fly': 'darkred'
        }

        # 收集实际存在的障碍物类型
        actual_types = set(obstacle.type for obstacle in dynamic_env.dynamic_obstacles)

        for obstacle in dynamic_env.dynamic_obstacles:
            ox, oy, oz = dynamic_env.predict_obstacle_position(obstacle, time_sec)
            color = colors_map.get(obstacle.type, 'red')

            # 绘制障碍物（球体）
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = obstacle.size * np.outer(np.cos(u), np.sin(v)) + ox
            y_sphere = obstacle.size * np.outer(np.sin(u), np.sin(v)) + oy
            z_sphere = obstacle.size * np.outer(np.ones(np.size(u)), np.cos(v)) + oz

            ax.plot_surface(x_sphere, y_sphere, z_sphere,
                          color=color, alpha=0.4,
                          edgecolor='black', linewidth=0.5)

            # 绘制速度向量
            ax.quiver(ox, oy, oz, obstacle.vx, obstacle.vy, obstacle.vz,
                     length=20, normalize=True, color='black', alpha=0.7)

        # 添加图例（只显示实际存在的类型）
        labels_map = {
            'drone': '无人机',
            'helicopter': '直升机',
            'bird_flock': '鸟群',
            'temporary_no_fly': '禁飞区'
        }
        legend_elements = [
            Patch(facecolor=colors_map[t], alpha=0.5, label=labels_map[t])
            for t in actual_types
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_xlabel('Y (米)', fontsize=11)
        ax.set_ylabel('X (米)', fontsize=11)
        ax.set_zlabel('高度 (米)', fontsize=11)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, env.size_z)
        ax.set_title(f'动态障碍物3D图 (t={time_sec:.1f}s)', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  动态障碍物3D图已保存: {filename}")
        plt.close()

    @staticmethod
    def plot_wind_cost_heatmap(env, wind_field, path: List,
                               filename='v5_wind_cost_heatmap.png',
                               height=100, time_sec=0.0):
        """
        绘制风场代价热力图（完整版本）

        说明:
            - 彩色区域: 风场代价热力图（包括建筑物区域）
              绿色=低代价(顺风), 蓝色=中等, 红色=高代价(逆风)
            - 三色矩形: 建筑物(障碍物) - 蓝色=住宅, 绿色=公共, 红色=政府
            - 蓝色线条: 飞行路径
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        # 生成网格（10米间隔）
        x_range = np.arange(0, int(env.size_x) + 10, 10)
        y_range = np.arange(0, int(env.size_y) + 10, 10)
        print(f"    网格点数量: {len(x_range)} x {len(y_range)} = {len(x_range)*len(y_range)}")

        # 计算路径方向（从起点到终点的整体方向）
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

        # 计算风场代价（所有点都计算，包括建筑物区域）
        cost_grid = np.zeros((len(x_range), len(y_range)), dtype=float)
        cost_min = float('inf')
        cost_max = float('-inf')

        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                # 不检查是否是建筑物，所有点都计算
                cost = wind_field.calculate_wind_cost(
                    x, y, height, path_dir, time_sec
                )
                cost_grid[i, j] = cost
                cost_min = min(cost_min, cost)
                cost_max = max(cost_max, cost)

        print(f"    代价范围: [{cost_min:.4f}, {cost_max:.4f}]")

        # 绘制完整热力图（所有点都有颜色）
        # 颜色映射需要: 绿色(低) -> 蓝色(中) -> 红色(高)
        # 使用自定义颜色映射
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#2ecc71', '#3498db', '#e74c3c']  # 绿 -> 蓝 -> 红
        custom_cmap = LinearSegmentedColormap.from_list('wind_cost', colors_list, N=100)
        im = ax.imshow(cost_grid, origin='lower', cmap=custom_cmap,
                      extent=[0, env.size_y, 0, env.size_x],
                      vmin=cost_min, vmax=cost_max, alpha=0.6, zorder=1)

        # 绘制建筑物（三色矩形，完全不透明）
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
                facecolor=color, edgecolor='black', alpha=1.0, linewidth=1, zorder=10
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
        cbar.set_label('风场代价 (绿色=低, 蓝色=中, 红色=高)', fontsize=12)

        # 添加图例说明
        legend_elements = [
            Patch(facecolor='lime', alpha=0.6, label='低代价 (顺风)'),
            Patch(facecolor='#3498db', alpha=0.6, label='中等代价 (侧风)'),
            Patch(facecolor='red', alpha=0.6, label='高代价 (逆风)'),
            Patch(facecolor='blue', edgecolor='black', alpha=1.0, label='住宅建筑'),
            Patch(facecolor='green', edgecolor='black', alpha=1.0, label='公共建筑'),
            Patch(facecolor='red', edgecolor='black', alpha=1.0, label='政府建筑'),
            plt.Line2D([0], [0], color='blue', linewidth=4, label='飞行路径')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 bbox_to_anchor=(1.32, 1))

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (米)', fontsize=13)
        ax.set_ylabel('X (米)', fontsize=13)
        ax.set_title(f'风场代价热力图 (z={height}m, t={time_sec:.1f}s)\n'
                    f'绿色: 顺风 (好) | 蓝色: 侧风 (可) | 红色: 逆风 (差)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', zorder=0)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  风场代价热力图已保存: {filename}")
        plt.close()
    
    @staticmethod
    def plot_path_with_wind_3d(env, wind_field, path: List, 
                                 start_pos: tuple, goal_pos: tuple,
                                 filename='v5_path_with_wind_3d.png',
                                 time_sec=0.0):
        """
        绘制路径和风场的3D图（继承V4）
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制建筑物（线框，全部显示）
        for building in env.buildings:
            x, y, z = building['y'], building['x'], 0
            dx, dy, dz = building['width'], building['length'], building['height']

            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            ax.plot(xx, yy, [z]*5, color='gray', linewidth=0.3, alpha=0.3)
            ax.plot(xx, yy, [z+dz]*5, color='gray', linewidth=0.3, alpha=0.3)
        
        # 绘制路径
        if path:
            x_coords = [p[1] for p in path]
            y_coords = [p[0] for p in path]
            z_coords = [p[2] for p in path]
            ax.plot(x_coords, y_coords, z_coords, color='blue', 
                   linewidth=3, label='路径', alpha=0.9, zorder=100)
        
        # 绘制风场向量（在路径上）
        if path and len(path) > 10:
            for i in range(0, len(path), 10):
                p = path[i]
                vx, vy, vz = wind_field.get_wind_at(p[0], p[1], p[2], time_sec)
                speed = math.sqrt(vx**2 + vy**2 + vz**2)
                
                ax.quiver(p[1], p[0], p[2], 
                         vy, vx, vz,
                         length=speed*3, normalize=False,
                         color='red', alpha=0.6, 
                         arrow_length_ratio=0.3, linewidth=1.5)
        
        # 绘制起点和终点
        ax.scatter([start_pos[1]], [start_pos[0]], [start_pos[2]],
                  c='green', s=200, marker='*', 
                  edgecolors='black', linewidths=2, 
                  label='起点', zorder=150)
        ax.scatter([goal_pos[1]], [goal_pos[0]], [goal_pos[2]],
                  c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=2,
                  label='终点', zorder=150)
        
        ax.set_xlabel('Y (米)', fontsize=12)
        ax.set_ylabel('X (米)', fontsize=12)
        ax.set_zlabel('高度 (米)', fontsize=12)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, env.size_z)
        ax.set_title(f'路径与风场 (t={time_sec:.1f}s)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  路径风场3D图已保存: {filename}")
        plt.close()
