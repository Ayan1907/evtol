#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4版本可视化系统 - 动态环境与风场可视化
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zhplot  # 中文支持
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from typing import List, Dict
import math


class V4Visualizer:
    """V4版本可视化类"""
    
    @staticmethod
    def plot_wind_field_2d(env, wind_field, filename='v4_wind_field_2d.png',
                          grid_spacing=100, time_sec=0.0, path=None):
        """
        绘制2D风场图（高度100m层）

        参数:
            env: 城市环境
            wind_field: 风场对象
            filename: 输出文件名
            grid_spacing: 网格间距（米）
            time_sec: 时间（秒）
            path: 飞行路径（可选）
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        # 绘制建筑物（显示所有建筑）
        for building in env.buildings:
            rect = Rectangle(
                (building['y'], building['x']),
                building['width'],
                building['length'],
                facecolor='gray', edgecolor='black', alpha=0.6, linewidth=0.8
            )
            ax.add_patch(rect)

        # 绘制路径（如果有）
        if path:
            x_coords = [p.y for p in path]
            y_coords = [p.x for p in path]
            ax.plot(x_coords, y_coords, color='blue',
                   linewidth=3, label='Flight Path', alpha=0.9, zorder=100)

            # 标记起点和终点
            ax.scatter([path[0].y], [path[0].x], color='green', s=200,
                      marker='o', label='Start', zorder=101, edgecolors='white', linewidth=2)
            ax.scatter([path[-1].y], [path[-1].x], color='red', s=200,
                      marker='s', label='Goal', zorder=101, edgecolors='white', linewidth=2)

        # 生成风场网格
        x_range = range(0, env.size_x, grid_spacing)
        y_range = range(0, env.size_y, grid_spacing)

        vx_grid = []
        vy_grid = []
        x_grid = []
        y_grid = []

        for x in x_range:
            for y in y_range:
                if env.is_free(x, y, 100):  # 只在100m高度
                    vx, vy, vz = wind_field.get_wind_at(x, y, 100, time_sec)
                    vx_grid.append(vx)
                    vy_grid.append(vy)
                    x_grid.append(y)
                    y_grid.append(x)

        # 绘制风向向量
        if len(vx_grid) > 0:
            # 计算风速大小用于颜色
            wind_speeds = np.sqrt(np.array(vx_grid)**2 + np.array(vy_grid)**2)

            # 归一化用于箭头长度
            max_speed = np.max(wind_speeds) if len(wind_speeds) > 0 else 1.0

            # 绘制箭头
            for i in range(len(x_grid)):
                arrow_len = (wind_speeds[i] / max_speed) * 30  # 最大箭头长度30
                # 根据风速设置颜色
                color = plt.cm.jet(wind_speeds[i] / 15.0)  # 归一化到0-1
                ax.arrow(x_grid[i], y_grid[i],
                        vx_grid[i]/wind_speeds[i] * arrow_len if wind_speeds[i] > 0 else 0,
                        vy_grid[i]/wind_speeds[i] * arrow_len if wind_speeds[i] > 0 else 0,
                        head_width=4, head_length=6, fc=color,
                        ec='black', alpha=0.8, linewidth=0.5)

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=15))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Wind Speed (m/s)', fontsize=13)

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (m)', fontsize=13)
        ax.set_ylabel('X (m)', fontsize=13)
        ax.set_title(f'Urban Wind Field (z=100m, t={time_sec:.1f}s)', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Wind field plot saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_dynamic_obstacles_3d(env, dynamic_env, path: List, 
                                  time_sec: float, filename='v4_dynamic_obstacles_3d.png'):
        """
        绘制3D动态障碍物图
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制建筑物（线框）
        for building in env.buildings[:30]:
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
            x_coords = [p.y for p in path]
            y_coords = [p.x for p in path]
            z_coords = [p.z for p in path]
            ax.plot(x_coords, y_coords, z_coords, color='blue', 
                   linewidth=2, label='Path', alpha=0.8, zorder=100)
        
        # 绘制动态障碍物
        colors_map = {
            'drone': 'red',
            'helicopter': 'orange',
            'bird_flock': 'purple',
            'temporary_no_fly': 'darkred'
        }
        
        for obstacle in dynamic_env.dynamic_obstacles:
            # 检查时间有效性
            if time_sec < obstacle.valid_time_start or time_sec > obstacle.valid_time_end:
                continue
            
            # 预测位置
            ox, oy, oz = dynamic_env.predict_obstacle_position(obstacle, time_sec)
            
            color = colors_map.get(obstacle.type, 'red')
            
            # 绘制障碍物（球体表示）
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
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Drone'),
            Patch(facecolor='orange', alpha=0.5, label='Helicopter'),
            Patch(facecolor='purple', alpha=0.5, label='Bird Flock'),
            Patch(facecolor='darkred', alpha=0.5, label='No-Fly Zone')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        ax.set_xlabel('Y (m)', fontsize=11)
        ax.set_ylabel('X (m)', fontsize=11)
        ax.set_zlabel('Height (m)', fontsize=11)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, env.size_z)
        ax.set_title(f'Dynamic Obstacles (t={time_sec:.1f}s)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Dynamic obstacles plot saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_wind_cost_heatmap(env, wind_field, path: List,
                               filename='v4_wind_cost_heatmap.png',
                               height=100, time_sec=0.0):
        """
        绘制风场代价热力图

        说明:
            - 彩色区域: 风场代价热力图
              绿色=低代价(顺风), 蓝色=中等, 红色=高代价(逆风)
            - 灰色矩形: 建筑物(障碍物)
            - 白色线条: 飞行路径
        """
        fig, ax = plt.subplots(figsize=(16, 16))

        # 生成网格
        x_range = np.arange(0, env.size_x, 20)
        y_range = np.arange(0, env.size_y, 20)
        X, Y = np.meshgrid(y_range, x_range)  # 注意：Y对应y坐标，X对应x坐标

        # 计算风场代价（使用浮点型数组）
        cost_grid = np.zeros(X.shape, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = x_range[i]
                y = y_range[j]
                if env.is_free(x, y, height):
                    # 估算飞行方向（指向中心）
                    center_x, center_y = env.size_x / 2, env.size_y / 2
                    dx = center_x - x
                    dy = center_y - y
                    dz = 0
                    norm = math.sqrt(dx**2 + dy**2 + dz**2)
                    if norm > 0:
                        dx, dy, dz = dx/norm, dy/norm, dz/norm

                    cost = wind_field.calculate_wind_cost(
                        x, y, height, (dx, dy, dz), time_sec
                    )
                    cost_grid[i, j] = cost
                else:
                    cost_grid[i, j] = np.nan

        # 绘制热力图（使用反转的jet colormap，让低代价为绿色，高代价为红色）
        im = ax.imshow(cost_grid, origin='lower', cmap='RdYlGn_r',
                      extent=[0, env.size_y, 0, env.size_x],
                      vmin=0, vmax=1, alpha=0.7)

        # 绘制建筑物
        for building in env.buildings:  # 显示所有建筑
            rect = Rectangle(
                (building['y'], building['x']),
                building['width'],
                building['length'],
                facecolor='gray', edgecolor='black', alpha=0.8, linewidth=1
            )
            ax.add_patch(rect)

        # 绘制路径
        if path:
            x_coords = [p.y for p in path]
            y_coords = [p.x for p in path]
            ax.plot(x_coords, y_coords, color='white',
                   linewidth=3, label='Flight Path', alpha=0.95, zorder=100)

            # 标记起点和终点
            ax.scatter([path[0].y], [path[0].x], color='lime', s=250,
                      marker='o', label='Start', zorder=101, edgecolors='black', linewidth=2)
            ax.scatter([path[-1].y], [path[-1].x], color='red', s=250,
                      marker='s', label='Goal', zorder=101, edgecolors='black', linewidth=2)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Wind Cost (Green=Low, Yellow=Medium, Red=High)', fontsize=12)

        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lime', alpha=0.7, label='Low Wind Cost (Tailwind)'),
            Patch(facecolor='yellow', alpha=0.7, label='Medium Wind Cost (Crosswind)'),
            Patch(facecolor='red', alpha=0.7, label='High Wind Cost (Headwind)'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.8, label='Buildings (Obstacles)'),
            plt.Line2D([0], [0], color='white', linewidth=3, label='Flight Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 bbox_to_anchor=(1.32, 1))

        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (m)', fontsize=13)
        ax.set_ylabel('X (m)', fontsize=13)
        ax.set_title(f'Wind Cost Heatmap (z={height}m, t={time_sec:.1f}s)\n'
                    f'Green: Tailwind (Good) | Yellow: Crosswind (OK) | Red: Headwind (Bad)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Wind cost heatmap saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_path_with_wind_3d(env, wind_field, path: List, 
                                 start_pos: tuple, goal_pos: tuple,
                                 filename='v4_path_with_wind_3d.png',
                                 time_sec=0.0):
        """
        绘制路径和风场的3D图
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制建筑物（简化）
        for building in env.buildings[:20]:
            x, y, z = building['y'], building['x'], 0
            dx, dy, dz = building['width'], building['length'], building['height']
            
            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            ax.plot(xx, yy, [z]*5, color='gray', linewidth=0.3, alpha=0.3)
            ax.plot(xx, yy, [z+dz]*5, color='gray', linewidth=0.3, alpha=0.3)
        
        # 绘制路径
        if path:
            x_coords = [p.y for p in path]
            y_coords = [p.x for p in path]
            z_coords = [p.z for p in path]
            ax.plot(x_coords, y_coords, z_coords, color='blue', 
                   linewidth=3, label='Path', alpha=0.9, zorder=100)
        
        # 绘制风场向量（在路径上）
        if path and len(path) > 10:
            # 每隔10个节点绘制一个风向量
            for i in range(0, len(path), 10):
                p = path[i]
                vx, vy, vz = wind_field.get_wind_at(p.x, p.y, p.z, time_sec)
                speed = math.sqrt(vx**2 + vy**2 + vz**2)
                
                # 绘制风向量
                ax.quiver(p.y, p.x, p.z, 
                         vy, vx, vz,
                         length=speed*3, normalize=False,
                         color='red', alpha=0.6, 
                         arrow_length_ratio=0.3, linewidth=1.5)
        
        # 绘制起点和终点
        ax.scatter([start_pos[1]], [start_pos[0]], [start_pos[2]],
                  c='green', s=200, marker='*', 
                  edgecolors='black', linewidths=2, 
                  label='Start', zorder=150)
        ax.scatter([goal_pos[1]], [goal_pos[0]], [goal_pos[2]],
                  c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=2,
                  label='Goal', zorder=150)
        
        ax.set_xlabel('Y (m)', fontsize=12)
        ax.set_ylabel('X (m)', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, env.size_z)
        ax.set_title(f'Path with Wind Field (t={time_sec:.1f}s)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Path with wind plot saved: {filename}")
        plt.close()
