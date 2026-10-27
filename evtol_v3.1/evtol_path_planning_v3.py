#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V3版本
整合v2的基础路径搜索和低碳优化
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_bearing

@dataclass
class CityEnvironment:
    """城市环境定义"""
    size_x: int
    size_y: int
    size_z: int
    buildings: List[Dict] = field(default_factory=list)
    use_spatial_index: bool = True
    
    def generate_random_buildings(self, n_buildings: int, reserved_zones=None, min_distance: float = 20.0):
        """生成随机建筑物（支持保留区域）"""
        self.buildings = []
        
        if reserved_zones is None:
            reserved_zones = []
        
        # 尝试生成n_buildings栋建筑物
        max_attempts = n_buildings * 10  # 最多尝试次数
        attempts = 0
        
        while len(self.buildings) < n_buildings and attempts < max_attempts:
            attempts += 1
            
            # 随机位置
            x = np.random.randint(50, self.size_x - 50)
            y = np.random.randint(50, self.size_y - 50)
            z = np.random.randint(20, 80)  # 高度20-80米
            width = np.random.randint(30, 80)
            length = np.random.randint(30, 80)
            
            # 检查是否与保留区域重叠
            in_reserved = False
            for (rz_x, rz_y, radius) in reserved_zones:
                dist = np.sqrt((x - rz_x)**2 + (y - rz_y)**2)
                if dist < radius + max(width, length) / 2:
                    in_reserved = True
                    break
            
            if in_reserved:
                continue
            
            # 检查是否与现有建筑物重叠
            overlap = False
            for b in self.buildings:
                dx = abs(x - b['x'])
                dy = abs(y - b['y'])
                if dx < (width + b['width']) / 2 + min_distance and \
                   dy < (length + b['length']) / 2 + min_distance:
                    overlap = True
                    break
            
            if overlap:
                continue
            
            # 添加建筑物
            building = {
                'x': x, 'y': y, 'z': z,
                'width': width, 'length': length,
                'height': 50  # 默认高度50米
            }
            self.buildings.append(building)
    
    def is_free(self, x: float, y: float, z: float) -> bool:
        """检查坐标是否空闲"""
        for b in self.buildings:
            if (b['x'] <= x <= b['x'] + b['width'] and
                b['y'] <= y <= b['y'] + b['length'] and
                b['z'] <= z <= b['z'] + 50):  # 假设建筑物高度50m
                return False
        return True


class LowCarbonAStarV3:
    """低碳A*算法V3版本"""
    
    def __init__(self, env: CityEnvironment, 
                 omega1: float = 0.7, omega2: float = 0.3,
                 alpha1: float = 0.3, alpha2: float = 0.2, 
                 alpha3: float = 0.25, alpha4: float = 0.25):
        self.env = env
        self.weights = {
            'omega1': omega1,  # 长度权重
            'omega2': omega2,  # 能量权重
            'alpha1': alpha1,  # 距离系数
            'alpha2': alpha2,  # 能量系数
            'alpha3': alpha3,  # 垂直变化系数
            'alpha4': alpha4   # 转向角系数
        }
        self.directions = self._init_directions()
    
    def _init_directions(self) -> List[Tuple[int, int, int]]:
        """初始化18种移动方向"""
        basic = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        diagonals = [(dx,dy,dz) for dx in [-1,0,1] 
                    for dy in [-1,0,1] for dz in [-1,0,1] 
                    if (dx,dy,dz) != (0,0,0) and (dx,dy,dz) not in basic]
        return basic + diagonals
    
    def find_path(self, start: Tuple[int,int,int], goal: Tuple[int,int,int], 
                 max_iterations: int = 200000) -> List[LowCarbonNode]:
        """主搜索函数 - 改进版本"""
        start_node = LowCarbonNode(start[0], start[1], start[2])
        start_node.h = self._calculate_h_cost(start_node, goal)
        start_node.g = 0.0
        start_node.f = start_node.g + start_node.h * (self.weights['omega1'] + self.weights['omega2'])
        
        open_set = [start_node]
        closed_set = set()
        
        print(f"  搜索中... 最大迭代次数: {max_iterations}")
        
        for iteration in range(max_iterations):
            if not open_set:
                print(f"  搜索失败: open_set为空，迭代 {iteration} 次")
                break
                
            current = min(open_set, key=lambda n: n.f)
            
            if self._is_goal(current, goal):
                print(f"  找到路径! 迭代次数: {iteration}")
                return self._reconstruct_path(current)
            
            open_set.remove(current)
            closed_set.add((current.x, current.y, current.z))
            
            # 自适应方向切换：距离目标小于100米时只用6个基本方向
            dist_to_goal = self._calculate_h_cost(current, goal)
            if dist_to_goal < 100:
                directions_to_use = self.directions[:6]  # 只用6个基本方向
            else:
                directions_to_use = self.directions  # 使用全部18个方向
            
            for direction in directions_to_use:
                neighbor = self._get_neighbor(current, direction)
                if not neighbor:
                    continue
                
                neighbor_key = (neighbor.x, neighbor.y, neighbor.z)
                if neighbor_key in closed_set:
                    continue
                
                # 获取祖父节点用于计算转向角
                grandparent = None
                if current.parent:
                    grandparent = current.parent
                
                # 更新低碳指标
                update_node_metrics(current, neighbor, grandparent)
                
                # 设置父子关系
                neighbor.parent = current
                
                # 计算g值和f值
                neighbor.h = self._calculate_h_cost(neighbor, goal)
                neighbor.g = self._calculate_g_cost(neighbor)
                neighbor.f = neighbor.g + neighbor.h * (self.weights['omega1'] + self.weights['omega2'])
                
                # 检查是否已在open_set中，如果有则更新
                in_open = False
                for i, node in enumerate(open_set):
                    if node.x == neighbor.x and node.y == neighbor.y and node.z == neighbor.z:
                        if neighbor.g < node.g:  # 找到更好的路径
                            open_set[i] = neighbor
                        in_open = True
                        break
                
                if not in_open:
                    open_set.append(neighbor)
            
            # 每处理10000个节点输出进度
            if iteration % 10000 == 0:
                print(f"    进度: {iteration} 次迭代, open_set大小: {len(open_set)}")
        
        print(f"  搜索失败: 达到最大迭代次数 {max_iterations}")
        return []
    
    def _calculate_g_cost(self, node: LowCarbonNode) -> float:
        """计算低碳综合成本 g"""
        # 能量估算：假设基础能耗为0.001，爬升能耗加倍，转向能耗增加
        energy = node.accumulated_length * 0.001  # 基础能耗
        energy += node.accumulated_vertical_change * 0.0005  # 爬升能耗
        energy += node.accumulated_turn_angle * 0.1  # 转向能耗
        
        g_cost = (
            self.weights['alpha1'] * node.accumulated_length +
            self.weights['alpha2'] * energy +
            self.weights['alpha3'] * node.accumulated_vertical_change +
            self.weights['alpha4'] * node.accumulated_turn_angle
        )
        return g_cost
    
    def _calculate_h_cost(self, node: LowCarbonNode, goal: Tuple[int,int,int]) -> float:
        """计算启发式成本 h（弱启发式，让g值更主导）"""
        return np.sqrt((goal[0]-node.x)**2 + (goal[1]-node.y)**2 + (goal[2]-node.z)**2) * 0.5  # 降低h的影响
    
    def _get_neighbor(self, node: LowCarbonNode,
                     direction: Tuple[int,int,int]) -> LowCarbonNode:
        """获取有效邻居节点（改进版：更小的步长）"""
        step_xy = 20  # 水平步长20米
        step_z = 10   # 垂直步长10米
        
        x = node.x + direction[0] * step_xy
        y = node.y + direction[1] * step_xy
        z = node.z + direction[2] * step_z
        
        if (0 <= x < self.env.size_x and
            0 <= y < self.env.size_y and
            0 <= z < self.env.size_z and
            self.env.is_free(x, y, z)):
            return LowCarbonNode(x, y, z)
        return None
    
    def _is_goal(self, node: LowCarbonNode, goal: Tuple[int,int,int]) -> bool:
        """检查是否到达目标"""
        return (abs(node.x - goal[0]) < 10 and 
                abs(node.y - goal[1]) < 10 and 
                abs(node.z - goal[2]) < 5)
    
    def _reconstruct_path(self, node: LowCarbonNode) -> List[LowCarbonNode]:
        """重构路径"""
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]


def test_v3():
    """测试V3版本"""
    print("="*60)
    print("eVTOL路径规划V3测试")
    print("="*60)
    
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    
    # 测试不同配置
    configs = [
        {
            'name': 'Low-Carbon Balanced',
            'omega1': 0.7, 'omega2': 0.3,
            'alpha1': 0.3, 'alpha2': 0.2, 'alpha3': 0.25, 'alpha4': 0.25
        },
        {
            'name': 'Smooth Priority',
            'omega1': 0.8, 'omega2': 0.2,
            'alpha1': 0.2, 'alpha2': 0.2, 'alpha3': 0.3, 'alpha4': 0.3
        },
        {
            'name': 'Shortest Path',
            'omega1': 0.5, 'omega2': 0.5,
            'alpha1': 0.5, 'alpha2': 0.0, 'alpha3': 0.25, 'alpha4': 0.25
        }
    ]
    
    print("\n测试不同权重配置...")
    print("="*60)
    
    for i, config in enumerate(configs):
        print(f"\n[{config['name']}]")
        print(f"  ω₁={config['omega1']}, ω₂={config['omega2']}")
        print(f"  α₁={config['alpha1']}, α₂={config['alpha2']}, α₃={config['alpha3']}, α₄={config['alpha4']}")
        
        # 每次测试都重新生成环境（使用相同的随机种子）
        np.random.seed(42)
        env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)
        env.generate_random_buildings(n_buildings=50)
        
        if i == 0:  # 只在第一次显示环境信息
            print(f"\n地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
            print(f"建筑物数量: {len(env.buildings)}")
        
        planner = LowCarbonAStarV3(
            env,
            omega1=config['omega1'], omega2=config['omega2'],
            alpha1=config['alpha1'], alpha2=config['alpha2'],
            alpha3=config['alpha3'], alpha4=config['alpha4']
        )
        
        path = planner.find_path(start=(250,250,80), goal=(2250,2250,100))
        
        if path:
            print(f"  找到路径! 长度: {len(path)}步")
            print(f"  总转向角: {path[-1].accumulated_turn_angle:.4f} rad ({path[-1].accumulated_turn_angle*180/math.pi:.1f}°)")
            print(f"  总垂直变化: {path[-1].accumulated_vertical_change:.2f} m")
            print(f"  累计长度: {path[-1].accumulated_length:.2f} m")
        else:
            print(f"  路径搜索失败")


class V3Visualizer:
    """V3版本可视化类 - 简化版"""
    
    @staticmethod
    def plot_path_comparison_2d(env: CityEnvironment, paths: List, config_names: List[str], 
                                start: Tuple, goal: Tuple, filename='v3_path_comparison_2d.png'):
        """绘制2D路径对比"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Patch
        
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # 绘制建筑物
        for building in env.buildings:
            rect = Rectangle(
                (building['y'], building['x']),
                building['width'],
                building['length'],
                facecolor='gray',
                edgecolor='black',
                alpha=0.5,
                linewidth=1
            )
            ax.add_patch(rect)
            
            # 显示建筑物高度
            ax.text(building['y'] + building['width']/2, 
                   building['x'] + building['length']/2,
                   f'{building["z"]}m',
                   ha='center', va='center',
                   fontsize=6, color='black',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        # 绘制起点和终点
        ax.scatter(start[1], start[0], c='lime', s=300, marker='*', 
                  edgecolors='black', linewidths=2, label='Start', zorder=10)
        ax.scatter(goal[1], goal[0], c='red', s=300, marker='*', 
                  edgecolors='black', linewidths=2, label='Goal', zorder=10)
        
        # 绘制路径
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (path, name) in enumerate(zip(paths, config_names)):
            if path:
                x_coords = [p.x for p in path]
                y_coords = [p.y for p in path]
                ax.plot(y_coords, x_coords, color=colors[i % len(colors)],
                       linewidth=2.5, label=name, alpha=0.9, zorder=5)
        
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (m)', fontsize=13)
        ax.set_ylabel('X (m)', fontsize=13)
        ax.set_title(f'Low-Carbon A* Path Planning Comparison (2D View)\nMap: {env.size_x}m × {env.size_y}m × {env.size_z}m | Buildings: {len(env.buildings)}', 
                    fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                fontsize=10, framealpha=0.95, ncol=4)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  2D comparison plot saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_path_3d(env: CityEnvironment, path: List, config_name: str,
                    start: Tuple, goal: Tuple, filename='v3_path_3d.png'):
        """绘制3D路径"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制建筑物（简化为线框）
        for building in env.buildings[:30]:  # 限制数量以提高性能
            x, y, z = building['y'], building['x'], 0
            dx, dy, dz = building['width'], building['length'], 50  # 默认高度50m
            
            # 绘制建筑物框架
            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            ax.plot(xx, yy, [z]*5, color='gray', linewidth=1, alpha=0.6)
            ax.plot(xx, yy, [z+dz]*5, color='gray', linewidth=1, alpha=0.6)
            for i in range(5):
                ax.plot([xx[i], xx[i]], [yy[i], yy[i]], [z, z+dz],
                       color='gray', linewidth=1, alpha=0.4)
        
        # 绘制路径
        if path:
            x_coords = [p.x for p in path]
            y_coords = [p.y for p in path]
            z_coords = [p.z for p in path]
            ax.plot(y_coords, x_coords, z_coords, color='red', 
                   linewidth=3, label=config_name, alpha=0.9, zorder=100)
        
        # 绘制起点和终点
        ax.scatter([start[1]], [start[0]], [start[2]],
                  c='lime', s=200, marker='*', edgecolors='black',
                  linewidths=2, label='Start', zorder=150)
        ax.scatter([goal[1]], [goal[0]], [goal[2]],
                  c='red', s=200, marker='*', edgecolors='black',
                  linewidths=2, label='Goal', zorder=150)
        
        ax.set_xlabel('Y (m)', fontsize=11)
        ax.set_ylabel('X (m)', fontsize=11)
        ax.set_zlabel('Height (m)', fontsize=11)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, env.size_z)
        ax.set_title(f'Low-Carbon A* Path Planning (3D View) - {config_name}', fontsize=13)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  3D path plot saved: {filename}")
        plt.close()


def test_v3_with_visualization():
    """测试V3版本（含可视化）"""
    print("="*60)
    print("eVTOL路径规划V3测试（含可视化）")
    print("="*60)
    
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    
    # 创建环境（使用v2的参数）
    env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)
    
    # 设置起点和终点（更有挑战性的场景）
    start = (250, 250, 100)
    goal = (2250, 2250, 80)
    
    # 设置保留区域（v2的做法）
    reserved_zones = [
        (start[0], start[1], 50),
        (goal[0], goal[1], 50)
    ]
    
    # 生成250栋建筑物（v2的设置）
    print(f"地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    env.generate_random_buildings(n_buildings=250, reserved_zones=reserved_zones)
    print(f"建筑物数量: {len(env.buildings)}")
    
    # 测试不同配置（使用相同环境，不同权重）
    configs = [
        {
            'name': 'Low-Carbon Balanced',
            'omega1': 0.7, 'omega2': 0.3,
            'alpha1': 0.3, 'alpha2': 0.2, 'alpha3': 0.25, 'alpha4': 0.25
        },
        {
            'name': 'Smooth Priority',
            'omega1': 0.8, 'omega2': 0.2,
            'alpha1': 0.2, 'alpha2': 0.2, 'alpha3': 0.3, 'alpha4': 0.3
        },
        {
            'name': 'Shortest Path',
            'omega1': 0.5, 'omega2': 0.5,
            'alpha1': 0.5, 'alpha2': 0.0, 'alpha3': 0.25, 'alpha4': 0.25
        }
    ]
    
    print("\n测试不同权重配置...")
    print("="*60)
    print("  注意：所有配置使用相同的建筑物分布，仅调整权重参数")
    print("="*60)
    
    paths = []
    config_names = []
    results = []
    
    # 使用固定种子生成环境（所有配置共享）
    np.random.seed(42)
    env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)
    env.generate_random_buildings(n_buildings=250, reserved_zones=reserved_zones)
    
    for i, config in enumerate(configs):
        print(f"\n[{config['name']}]")
        print(f"  ω₁={config['omega1']}, ω₂={config['omega2']}")
        print(f"  α₁={config['alpha1']}, α₂={config['alpha2']}, α₃={config['alpha3']}, α₄={config['alpha4']}")
        
        planner = LowCarbonAStarV3(
            env,
            omega1=config['omega1'], omega2=config['omega2'],
            alpha1=config['alpha1'], alpha2=config['alpha2'],
            alpha3=config['alpha3'], alpha4=config['alpha4']
        )
        
        path = planner.find_path(start=start, goal=goal)
        
        if path:
            paths.append(path)
            config_names.append(config['name'])
            
            print(f"  找到路径! 长度: {len(path)}步")
            print(f"  总转向角: {path[-1].accumulated_turn_angle:.4f} rad ({path[-1].accumulated_turn_angle*180/math.pi:.1f}°)")
            print(f"  总垂直变化: {path[-1].accumulated_vertical_change:.2f} m")
            print(f"  累计长度: {path[-1].accumulated_length:.2f} m")
            
            results.append({
                'name': config['name'],
                'length': len(path),
                'turn_angle': path[-1].accumulated_turn_angle,
                'vertical_change': path[-1].accumulated_vertical_change
            })
        else:
            print(f"  路径搜索失败")
            paths.append(None)
            config_names.append(config['name'])
    
    # 对比结果
    print("\n" + "="*60)
    print("性能对比:")
    print("="*60)
    print(f"{'配置':<15} {'路径长度':<10} {'转向角(rad)':<15} {'垂直变化(m)':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<15} {r['length']:<10} {r['turn_angle']:<15.4f} {r['vertical_change']:<15.2f}")
    print("="*60)
    
    # 生成可视化
    if any(paths):
        print("\n生成可视化...")
        output_dir = "evtol_v3_results"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 2D对比图
        valid_paths = [p for p in paths if p]
        valid_names = [name for p, name in zip(paths, config_names) if p]
        if valid_paths:
            V3Visualizer.plot_path_comparison_2d(
                env, valid_paths, valid_names, start, goal,
                filename=os.path.join(output_dir, 'v3_path_comparison_2d.png')
            )
        
        # 3D路径图（只绘制第一个成功的路径）
        for i, (path, name) in enumerate(zip(paths, config_names)):
            if path:
                V3Visualizer.plot_path_3d(
                    env, path, name, start, goal,
                    filename=os.path.join(output_dir, f'v3_path_3d_{name}.png')
                )
                break  # 只绘制一个
        
        print(f"\nAll plots saved to: {output_dir}/")
    
    return results


if __name__ == "__main__":
    test_v3_with_visualization()