#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V3标准版
整合v2的基础路径搜索、低碳优化和eVTOL标准几何约束
基于《客运eVTOL起降场技术要求（征求意见稿）》
"""

import numpy as np
import math
import json
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_bearing


@dataclass
class EVTOLStandard:
    """eVTOL标准参数定义"""
    # 机型参数（以亿航EH216-S为例）
    D: float = 5.63  # 航空器全尺寸(米)
    L: float = 5.73  # 全长(米)
    W: float = 5.61  # 全宽(米)
    H: float = 1.93  # 高度(米)
    MTOW: float = 600  # 最大起飞重量(kg)
    
    # 起降区尺寸约束（标准5.2.3和5.3.3条）
    FATO_diameter: float = field(init=False)  # 最终进近和起飞区直径
    TLOF_diameter: float = field(init=False)  # 接地和离地区直径
    safety_zone_extension: float = 3.0  # 安全区超出FATO边界(米)
    
    # 无障碍空间约束（标准5.4.2和6.2条）
    OFV_height: float = 30.0  # 悬停高度(米)
    OFV_shape: str = 'cylinder'  # 无障碍空间形状: 'cylinder' 或 'inverted_frustum'
    
    # 障碍物限制面约束（标准6.1.1条）
    approach_surface_slope: float = 0.083  # 进近面坡度 (1:12)
    climb_surface_slope: float = 0.1  # 爬升面坡度 (1:10)
    transition_surface_slope: float = 0.143  # 过渡面坡度 (1:7)
    
    def __post_init__(self):
        self.FATO_diameter = 1.5 * self.D
        self.TLOF_diameter = 1.0 * self.D
    
    def get_TLOF_radius(self) -> float:
        """获取TLOF半径"""
        return self.TLOF_diameter / 2
    
    def get_safety_zone_radius(self) -> float:
        """获取安全区半径"""
        return self.FATO_diameter / 2 + self.safety_zone_extension

@dataclass
class Vertiport:
    """起降场定义"""
    x: float
    y: float
    z: float  # 起降场高度
    standard: EVTOLStandard
    
    def get_TLOF_center(self) -> Tuple[float, float, float]:
        """获取TLOF中心坐标"""
        return (self.x, self.y, self.z)
    
    def get_TLOF_radius(self) -> float:
        """获取TLOF半径"""
        return self.standard.get_TLOF_radius()
    
    def get_safety_zone_radius(self) -> float:
        """获取安全区半径"""
        return self.standard.get_safety_zone_radius()
    
    def check_in_TLOF(self, x: float, y: float, z: float) -> bool:
        """检查点是否在TLOF范围内"""
        distance = np.sqrt((x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2)
        return distance <= self.get_TLOF_radius()
    
    def check_in_safety_zone(self, x: float, y: float, z: float) -> bool:
        """检查点是否在安全区内"""
        horizontal_dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        
        # 放宽约束：在OFV外部也不强制要求在安全区内
        # 只检查垂直约束：固定物体高度受限
        if horizontal_dist > self.get_safety_zone_radius():
            # 在安全区外，需要满足障碍物限制面约束
            # 这个检查在check_approach_surface_clearance中进行
            pass
        
        # OFV内的约束
        if horizontal_dist <= self.get_safety_zone_radius() and z <= self.z + self.standard.OFV_height:
            return True
        
        # OFV外或高度超过OFV
        if z > self.z + self.standard.OFV_height:
            # 超过OFV高度后需要满足无障碍空间约束
            if self.standard.OFV_shape == 'cylinder':
                return horizontal_dist <= self.get_safety_zone_radius()
            else:  # inverted_frustum
                # 倒棱台形：高度越高，水平范围越大
                height_above = z - self.z
                allowed_radius = self.get_safety_zone_radius() + height_above * self.standard.approach_surface_slope
                return horizontal_dist <= allowed_radius
        
        return True
    
    def check_approach_surface_clearance(self, x: float, y: float, z: float, is_departure: bool) -> bool:
        """检查是否满足障碍物限制面约束"""
        horizontal_dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        
        if horizontal_dist <= self.get_safety_zone_radius():
            return True  # 安全区内不受限制面约束
        
        slope = self.standard.climb_surface_slope if is_departure else self.standard.approach_surface_slope
        min_height = self.z + horizontal_dist * slope
        
        # 航迹必须位于障碍物限制面之上
        return z >= min_height
    
    def get_OFV_clearance_constraint(self, x: float, y: float, z: float) -> float:
        """获取无障碍空间约束值（越小越好）"""
        horizontal_dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        
        if z <= self.z + self.standard.OFV_height:
            # OFV内部：距离中心越近越好
            return horizontal_dist / self.get_safety_zone_radius()
        else:
            # OFV外部：需要满足斜面约束
            height_above = z - self.z
            min_horizontal_dist = 0
            if self.standard.OFV_shape == 'cylinder':
                min_horizontal_dist = self.get_safety_zone_radius()
            else:
                min_horizontal_dist = self.get_safety_zone_radius() + height_above * self.standard.approach_surface_slope
            
            return max(0, (horizontal_dist - min_horizontal_dist) / min_horizontal_dist)


@dataclass
class CityEnvironment:
    """城市环境定义（支持eVTOL标准约束）"""
    size_x: int
    size_y: int
    size_z: int
    buildings: List[Dict] = field(default_factory=list)
    use_spatial_index: bool = True
    map_file: str = None
    
    def load_from_json(self, map_file: str):
        """从JSON文件加载地图"""
        self.map_file = map_file
        with open(map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.size_x = data['map_info']['size_x']
        self.size_y = data['map_info']['size_y']
        self.size_z = data['map_info']['size_z']
        self.buildings = data['buildings']
        
        return data
    
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
                b['z'] <= z <= b['z'] + b['height']):
                return False
        return True
    
    def get_building_at(self, x: float, y: float, z: float) -> Dict:
        """获取指定位置的建筑物"""
        for b in self.buildings:
            if (b['x'] <= x <= b['x'] + b['width'] and
                b['y'] <= y <= b['y'] + b['length'] and
                b['z'] <= z <= b['z'] + b['height']):
                return b
        return None


class LowCarbonAStarV3Standard:
    """低碳A*算法V3标准版（集成eVTOL几何约束）"""
    
    def __init__(self, env: CityEnvironment, standard: EVTOLStandard,
                 start_vertiport: Vertiport, goal_vertiport: Vertiport,
                 omega1: float = 0.7, omega2: float = 0.3,
                 alpha1: float = 0.3, alpha2: float = 0.2, 
                 alpha3: float = 0.25, alpha4: float = 0.25,
                 standard_weight: float = 0.5):
        self.env = env
        self.standard = standard
        self.start_vertiport = start_vertiport
        self.goal_vertiport = goal_vertiport
        self.standard_weight = standard_weight  # eVTOL标准约束的权重
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
    
    def find_path(self, max_iterations: int = 200000) -> List[LowCarbonNode]:
        """主搜索函数 - 标准版"""
        start = (self.start_vertiport.x, self.start_vertiport.y, self.start_vertiport.z)
        goal = (self.goal_vertiport.x, self.goal_vertiport.y, self.goal_vertiport.z)
        
        start_node = LowCarbonNode(start[0], start[1], start[2])
        start_node.h = self._calculate_h_cost(start_node, goal)
        start_node.g = 0.0
        start_node.f = start_node.g + start_node.h * (self.weights['omega1'] + self.weights['omega2'])
        
        open_set = [start_node]
        closed_set = set()
        
        print(f"  搜索中... 最大迭代次数: {max_iterations}")
        print(f"  eVTOL标准约束权重: {self.standard_weight}")
        
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
                
                # 检查eVTOL标准约束
                if not self._check_standard_constraints(neighbor):
                    continue
                
                # 获取祖父节点用于计算转向角
                grandparent = None
                if current.parent:
                    grandparent = current.parent
                
                # 更新低碳指标
                update_node_metrics(current, neighbor, grandparent)
                
                # 设置父子关系
                neighbor.parent = current
                
                # 计算g值和f值（加入eVTOL标准约束成本）
                neighbor.h = self._calculate_h_cost(neighbor, goal)
                neighbor.g = self._calculate_g_cost(neighbor) + \
                           self._calculate_standard_cost(neighbor, current)
                neighbor.f = neighbor.g + neighbor.h * (self.weights['omega1'] + self.weights['omega2'])
                
                # 检查是否已在open_set中，如果有则更新
                in_open = False
                for i, node in enumerate(open_set):
                    if node.x == neighbor.x and node.y == neighbor.y and node.z == node.z:
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
    
    def _check_standard_constraints(self, node: LowCarbonNode) -> bool:
        """检查eVTOL标准约束"""
        # 1. 检查是否在建筑物内
        if not self.env.is_free(node.x, node.y, node.z):
            return False
        
        # 2. 检查起点/终点的安全区约束（仅在接近时检查）
        dist_to_start = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                (node.y - self.start_vertiport.y)**2)
        
        dist_to_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                               (node.y - self.goal_vertiport.y)**2)
        
        # 起点：起飞阶段，需要满足爬升面约束（只在接近50米内检查）
        if dist_to_start < 50:
            if not self.start_vertiport.check_approach_surface_clearance(
                node.x, node.y, node.z, is_departure=True):
                return False
        
        # 终点：进近阶段，需要满足进近面约束（只在接近50米内检查）
        if dist_to_goal < 50:
            if not self.goal_vertiport.check_approach_surface_clearance(
                node.x, node.y, node.z, is_departure=False):
                return False
        
        # 3. 检查无障碍空间约束（OFV）（只在接近30米内检查）
        if dist_to_start < 30:
            if not self.start_vertiport.check_in_safety_zone(node.x, node.y, node.z):
                return False
        
        if dist_to_goal < 30:
            if not self.goal_vertiport.check_in_safety_zone(node.x, node.y, node.z):
                return False
        
        return True
    
    def _calculate_standard_cost(self, node: LowCarbonNode, current: LowCarbonNode) -> float:
        """计算eVTOL标准约束成本"""
        cost = 0.0
        
        # 1. 无障碍空间约束成本
        dist_to_start = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                (node.y - self.start_vertiport.y)**2 +
                                (node.z - self.start_vertiport.z)**2)
        dist_to_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                               (node.y - self.goal_vertiport.y)**2 +
                               (node.z - self.goal_vertiport.z)**2)
        
        if dist_to_start < 100:
            ofv_cost = self.start_vertiport.get_OFV_clearance_constraint(node.x, node.y, node.z)
            cost += ofv_cost * 100  # 权重100
        
        if dist_to_goal < 100:
            ofv_cost = self.goal_vertiport.get_OFV_clearance_constraint(node.x, node.y, node.z)
            cost += ofv_cost * 100  # 权重100
        
        # 2. 障碍物限制面约束成本（鼓励在限制面上方飞行）
        if dist_to_start < 200:
            horizontal_dist = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                      (node.y - self.start_vertiport.y)**2)
            min_height = self.start_vertiport.z + horizontal_dist * self.standard.climb_surface_slope
            if node.z < min_height:
                cost += (min_height - node.z) * 10  # 低于限制面的惩罚
        
        if dist_to_goal < 200:
            horizontal_dist = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                                      (node.y - self.goal_vertiport.y)**2)
            min_height = self.goal_vertiport.z + horizontal_dist * self.standard.approach_surface_slope
            if node.z < min_height:
                cost += (min_height - node.z) * 10  # 低于限制面的惩罚
        
        return cost * self.standard_weight
    
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
        """检查是否到达目标（严格满足TLOF约束）"""
        return (abs(node.x - goal[0]) < 5 and 
                abs(node.y - goal[1]) < 5 and 
                abs(node.z - goal[2]) < 3)
    
    def _reconstruct_path(self, node: LowCarbonNode) -> List[LowCarbonNode]:
        """重构路径"""
        path = []
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]


class V3StandardVisualizer:
    """V3标准版可视化类"""
    
    @staticmethod
    def plot_path_comparison_2d(env: CityEnvironment, paths: List, config_names: List[str], 
                                start: Tuple, goal: Tuple, 
                                start_vertiport: Vertiport, goal_vertiport: Vertiport,
                                filename='v3_standard_path_comparison_2d.png'):
        """绘制2D路径对比（含eVTOL标准约束可视化）"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle, Patch
        
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
        
        # 绘制起点和终点
        ax.scatter(start[1], start[0], c='lime', s=300, marker='*', 
                  edgecolors='black', linewidths=2, label='Start', zorder=10)
        ax.scatter(goal[1], goal[0], c='red', s=300, marker='*', 
                  edgecolors='black', linewidths=2, label='Goal', zorder=10)
        
        # 绘制TLOF区域
        tlof_start = Circle((start[1], start[0]), start_vertiport.get_TLOF_radius(),
                           facecolor='lime', alpha=0.3, edgecolor='green', linewidth=2,
                           label='TLOF Start', zorder=5)
        tlof_goal = Circle((goal[1], goal[0]), goal_vertiport.get_TLOF_radius(),
                         facecolor='red', alpha=0.3, edgecolor='darkred', linewidth=2,
                         label='TLOF Goal', zorder=5)
        ax.add_patch(tlof_start)
        ax.add_patch(tlof_goal)
        
        # 绘制安全区
        safety_start = Circle((start[1], start[0]), start_vertiport.get_safety_zone_radius(),
                             facecolor='yellow', alpha=0.2, edgecolor='orange', linewidth=2,
                             linestyle='--', label='Safety Zone', zorder=4)
        safety_goal = Circle((goal[1], goal[0]), goal_vertiport.get_safety_zone_radius(),
                            facecolor='yellow', alpha=0.2, edgecolor='orange', linewidth=2,
                            linestyle='--', zorder=4)
        ax.add_patch(safety_start)
        ax.add_patch(safety_goal)
        
        # 绘制路径
        colors = ['blue', 'green', 'orange', 'purple']
        for i, (path, name) in enumerate(zip(paths, config_names)):
            if path:
                x_coords = [p.x for p in path]
                y_coords = [p.y for p in path]
                ax.plot(y_coords, x_coords, color=colors[i % len(colors)],
                       linewidth=2.5, label=name, alpha=0.9, zorder=8)
        
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel('Y (m)', fontsize=13)
        ax.set_ylabel('X (m)', fontsize=13)
        ax.set_title(f'eVTOL Path Planning with Standard Constraints (2D View)\n'
                    f'Map: {env.size_x}m × {env.size_y}m × {env.size_z}m | '
                    f'Buildings: {len(env.buildings)}', fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                fontsize=9, framealpha=0.95, ncol=5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  2D comparison plot saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_path_3d(env: CityEnvironment, path: List, config_name: str,
                    start: Tuple, goal: Tuple,
                    start_vertiport: Vertiport, goal_vertiport: Vertiport,
                    filename='v3_standard_path_3d.png'):
        """绘制3D路径（含eVTOL标准约束可视化）"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import Circle
        import mpl_toolkits.mplot3d.art3d as art3d
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制建筑物（简化为线框）
        for building in env.buildings[:50]:  # 限制数量以提高性能
            x, y, z = building['y'], building['x'], 0
            dx, dy, dz = building['width'], building['length'], building['height']
            
            # 绘制建筑物框架
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
            ax.plot(x_coords, y_coords, z_coords, color='red', 
                   linewidth=3, label=config_name, alpha=0.9, zorder=100)
        
        # 绘制起点和终点
        ax.scatter([start[1]], [start[0]], [start[2]],
                  c='lime', s=200, marker='*', edgecolors='black',
                  linewidths=2, label='Start', zorder=150)
        ax.scatter([goal[1]], [goal[0]], [goal[2]],
                  c='red', s=200, marker='*', edgecolors='black',
                  linewidths=2, label='Goal', zorder=150)
        
        # 绘制OFV（无障碍空间）
        # 起点OFV
        theta = np.linspace(0, 2*np.pi, 50)
        r_start = start_vertiport.get_safety_zone_radius()
        for h in np.linspace(0, start_vertiport.standard.OFV_height, 5):
            x_circ = start[1] + r_start * np.cos(theta)
            y_circ = start[0] + r_start * np.sin(theta)
            z_circ = np.full_like(x_circ, start[2] + h)
            ax.plot(x_circ, y_circ, z_circ, color='lime', alpha=0.3, linewidth=1)
        
        # 终点OFV
        r_goal = goal_vertiport.get_safety_zone_radius()
        for h in np.linspace(0, goal_vertiport.standard.OFV_height, 5):
            x_circ = goal[1] + r_goal * np.cos(theta)
            y_circ = goal[0] + r_goal * np.sin(theta)
            z_circ = np.full_like(x_circ, goal[2] + h)
            ax.plot(x_circ, y_circ, z_circ, color='red', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Y (m)', fontsize=11)
        ax.set_ylabel('X (m)', fontsize=11)
        ax.set_zlabel('Height (m)', fontsize=11)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, env.size_z)
        ax.set_title(f'eVTOL Path Planning with Standard Constraints (3D View) - {config_name}', fontsize=13)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  3D path plot saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(results: List[Dict], paths: List, 
                                  filename='v3_standard_metrics_comparison.png'):
        """绘制性能指标对比分析"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        configs = [r['name'] for r in results]
        path_lengths = [r['length'] for r in results]
        turn_angles = [r['turn_angle'] * 180/math.pi for r in results]
        vertical_changes = [r['vertical_change'] for r in results]
        
        # 计算能耗估算
        energy_estimates = []
        for path in paths:
            if path:
                end_node = path[-1]
                energy = (end_node.accumulated_length * 0.001 + 
                         end_node.accumulated_vertical_change * 0.0005 +
                         end_node.accumulated_turn_angle * 0.1)
                energy_estimates.append(energy * 1000)  # 转换为相对单位
            else:
                energy_estimates.append(0)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 路径长度对比
        ax1 = plt.subplot(2, 3, 1)
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars1 = ax1.bar(range(len(configs)), path_lengths, color=colors[:len(configs)], 
                       edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Path Length (steps)', fontsize=11)
        ax1.set_title('Path Length Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=15, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # 2. 转向角对比
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(range(len(configs)), turn_angles, 
                       color=colors[:len(configs)], edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Turn Angle (degrees)', fontsize=11)
        ax2.set_title('Total Turn Angle Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}°', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # 3. 垂直变化对比
        ax3 = plt.subplot(2, 3, 3)
        bars3 = ax3.bar(range(len(configs)), vertical_changes,
                       color=colors[:len(configs)], edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Vertical Change (m)', fontsize=11)
        ax3.set_title('Total Vertical Change Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs, rotation=15, ha='right')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # 4. 能耗估算对比
        ax4 = plt.subplot(2, 3, 4)
        bars4 = ax4.bar(range(len(configs)), energy_estimates,
                       color=colors[:len(configs)], edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Estimated Energy (relative units)', fontsize=11)
        ax4.set_title('Energy Consumption Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs, rotation=15, ha='right')
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # 5. 综合雷达图
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        # 归一化数据
        norm_length = [l/max(path_lengths) for l in path_lengths]
        norm_turn = [t/max(turn_angles) for t in turn_angles]
        norm_vert = [v/max(vertical_changes) for v in vertical_changes]
        norm_energy = [e/max(energy_estimates) for e in energy_estimates]
        
        categories = ['Length', 'Turn\nAngle', 'Vertical\nChange', 'Energy']
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        
        for i, config_name in enumerate(configs):
            values = [1-norm_length[i], 1-norm_turn[i], 1-norm_vert[i], 1-norm_energy[i]]
            values += values[:1]
            
            ax5.plot(angles, values, 'o-', linewidth=2, 
                    label=config_name, color=colors[i % len(colors)])
            ax5.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories, fontsize=9)
        ax5.set_ylim(0, 1)
        ax5.set_title('Overall Performance Radar\n(Higher is Better)', 
                      fontsize=12, fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. 综合评分（加权）
        ax6 = plt.subplot(2, 3, 6)
        scores = []
        for i in range(len(configs)):
            # 综合评分：长度和能耗越低越好，转向角和垂直变化越低越好
            score = (1-norm_length[i]) * 0.3 + (1-norm_turn[i]) * 0.25 + \
                    (1-norm_vert[i]) * 0.2 + (1-norm_energy[i]) * 0.25
            scores.append(score * 100)
        
        bars6 = ax6.bar(range(len(configs)), scores,
                       color=colors[:len(configs)], edgecolor='black', linewidth=1.5)
        ax6.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Overall Score (out of 100)', fontsize=11)
        ax6.set_title('Comprehensive Performance Score', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(len(configs)))
        ax6.set_xticklabels(configs, rotation=15, ha='right')
        ax6.grid(axis='y', alpha=0.3, linestyle='--')
        ax6.set_ylim(0, 100)
        
        for i, bar in enumerate(bars6):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Metrics comparison plot saved: {filename}")
        plt.close()
    
    @staticmethod
    def plot_energy_profile(path: List, config_name: str,
                             filename='v3_standard_energy_profile.png'):
        """绘制能耗剖面图"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 提取数据
        indices = list(range(len(path)))
        heights = [p.z for p in path]
        
        # 计算每段的能耗贡献
        segment_distances = []
        segment_vertical = []
        segment_turns = []
        
        for i in range(len(path)-1):
            prev = path[i]
            curr = path[i+1]
            
            dist = np.sqrt((curr.x-prev.x)**2 + (curr.y-prev.y)**2 + (curr.z-prev.z)**2)
            vert = abs(curr.z - prev.z)
            
            segment_distances.append(dist)
            segment_vertical.append(vert)
            
            # 估算转向角
            if i > 0:
                vec1 = np.array([prev.x - path[i-1].x, prev.y - path[i-1].y, prev.z - path[i-1].z])
                vec2 = np.array([curr.x - prev.x, curr.y - prev.y, curr.z - prev.z])
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    turn = np.arccos(cos_angle)
                else:
                    turn = 0
                segment_turns.append(turn)
            else:
                segment_turns.append(0)
        
        # 1. 高度变化
        ax1 = axes[0]
        ax1.plot(indices, heights, color='#2E86AB', linewidth=2, marker='o', 
                markersize=3, alpha=0.8)
        ax1.fill_between(indices, heights, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Path Segment Index', fontsize=11)
        ax1.set_ylabel('Altitude (m)', fontsize=11)
        ax1.set_title(f'Altitude Profile - {config_name}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 能耗分量
        ax2 = axes[1]
        ax2.plot(indices[1:], [d*0.001 for d in segment_distances], 
                label='Distance Energy', color='#A23B72', linewidth=2, alpha=0.8)
        ax2.plot(indices[1:], [v*0.0005 for v in segment_vertical], 
                label='Vertical Energy', color='#F18F01', linewidth=2, alpha=0.8)
        ax2.plot(indices[1:], [t*0.1 for t in segment_turns], 
                label='Turn Energy', color='#C73E1D', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Path Segment Index', fontsize=11)
        ax2.set_ylabel('Energy Component (relative)', fontsize=11)
        ax2.set_title('Energy Components Breakdown', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. 累计能耗
        ax3 = axes[2]
        cumulative_energy = [0]
        for i in range(len(segment_distances)):
            e = segment_distances[i]*0.001 + segment_vertical[i]*0.0005 + segment_turns[i]*0.1
            cumulative_energy.append(cumulative_energy[-1] + e)
        
        ax3.plot(indices, cumulative_energy, color='#2E86AB', linewidth=2.5, alpha=0.8)
        ax3.fill_between(indices, cumulative_energy, alpha=0.3, color='#2E86AB')
        ax3.set_xlabel('Path Segment Index', fontsize=11)
        ax3.set_ylabel('Cumulative Energy (relative)', fontsize=11)
        ax3.set_title(f'Cumulative Energy Consumption - {config_name}', 
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Energy profile plot saved: {filename}")
        plt.close()


def test_v3_standard():
    """测试V3标准版"""
    print("="*60)
    print("eVTOL路径规划V3标准版测试")
    print("="*60)
    
    # 创建eVTOL标准参数
    standard = EVTOLStandard()
    print(f"\neVTOL标准参数:")
    print(f"  机型: EH216-S")
    print(f"  全尺寸D: {standard.D}m")
    print(f"  TLOF直径: {standard.TLOF_diameter:.2f}m (半径: {standard.get_TLOF_radius():.2f}m)")
    print(f"  FATO直径: {standard.FATO_diameter:.2f}m")
    print(f"  安全区半径: {standard.get_safety_zone_radius():.2f}m")
    print(f"  OFV高度: {standard.OFV_height}m")
    
    # 从JSON加载地图
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('/home/ayan1907/evtol/test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    print(f"\n地图信息:")
    print(f"  尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    print(f"  建筑物数量: {len(env.buildings)}")
    print(f"  起始点: {start_pos}")
    print(f"  终点: {goal_pos}")
    
    # 创建起降场
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    # 测试不同配置
    configs = [
        {
            'name': 'Standard Compliance',
            'omega1': 0.5, 'omega2': 0.5,
            'alpha1': 0.3, 'alpha2': 0.2, 'alpha3': 0.25, 'alpha4': 0.25,
            'standard_weight': 0.8  # 高标准约束权重
        },
        {
            'name': 'Balanced',
            'omega1': 0.7, 'omega2': 0.3,
            'alpha1': 0.3, 'alpha2': 0.2, 'alpha3': 0.25, 'alpha4': 0.25,
            'standard_weight': 0.5  # 中等标准约束权重
        },
        {
            'name': 'Performance Priority',
            'omega1': 0.8, 'omega2': 0.2,
            'alpha1': 0.5, 'alpha2': 0.0, 'alpha3': 0.25, 'alpha4': 0.25,
            'standard_weight': 0.3  # 低标准约束权重
        }
    ]
    
    print("\n测试不同权重配置...")
    print("="*60)
    
    paths = []
    config_names = []
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n[{config['name']}]")
        print(f"  ω₁={config['omega1']}, ω₂={config['omega2']}")
        print(f"  α₁={config['alpha1']}, α₂={config['alpha2']}, α₃={config['alpha3']}, α₄={config['alpha4']}")
        print(f"  标准约束权重: {config['standard_weight']}")
        
        planner = LowCarbonAStarV3Standard(
            env, standard, start_vertiport, goal_vertiport,
            omega1=config['omega1'], omega2=config['omega2'],
            alpha1=config['alpha1'], alpha2=config['alpha2'],
            alpha3=config['alpha3'], alpha4=config['alpha4'],
            standard_weight=config['standard_weight']
        )
        
        path = planner.find_path()
        
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
    print(f"{'配置':<20} {'路径长度':<10} {'转向角(rad)':<15} {'垂直变化(m)':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r['length']:<10} {r['turn_angle']:<15.4f} {r['vertical_change']:<15.2f}")
    print("="*60)
    
    # 生成可视化
    if any(paths):
        print("\n生成可视化...")
        output_dir = "evtol_v3_standard_results"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 2D对比图
        valid_paths = [p for p in paths if p]
        valid_names = [name for p, name in zip(paths, config_names) if p]
        if valid_paths:
            V3StandardVisualizer.plot_path_comparison_2d(
                env, valid_paths, valid_names, start_pos, goal_pos,
                start_vertiport, goal_vertiport,
                filename=os.path.join(output_dir, 'v3_standard_path_comparison_2d.png')
            )
        
        # 3D路径图（绘制所有成功的路径）
        for i, (path, name) in enumerate(zip(paths, config_names)):
            if path:
                V3StandardVisualizer.plot_path_3d(
                    env, path, name, start_pos, goal_pos,
                    start_vertiport, goal_vertiport,
                    filename=os.path.join(output_dir, f'v3_standard_path_3d_{name}.png')
                )
                
                # 能耗剖面图（每个配置生成一张）
                V3StandardVisualizer.plot_energy_profile(
                    path, name,
                    filename=os.path.join(output_dir, f'v3_standard_energy_profile_{name}.png')
                )
        
        # 性能指标综合对比图
        if valid_paths:
            V3StandardVisualizer.plot_metrics_comparison(
                results, valid_paths,
                filename=os.path.join(output_dir, 'v3_standard_metrics_comparison.png')
            )
        
        print(f"\nAll plots saved to: {output_dir}/")
    
    return results


if __name__ == "__main__":
    test_v3_standard()
