#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V4版本 - 动态环境实时避障与城市风场
整合V3的基础功能，新增：
1. 动态障碍物（移动物体、临时禁飞区）
2. 城市风场模型（建筑物效应、时变风场）
3. 实时重规划能力
4. 风场对能耗的影响
"""

import numpy as np
import math
import json
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_bearing
import time

# 导入V3标准版的核心类
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment


@dataclass
class WindField:
    """城市风场模型"""
    
    def __init__(self, env: CityEnvironment, 
                 base_wind_speed: float = 5.0,  # 基础风速 m/s
                 base_wind_direction: float = 0.0,  # 基础风向（度，0=东风）
                 turbulence_intensity: float = 0.1,  # 湍流强度
                 time_variation: bool = True):  # 是否启用时间变化
        """
        初始化城市风场
        
        参数:
            env: 城市环境
            base_wind_speed: 基础风速 (m/s)
            base_wind_direction: 基础风向 (度，0=东风，90=南风，180=西风，270=北风)
            turbulence_intensity: 湍流强度 (0-1)
            time_variation: 是否启用时间变化
        """
        self.env = env
        self.base_wind_speed = base_wind_speed
        self.base_wind_direction = base_wind_direction
        self.turbulence_intensity = turbulence_intensity
        self.time_variation = time_variation
        self.time_seed = 0.0
        
        # 建筑物风场效应参数
        self.building_wake_factor = 0.7  # 建筑物尾流速度衰减因子
        self.building_wake_length = 100.0  # 建筑物尾流长度 (m)
        self.building_channeling_factor = 1.5  # 建筑物通道效应加速因子
    
    def get_wind_at(self, x: float, y: float, z: float, 
                   time_sec: float = 0.0) -> Tuple[float, float, float]:
        """
        获取指定位置的风速向量
        
        返回: (vx, vy, vz) 单位: m/s
        """
        # 1. 基础风场
        wind_dir_rad = math.radians(self.base_wind_direction)
        vx_base = self.base_wind_speed * math.cos(wind_dir_rad)
        vy_base = self.base_wind_speed * math.sin(wind_dir_rad)
        vz_base = 0.0  # 假设垂直风为0
        
        # 2. 时间变化（模拟阵风）
        if self.time_variation:
            gust_factor = 1.0 + 0.3 * math.sin(time_sec * 0.1 + self.time_seed)
            vx_base *= gust_factor
            vy_base *= gust_factor
        
        # 3. 建筑物效应
        vx_building, vy_building, vz_building = self._calculate_building_effect(
            x, y, z, vx_base, vy_base
        )
        
        # 4. 湍流
        if self.turbulence_intensity > 0:
            np.random.seed(int(time_sec + x + y + z))
            vx_turb = vx_base * self.turbulence_intensity * (np.random.rand() - 0.5) * 2
            vy_turb = vy_base * self.turbulence_intensity * (np.random.rand() - 0.5) * 2
            vz_turb = vz_base * self.turbulence_intensity * (np.random.rand() - 0.5) * 2
        else:
            vx_turb = vy_turb = vz_turb = 0.0
        
        # 合成风场
        vx = vx_base + vx_building + vx_turb
        vy = vy_base + vy_building + vy_turb
        vz = vz_base + vz_building + vz_turb
        
        return vx, vy, vz
    
    def _calculate_building_effect(self, x: float, y: float, z: float,
                                  vx_base: float, vy_base: float) -> Tuple[float, float, float]:
        """
        计算建筑物对风场的影响
        
        效应包括：
        1. 尾流效应：建筑物下游风速降低
        2. 通道效应：建筑物间风速增加
        3. 阻挡效应：建筑物上游风速降低
        """
        vx_effect = 0.0
        vy_effect = 0.0
        vz_effect = 0.0
        
        # 计算风向上的单位向量
        wind_speed = math.sqrt(vx_base**2 + vy_base**2)
        if wind_speed > 0:
            wind_dir_x = vx_base / wind_speed
            wind_dir_y = vy_base / wind_speed
        else:
            wind_dir_x = wind_dir_y = 0.0
        
        # 检查每个建筑物的影响
        for building in self.env.buildings:
            bx_center = building['x'] + building['width'] / 2
            by_center = building['y'] + building['length'] / 2
            bz_top = building['z'] + building['height']
            
            # 检查是否在建筑物高度范围内
            if z < building['z'] or z > bz_top + 50:
                continue
            
            # 计算相对于建筑物的位置
            dx = x - bx_center
            dy = y - by_center
            
            # 检查是否在尾流区（建筑物下游）
            dist_downstream = dx * wind_dir_x + dy * wind_dir_y
            if dist_downstream > 0 and dist_downstream < self.building_wake_length:
                # 尾流效应：速度衰减
                lateral_dist = abs(dx * wind_dir_y - dy * wind_dir_x)
                building_width = max(building['width'], building['length'])
                
                if lateral_dist < building_width * 2:
                    wake_factor = 1.0 - (1.0 - self.building_wake_factor) * \
                                (1.0 - dist_downstream / self.building_wake_length) * \
                                math.exp(-lateral_dist / building_width)
                    vx_effect -= vx_base * (1.0 - wake_factor)
                    vy_effect -= vy_base * (1.0 - wake_factor)
            
            # 检查是否在阻挡区（建筑物上游）
            dist_upstream = -(dx * wind_dir_x + dy * wind_dir_y)
            if dist_upstream > 0 and dist_upstream < 50:
                lateral_dist = abs(dx * wind_dir_y - dy * wind_dir_x)
                building_width = max(building['width'], building['length'])
                
                if lateral_dist < building_width:
                    # 阻挡效应：速度降低
                    block_factor = math.exp(-dist_upstream / 20.0)
                    vx_effect -= vx_base * block_factor * 0.3
                    vy_effect -= vy_base * block_factor * 0.3
        
        # 通道效应：建筑物之间加速
        channel_factor = self._calculate_channeling_effect(x, y, z)
        if channel_factor > 1.0:
            vx_effect += vx_base * (channel_factor - 1.0) * 0.5
            vy_effect += vy_base * (channel_factor - 1.0) * 0.5
        
        return vx_effect, vy_effect, vz_effect
    
    def _calculate_channeling_effect(self, x: float, y: float, z: float) -> float:
        """计算建筑物通道效应"""
        # 简化实现：计算周围建筑物的遮挡比例
        building_count = 0
        total_width = 0.0
        
        for building in self.env.buildings:
            bx_center = building['x'] + building['width'] / 2
            by_center = building['y'] + building['length'] / 2
            bz_top = building['z'] + building['height']
            
            # 检查高度范围
            if z < building['z'] or z > bz_top:
                continue
            
            # 检查水平距离
            dist = math.sqrt((x - bx_center)**2 + (y - by_center)**2)
            if dist < 100:
                building_count += 1
                total_width += max(building['width'], building['length'])
        
        if building_count > 1:
            # 有多个建筑物时，可能存在通道效应
            avg_width = total_width / building_count
            channel_factor = 1.0 + min(avg_width / 50.0, 
                                      self.building_channeling_factor - 1.0)
            return min(channel_factor, self.building_channeling_factor)
        
        return 1.0
    
    def calculate_wind_cost(self, x: float, y: float, z: float,
                           desired_direction: Tuple[float, float, float],
                           time_sec: float = 0.0) -> float:
        """
        计算风场对路径的代价
        
        参数:
            x, y, z: 位置
            desired_direction: 期望的飞行方向向量
            time_sec: 时间
            
        返回: 风场代价 (0-1, 越小越好)
        """
        # 获取当前风场
        vx, vy, vz = self.get_wind_at(x, y, z, time_sec)
        wind_speed = math.sqrt(vx**2 + vy**2 + vz**2)
        
        # 获取期望飞行方向
        dx, dy, dz = desired_direction
        desired_speed = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if desired_speed == 0 or wind_speed == 0:
            return 0.0
        
        # 计算逆风比例（逆风增加能耗）
        desired_norm = np.array([dx, dy, dz]) / desired_speed
        wind_norm = np.array([vx, vy, vz]) / wind_speed
        
        # 点积：正值=顺风，负值=逆风
        dot_product = np.dot(desired_norm, wind_norm)
        
        # 逆风惩罚：逆风越强，代价越高
        if dot_product < 0:
            headwind_cost = abs(dot_product) * 2.0  # 逆风加倍惩罚
        else:
            headwind_cost = 0.0  # 顺风无惩罚
        
        # 侧风惩罚：侧风增加操控难度
        crosswind = abs(1.0 - abs(dot_product))
        crosswind_cost = crosswind * 0.5
        
        # 强风惩罚：风速过大会增加风险
        if wind_speed > 15.0:  # 超过15m/s（6级风）
            strong_wind_cost = (wind_speed - 15.0) * 0.1
        else:
            strong_wind_cost = 0.0
        
        total_cost = headwind_cost + crosswind_cost + strong_wind_cost
        return min(total_cost, 1.0)  # 上限为1.0


@dataclass
class DynamicObstacle:
    """动态障碍物"""
    id: int
    type: str  # 'drone', 'helicopter', 'bird_flock', 'temporary_no_fly'
    x: float
    y: float
    z: float
    vx: float = 0.0  # 速度x
    vy: float = 0.0  # 速度y
    vz: float = 0.0  # 速度z
    size: float = 5.0  # 影响半径（米）
    valid_time_start: float = 0.0  # 有效开始时间（秒）
    valid_time_end: float = 999999.0  # 有效结束时间（秒）
    predictability: float = 1.0  # 可预测性（0-1，1=完全可预测）


class DynamicEnvironment:
    """动态环境管理器"""
    
    def __init__(self, env: CityEnvironment):
        self.env = env
        self.static_obstacles = env.buildings
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.wind_field: Optional[WindField] = None
        self.current_time = 0.0  # 当前时间（秒）
        
        # 安全缓冲距离
        self.static_obstacle_buffer = 10.0  # 静态障碍物缓冲
        self.dynamic_obstacle_buffer = 20.0  # 动态障碍物缓冲
        
    def set_wind_field(self, wind_field: WindField):
        """设置风场"""
        self.wind_field = wind_field
    
    def add_dynamic_obstacle(self, obstacle: DynamicObstacle):
        """添加动态障碍物"""
        self.dynamic_obstacles.append(obstacle)
    
    def predict_obstacle_position(self, obstacle: DynamicObstacle, 
                                  time_sec: float) -> Tuple[float, float, float]:
        """
        预测动态障碍物在指定时间的位置
        
        简化模型：假设匀速直线运动
        """
        dt = time_sec - self.current_time
        x_pred = obstacle.x + obstacle.vx * dt
        y_pred = obstacle.y + obstacle.vy * dt
        z_pred = obstacle.z + obstacle.vz * dt
        return x_pred, y_pred, z_pred
    
    def is_safe(self, x: float, y: float, z: float, 
                time_sec: Optional[float] = None) -> bool:
        """
        检查位置是否安全（避开静态和动态障碍物）
        
        参数:
            x, y, z: 检查位置
            time_sec: 时间（如果为None，使用当前时间）
        """
        if time_sec is None:
            time_sec = self.current_time
        
        # 1. 检查静态障碍物（建筑物）
        for building in self.static_obstacles:
            # 扩展缓冲区
            bx_min = building.x - self.static_obstacle_buffer
            bx_max = building.x + building.width + self.static_obstacle_buffer
            by_min = building.y - self.static_obstacle_buffer
            by_max = building.y + building.length + self.static_obstacle_buffer
            bz_min = building.z
            bz_max = building.z + building.height + self.static_obstacle_buffer
            
            if bx_min <= x <= bx_max and by_min <= y <= by_max and bz_min <= z <= bz_max:
                return False
        
        # 2. 检查动态障碍物
        for obstacle in self.dynamic_obstacles:
            # 检查时间有效性
            if time_sec < obstacle.valid_time_start or time_sec > obstacle.valid_time_end:
                continue
            
            # 预测障碍物位置
            ox, oy, oz = self.predict_obstacle_position(obstacle, time_sec)
            
            # 检查碰撞（考虑缓冲区）
            dist = math.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            if dist < obstacle.size + self.dynamic_obstacle_buffer:
                return False
        
        return True
    
    def update_time(self, time_sec: float):
        """更新当前时间"""
        self.current_time = time_sec
    
    def get_all_obstacles_at_time(self, time_sec: float) -> List[Dict]:
        """
        获取指定时间所有障碍物的位置信息（用于可视化）
        """
        obstacles = []
        
        # 静态障碍物
        for building in self.static_obstacles:
            obstacles.append({
                'type': 'static',
                'x': building.x,
                'y': building.y,
                'z': building.z,
                'width': building.width,
                'length': building.length,
                'height': building.height
            })
        
        # 动态障碍物
        for obstacle in self.dynamic_obstacles:
            if time_sec < obstacle.valid_time_start or time_sec > obstacle.valid_time_end:
                continue
            
            ox, oy, oz = self.predict_obstacle_position(obstacle, time_sec)
            obstacles.append({
                'type': 'dynamic',
                'id': obstacle.id,
                'obstacle_type': obstacle.type,
                'x': ox,
                'y': oy,
                'z': oz,
                'size': obstacle.size
            })
        
        return obstacles


class LowCarbonAStarV4:
    """低碳A*算法V4版本 - 动态环境实时避障与风场"""
    
    def __init__(self, env: CityEnvironment, standard: EVTOLStandard,
                 start_vertiport: Vertiport, goal_vertiport: Vertiport,
                 dynamic_env: DynamicEnvironment,
                 omega1: float = 0.7, omega2: float = 0.3,
                 alpha1: float = 0.3, alpha2: float = 0.2, 
                 alpha3: float = 0.25, alpha4: float = 0.25,
                 alpha5: float = 0.2,  # 风场权重（新增）
                 standard_weight: float = 0.5):
        self.env = env
        self.standard = standard
        self.start_vertiport = start_vertiport
        self.goal_vertiport = goal_vertiport
        self.dynamic_env = dynamic_env
        self.standard_weight = standard_weight
        self.weights = {
            'omega1': omega1,  # 长度权重
            'omega2': omega2,  # 能量权重
            'alpha1': alpha1,  # 距离系数
            'alpha2': alpha2,  # 能量系数
            'alpha3': alpha3,  # 垂直变化系数
            'alpha4': alpha4,  # 转向角系数
            'alpha5': alpha5   # 风场系数（新增）
        }
        self.directions = self._init_directions()
        self.replan_count = 0  # 重规划次数
    
    def _init_directions(self) -> List[Tuple[int, int, int]]:
        """初始化18种移动方向"""
        basic = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        diagonals = [(dx,dy,dz) for dx in [-1,0,1] 
                    for dy in [-1,0,1] for dz in [-1,0,1] 
                    if (dx,dy,dz) != (0,0,0) and (dx,dy,dz) not in basic]
        return basic + diagonals
    
    def find_path(self, max_iterations: int = 200000, 
                 current_time: float = 0.0) -> List[LowCarbonNode]:
        """
        主搜索函数 - V4版本（动态环境）
        
        参数:
            max_iterations: 最大迭代次数
            current_time: 当前时间（秒）
        """
        self.dynamic_env.update_time(current_time)
        
        start = (self.start_vertiport.x, self.start_vertiport.y, self.start_vertiport.z)
        goal = (self.goal_vertiport.x, self.goal_vertiport.y, self.goal_vertiport.z)
        
        start_node = LowCarbonNode(start[0], start[1], start[2])
        start_node.h = self._calculate_h_cost(start_node, goal)
        start_node.g = 0.0
        start_node.f = start_node.g + start_node.h * (self.weights['omega1'] + self.weights['omega2'])
        
        open_set = [start_node]
        closed_set = set()
        
        print(f"  V4动态搜索... 时间: {current_time:.1f}s")
        print(f"  动态障碍物数量: {len(self.dynamic_env.dynamic_obstacles)}")
        
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
            
            # 自适应方向切换
            dist_to_goal = self._calculate_h_cost(current, goal)
            if dist_to_goal < 100:
                directions_to_use = self.directions[:6]
            else:
                directions_to_use = self.directions
            
            for direction in directions_to_use:
                neighbor = self._get_neighbor(current, direction, current_time)
                if not neighbor:
                    continue
                
                neighbor_key = (neighbor.x, neighbor.y, neighbor.z)
                if neighbor_key in closed_set:
                    continue
                
                # 检查安全约束（包括动态障碍物）
                if not self._check_safety_constraints(neighbor, current_time):
                    continue
                
                # 检查eVTOL标准约束
                if not self._check_standard_constraints(neighbor):
                    continue
                
                # 计算风场代价
                wind_cost = self._calculate_wind_cost(current, neighbor, current_time)
                
                # 获取祖父节点用于计算转向角
                grandparent = None
                if current.parent:
                    grandparent = current.parent
                
                # 更新低碳指标
                update_node_metrics(current, neighbor, grandparent)
                
                # 设置父子关系
                neighbor.parent = current
                
                # 计算g值和f值（加入风场成本）
                neighbor.h = self._calculate_h_cost(neighbor, goal)
                neighbor.g = self._calculate_g_cost(neighbor) + \
                           self._calculate_standard_cost(neighbor, current) + \
                           wind_cost * self.weights['alpha5'] * 100
                neighbor.f = neighbor.g + neighbor.h * (self.weights['omega1'] + self.weights['omega2'])
                
                # 检查是否已在open_set中
                in_open = False
                for i, node in enumerate(open_set):
                    if node.x == neighbor.x and node.y == neighbor.y and node.z == neighbor.z:
                        if neighbor.g < node.g:
                            open_set[i] = neighbor
                        in_open = True
                        break
                
                if not in_open:
                    open_set.append(neighbor)
            
            if iteration % 10000 == 0:
                print(f"    进度: {iteration} 次迭代, open_set大小: {len(open_set)}")
        
        print(f"  搜索失败: 达到最大迭代次数 {max_iterations}")
        return []
    
    def _check_safety_constraints(self, node: LowCarbonNode, time: float) -> bool:
        """检查安全约束（包括动态障碍物）"""
        return self.dynamic_env.is_safe(node.x, node.y, node.z, time)
    
    def _calculate_wind_cost(self, current: LowCarbonNode, 
                             neighbor: LowCarbonNode, time: float) -> float:
        """计算风场代价"""
        if not self.dynamic_env.wind_field:
            return 0.0
        
        # 计算飞行方向
        dx = neighbor.x - current.x
        dy = neighbor.y - current.y
        dz = neighbor.z - current.z
        
        # 获取风场代价
        wind_cost = self.dynamic_env.wind_field.calculate_wind_cost(
            current.x + dx/2, current.y + dy/2, current.z + dz/2,
            (dx, dy, dz), time
        )
        
        return wind_cost
    
    def _check_standard_constraints(self, node: LowCarbonNode) -> bool:
        """检查eVTOL标准约束（与V3相同）"""
        # 1. 检查是否在建筑物内（已在is_safe中检查）
        
        # 2. 检查起点/终点的安全区约束
        dist_to_start = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                (node.y - self.start_vertiport.y)**2)
        
        dist_to_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                               (node.y - self.goal_vertiport.y)**2)
        
        # 起点：起飞阶段
        if dist_to_start < 50:
            if not self.start_vertiport.check_approach_surface_clearance(
                node.x, node.y, node.z, is_departure=True):
                return False
        
        # 终点：进近阶段
        if dist_to_goal < 50:
            if not self.goal_vertiport.check_approach_surface_clearance(
                node.x, node.y, node.z, is_departure=False):
                return False
        
        # 3. 检查无障碍空间约束
        if dist_to_start < 30:
            if not self.start_vertiport.check_in_safety_zone(node.x, node.y, node.z):
                return False
        
        if dist_to_goal < 30:
            if not self.goal_vertiport.check_in_safety_zone(node.x, node.y, node.z):
                return False
        
        return True
    
    def _calculate_standard_cost(self, node: LowCarbonNode, 
                                 current: LowCarbonNode) -> float:
        """计算eVTOL标准约束成本（与V3相同）"""
        cost = 0.0
        
        dist_to_start = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                (node.y - self.start_vertiport.y)**2 +
                                (node.z - self.start_vertiport.z)**2)
        dist_to_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                               (node.y - self.goal_vertiport.y)**2 +
                               (node.z - self.goal_vertiport.z)**2)
        
        if dist_to_start < 100:
            ofv_cost = self.start_vertiport.get_OFV_clearance_constraint(
                node.x, node.y, node.z)
            cost += ofv_cost * 100
        
        if dist_to_goal < 100:
            ofv_cost = self.goal_vertiport.get_OFV_clearance_constraint(
                node.x, node.y, node.z)
            cost += ofv_cost * 100
        
        if dist_to_start < 200:
            horizontal_dist = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                      (node.y - self.start_vertiport.y)**2)
            min_height = self.start_vertiport.z + \
                        horizontal_dist * self.standard.climb_surface_slope
            if node.z < min_height:
                cost += (min_height - node.z) * 10
        
        if dist_to_goal < 200:
            horizontal_dist = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                                      (node.y - self.goal_vertiport.y)**2)
            min_height = self.goal_vertiport.z + \
                        horizontal_dist * self.standard.approach_surface_slope
            if node.z < min_height:
                cost += (min_height - node.z) * 10
        
        return cost * self.standard_weight
    
    def _calculate_g_cost(self, node: LowCarbonNode) -> float:
        """计算低碳综合成本 g（与V3相同）"""
        energy = node.accumulated_length * 0.001
        energy += node.accumulated_vertical_change * 0.0005
        energy += node.accumulated_turn_angle * 0.1
        
        g_cost = (
            self.weights['alpha1'] * node.accumulated_length +
            self.weights['alpha2'] * energy +
            self.weights['alpha3'] * node.accumulated_vertical_change +
            self.weights['alpha4'] * node.accumulated_turn_angle
        )
        return g_cost
    
    def _calculate_h_cost(self, node: LowCarbonNode, goal: Tuple[int,int,int]) -> float:
        """计算启发式成本 h"""
        return np.sqrt((goal[0]-node.x)**2 + (goal[1]-node.y)**2 + 
                      (goal[2]-node.z)**2) * 0.5
    
    def _get_neighbor(self, node: LowCarbonNode, direction: Tuple[int,int,int],
                     time: float) -> Optional[LowCarbonNode]:
        """获取有效邻居节点"""
        step_xy = 20
        step_z = 10
        
        x = node.x + direction[0] * step_xy
        y = node.y + direction[1] * step_xy
        z = node.z + direction[2] * step_z
        
        if (0 <= x < self.env.size_x and
            0 <= y < self.env.size_y and
            0 <= z < self.env.size_z and
            self.dynamic_env.is_safe(x, y, z, time)):
            return LowCarbonNode(x, y, z)
        return None
    
    def _is_goal(self, node: LowCarbonNode, goal: Tuple[int,int,int]) -> bool:
        """检查是否到达目标"""
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


def test_v4_basic():
    """V4基础测试 - 动态环境与风场"""
    print("="*60)
    print("eVTOL路径规划V4 - 动态环境与风场测试")
    print("="*60)
    
    # 创建eVTOL标准参数
    standard = EVTOLStandard()
    
    # 从JSON加载地图
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('/home/ayan1907/evtol/data/test_maps/test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    # 创建起降场
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    # 创建动态环境
    dynamic_env = DynamicEnvironment(env)
    
    # 创建风场（东南风，风速8m/s）
    wind_field = WindField(
        env, base_wind_speed=8.0, base_wind_direction=135.0,
        turbulence_intensity=0.15, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    
    print(f"\n环境配置:")
    print(f"  风速: {wind_field.base_wind_speed} m/s")
    print(f"  风向: {wind_field.base_wind_direction}° (东南风)")
    print(f"  湍流强度: {wind_field.turbulence_intensity}")
    
    # 添加动态障碍物
    dynamic_env.add_dynamic_obstacle(
        DynamicObstacle(
            id=1, type='drone', x=1250, y=1250, z=100,
            vx=5.0, vy=5.0, vz=0.0, size=15.0,
            valid_time_start=0.0, valid_time_end=300.0
        )
    )
    
    dynamic_env.add_dynamic_obstacle(
        DynamicObstacle(
            id=2, type='helicopter', x=1500, y=1800, z=80,
            vx=-3.0, vy=-2.0, vz=0.5, size=30.0,
            valid_time_start=50.0, valid_time_end=200.0
        )
    )
    
    print(f"  动态障碍物: {len(dynamic_env.dynamic_obstacles)} 个")
    
    # 创建规划器
    planner = LowCarbonAStarV4(
        env, standard, start_vertiport, goal_vertiport, dynamic_env,
        omega1=0.7, omega2=0.3,
        alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25, alpha5=0.2,
        standard_weight=0.5
    )
    
    # 搜索路径（当前时间0秒）
    path = planner.find_path(max_iterations=200000, current_time=0.0)
    
    if path:
        print(f"\n成功找到路径!")
        print(f"  路径长度: {len(path)} 步")
        print(f"  总转向角: {path[-1].accumulated_turn_angle:.4f} rad "
              f"({path[-1].accumulated_turn_angle*180/math.pi:.1f}°)")
        print(f"  总垂直变化: {path[-1].accumulated_vertical_change:.2f} m")
        print(f"  累计长度: {path[-1].accumulated_length:.2f} m")
    else:
        print("\n路径搜索失败")
    
    return path


if __name__ == "__main__":
    test_v4_basic()
