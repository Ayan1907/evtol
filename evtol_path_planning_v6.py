#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V6版本 - 国标合规修复 + 集成改进风场
基于V5,修复：
1. 删除alpha6 - 国标高度约束改为100%硬约束
2. 修复OFV范围 - OFV只检查起降场周边50m范围(国标要求)
3. 集成改进风场 - 时空相关性 + 增强建筑物效应
保留V5所有功能：动态障碍物、城市风场、风场代价、完整国标合规等
"""

import numpy as np
import math
import json
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_bearing
import time

# 导入V4的核心类
from evtol_path_planning_v4 import (
    WindField, DynamicObstacle, DynamicEnvironment
)
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment

# 导入改进风场
from improved_wind_field import ImprovedWindField


@dataclass
class FlightProfile:
    """
    飞行剖面配置 - 基于GB 38031-2022国标
    """
    # OFV无障碍空间
    ofv_height: float = 30.0  # OFV高度 (m)
    
    # 巡航高度层
    cruise_min: float = 200.0  # 巡航层最小高度 (m)
    cruise_max: float = 300.0  # 巡航层最大高度 (m)
    
    # 起升/下降阶段距离阈值
    climb_distance_threshold: float = 500.0  # 起升阶段水平距离 (m)
    approach_distance_threshold: float = 500.0  # 下降阶段水平距离 (m)
    
    # 限制面坡度
    climb_surface_slope: float = 0.1  # 爬升面 (1:10)
    approach_surface_slope: float = 0.083  # 进近面 (1:12)
    transition_surface_slope: float = 0.143  # 过渡面 (1:7)
    
    # 允许的高度偏差
    height_tolerance: float = 10.0  # 高度容差 (m)
    
    def get_flight_phase(self, current_pos: Tuple[float, float, float],
                       start_pos: Tuple[float, float, float],
                       goal_pos: Tuple[float, float, float]) -> str:
        """
        判断当前飞行阶段
        
        返回: 'climb' | 'cruise' | 'approach'
        """
        dist_to_start = math.sqrt(
            (current_pos[0] - start_pos[0])**2 +
            (current_pos[1] - start_pos[1])**2
        )
        dist_to_goal = math.sqrt(
            (current_pos[0] - goal_pos[0])**2 +
            (current_pos[1] - goal_pos[1])**2
        )
        
        if dist_to_start < self.climb_distance_threshold:
            return 'climb'
        elif dist_to_goal < self.approach_distance_threshold:
            return 'approach'
        else:
            return 'cruise'
    
    def get_allowed_height_range(self, current_pos: Tuple[float, float, float],
                                 start_pos: Tuple[float, float, float],
                                 goal_pos: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        根据国标获取允许的高度范围
        
        返回: (min_height, max_height)
        """
        phase = self.get_flight_phase(current_pos, start_pos, goal_pos)
        
        if phase == 'climb':
            # 起升阶段: 按爬升面坡度爬升到巡航高度
            horizontal_dist = math.sqrt(
                (current_pos[0] - start_pos[0])**2 +
                (current_pos[1] - start_pos[1])**2
            )
            # 计算限制面要求的最小高度
            min_height = start_pos[2] + horizontal_dist * self.climb_surface_slope
            # 确保不低于OFV
            min_height = max(min_height, start_pos[2])
            # V5修复: 起升阶段结束时必须至少到达cruise_min
            if horizontal_dist >= self.climb_distance_threshold * 0.8:
                # 在起升阶段后期,强制向巡航层靠拢
                min_height = max(min_height, self.cruise_min * 0.9)
            max_height = self.cruise_max
            
        elif phase == 'approach':
            # 下降阶段: 按进近面坡度下降到目标高度
            horizontal_dist = math.sqrt(
                (current_pos[0] - goal_pos[0])**2 +
                (current_pos[1] - goal_pos[1])**2
            )
            min_height = goal_pos[2] + horizontal_dist * self.approach_surface_slope
            max_height = self.cruise_max
            # 确保不低于OFV
            min_height = max(min_height, goal_pos[2])
            
        else:  # cruise
            # 巡航阶段: 固定在巡航层
            min_height = self.cruise_min
            max_height = self.cruise_max
        
        return (min_height, max_height)
    
    def check_height_constraint(self, current_pos: Tuple[float, float, float],
                               start_pos: Tuple[float, float, float],
                               goal_pos: Tuple[float, float, float]) -> Tuple[bool, float]:
        """
        检查高度是否符合国标约束
        
        返回: (is_valid, violation_amount)
               violation_amount > 0 表示超出约束的量
        """
        min_h, max_h = self.get_allowed_height_range(
            current_pos, start_pos, goal_pos
        )
        
        current_height = current_pos[2]
        
        if current_height < min_h - self.height_tolerance:
            violation = min_h - current_height
            return (False, violation)
        elif current_height > max_h + self.height_tolerance:
            violation = current_height - max_h
            return (False, violation)
        else:
            return (True, 0.0)


class LowCarbonAStarV6:
    """
    低碳A*算法V6版本 - 国标合规修复 + 集成改进风场
    """
    
    def __init__(self, env: CityEnvironment, standard: EVTOLStandard,
                 start_vertiport: Vertiport, goal_vertiport: Vertiport,
                 dynamic_env: DynamicEnvironment,
                 flight_profile: Optional[FlightProfile] = None,
                 omega1: float = 0.7, omega2: float = 0.3,
                 alpha1: float = 0.3, alpha2: float = 0.2,
                 alpha3: float = 0.25, alpha4: float = 0.25,
                 alpha5: float = 0.25,  # 风场权重(V6增强)
                 # V6修复: 删除alpha6 - 国标高度约束改为100%硬约束
                 standard_weight: float = 0.5,
                 use_improved_wind: bool = True):  # V6新增: 是否使用改进风场
        """
        初始化V6规划器

        参数:
            use_improved_wind: 是否使用改进风场(True)或标准风场(False)
        """
        self.env = env
        self.standard = standard
        self.start_vertiport = start_vertiport
        self.goal_vertiport = goal_vertiport
        self.dynamic_env = dynamic_env
        self.standard_weight = standard_weight
        
        # 飞行剖面配置
        self.flight_profile = flight_profile if flight_profile else FlightProfile()
        
        self.weights = {
            'omega1': omega1,  # 长度权重
            'omega2': omega2,  # 能量权重
            'alpha1': alpha1,  # 距离系数
            'alpha2': alpha2,  # 能量系数
            'alpha3': alpha3,  # 垂直变化系数
            'alpha4': alpha4,  # 转向角系数
            'alpha5': alpha5   # 风场系数
            # V6修复: 删除alpha6
        }
        self.directions = self._init_directions()
        self.replan_count = 0

        # V6新增: 集成改进风场
        self.use_improved_wind = use_improved_wind
        if use_improved_wind and hasattr(dynamic_env, 'wind_field'):
            print("  使用改进风场模型(ImprovedWindField)")
            # 创建改进风场,复制原风场的基本参数
            old_wind_field = dynamic_env.wind_field
            improved_wind = ImprovedWindField(
                base_wind_speed=old_wind_field.base_wind_speed,
                base_wind_direction=old_wind_field.base_wind_direction,
                turbulence_intensity=old_wind_field.turbulence_intensity
            )
            # 设置环境
            improved_wind.set_environment(env)
            # 替换风场
            dynamic_env.set_wind_field(improved_wind)

        # V6新增: 噪声约束(硬约束,不加入目标函数)
        from noise_model import NoiseModel
        from noise_constraint import NoiseConstraint, NoiseConstraintConfig
        self.noise_model = NoiseModel(dynamic_env, source_level=85.0, reference_distance=100.0)
        noise_config = NoiseConstraintConfig(
            max_average_noise=95.0,  # 提高阈值，避免过于严格
            max_peak_noise=105.0,  # 提高峰值阈值
            nighttime_strict=False  # 暂时关闭夜间严格模式
        )
        self.noise_constraint = NoiseConstraint(self.noise_model, noise_config)
    
    def _init_directions(self) -> List[Tuple[int, int, int]]:
        """初始化26种移动方向"""
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if (dx, dy, dz) != (0, 0, 0):
                        directions.append((dx, dy, dz))
        return directions
    
    def find_path(self, max_iterations: int = 200000, 
                 current_time: float = 0.0) -> List[LowCarbonNode]:
        """
        主搜索函数 - V5版本(完整国标合规)
        
        参数:
            max_iterations: 最大迭代次数
            current_time: 当前时间(秒)
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
        
        if getattr(self, 'verbose', True):
            print(f"  V5国标合规搜索... 时间: {current_time:.1f}s")
            print(f"  动态障碍物数量: {len(self.dynamic_env.dynamic_obstacles)}")
            print(f"  巡航高度层: {self.flight_profile.cruise_min}-{self.flight_profile.cruise_max}m")
        
        for iteration in range(max_iterations):
            if not open_set:
                if getattr(self, 'verbose', True):
                    print(f"  搜索失败: open_set为空，迭代 {iteration} 次")
                break
                
            current = min(open_set, key=lambda n: n.f)
            
            if self._is_goal(current, goal):
                if getattr(self, 'verbose', True):
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
                
                # 检查安全约束(包括动态障碍物)
                if not self._check_safety_constraints(neighbor, current_time):
                    continue

                # V6新增: 检查噪声约束(硬约束)
                if not self._check_noise_constraint(neighbor, current_time):
                    continue

                # V6修复: 国标高度约束必须是100%硬约束
                # V5新增: 检查国标高度约束(全程应用)
                if not self._check_standard_height_constraint(neighbor):
                    continue
                
                # 检查eVTOL标准约束(OFV、限制面等)
                if not self._check_standard_constraints(neighbor):
                    continue
                
                # 计算风场代价
                wind_cost = self._calculate_wind_cost(current, neighbor, current_time)
                
                # V6修复: 删除高度约束代价计算 - 国标高度约束现在是100%硬约束
                # V5新增: 计算国标高度约束代价 (已删除)
                
                # 获取祖父节点用于计算转向角
                grandparent = None
                if current.parent:
                    grandparent = current.parent
                
                # 更新低碳指标
                update_node_metrics(current, neighbor, grandparent)
                
                # 设置父子关系
                neighbor.parent = current
                
                # V6修复: 删除height_violation_cost - 国标高度约束现在是硬约束
                # 计算g值和f值(V6: 删除alpha6和height_violation_cost)
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
            
            if getattr(self, 'verbose', True) and iteration % 10000 == 0:
                print(f"    进度: {iteration} 次迭代, open_set大小: {len(open_set)}")
        
        if getattr(self, 'verbose', True):
            print(f"  搜索失败: 达到最大迭代次数 {max_iterations}")
        return []
    
    def _check_standard_height_constraint(self, node: LowCarbonNode) -> bool:
        """
        V6修复: 国标高度约束 - 100%硬约束
        
        不满足国标高度约束的节点直接拒绝(硬约束)
        """
        current_pos = (node.x, node.y, node.z)
        start_pos = (self.start_vertiport.x, self.start_vertiport.y, self.start_vertiport.z)
        goal_pos = (self.goal_vertiport.x, self.goal_vertiport.y, self.goal_vertiport.z)
        
        is_valid, _ = self.flight_profile.check_height_constraint(
            current_pos, start_pos, goal_pos
        )
        
        return is_valid
    
    def _calculate_height_violation_cost(self, node: LowCarbonNode) -> float:
        """
        V6已删除: 此函数不再使用

        国标高度约束现在是100%硬约束,不需要代价计算
        """
        return 0.0

    def _check_safety_constraints(self, node: LowCarbonNode, time: float) -> bool:
        """检查安全约束(包括动态障碍物)"""
        return self.dynamic_env.is_safe(node.x, node.y, node.z, time)

    def _check_noise_constraint(self, node: LowCarbonNode, current_time: float) -> bool:
        """
        V6新增: 检查噪声约束(硬约束)

        如果噪声超标,直接返回False (节点不可达)

        这样规划出的路径天然满足噪声要求,不需要在目标函数中体现
        """
        current_hour = int(current_time / 3600) % 24

        return self.noise_constraint.check_noise_constraint(
            node.x, node.y, node.z,
            current_hour=current_hour
        )
    
    def _calculate_wind_cost(self, current: LowCarbonNode,
                             neighbor: LowCarbonNode, time: float) -> float:
        """
        V6新增: 计算风场代价 - 支持改进风场
        """
        if not self.dynamic_env.wind_field:
            return 0.0

        # 计算飞行方向
        dx = neighbor.x - current.x
        dy = neighbor.y - current.y
        dz = neighbor.z - current.z

        # 计算中点位置的风场代价
        mid_x = current.x + dx / 2
        mid_y = current.y + dy / 2
        mid_z = current.z + dz / 2

        # 获取风场代价
        wind_cost = self.dynamic_env.wind_field.calculate_wind_cost(
            mid_x, mid_y, mid_z,
            (dx, dy, dz), time
        )

        return wind_cost

    def print_wind_field_info(self):
        """
        V6新增: 打印风场信息
        """
        if hasattr(self.dynamic_env, 'wind_field') and self.dynamic_env.wind_field:
            wf = self.dynamic_env.wind_field
            print(f"\n风场配置:")
            print(f"  类型: 改进风场(ImprovedWindField)" if isinstance(wf, ImprovedWindField) else f"  类型: 标准风场(WindField)")
            print(f"  基础风速: {wf.base_wind_speed:.1f} m/s")
            print(f"  基础风向: {wf.base_wind_direction:.1f}°")
            print(f"  湍流强度: {wf.turbulence_intensity:.2f}")

            if isinstance(wf, ImprovedWindField):
                print(f"  阵风幅度: {wf.gust_amplitude:.1f} m/s")
                print(f"  阵风频率: {wf.gust_frequency:.2f} Hz")
                print(f"  风摆幅度: {wf.wind_variation_amplitude:.1f}°")
                print(f"  空间相关尺度: {wf.spatial_scale:.4f}")
                print(f"  建筑物尾流因子: {wf.building_wake_factor:.2f}")
                print(f"  建筑物尾流长度: {wf.building_wake_length:.1f}m")
                print(f"  通道效应因子: {wf.building_channel_factor:.2f}")
        else:
            print(f"\n风场: 未启用")

    def _check_standard_constraints(self, node: LowCarbonNode) -> bool:
        """
        V6修复: 检查eVTOL标准约束
        
        V6改进:
        - 限制面全程应用(起降场1000m范围)
        - OFV检查严格限制在起降场50m范围内(国标要求)
        """
        # 1. 检查起点/终点的安全区约束
        dist_to_start = np.sqrt((node.x - self.start_vertiport.x)**2 +
                                (node.y - self.start_vertiport.y)**2)

        dist_to_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 +
                               (node.y - self.goal_vertiport.y)**2)

        # V5: 全程应用起降场限制面检查
        # 起点: 起飞阶段
        if dist_to_start < 1000:  # 扩大检查范围
            if not self.start_vertiport.check_approach_surface_clearance(
                node.x, node.y, node.z, is_departure=True):
                return False

        # 终点: 进近阶段
        if dist_to_goal < 1000:  # 扩大检查范围
            if not self.goal_vertiport.check_approach_surface_clearance(
                node.x, node.y, node.z, is_departure=False):
                return False

        # V6修复: OFV只检查起降场周边50m范围(国标要求)
        if dist_to_start < 50:  # 严格50m范围
            if not self.start_vertiport.check_in_safety_zone(node.x, node.y, node.z):
                return False

        if dist_to_goal < 50:  # 严格50m范围
            if not self.goal_vertiport.check_in_safety_zone(node.x, node.y, node.z):
                return False

        return True
    
    def _calculate_standard_cost(self, node: LowCarbonNode,
                                 current: LowCarbonNode) -> float:
        """
        V6修复: 计算eVTOL标准约束成本

        V6改进:
        - OFV严格限制在50m范围(国标要求)
        - 限制面检查范围1000m并全程应用
        """
        cost = 0.0

        dist_to_start = np.sqrt((node.x - self.start_vertiport.x)**2 +
                                (node.y - self.start_vertiport.y)**2 +
                                (node.z - self.start_vertiport.z)**2)
        dist_to_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 +
                               (node.y - self.goal_vertiport.y)**2 +
                               (node.z - self.goal_vertiport.z)**2)

        # V6修复: OFV成本严格限制在50m范围(国标要求)
        if dist_to_start < 50:  # 严格50m范围
            ofv_cost = self.start_vertiport.get_OFV_clearance_constraint(
                node.x, node.y, node.z)
            cost += ofv_cost * 100

        if dist_to_goal < 50:  # 严格50m范围
            ofv_cost = self.goal_vertiport.get_OFV_clearance_constraint(
                node.x, node.y, node.z)
            cost += ofv_cost * 100

        # 限制面成本(扩大范围并全程应用)
        # 起点爬升面
        horizontal_dist_start = np.sqrt((node.x - self.start_vertiport.x)**2 + 
                                      (node.y - self.start_vertiport.y)**2)
        if horizontal_dist_start < 1000:  # 扩大到1000m
            min_height = self.start_vertiport.z + \
                        horizontal_dist_start * self.standard.climb_surface_slope
            if node.z < min_height:
                cost += (min_height - node.z) * 10
        
        # 终点进近面
        horizontal_dist_goal = np.sqrt((node.x - self.goal_vertiport.x)**2 + 
                                      (node.y - self.goal_vertiport.y)**2)
        if horizontal_dist_goal < 1000:  # 扩大到1000m
            min_height = self.goal_vertiport.z + \
                        horizontal_dist_goal * self.standard.approach_surface_slope
            if node.z < min_height:
                cost += (min_height - node.z) * 10
        
        return cost * self.standard_weight
    
    def _calculate_g_cost(self, node: LowCarbonNode) -> float:
        """计算低碳综合成本 g"""
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
        """
        获取有效邻居节点
        
        V5改进: 考虑国标高度约束,限制垂直方向
        """
        step_xy = 50
        step_z = 20
        
        x = node.x + direction[0] * step_xy
        y = node.y + direction[1] * step_xy
        z = node.z + direction[2] * step_z
        
        # 基础边界检查
        if not (0 <= x < self.env.size_x and
                0 <= y < self.env.size_y and
                0 <= z < self.env.size_z):
            return None
        
        # V5新增: 预先检查高度约束,快速剪枝
        current_pos = (x, y, z)
        start_pos = (self.start_vertiport.x, self.start_vertiport.y, self.start_vertiport.z)
        goal_pos = (self.goal_vertiport.x, self.goal_vertiport.y, self.goal_vertiport.z)
        
        phase = self.flight_profile.get_flight_phase(current_pos, start_pos, goal_pos)
        min_h, max_h = self.flight_profile.get_allowed_height_range(
            current_pos, start_pos, goal_pos
        )
        
        # V5修复: 检查高度约束
        if z < min_h - 30 or z > max_h + 30:  # 放宽容差,避免搜索空间过小
            return None
        
        # 巡航阶段额外检查
        if phase == 'cruise':
            # 巡航阶段: 尝试保持在巡航层内,但不强制
            # 通过代价函数引导,而不是硬约束
            pass
        
        # 安全性检查
        if self.dynamic_env.is_safe(x, y, z, time):
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
    
    def analyze_path_profile(self, path: List[LowCarbonNode]) -> Dict:
        """
        V5新增: 分析路径的飞行剖面
        
        返回: 包含各阶段统计信息的字典
        """
        if not path:
            return {}
        
        start_pos = (self.start_vertiport.x, self.start_vertiport.y, self.start_vertiport.z)
        goal_pos = (self.goal_vertiport.x, self.goal_vertiport.y, self.goal_vertiport.z)
        
        climb_nodes = []
        cruise_nodes = []
        approach_nodes = []
        
        for node in path:
            phase = self.flight_profile.get_flight_phase(
                (node.x, node.y, node.z), start_pos, goal_pos
            )
            
            if phase == 'climb':
                climb_nodes.append(node)
            elif phase == 'cruise':
                cruise_nodes.append(node)
            else:  # approach
                approach_nodes.append(node)
        
        return {
            'climb': {
                'count': len(climb_nodes),
                'min_height': min(n.z for n in climb_nodes) if climb_nodes else 0,
                'max_height': max(n.z for n in climb_nodes) if climb_nodes else 0,
                'avg_height': np.mean([n.z for n in climb_nodes]) if climb_nodes else 0
            },
            'cruise': {
                'count': len(cruise_nodes),
                'min_height': min(n.z for n in cruise_nodes) if cruise_nodes else 0,
                'max_height': max(n.z for n in cruise_nodes) if cruise_nodes else 0,
                'avg_height': np.mean([n.z for n in cruise_nodes]) if cruise_nodes else 0
            },
            'approach': {
                'count': len(approach_nodes),
                'min_height': min(n.z for n in approach_nodes) if approach_nodes else 0,
                'max_height': max(n.z for n in approach_nodes) if approach_nodes else 0,
                'avg_height': np.mean([n.z for n in approach_nodes]) if approach_nodes else 0
            }
        }


def test_v5_basic(map_file='/home/ayan1907/evtol/test_map_v3_compatible.json', save_result=False):
    """
    V5基础测试 - 完整国标合规

    参数:
        map_file: 地图JSON文件路径
        save_result: 是否保存结果到JSON文件
    """
    print("="*60)
    print("eVTOL路径规划V5 - 完整国标合规测试")
    print("="*60)

    # 创建eVTOL标准参数
    standard = EVTOLStandard()

    # 从JSON加载地图(保留V4的地图加载方式)
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json(map_file)

    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])

    print(f"\n地图信息:")
    print(f"  地图文件: {map_file}")
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")
    print(f"  建筑物数量: {map_data['map_info']['n_buildings']} 栋")
    
    # 创建起降场
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    
    # 创建动态环境(保留V4的动态环境)
    dynamic_env = DynamicEnvironment(env)
    
    # 创建风场(保留V4的风场)
    wind_field = WindField(
        env, base_wind_speed=8.0, base_wind_direction=135.0,
        turbulence_intensity=0.15, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    
    print(f"\n环境配置:")
    print(f"  风速: {wind_field.base_wind_speed} m/s")
    print(f"  风向: {wind_field.base_wind_direction}° (东南风)")
    print(f"  湍流强度: {wind_field.turbulence_intensity}")
    
    # 添加动态障碍物(保留V4的配置)
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
    
    # V5新增: 创建飞行剖面配置
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=150.0,   # 降低以适应地图size_z=200
        cruise_max=190.0,
        climb_distance_threshold=500.0,
        approach_distance_threshold=500.0
    )
    
    print(f"\n飞行剖面配置(V5新增):")
    print(f"  巡航高度层: {flight_profile.cruise_min}-{flight_profile.cruise_max}m")
    print(f"  起升阈值: {flight_profile.climb_distance_threshold}m")
    print(f"  下降阈值: {flight_profile.approach_distance_threshold}m")

    # 创建规划器(V6)
    planner = LowCarbonAStarV6(
        env, standard, start_vertiport, goal_vertiport, dynamic_env, flight_profile,
        omega1=0.7, omega2=0.3,
        alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25,
        alpha5=0.2,  # 风场权重(保留V4)
        # V6修复: 删除alpha6 - 国标高度约束改为100%硬约束
        standard_weight=0.5
    )

    print(f"\n算法配置:")
    print(f"  ω₁={planner.weights['omega1']}, ω₂={planner.weights['omega2']}")
    print(f"  α₁={planner.weights['alpha1']}, α₂={planner.weights['alpha2']}")
    print(f"  α₃={planner.weights['alpha3']}, α₄={planner.weights['alpha4']}")
    print(f"  α₅={planner.weights['alpha5']} (风场权重)")
    # V6修复: 删除alpha6打印
    print(f"  标准约束权重: {planner.standard_weight}")

    # 搜索路径
    path = planner.find_path(max_iterations=200000, current_time=0.0)

    if path:
        print(f"\n{'='*60}")
        print(f"成功找到路径!")
        print(f"{'='*60}")
        print(f"  路径长度: {len(path)} 步")
        print(f"  总转向角: {path[-1].accumulated_turn_angle:.4f} rad "
              f"({path[-1].accumulated_turn_angle*180/math.pi:.1f}°)")
        print(f"  总垂直变化: {path[-1].accumulated_vertical_change:.2f} m")
        print(f"  累计长度: {path[-1].accumulated_length:.2f} m")

        # V5新增: 飞行剖面分析
        print(f"\n飞行剖面分析:")
        print(f"{'='*60}")
        profile = planner.analyze_path_profile(path)

        print(f"  起升阶段: {profile['climb']['count']} 个节点")
        print(f"    高度范围: {profile['climb']['min_height']:.1f} - {profile['climb']['max_height']:.1f}m")
        print(f"    平均高度: {profile['climb']['avg_height']:.1f}m")

        print(f"  巡航阶段: {profile['cruise']['count']} 个节点")
        print(f"    高度范围: {profile['cruise']['min_height']:.1f} - {profile['cruise']['max_height']:.1f}m")
        print(f"    平均高度: {profile['cruise']['avg_height']:.1f}m")

        print(f"  下降阶段: {profile['approach']['count']} 个节点")
        print(f"    高度范围: {profile['approach']['min_height']:.1f} - {profile['approach']['max_height']:.1f}m")
        print(f"    平均高度: {profile['approach']['avg_height']:.1f}m")

        # 路径平滑优化
        print(f"\n路径平滑优化...")
        path_before_stats = {
            'nodes': len(path),
            'turn_angle': path[-1].accumulated_turn_angle * 180 / np.pi,
            'length': path[-1].accumulated_length
        }
        path = smooth_path_nodes(path, start_pos, goal_pos, flight_profile, planner)

        path_after_stats = {
            'nodes': len(path),
            'turn_angle': path[-1].accumulated_turn_angle * 180 / np.pi,
            'length': path[-1].accumulated_length
        }

        print(f"  平滑前: 节点{path_before_stats['nodes']}, 转向{path_before_stats['turn_angle']:.1f}°, 长度{path_before_stats['length']:.2f}m")
        print(f"  平滑后: 节点{path_after_stats['nodes']}, 转向{path_after_stats['turn_angle']:.1f}°, 长度{path_after_stats['length']:.2f}m")
        print(f"  变化: 节点{path_after_stats['nodes']-path_before_stats['nodes']:+d}, "
              f"转向{path_after_stats['turn_angle']-path_before_stats['turn_angle']:+.1f}°, "
              f"长度{path_after_stats['length']-path_before_stats['length']:+.2f}m")
        print(f"{'='*60}")

        # 保存结果到JSON
        if save_result:
            result_file = map_file.replace('.json', '_result.json')
            result_data = {
                'map_file': map_file,
                'map_info': map_data.get('map_info', {}),
                'start': list(start_pos),
                'goal': list(goal_pos),
                'path_before': {
                    'nodes': path_before_stats['nodes'],
                    'turn_angle': path_before_stats['turn_angle'],
                    'length': path_before_stats['length']
                },
                'path_after': {
                    'nodes': path_after_stats['nodes'],
                    'turn_angle': path_after_stats['turn_angle'],
                    'length': path_after_stats['length']
                },
                'path_data': [
                    {
                        'x': node.x,
                        'y': node.y,
                        'z': node.z,
                        'accumulated_length': node.accumulated_length,
                        'accumulated_turn_angle': node.accumulated_turn_angle,
                        'accumulated_vertical_change': node.accumulated_vertical_change
                    }
                    for node in path
                ],
                'flight_profile': {
                    'cruise_min': flight_profile.cruise_min,
                    'cruise_max': flight_profile.cruise_max,
                    'climb_distance_threshold': flight_profile.climb_distance_threshold,
                    'approach_distance_threshold': flight_profile.approach_distance_threshold
                },
                'weights': planner.weights
            }

            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"\n  结果已保存到: {result_file}")

    else:
        print("\n路径搜索失败")

    return path


def smooth_path_nodes(path: List[LowCarbonNode],
                      start_pos: Tuple[float, float, float],
                      goal_pos: Tuple[float, float, float],
                      flight_profile: FlightProfile,
                      planner: 'LowCarbonAStarV5',
                      turn_threshold: float = 60.0,
                      max_iterations: int = 10) -> List[LowCarbonNode]:
    """
    通用路径平滑算法 - 基于几何特征而非硬编码坐标

    核心思想：
    1. 检测所有大转弯节点（超过turn_threshold度）
    2. 评估直接连接前后节点是否可行（检查约束）
    3. 如果可行且代价更低，则移除中间节点
    4. 递归优化直到收敛

    参数：
        turn_threshold: 转弯角度阈值（度），超过此值认为是需要优化的转弯
        max_iterations: 最大迭代次数，防止无限循环
    """
    if len(path) < 10:
        return path

    goal_node = path[-1]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        improved = False

        # 遍历所有可能的转弯点
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            nxt = path[i + 1]

            # 计算三个连续节点的转向角
            dx1 = curr.x - prev.x
            dy1 = curr.y - prev.y
            dz1 = curr.z - prev.z

            dx2 = nxt.x - curr.x
            dy2 = nxt.y - curr.y
            dz2 = nxt.z - curr.z

            # 只考虑水平方向的转向角
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)

            turn = abs(angle2 - angle1)
            if turn > np.pi:
                turn = 2 * np.pi - turn

            # 如果转弯角度超过阈值，尝试优化
            if turn > np.radians(turn_threshold):
                # 检查直接连接prev和nxt是否可行
                if _is_direct_connection_feasible(
                    prev, nxt, goal_pos, flight_profile, planner
                ):
                    # 计算直接连接的代价
                    direct_cost = _calculate_segment_cost(prev, nxt, flight_profile)

                    # 计算经过curr的代价
                    indirect_cost = _calculate_segment_cost(prev, curr, flight_profile) + \
                                    _calculate_segment_cost(curr, nxt, flight_profile)

                    # 如果直接连接更优或相差不大，则移除curr
                    if direct_cost < indirect_cost * 1.05:  # 允许5%的代价增加
                        # 创建新路径，跳过curr
                        new_path = path[:i] + path[i+1:]

                        # 插入中间节点以保持平滑（可选）
                        if i + 1 < len(new_path):
                            last_good = new_path[i - 1]
                            target_node = new_path[i]

                            # 插入2个中间节点使路径更平滑
                            num_interp = 2
                            interpolated_nodes = []

                            for step_idx in range(1, num_interp + 1):
                                alpha = step_idx / (num_interp + 1)
                                x = last_good.x + (target_node.x - last_good.x) * alpha
                                y = last_good.y + (target_node.y - last_good.y) * alpha

                                # 高度符合飞行剖面约束
                                z = _calculate_safe_height(
                                    (x, y), last_good.z, target_node.z,
                                    goal_pos, flight_profile
                                )

                                new_node = LowCarbonNode(int(x), int(y), int(z))
                                new_node.parent = last_good
                                seg_len = np.sqrt(
                                    (x - last_good.x)**2 +
                                    (y - last_good.y)**2 +
                                    (z - last_good.z)**2
                                )
                                new_node.accumulated_length = last_good.accumulated_length + seg_len
                                new_node.accumulated_vertical_change = last_good.accumulated_vertical_change + abs(z - last_good.z)
                                new_node.accumulated_energy = new_node.accumulated_length * 0.001

                                # 更新转向角
                                if last_good.parent:
                                    dx_prev = last_good.x - last_good.parent.x
                                    dy_prev = last_good.y - last_good.parent.y
                                    dx_curr = x - last_good.x
                                    dy_curr = y - last_good.y

                                    if (dx_prev != 0 or dy_prev != 0) and (dx_curr != 0 or dy_curr != 0):
                                        angle_prev = np.arctan2(dy_prev, dx_prev)
                                        angle_curr = np.arctan2(dy_curr, dx_curr)
                                        turn_angle = abs(angle_curr - angle_prev)
                                        if turn_angle > np.pi:
                                            turn_angle = 2 * np.pi - turn_angle
                                        new_node.accumulated_turn_angle = last_good.accumulated_turn_angle + turn_angle

                                interpolated_nodes.append(new_node)
                                last_good = new_node

                            # 插入到路径中
                            new_path = new_path[:i] + interpolated_nodes + new_path[i:]

                        # 更新路径并重新计算所有累积指标
                        new_path = _recalculate_path_metrics(new_path)

                        path = new_path
                        improved = True
                        print(f"    迭代{iteration}: 移除节点{i}（转向{turn*180/np.pi:.1f}°）, "
                              f"路径长度 {len(path)}")
                        break

        if not improved:
            break

    print(f"    平滑完成: 迭代{iteration}次, 最终节点数: {len(path)}")
    return path


def _is_direct_connection_feasible(
    start_node: LowCarbonNode,
    end_node: LowCarbonNode,
    goal_pos: Tuple[float, float, float],
    flight_profile: FlightProfile,
    planner: 'LowCarbonAStarV5'
) -> bool:
    """
    检查直接连接两个节点是否可行

    检查：
    1. 距离不过长（避免跨越过大区域）
    2. 所有中间点符合飞行剖面约束
    3. 所有中间点在安全区域内
    """
    # 检查距离
    dist = np.sqrt(
        (end_node.x - start_node.x)**2 +
        (end_node.y - start_node.y)**2 +
        (end_node.z - start_node.z)**2
    )

    if dist > 500:  # 单段距离不超过500m
        return False

    # 抽样检查中间点（每50m检查一个点）
    num_samples = max(1, int(dist / 50))

    for i in range(1, num_samples):
        alpha = i / num_samples
        x = start_node.x + (end_node.x - start_node.x) * alpha
        y = start_node.y + (end_node.y - start_node.y) * alpha
        z = start_node.z + (end_node.z - start_node.z) * alpha

        # 检查高度约束
        current_pos = (x, y, z)
        start_pos = (planner.start_vertiport.x,
                    planner.start_vertiport.y,
                    planner.start_vertiport.z)
        is_valid, _ = flight_profile.check_height_constraint(
            current_pos, start_pos, goal_pos
        )

        if not is_valid:
            return False

        # 检查安全约束（静态障碍物）
        if not planner.dynamic_env.is_safe(x, y, z, 0.0):
            return False

    return True


def _calculate_segment_cost(
    start_node: LowCarbonNode,
    end_node: LowCarbonNode,
    flight_profile: FlightProfile
) -> float:
    """
    计算两个节点之间的代价

    代价包括：
    1. 距离
    2. 高度变化
    3. 转向角
    """
    dist = np.sqrt(
        (end_node.x - start_node.x)**2 +
        (end_node.y - start_node.y)**2 +
        (end_node.z - start_node.z)**2
    )

    height_change = abs(end_node.z - start_node.z)

    # 简化的代价计算
    cost = dist * 1.0 + height_change * 2.0

    return cost


def _calculate_safe_height(
    xy_pos: Tuple[float, float],
    start_height: float,
    end_height: float,
    goal_pos: Tuple[float, float, float],
    flight_profile: FlightProfile
) -> float:
    """
    计算指定位置的安全高度

    符合飞行剖面约束
    """
    x, y = xy_pos

    # 计算到终点的距离
    dist_to_goal = np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)

    # 如果在下降阶段，使用进近面坡度
    if dist_to_goal < flight_profile.approach_distance_threshold:
        min_h = goal_pos[2] + dist_to_goal * flight_profile.approach_surface_slope
        min_h = max(min_h, goal_pos[2])
    else:
        min_h = flight_profile.cruise_min

    # 高度在起始和结束高度之间插值
    interp_height = (start_height + end_height) / 2

    # 确保不低于最小高度
    safe_height = max(interp_height, min_h)

    # 确保不超过巡航高度上限
    safe_height = min(safe_height, flight_profile.cruise_max)

    return int(safe_height)


def _recalculate_path_metrics(path: List[LowCarbonNode]) -> List[LowCarbonNode]:
    """
    重新计算路径中所有节点的累积指标

    包括：
    1. accumulated_length
    2. accumulated_vertical_change
    3. accumulated_turn_angle
    4. accumulated_energy
    """
    if not path:
        return path

    # 重新计算所有累积指标
    for i in range(len(path)):
        if i == 0:
            path[i].parent = None
            path[i].accumulated_length = 0.0
            path[i].accumulated_vertical_change = 0.0
            path[i].accumulated_turn_angle = 0.0
            path[i].accumulated_energy = 0.0
        else:
            path[i].parent = path[i - 1]

            # 计算长度
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            dz = path[i].z - path[i - 1].z
            seg_len = np.sqrt(dx**2 + dy**2 + dz**2)

            path[i].accumulated_length = path[i - 1].accumulated_length + seg_len
            path[i].accumulated_vertical_change = path[i - 1].accumulated_vertical_change + abs(dz)
            path[i].accumulated_energy = path[i].accumulated_length * 0.001

            # 计算转向角
            if i > 1:
                dx_prev = path[i - 1].x - path[i - 2].x
                dy_prev = path[i - 1].y - path[i - 2].y
                dx_curr = dx
                dy_curr = dy

                if (dx_prev != 0 or dy_prev != 0) and (dx_curr != 0 or dy_curr != 0):
                    angle_prev = np.arctan2(dy_prev, dx_prev)
                    angle_curr = np.arctan2(dy_curr, dx_curr)
                    turn_angle = abs(angle_curr - angle_prev)
                    if turn_angle > np.pi:
                        turn_angle = 2 * np.pi - turn_angle

                    path[i].accumulated_turn_angle = path[i - 1].accumulated_turn_angle + turn_angle
                else:
                    path[i].accumulated_turn_angle = path[i - 1].accumulated_turn_angle
            else:
                path[i].accumulated_turn_angle = 0.0

    return path


def create_v6_planner(map_file: str,
                      wind_speed: float = 8.0,
                      wind_direction: float = 315.0,
                      cruise_min: float = 120.0,
                      cruise_max: float = 180.0,
                      climb_distance: float = 400.0,
                      approach_distance: float = 400.0,
                      **kwargs) -> Optional[LowCarbonAStarV6]:
    """
    便捷函数: 创建V6规划器 - 集成改进风场

    参数:
        map_file: 地图文件路径
        wind_speed: 基础风速(m/s)
        wind_direction: 基础风向(度)
        cruise_min: 巡航层最小高度
        cruise_max: 巡航层最大高度
        climb_distance: 起升距离阈值
        approach_distance: 下降距离阈值
        **kwargs: 传递给LowCarbonAStarV6的其他参数(omega1, omega2, alpha1-5等)

    返回: 规划器对象,失败返回None
    """
    try:
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

        # 添加风场 - 先添加标准风场,V6会自动替换为改进风场
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

        # 5. 创建规划器 - 只传递有效的参数
        planner = LowCarbonAStarV6(
            env, standard, start_vertiport, goal_vertiport,
            dynamic_env, flight_profile,
            use_improved_wind=True,  # V6新增: 使用改进风场
            **kwargs  # 允许传递其他参数如omega1, omega2, alpha1-5等
        )

        return planner

    except Exception as e:
        print(f"创建V6规划器失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_v5_basic()
