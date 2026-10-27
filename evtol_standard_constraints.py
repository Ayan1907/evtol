#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL标准约束模型 - V5核心模块
基于GB 38031-2022《电动垂直起降航空器起降场技术要求》
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from typing_extensions import Literal


@dataclass
class VertiportStandard:
    """
    eVTOL起降场标准参数
    基于GB 38031-2022《电动垂直起降航空器起降场技术要求》
    """
    # 航空器尺寸
    aircraft_length: float = 5.73  # 航空器全长 (m)
    aircraft_width: float = 5.61  # 航空器全宽 (m)
    aircraft_height: float = 1.93  # 航空器高度 (m)
    aircraft_diameter: float = 5.63  # 航空器最大尺寸 (m)
    
    # TLOF (Touchdown and Lift-Off Area) - 接地和离地区
    TLOF_diameter: float = field(init=False)  # TLOF直径 (m)
    TLOF_radius: float = field(init=False)  # TLOF半径 (m)
    
    # FATO (Final Approach and Take-Off Area) - 最终进近和起飞区
    FATO_diameter: float = field(init=False)  # FATO直径 (m)
    FATO_radius: float = field(init=False)  # FATO半径 (m)
    
    # Safety Zone - 安全区
    safety_zone_extension: float = 3.0  # 安全区超出FATO边界 (m)
    safety_zone_radius: float = field(init=False)  # 安全区半径 (m)
    
    # OFV (Obstacle Free Volume) - 无障碍空间
    OFV_height: float = 30.0  # OFV高度 (m) - 标准值
    OFV_shape: Literal['cylinder', 'inverted_frustum'] = 'cylinder'  # OFV形状
    
    # 障碍物限制面 (Obstacle Limitation Surfaces)
    # 爬升面 (Climb Surface)
    climb_surface_slope: float = 0.1  # 爬升面坡度 (1:10)
    climb_surface_length: float = 3000.0  # 爬升面水平长度 (m)
    
    # 进近面 (Approach Surface)
    approach_surface_slope: float = 0.083  # 进近面坡度 (1:12)
    approach_surface_length: float = 3000.0  # 进近面水平长度 (m)
    
    # 过渡面 (Transition Surface)
    transition_surface_slope: float = 0.143  # 过渡面坡度 (1:7)
    transition_surface_length: float = 150.0  # 过渡面水平长度 (m)
    
    def __post_init__(self):
        """计算派生参数"""
        # TLOF尺寸 = 航空器最大尺寸
        self.TLOF_diameter = self.aircraft_diameter
        self.TLOF_radius = self.TLOF_diameter / 2.0
        
        # FATO尺寸 = 1.5 × 航空器最大尺寸
        self.FATO_diameter = 1.5 * self.aircraft_diameter
        self.FATO_radius = self.FATO_diameter / 2.0
        
        # 安全区尺寸
        self.safety_zone_radius = self.FATO_radius + self.safety_zone_extension
    
    def get_minimum_safe_height(self, 
                               horizontal_distance: float,
                               surface_type: str = 'climb') -> float:
        """
        计算给定水平距离下的最小安全高度
        
        参数:
            horizontal_distance: 水平距离 (m)
            surface_type: 'climb'(爬升面), 'approach'(进近面), 'transition'(过渡面)
            
        返回:
            最小安全高度 (m)
        """
        if surface_type == 'climb':
            slope = self.climb_surface_slope
        elif surface_type == 'approach':
            slope = self.approach_surface_slope
        elif surface_type == 'transition':
            slope = self.transition_surface_slope
        else:
            raise ValueError(f"Unknown surface type: {surface_type}")
        
        return horizontal_distance * slope
    
    def get_OFV_limits(self, center: Tuple[float, float, float]) -> Dict:
        """
        获取OFV边界
        
        参数:
            center: OFV中心 (x, y, z)
            
        返回:
            OFV边界字典
        """
        cx, cy, cz = center
        
        return {
            'x_center': cx,
            'y_center': cy,
            'z_bottom': cz,
            'z_top': cz + self.OFV_height,
            'radius': self.safety_zone_radius
        }


@dataclass
class SafetyMargins:
    """安全裕度参数"""
    # 静态障碍物缓冲
    static_obstacle_buffer: float = 10.0  # 静态障碍物缓冲距离 (m)
    
    # 动态障碍物缓冲
    dynamic_obstacle_buffer_base: float = 10.0  # 动态障碍物基础缓冲 (m)
    prediction_uncertainty: float = 3.0  # 预测不确定性 (m)
    reaction_time_margin: float = 2.0  # 反应时间裕度 (m)
    velocity_factor: float = 0.5  # 速度相关因子 (s)
    
    # 分项
    navigation_uncertainty: float = 3.0  # 导航不确定性 (m)
    gps_error: float = 5.0  # GPS误差 (m)
    building_geometric_error: float = 2.0  # 建筑物几何误差 (m)
    control_error: float = 2.0  # 控制误差 (m)
    
    def calculate_dynamic_buffer(self, obstacle_velocity: float) -> float:
        """
        计算动态障碍物缓冲距离
        
        d_dynamic = d_static + d_pred + d_react + d_vel * v
        
        参数:
            obstacle_velocity: 障碍物速度 (m/s)
            
        返回:
            动态缓冲距离 (m)
        """
        return (self.dynamic_obstacle_buffer_base + 
                self.prediction_uncertainty + 
                self.reaction_time_margin + 
                self.velocity_factor * obstacle_velocity)
    
    def get_total_static_buffer(self) -> float:
        """
        获取总静态缓冲距离
        """
        return (self.navigation_uncertainty + 
                self.gps_error + 
                self.building_geometric_error + 
                self.control_error)


@dataclass
class Building:
    """建筑物模型 (AABB包围盒)"""
    id: str
    x: float  # 西南角X坐标 (m)
    y: float  # 西南角Y坐标 (m)
    z: float  # 基础高度 (m)
    width: float  # X方向宽度 (m)
    length: float  # Y方向长度 (m)
    height: float  # 建筑物高度 (m)
    building_type: str = "unknown"  # 建筑物类型
    
    def get_aabb(self, buffer: float = 0.0) -> Dict:
        """
        获取AABB包围盒 (含缓冲区)
        
        参数:
            buffer: 缓冲距离 (m)
            
        返回:
            AABB边界字典
        """
        return {
            'x_min': self.x - buffer,
            'x_max': self.x + self.width + buffer,
            'y_min': self.y - buffer,
            'y_max': self.y + self.length + buffer,
            'z_min': self.z,
            'z_max': self.z + self.height + buffer
        }
    
    def contains_point(self, x: float, y: float, z: float, 
                     buffer: float = 0.0) -> bool:
        """
        检查点是否在建筑物内 (含缓冲区)
        
        参数:
            x, y, z: 点坐标 (m)
            buffer: 缓冲距离 (m)
            
        返回:
            True如果在建筑物内
        """
        aabb = self.get_aabb(buffer)
        return (aabb['x_min'] <= x <= aabb['x_max'] and
                aabb['y_min'] <= y <= aabb['y_max'] and
                aabb['z_min'] <= z <= aabb['z_max'])
    
    def distance_to_point(self, x: float, y: float, z: float,
                         buffer: float = 0.0) -> float:
        """
        计算点到建筑物边界的最小距离
        
        参数:
            x, y, z: 点坐标 (m)
            buffer: 缓冲距离 (m)
            
        返回:
            最小距离 (m), 负值表示在建筑物内
        """
        aabb = self.get_aabb(buffer)
        
        # 计算到AABB的距离
        dx = max(aabb['x_min'] - x, 0, x - aabb['x_max'])
        dy = max(aabb['y_min'] - y, 0, y - aabb['y_max'])
        dz = max(aabb['z_min'] - z, 0, z - aabb['z_max'])
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 如果点在AABB内，返回负值
        if aabb['x_min'] <= x <= aabb['x_max'] and \
           aabb['y_min'] <= y <= aabb['y_max'] and \
           aabb['z_min'] <= z <= aabb['z_max']:
            distance = -distance
        
        return distance


class SpatialGrid:
    """空间网格索引 - 加速碰撞检测"""
    
    def __init__(self, buildings: List[Building], 
                 cell_size: float = 50.0):
        """
        初始化空间网格
        
        参数:
            buildings: 建筑物列表
            cell_size: 网格单元尺寸 (m)
        """
        self.cell_size = cell_size
        self.buildings = buildings
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}
        
        self._build_grid()
    
    def _build_grid(self):
        """构建空间网格"""
        for i, building in enumerate(self.buildings):
            aabb = building.get_aabb(buffer=0)
            
            # 计算建筑物覆盖的网格索引
            ix_min = int(np.floor(aabb['x_min'] / self.cell_size))
            ix_max = int(np.floor(aabb['x_max'] / self.cell_size))
            iy_min = int(np.floor(aabb['y_min'] / self.cell_size))
            iy_max = int(np.floor(aabb['y_max'] / self.cell_size))
            iz_min = int(np.floor(aabb['z_min'] / self.cell_size))
            iz_max = int(np.floor(aabb['z_max'] / self.cell_size))
            
            # 将建筑物ID添加到相关网格
            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    for iz in range(iz_min, iz_max + 1):
                        key = (ix, iy, iz)
                        if key not in self.grid:
                            self.grid[key] = []
                        self.grid[key].append(i)
    
    def get_nearby_buildings(self, x: float, y: float, z: float,
                            buffer: float = 0.0) -> List[int]:
        """
        获取附近的建筑物索引
        
        参数:
            x, y, z: 点坐标 (m)
            buffer: 缓冲距离 (m)
            
        返回:
            附近建筑物索引列表
        """
        # 计算查询点所在的网格
        ix = int(np.floor(x / self.cell_size))
        iy = int(np.floor(y / self.cell_size))
        iz = int(np.floor(z / self.cell_size))
        
        # 搜索邻近网格 (考虑缓冲区)
        search_radius = int(np.ceil(buffer / self.cell_size))
        
        nearby_indices = set()
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    key = (ix + dx, iy + dy, iz + dz)
                    if key in self.grid:
                        nearby_indices.update(self.grid[key])
        
        return list(nearby_indices)


@dataclass
class DynamicObstacle:
    """动态障碍物"""
    id: int
    type: str  # 'drone', 'helicopter', 'bird_flock', 'temporary_no_fly'
    
    # 初始位置
    x: float
    y: float
    z: float
    
    # 速度向量 (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # 几何
    size: float = 5.0  # 影响半径 (m)
    
    # 时间有效性
    valid_time_start: float = 0.0  # 有效开始时间 (s)
    valid_time_end: float = 999999.0  # 有效结束时间 (s)
    
    # 可预测性
    predictability: float = 1.0  # 可预测性 (0-1)
    
    def get_velocity_magnitude(self) -> float:
        """获取速度大小"""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def predict_position(self, current_time: float, 
                        target_time: float) -> Tuple[float, float, float]:
        """
        预测未来位置 (匀速直线模型)
        
        参数:
            current_time: 当前时间 (s)
            target_time: 目标时间 (s)
            
        返回:
            预测位置 (x, y, z) (m)
        """
        dt = target_time - current_time
        
        if self.valid_time_start <= target_time <= self.valid_time_end:
            x_pred = self.x + self.vx * dt
            y_pred = self.y + self.vy * dt
            z_pred = self.z + self.vz * dt
            return (x_pred, y_pred, z_pred)
        else:
            # 超出有效时间范围，返回无效位置
            return (float('inf'), float('inf'), float('inf'))
    
    def distance_to_point(self, x: float, y: float, z: float,
                         time_sec: float) -> float:
        """
        计算指定时间点到动态障碍物的距离
        
        参数:
            x, y, z: 点坐标 (m)
            time_sec: 时间 (s)
            
        返回:
            距离 (m)
        """
        if time_sec < self.valid_time_start or time_sec > self.valid_time_end:
            return float('inf')
        
        x_obs, y_obs, z_obs = self.predict_position(time_sec, time_sec)
        distance = np.sqrt((x - x_obs)**2 + (y - y_obs)**2 + (z - z_obs)**2)
        
        return distance


class SafetyConstraints:
    """综合安全约束检查"""
    
    def __init__(self, 
                 standard: VertiportStandard,
                 safety_margins: SafetyMargins,
                 buildings: List[Building]):
        """
        初始化安全约束检查器
        
        参数:
            standard: eVTOL标准参数
            safety_margins: 安全裕度参数
            buildings: 建筑物列表
        """
        self.standard = standard
        self.safety_margins = safety_margins
        self.buildings = buildings
        
        # 构建空间索引
        self.spatial_grid = SpatialGrid(
            buildings, 
            cell_size=50.0
        )
    
    def check_static_collision(self, 
                               x: float, y: float, z: float,
                               buffer: float = None) -> Tuple[bool, float]:
        """
        检查静态障碍物碰撞
        
        参数:
            x, y, z: 点坐标 (m)
            buffer: 缓冲距离 (m), None表示使用默认值
            
        返回:
            (是否碰撞, 最小距离)
        """
        if buffer is None:
            buffer = self.safety_margins.get_total_static_buffer()
        
        # 使用空间索引快速筛选
        nearby_indices = self.spatial_grid.get_nearby_buildings(
            x, y, z, buffer
        )
        
        min_distance = float('inf')
        
        for idx in nearby_indices:
            building = self.buildings[idx]
            distance = building.distance_to_point(x, y, z, buffer)
            
            if distance >= 0:
                min_distance = min(min_distance, distance)
            else:
                # 在建筑物内，直接返回碰撞
                return True, 0.0
        
        return False, min_distance
    
    def check_dynamic_collision(self,
                                x: float, y: float, z: float,
                                time_sec: float,
                                dynamic_obstacles: List[DynamicObstacle]) -> Tuple[bool, float]:
        """
        检查动态障碍物碰撞
        
        参数:
            x, y, z: 点坐标 (m)
            time_sec: 时间 (s)
            dynamic_obstacles: 动态障碍物列表
            
        返回:
            (是否安全, 最小距离) - True表示安全(无碰撞)
        """
        # 如果没有动态障碍物，直接返回安全
        if not dynamic_obstacles:
            return True, float('inf')
        
        min_distance = float('inf')
        
        for obstacle in dynamic_obstacles:
            # 检查时间有效性
            if time_sec < obstacle.valid_time_start or time_sec > obstacle.valid_time_end:
                continue
            
            # 计算所需缓冲距离
            velocity = obstacle.get_velocity_magnitude()
            buffer = self.safety_margins.calculate_dynamic_buffer(velocity)
            
            # 计算距离
            distance = obstacle.distance_to_point(x, y, z, time_sec)
            
            if distance < buffer + obstacle.size:
                # 碰撞
                return False, distance
            
            min_distance = min(min_distance, distance - buffer - obstacle.size)
        
        return True, min_distance
    
    def check_OFV_compliance(self,
                            x: float, y: float, z: float,
                            vertiport_center: Tuple[float, float, float]) -> Tuple[bool, float]:
        """
        检查OFV (无障碍空间) 约束
        
        参数:
            x, y, z: 点坐标 (m)
            vertiport_center: 起降场中心 (x, y, z)
            
        返回:
            (是否合规, 违约距离)
        """
        cx, cy, cz = vertiport_center
        ofv_limits = self.standard.get_OFV_limits(vertiport_center)
        
        # 检查水平距离
        horizontal_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # 检查高度
        in_ofv_height = (cz <= z <= ofv_limits['z_top'])
        
        # 检查是否在OFV圆柱内
        in_cylinder = (horizontal_dist <= ofv_limits['radius']) and in_ofv_height
        
        if in_cylinder:
            # 在OFV内 - 允许通过
            violation_distance = 0.0
            return True, violation_distance
        else:
            # 在OFV外
            if horizontal_dist > ofv_limits['radius']:
                violation_distance = horizontal_dist - ofv_limits['radius']
            elif z < cz:
                violation_distance = cz - z
            else:
                violation_distance = 0.0
            
            return False, violation_distance
    
    def check_obstacle_limitation_surface(self,
                                          x: float, y: float, z: float,
                                          vertiport_center: Tuple[float, float, float],
                                          surface_type: str = 'climb') -> Tuple[bool, float]:
        """
        检查障碍物限制面约束
        
        参数:
            x, y, z: 点坐标 (m)
            vertiport_center: 起降场中心 (x, y, z)
            surface_type: 'climb', 'approach', 'transition'
            
        返回:
            (是否合规, 违约高度)
        """
        cx, cy, cz = vertiport_center
        
        # 计算水平距离
        horizontal_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # 计算最小安全高度
        min_safe_height = self.standard.get_minimum_safe_height(
            horizontal_dist, surface_type
        )
        
        # 检查是否满足高度约束
        actual_height_above_vertiport = z - cz
        
        if actual_height_above_vertiport >= min_safe_height:
            return True, 0.0
        else:
            violation_height = min_safe_height - actual_height_above_vertiport
            return False, violation_height
    
    def check_all_constraints(self,
                             x: float, y: float, z: float,
                             time_sec: float,
                             dynamic_obstacles: List[DynamicObstacle],
                             vertiport_center: Tuple[float, float, float],
                             check_OFV: bool = True,
                             check_surfaces: bool = True) -> Dict:
        """
        检查所有约束
        
        参数:
            x, y, z: 点坐标 (m)
            time_sec: 时间 (s)
            dynamic_obstacles: 动态障碍物列表
            vertiport_center: 起降场中心
            check_OFV: 是否检查OFV约束
            check_surfaces: 是否检查障碍物限制面约束
            
        返回:
            约束检查结果字典
        """
        # 静态障碍物检查
        static_collision, static_distance = self.check_static_collision(x, y, z)
        static_safe = not static_collision  # 转换:碰撞=False, 无碰撞=True
        
        # 动态障碍物检查
        dynamic_safe, dynamic_distance = self.check_dynamic_collision(
            x, y, z, time_sec, dynamic_obstacles
        )
        
        # OFV约束检查
        ofv_compliant, ofv_violation = self.check_OFV_compliance(
            x, y, z, vertiport_center
        )
        
        # 障碍物限制面检查
        surface_compliant, surface_violation = self.check_obstacle_limitation_surface(
            x, y, z, vertiport_center, surface_type='climb'
        )
        
        # 综合判断
        all_safe = (static_safe and
                   dynamic_safe and
                   (not check_OFV or ofv_compliant) and
                   (not check_surfaces or surface_compliant))
        
        return {
            'all_safe': all_safe,
            'static_safe': static_safe,
            'static_distance': static_distance,
            'dynamic_safe': dynamic_safe,
            'dynamic_distance': dynamic_distance,
            'ofv_compliant': ofv_compliant,
            'ofv_violation': ofv_violation,
            'surface_compliant': surface_compliant,
            'surface_violation': surface_violation,
            'min_distance': min(static_distance, dynamic_distance)
        }
    
    def calculate_constraint_cost(self,
                                 x: float, y: float, z: float,
                                 time_sec: float,
                                 dynamic_obstacles: List[DynamicObstacle],
                                 vertiport_center: Tuple[float, float, float]) -> float:
        """
        计算约束违反代价 (用于软约束)
        
        参数:
            x, y, z: 点坐标 (m)
            time_sec: 时间 (s)
            dynamic_obstacles: 动态障碍物列表
            vertiport_center: 起降场中心
            
        返回:
            约束代价 (0-1, 越高越差)
        """
        result = self.check_all_constraints(
            x, y, z, time_sec, dynamic_obstacles, vertiport_center
        )
        
        if result['all_safe']:
            return 0.0
        
        cost = 0.0
        
        # 静态障碍物代价
        if not result['static_safe']:
            cost += 1.0
        
        # 动态障碍物代价
        if not result['dynamic_safe']:
            cost += 1.0
        
        # OFV代价 (软约束)
        if result['ofv_violation'] > 0:
            cost += min(result['ofv_violation'] / 10.0, 0.3)
        
        # 障碍物限制面代价 (软约束)
        if result['surface_violation'] > 0:
            cost += min(result['surface_violation'] / 20.0, 0.2)
        
        return min(cost, 1.0)
