#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V5.1版本 - 集成改进风场
基于V5,改进：
1. 集成改进风场模型(时空相关性 + 增强建筑物效应)
2. Perlin噪声实现空间相关风场
3. 增强建筑物效应(尾流、阻塞、通道、涡流)
4. 风摆和阵风的时间变化
5. 更精确的风场代价计算
"""

import numpy as np
import math
import json
import os
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_bearing
import time

# 导入V5的核心类
from evtol_path_planning_v5 import (
    LowCarbonAStarV5, FlightProfile
)
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import DynamicEnvironment, DynamicObstacle

# 导入改进风场
from improved_wind_field import ImprovedWindField


class LowCarbonAStarV51(LowCarbonAStarV5):
    """
    低碳A*算法V5.1版本 - 集成改进风场
    继承V5所有功能,使用ImprovedWindField替代原来的WindField
    """
    
    def __init__(self, env: CityEnvironment, standard: EVTOLStandard,
                 start_vertiport: Vertiport, goal_vertiport: Vertiport,
                 dynamic_env: DynamicEnvironment,
                 flight_profile: Optional[FlightProfile] = None,
                 omega1: float = 0.7, omega2: float = 0.3,
                 alpha1: float = 0.3, alpha2: float = 0.2, 
                 alpha3: float = 0.25, alpha4: float = 0.25,
                 alpha5: float = 0.25,  # 风场权重(V5.1增强)
                 alpha6: float = 0.3,  # 国标高度约束权重
                 standard_weight: float = 0.5,
                 use_improved_wind: bool = True):  # V5.1新增参数
        """
        初始化V5.1规划器
        
        参数:
            use_improved_wind: 是否使用改进风场(True)或V4原风场(False)
        """
        super().__init__(env, standard, start_vertiport, goal_vertiport, 
                        dynamic_env, flight_profile,
                        omega1, omega2, alpha1, alpha2, alpha3, alpha4,
                        alpha5, alpha6, standard_weight)
        
        self.use_improved_wind = use_improved_wind
        
        # 如果需要,替换为改进风场
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
    
    def _calculate_wind_cost(self, current: LowCarbonNode, 
                             neighbor: LowCarbonNode, time: float) -> float:
        """
        计算风场代价 - V5.1改进
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
        """打印风场信息"""
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


def create_v51_planner(map_file: str, 
                      use_improved_wind: bool = True,
                      wind_speed: float = 8.0,
                      wind_direction: float = 315.0,
                      cruise_min: float = 120.0,
                      cruise_max: float = 180.0,
                      climb_distance: float = 400.0,
                      approach_distance: float = 400.0,
                      **kwargs) -> Optional[LowCarbonAStarV51]:
    """
    便捷函数: 创建V5.1规划器
    
    参数:
        map_file: 地图文件路径
        use_improved_wind: 是否使用改进风场
        wind_speed: 基础风速(m/s)
        wind_direction: 基础风向(度)
        cruise_min: 巡航层最小高度
        cruise_max: 巡航层最大高度
        climb_distance: 起升距离阈值
        approach_distance: 下降距离阈值
        **kwargs: 传递给LowCarbonAStarV51的其他参数(omega1, omega2, alpha1-6等)
        
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
        
        # 添加改进风场或标准风场
        if use_improved_wind:
            wind_field = ImprovedWindField(
                base_wind_speed=wind_speed,
                base_wind_direction=wind_direction,
                turbulence_intensity=0.15
            )
            wind_field.set_environment(env)
            dynamic_env.set_wind_field(wind_field)
        else:
            from evtol_path_planning_v4 import WindField
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
        planner = LowCarbonAStarV51(
            env, standard, start_vertiport, goal_vertiport, 
            dynamic_env, flight_profile,
            use_improved_wind=use_improved_wind,
            omega1=kwargs.get('omega1', 0.7),
            omega2=kwargs.get('omega2', 0.3),
            alpha1=kwargs.get('alpha1', 0.3),
            alpha2=kwargs.get('alpha2', 0.2),
            alpha3=kwargs.get('alpha3', 0.25),
            alpha4=kwargs.get('alpha4', 0.25),
            alpha5=kwargs.get('alpha5', 0.25),
            alpha6=kwargs.get('alpha6', 0.3),
            standard_weight=kwargs.get('standard_weight', 0.5)
        )
        
        return planner
        
    except Exception as e:
        print(f"创建V5.1规划器失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("V5.1路径规划 - 集成改进风场")
    print("=" * 70)
    
    # 测试地图
    test_map = 'test_map_v3_compatible.json'
    
    if os.path.exists(test_map):
        print(f"\n测试地图: {test_map}")
        
        # 创建V5.1规划器
        planner = create_v51_planner(
            test_map,
            use_improved_wind=True,
            wind_speed=8.0,
            wind_direction=315.0,
            omega1=0.7, omega2=0.3,
            alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25,
            alpha5=0.25,  # 风场权重
            alpha6=0.3,   # 国标高度约束权重
            standard_weight=0.5
        )
        
        if planner:
            # 打印风场信息
            planner.print_wind_field_info()
            
            # 运行规划
            print(f"\n开始路径规划...")
            start_time = time.time()
            path = planner.find_path(max_iterations=200000, current_time=0.0)
            planning_time = time.time() - start_time
            
            if path:
                print(f"\n规划成功!")
                print(f"  路径长度: {len(path)} 个节点")
                print(f"  规划时间: {planning_time:.2f}s")
                
                # 统计信息
                if len(path) > 1:
                    total_dist = 0
                    total_energy = 0
                    for i in range(len(path) - 1):
                        dist = math.sqrt(
                            (path[i+1].x - path[i].x)**2 +
                            (path[i+1].y - path[i].y)**2 +
                            (path[i+1].z - path[i].z)**2
                        )
                        total_dist += dist
                        total_energy += path[i+1].accumulated_energy
                    
                    print(f"  总距离: {total_dist:.1f}m")
                    print(f"  累计能耗: {total_energy:.2f}")
            else:
                print(f"\n规划失败")
    else:
        print(f"地图文件不存在: {test_map}")
