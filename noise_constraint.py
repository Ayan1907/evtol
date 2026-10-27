#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声约束模块 - 作为路径规划的硬约束

设计理念:
1. 噪声不加入目标函数(避免复杂化)
2. 噪声超标区域在路径规划中直接标记为不可达(硬约束)
3. 这样规划出的路径天然满足噪声要求
4. 评估时噪声自然不会过大

参考论文:
- Minimizing Acoustic Noise for Quadruped Robots (2025): 室内噪声敏感环境优化
- Optimization of operational aircraft parameters Reducing Noise Emission (2008): 飞行路径噪声优化
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from noise_model import NoiseModel


@dataclass
class NoiseConstraintConfig:
    """噪声约束配置"""
    # 噪声阈值 (dBA)
    max_average_noise: float = 75.0      # 平均噪声上限
    max_peak_noise: float = 85.0         # 峰值噪声上限

    # 噪声敏感区域
    residential_threshold: float = 65.0  # 居民区噪声阈值
    commercial_threshold: float = 70.0  # 商业区噪声阈值
    quiet_zone_threshold: float = 55.0   # 安全区噪声阈值

    # 高度衰减因子 (噪声随高度衰减)
    height_decay_factor: float = 2.0

    # 夜间模式 (噪声限制更严格)
    nighttime_strict: bool = True
    nighttime_penalty: float = 1.5       # 夜间阈值惩罚系数


class NoiseConstraint:
    """
    噪声约束 - 作为路径规划的硬约束

    用法:
    1. 在A*搜索的邻居检查中调用
    2. 如果噪声超标,直接返回False(跳过该节点)
    3. 这样规划出的路径天然满足噪声要求
    """

    def __init__(self, noise_model: NoiseModel, config: NoiseConstraintConfig = None):
        """
        初始化噪声约束

        参数:
            noise_model: 噪声模型
            config: 噪声约束配置
        """
        self.noise_model = noise_model
        self.config = config or NoiseConstraintConfig()

    def check_noise_constraint(self, node_x: float, node_y: float, node_z: float,
                               current_hour: int = 10, is_nighttime: bool = False) -> bool:
        """
        检查节点是否满足噪声约束

        参数:
            node_x, node_y, node_z: 节点坐标
            current_hour: 当前小时 (0-23)
            is_nighttime: 是否为夜间

        返回:
            True: 满足约束,可以经过
            False: 不满足约束,不可达
        """
        # 1. 计算该点的噪声水平
        noise_level = self.noise_model.calculate_noise_level(
            node_x, node_y, node_z, speed=20.0
        )

        # 2. 夜间模式更严格
        threshold = self.config.max_average_noise
        if self.config.nighttime_strict and (is_nighttime or current_hour < 6 or current_hour >= 22):
            threshold = threshold / self.config.nighttime_penalty

        # 3. 检查是否超过阈值
        if noise_level > threshold:
            return False

        # 4. 检查峰值噪声
        if noise_level > self.config.max_peak_noise:
            return False

        # 5. 检查噪声敏感区域
        if not self._check_noise_sensitive_zones(node_x, node_y, node_z, noise_level, current_hour):
            return False

        return True

    def _check_noise_sensitive_zones(self, x: float, y: float, z: float,
                                     noise_level: float, current_hour: int) -> bool:
        """
        检查噪声敏感区域

        噪声敏感区域需要更严格的噪声控制
        """
        # 获取区域类型 (从environment获取)
        zone_type = self._get_zone_type(x, y)

        # 根据区域类型设置阈值
        if zone_type == 'residential':  # 居民区
            threshold = self.config.residential_threshold
        elif zone_type == 'commercial':  # 商业区
            threshold = self.config.commercial_threshold
        elif zone_type == 'quiet':  # 安全区
            threshold = self.config.quiet_zone_threshold
        else:
            return True  # 普通区域不额外限制

        # 夜间更严格
        if self.config.nighttime_strict and (current_hour < 6 or current_hour >= 22):
            threshold = threshold * 0.8  # 夜间降低20%阈值

        # 检查是否超标
        if noise_level > threshold:
            return False

        return True

    def _get_zone_type(self, x: float, y: float) -> str:
        """
        获取位置的区域类型

        简化版: 根据坐标判断区域类型
        实际应用中应该从地图数据中读取
        """
        # 这里简化处理,实际应该从地图数据读取
        # 假设:
        # - 人口密度 > 0.01: 居民区
        # - 0.005 < 人口密度 <= 0.01: 商业区
        # - 人口密度 <= 0.005: 普通区
        # - 标记为quiet_zone的区域: 安全区

        # 从noise_model获取环境
        env = self.noise_model.env

        # 尝试获取ground_risk_model
        ground_risk_model = None
        if hasattr(env, 'ground_risk_model'):
            ground_risk_model = env.ground_risk_model
        elif hasattr(env, 'env') and hasattr(env.env, 'ground_risk_model'):
            ground_risk_model = env.env.ground_risk_model

        if ground_risk_model:
            try:
                # 获取人口密度
                population_density = ground_risk_model.get_population_density(x, y)

                # 简化判断
                if population_density > 0.01:
                    return 'residential'
                elif population_density > 0.005:
                    return 'commercial'
                else:
                    return 'normal'
            except:
                pass

        # 默认：根据位置简单判断
        import math
        if hasattr(env, 'size_x') and hasattr(env, 'size_y'):
            if env.size_x > 0 and env.size_y > 0:
                center_x = env.size_x / 2
                center_y = env.size_y / 2
                dist_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = math.sqrt(center_x**2 + center_y**2)
                if dist_to_center < max_dist * 0.3:
                    return 'commercial'
                elif dist_to_center < max_dist * 0.7:
                    return 'residential'
                else:
                    return 'normal'

        return 'normal'  # 默认普通区域

    def calculate_noise_cost_for_evaluation(self, node_x: float, node_y: float,
                                            node_z: float, current_hour: int = 10) -> float:
        """
        计算噪声代价 (仅用于评估,不影响路径规划)

        返回:
            噪声代价 [0, 1]
        """
        noise_level = self.noise_model.calculate_noise_level(
            node_x, node_y, node_z, speed=20.0
        )

        # 归一化到 [0, 1]
        # 假设最大噪声为 100 dBA
        return min(noise_level / 100.0, 1.0)


def integrate_noise_constraint_to_astar():
    """
    集成噪声约束到A*搜索

    在evtol_path_planning_v6.py中的find_path函数里添加噪声约束检查

    示例代码:

    # 在_check_safety_constraints之后添加
    if not self._check_noise_constraint(neighbor, current_time):
        continue
    """
    pass


# 集成示例代码 (添加到evtol_path_planning_v6.py的LowCarbonAStarV6类中)
"""
# 在__init__中添加:
def __init__(self, ...):
    # ... 现有代码 ...

    # 新增: 噪声约束
    from noise_model import NoiseModel
    from noise_constraint import NoiseConstraint, NoiseConstraintConfig

    self.noise_model = NoiseModel(self.dynamic_env, self.flight_profile)
    noise_config = NoiseConstraintConfig(
        max_average_noise=75.0,
        max_peak_noise=85.0,
        nighttime_strict=True
    )
    self.noise_constraint = NoiseConstraint(self.noise_model, noise_config)

# 在find_path的邻居检查中添加 (在_check_safety_constraints之后):
if not self._check_noise_constraint(neighbor, current_time):
    continue

# 添加噪声约束检查方法:
def _check_noise_constraint(self, node: LowCarbonNode, current_time: float) -> bool:
    ''
    检查噪声约束

    如果噪声超标,直接返回False (硬约束)
    ''
    current_hour = int(current_time / 3600) % 24

    return self.noise_constraint.check_noise_constraint(
        node.x, node.y, node.z,
        current_hour=current_hour
    )
"""
