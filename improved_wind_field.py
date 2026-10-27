#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的城市风场模型 - 时空相关性 + 增强建筑物效应
基于学术论文的最佳实践改进：
1. 引入Perlin噪声实现空间相关性
2. 时间变化的风场（阵风、风摆）
3. 增强建筑物效应（尾流、阻塞、通道、涡流）
4. 考虑风速、逆风、顺风、侧风综合影响
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# Perlin噪声实现 - 用于空间相关的风场
# ============================================================================

class PerlinNoise2D:
    """2D Perlin噪声生成器 - 用于生成空间相关的风场"""
    
    def __init__(self, seed: int = 42):
        """初始化Perlin噪声"""
        self.seed = seed
        np.random.seed(seed)
        
        # 梯度向量
        self.gradients = {}
        self._generate_gradient_table()
    
    def _generate_gradient_table(self):
        """生成梯度表"""
        for i in range(256):
            angle = 2 * np.pi * i / 256
            self.gradients[i] = np.array([math.cos(angle), math.sin(angle)])
    
    def _fade(self, t: float) -> float:
        """平滑插值函数"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        """线性插值"""
        return a + t * (b - a)
    
    def _get_gradient(self, x: int, y: int) -> np.ndarray:
        """获取指定格点的梯度"""
        # 使用确定性哈希
        h = (x * 374761393 + y * 668265263) % 256
        return self.gradients[h]
    
    def noise(self, x: float, y: float) -> float:
        """计算2D Perlin噪声"""
        # 找到网格点
        x0 = int(math.floor(x))
        x1 = x0 + 1
        y0 = int(math.floor(y))
        y1 = y0 + 1
        
        # 计算局部坐标
        sx = x - x0
        sy = y - y0
        
        # 计算平滑权重
        wx = self._fade(sx)
        wy = self._fade(sy)
        
        # 计算四个角的梯度点积
        n00 = np.dot([sx, sy], self._get_gradient(x0, y0))
        n10 = np.dot([sx - 1, sy], self._get_gradient(x1, y0))
        n01 = np.dot([sx, sy - 1], self._get_gradient(x0, y1))
        n11 = np.dot([sx - 1, sy - 1], self._get_gradient(x1, y1))
        
        # 双线性插值
        nx0 = self._lerp(n00, n10, wx)
        nx1 = self._lerp(n01, n11, wx)
        value = self._lerp(nx0, nx1, wy)
        
        return value
    
    def fractal_noise(self, x: float, y: float, octaves: int = 4, 
                      persistence: float = 0.5, scale: float = 0.01) -> float:
        """分形噪声 - 多层叠加"""
        total = 0.0
        amplitude = 1.0
        max_value = 0.0
        frequency = scale
        
        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        return total / max_value


# ============================================================================
# 改进的城市风场模型
# ============================================================================

@dataclass
class ImprovedWindField:
    """改进的城市风场模型 - 时空相关性 + 增强建筑物效应"""
    
    # 基础风场参数
    base_wind_speed: float = 8.0  # 基础风速 m/s
    base_wind_direction: float = 315.0  # 基础风向（度，0=东风，90=南风，180=西风，270=北风）
    
    # 湍流和阵风参数
    turbulence_intensity: float = 0.15  # 湍流强度
    gust_amplitude: float = 3.0  # 阵风幅度 m/s
    gust_frequency: float = 0.05  # 阵风频率 Hz
    
    # 风摆参数（风向随时间变化）
    wind_variation_amplitude: float = 15.0  # 风向变化幅度（度）
    wind_variation_frequency: float = 0.01  # 风向变化频率 Hz
    
    # 空间相关性参数
    spatial_scale: float = 0.005  # 空间相关尺度
    spatial_octaves: int = 4  # 分形噪声层数
    
    # 建筑物效应参数（增强版）
    building_wake_factor: float = 0.3  # 建筑物尾流速度衰减因子（增强）
    building_wake_length: float = 300.0  # 建筑物尾流长度 (m)（增强）
    building_block_factor: float = 0.4  # 建筑物阻挡因子（增强）
    building_channel_factor: float = 2.0  # 建筑物通道效应加速因子（增强）
    building_vortex_cost: float = 0.5  # 建筑物涡流额外代价（增强）
    building_cost_weight: float = 0.4  # 建筑物效应权重（新增）
    
    # 风场代价权重
    headwind_weight: float = 0.8  # 逆风权重（增强）
    crosswind_weight: float = 0.5  # 侧风权重（增强）
    strong_wind_weight: float = 0.2  # 强风权重（增强）
    
    def __post_init__(self):
        """初始化后处理"""
        # 初始化Perlin噪声（用于空间相关性）
        self.perlin = PerlinNoise2D(seed=42)
        
        # 时间相关参数
        self.time_offset = 0.0
    
    def get_wind_at(self, x: float, y: float, z: float, 
                   time_sec: float = 0.0) -> Tuple[float, float, float]:
        """
        获取指定位置的风速向量 - 时空相关
        
        返回: (vx, vy, vz) 单位: m/s
        """
        # 1. 基础风向 - 随时间缓慢变化（风摆效应）
        wind_dir_variation = self.wind_variation_amplitude * math.sin(2 * math.pi * self.wind_variation_frequency * time_sec)
        current_wind_dir = math.radians(self.base_wind_direction + wind_dir_variation)
        
        # 2. 基础风速 - 阵风效应
        gust_factor = self.gust_amplitude * math.sin(2 * math.pi * self.gust_frequency * time_sec)
        current_wind_speed = self.base_wind_speed + gust_factor
        
        # 3. 空间相关性 - 使用Perlin噪声
        spatial_variation = self.perlin.fractal_noise(x, y, 
                                                       octaves=self.spatial_octaves,
                                                       scale=self.spatial_scale)
        # 空间噪声影响风速的±20%
        speed_variation = 1.0 + 0.2 * spatial_variation
        
        # 4. 基础风向量
        vx_base = current_wind_speed * speed_variation * math.cos(current_wind_dir)
        vy_base = current_wind_speed * speed_variation * math.sin(current_wind_dir)
        vz_base = 0.0
        
        # 5. 建筑物效应（增强版）
        vx_building, vy_building, vz_building = self._calculate_building_effect(
            x, y, z, vx_base, vy_base
        )
        
        # 6. 湍流（随机脉动）
        np.random.seed(int(time_sec * 100 + int(x) + int(y) + int(z)))
        if self.turbulence_intensity > 0:
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
        计算建筑物对风场的影响 - 增强版
        
        效应包括：
        1. 尾流效应：建筑物下游风速显著降低（40%）
        2. 通道效应：建筑物间风速增加（80%）
        3. 阻挡效应：建筑物上游风速降低（50%）
        4. 涡流效应：建筑物角落产生涡流
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
        if not hasattr(self, 'env') or not self.env.buildings:
            return vx_effect, vy_effect, vz_effect
        
        for building in self.env.buildings:
            # 处理不同格式的建筑物数据
            if isinstance(building, dict):
                bx_center = building['x'] + building['width'] / 2
                by_center = building['y'] + building['length'] / 2
                bz_top = building['z'] + building['height']
                bz_bottom = building['z']
                building_width = max(building['width'], building['length'])
                building_length = min(building['width'], building['length'])
            else:
                # 假设是Building对象
                bx_center = building.x + building.length / 2
                by_center = building.y + building.width / 2
                bz_top = building.z + building.height
                bz_bottom = building.z
                building_width = max(building.length, building.width)
                building_length = min(building.length, building.width)
            
            # 检查是否在建筑物高度范围内（±50m）
            if z < bz_bottom - 50 or z > bz_top + 50:
                continue
            
            # 计算相对于建筑物中心的位置
            dx = x - bx_center
            dy = y - by_center
            
            # ==== 1. 尾流效应（下游）====
            dist_downstream = dx * wind_dir_x + dy * wind_dir_y
            if dist_downstream > 0 and dist_downstream < self.building_wake_length:
                lateral_dist = abs(dx * wind_dir_y - dy * wind_dir_x)
                
                if lateral_dist < building_width * 3:
                    # 尾流衰减 - 随下游距离逐渐恢复
                    recovery = dist_downstream / self.building_wake_length
                    wake_factor = 1.0 - (1.0 - self.building_wake_factor) * (1.0 - recovery)
                    lateral_attenuation = math.exp(-lateral_dist / building_width)
                    
                    vx_effect -= vx_base * (1.0 - wake_factor) * lateral_attenuation
                    vy_effect -= vy_base * (1.0 - wake_factor) * lateral_attenuation
            
            # ==== 2. 阻挡效应（上游）====
            dist_upstream = -(dx * wind_dir_x + dy * wind_dir_y)
            if dist_upstream > 0 and dist_upstream < 80:  # 阻挡范围80m
                lateral_dist = abs(dx * wind_dir_y - dy * wind_dir_x)
                
                if lateral_dist < building_width:
                    # 阻挡衰减 - 指数衰减
                    block_factor = self.building_block_factor * math.exp(-dist_upstream / 30.0)
                    vx_effect -= vx_base * block_factor
                    vy_effect -= vy_base * block_factor
            
            # ==== 3. 涡流效应（建筑物角落）====
            # 简化：检查是否在建筑物角落附近
            if (abs(dx) > building_width * 0.8 and 
                abs(dy) > building_length * 0.8 and
                abs(dx) < building_width * 2 and
                abs(dy) < building_length * 2):
                # 涡流产生额外的垂直风
                corner_dist = math.sqrt(dx**2 + dy**2)
                vortex_strength = math.exp(-corner_dist / 50.0)
                vz_effect += wind_speed * vortex_strength * 0.3
        
        # ==== 4. 通道效应（建筑物之间）====
        channel_factor = self._calculate_channeling_effect(x, y, z)
        if channel_factor > 1.0:
            vx_effect += vx_base * (channel_factor - 1.0) * 0.7
            vy_effect += vy_base * (channel_factor - 1.0) * 0.7
        
        return vx_effect, vy_effect, vz_effect
    
    def _calculate_channeling_effect(self, x: float, y: float, z: float) -> float:
        """计算建筑物通道效应 - 增强版"""
        if not hasattr(self, 'env') or not self.env.buildings:
            return 1.0
        
        building_count = 0
        total_width = 0.0
        max_building_height = 0.0
        
        for building in self.env.buildings:
            if isinstance(building, dict):
                bx_center = building['x'] + building['width'] / 2
                by_center = building['y'] + building['length'] / 2
                bz_top = building['z'] + building['height']
                bz_bottom = building['z']
                building_w = max(building['width'], building['length'])
            else:
                bx_center = building.x + building.length / 2
                by_center = building.y + building.width / 2
                bz_top = building.z + building.height
                bz_bottom = building.z
                building_w = max(building.length, building.width)
            
            # 检查高度范围
            if z < bz_bottom or z > bz_top:
                continue
            
            # 检查水平距离
            dist = math.sqrt((x - bx_center)**2 + (y - by_center)**2)
            if dist < 150:  # 检查范围150m
                building_count += 1
                total_width += building_w
                max_building_height = max(max_building_height, bz_top)
        
        if building_count >= 2:
            # 多个建筑物时存在通道效应
            # 建筑物越密集，通道效应越强
            density_factor = min(building_count / 3.0, 1.5)
            avg_width = total_width / building_count
            
            # 高度越高，通道效应越弱
            height_factor = 1.0
            if z > 100:
                height_factor = max(0.5, 1.0 - (z - 100) / 200)
            
            channel_factor = 1.0 + min(avg_width / 30.0, 
                                       self.building_channel_factor - 1.0) * density_factor * height_factor
            return min(channel_factor, self.building_channel_factor)
        
        return 1.0
    
    def calculate_wind_cost(self, x: float, y: float, z: float,
                           desired_direction: Tuple[float, float, float],
                           time_sec: float = 0.0) -> float:
        """
        计算风场对路径的代价 - 改进版
        
        考虑：
        1. 逆风惩罚（最大权重）
        2. 侧风惩罚（中等权重）
        3. 强风惩罚
        4. 建筑物涡流惩罚
        5. 顺风奖励（负代价）
        
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
        
        # 计算风与飞行方向的夹角
        desired_norm = np.array([dx, dy, dz]) / desired_speed
        wind_norm = np.array([vx, vy, vz]) / wind_speed
        
        # 点积：正值=顺风，负值=逆风
        dot_product = np.dot(desired_norm, wind_norm)
        dot_product = max(-1.0, min(1.0, dot_product))  # 钳制到[-1, 1]
        
        # ==== 基础代价（基于风速）====
        # 风速越大，基础代价越高（表示操控难度增加）
        base_cost = min(wind_speed / 18.0, 0.5)  # 最大0.5
        
        # ==== 1. 逆风惩罚（最大权重）====
        # 逆风越强，代价越高；同时考虑风速
        if dot_product < 0:
            headwind_ratio = abs(dot_product)  # 0~1
            # 风速越大，逆风惩罚越大
            speed_factor = min(wind_speed / 15.0, 2.0)  # 最大2.0倍
            headwind_cost = headwind_ratio * speed_factor * 0.5
        else:
            headwind_cost = 0.0
        
        # ==== 2. 顺风奖励（负代价）====
        if dot_product > 0:
            tailwind_ratio = dot_product  # 0~1
            # 风速越大，顺风奖励越大
            speed_factor = min(wind_speed / 18.0, 1.2)
            tailwind_benefit = -tailwind_ratio * speed_factor * 0.4  # 负代价
        else:
            tailwind_benefit = 0.0
        
        # ==== 3. 侧风惩罚（中等权重）====
        # 侧风角度0~90度
        crosswind_angle = math.acos(abs(dot_product))  # 0~π/2
        crosswind_ratio = crosswind_angle / (math.pi / 2)  # 0~1
        # 完全侧风（90度）时惩罚最大
        crosswind_cost = crosswind_ratio * 0.6
        
        # ==== 4. 强风惩罚====
        if wind_speed > 12.0:  # 超过12m/s（5级风）
            strong_wind_cost = (wind_speed - 12.0) * 0.06
        else:
            strong_wind_cost = 0.0
        
        # ==== 5. 建筑物涡流惩罚====
        vortex_cost = self._calculate_vortex_cost(x, y, z)
        
        # ==== 6. 建筑物效应代价（基于风场扰动）====
        building_effect_cost = self._calculate_building_cost(x, y, z, wind_speed)
        
        # 总代价（加权和）
        total_cost = (
            base_cost +  # 基于风速的基础代价
            self.headwind_weight * headwind_cost +
            self.crosswind_weight * crosswind_cost +
            self.strong_wind_weight * strong_wind_cost +
            tailwind_benefit +  # 顺风奖励（负值）
            vortex_cost +
            self.building_cost_weight * building_effect_cost
        )
        
        # 保证代价在合理范围内
        return max(0.0, min(total_cost, 1.0))
    
    def _calculate_vortex_cost(self, x: float, y: float, z: float) -> float:
        """计算建筑物涡流代价"""
        if not hasattr(self, 'env') or not self.env.buildings:
            return 0.0
        
        max_vortex_cost = 0.0
        
        for building in self.env.buildings:
            if isinstance(building, dict):
                bx_center = building['x'] + building['width'] / 2
                by_center = building['y'] + building['length'] / 2
                bz_top = building['z'] + building['height']
                bz_bottom = building['z']
                building_width = max(building['width'], building['length'])
                building_length = min(building['width'], building['length'])
            else:
                bx_center = building.x + building.length / 2
                by_center = building.y + building.width / 2
                bz_top = building.z + building.height
                bz_bottom = building.z
                building_width = max(building.length, building.width)
                building_length = min(building.length, building.width)
            
            # 检查高度范围
            if z < bz_bottom or z > bz_top:
                continue
            
            # 检查是否在建筑物角落附近（涡流区）
            dx = abs(x - bx_center)
            dy = abs(y - by_center)
            
            if (dx > building_width * 0.7 and dx < building_width * 2 and
                dy > building_length * 0.7 and dy < building_length * 2):
                # 在角落附近，计算涡流强度
                corner_dist = math.sqrt(dx**2 + dy**2)
                vortex_strength = math.exp(-corner_dist / 40.0)
                max_vortex_cost = max(max_vortex_cost, 
                                     vortex_strength * self.building_vortex_cost)
        
        return max_vortex_cost
    
    def _calculate_building_cost(self, x: float, y: float, z: float, wind_speed: float) -> float:
        """计算建筑物对风场代价的综合影响"""
        if not hasattr(self, 'env') or not self.env.buildings:
            return 0.0
        
        total_building_cost = 0.0
        
        for building in self.env.buildings:
            if isinstance(building, dict):
                bx_center = building['x'] + building['width'] / 2
                by_center = building['y'] + building['length'] / 2
                bz_top = building['z'] + building['height']
                bz_bottom = building['z']
                building_width = max(building['width'], building['length'])
                building_length = min(building['width'], building['length'])
            else:
                bx_center = building.x + building.length / 2
                by_center = building.y + building.width / 2
                bz_top = building.z + building.height
                bz_bottom = building.z
                building_width = max(building.length, building.width)
                building_length = min(building.length, building.width)
            
            # 检查高度范围
            if z < bz_bottom or z > bz_top:
                continue
            
            # 计算距离
            dist = math.sqrt((x - bx_center)**2 + (y - by_center)**2)
            
            # 尾流区代价
            if dist < self.building_wake_length:
                wake_cost = (1.0 - dist / self.building_wake_length) * 0.3
                total_building_cost += wake_cost
            
            # 建筑物附近代价（湍流区）
            if dist < 100:
                proximity_cost = (1.0 - dist / 100.0) * 0.2
                total_building_cost += proximity_cost
        
        return min(total_building_cost, 1.0)
    
    def set_environment(self, env):
        """设置城市环境"""
        self.env = env


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("改进的城市风场模型测试")
    print("=" * 60)
    
    # 模拟环境
    class MockBuilding:
        def __init__(self, x, y, z, length, width, height, btype, color):
            self.x = x
            self.y = y
            self.z = z
            self.length = length
            self.width = width
            self.height = height
            self.btype = btype
            self.color = color
    
    class MockEnvironment:
        def __init__(self):
            self.buildings = []
            # 添加一些测试建筑物
            self.buildings.append(MockBuilding(200, 200, 0, 50, 50, 100, 1, 'blue'))
            self.buildings.append(MockBuilding(260, 200, 0, 50, 50, 120, 1, 'blue'))
            self.buildings.append(MockBuilding(200, 260, 0, 50, 50, 80, 2, 'green'))
    
    env = MockEnvironment()
    
    # 创建改进的风场
    wind_field = ImprovedWindField(
        base_wind_speed=8.0,
        base_wind_direction=315.0,
        turbulence_intensity=0.15
    )
    wind_field.set_environment(env)
    
    print(f"\n风场参数:")
    print(f"  基础风速: {wind_field.base_wind_speed} m/s")
    print(f"  基础风向: {wind_field.base_wind_direction}°")
    print(f"  湍流强度: {wind_field.turbulence_intensity}")
    
    # 测试1：空间相关性
    print("\n测试1: 空间相关性（不同位置的风场）")
    test_positions = [(100, 100, 150), (200, 200, 150), (300, 300, 150), (400, 400, 150)]
    for x, y, z in test_positions:
        vx, vy, vz = wind_field.get_wind_at(x, y, z, time_sec=0)
        speed = math.sqrt(vx**2 + vy**2)
        print(f"  ({x:4d}, {y:4d}, {z:3d}): 速度={speed:5.2f} m/s, 方向=({vx:6.2f}, {vy:6.2f})")
    
    # 测试2：时间变化性
    print("\n测试2: 时间变化性（同一位置不同时间）")
    test_times = [0, 10, 20, 30, 40, 50]
    x, y, z = 250, 250, 150
    for t in test_times:
        vx, vy, vz = wind_field.get_wind_at(x, y, z, time_sec=t)
        speed = math.sqrt(vx**2 + vy**2)
        print(f"  时间{t:3d}s: 速度={speed:5.2f} m/s, 方向=({vx:6.2f}, {vy:6.2f})")
    
    # 测试3：风场代价
    print("\n测试3: 风场代价（不同飞行方向）")
    test_directions = [
        ((1, 0, 0), "东"),
        ((0, 1, 0), "南"),
        ((-1, 0, 0), "西"),
        ((0, -1, 0), "北"),
        ((1, 1, 0), "东北")
    ]
    x, y, z = 250, 250, 150
    for direction, name in test_directions:
        cost = wind_field.calculate_wind_cost(x, y, z, direction, time_sec=0)
        print(f"  飞向{name:4s}: 代价={cost:.4f}")
    
    # 测试4：建筑物效应
    print("\n测试4: 建筑物效应")
    # 在建筑物上游
    print(f"  建筑物上游 (180, 200, 150):")
    vx, vy, vz = wind_field.get_wind_at(180, 200, 150, time_sec=0)
    speed = math.sqrt(vx**2 + vy**2)
    cost = wind_field.calculate_wind_cost(180, 200, 150, (1, 0, 0), 0)
    print(f"    风速={speed:.2f} m/s, 代价={cost:.4f}")
    
    # 在建筑物下游（尾流区）
    print(f"  建筑物下游 (280, 200, 150):")
    vx, vy, vz = wind_field.get_wind_at(280, 200, 150, time_sec=0)
    speed = math.sqrt(vx**2 + vy**2)
    cost = wind_field.calculate_wind_cost(280, 200, 150, (1, 0, 0), 0)
    print(f"    风速={speed:.2f} m/s, 代价={cost:.4f}")
    
    # 在建筑物角落（涡流区）
    print(f"  建筑物角落 (270, 270, 100):")
    vx, vy, vz = wind_field.get_wind_at(270, 270, 100, time_sec=0)
    speed = math.sqrt(vx**2 + vy**2 + vz**2)
    cost = wind_field.calculate_wind_cost(270, 270, 100, (1, 0, 0), 0)
    print(f"    风速={speed:.2f} m/s, 代价={cost:.4f}")
    
    # 在建筑物间（通道区）
    print(f"  建筑物通道 (230, 230, 100):")
    vx, vy, vz = wind_field.get_wind_at(230, 230, 100, time_sec=0)
    speed = math.sqrt(vx**2 + vy**2)
    cost = wind_field.calculate_wind_cost(230, 230, 100, (1, 0, 0), 0)
    print(f"    风速={speed:.2f} m/s, 代价={cost:.4f}")
    
    print("\n测试完成!")
