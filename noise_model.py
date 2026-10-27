"""
噪声影响模型 (C_noise)
基于A计权声级的噪声传播模型 - 评估路径段对地面噪声的影响
"""

import math
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class NoiseSource:
    """噪声源参数"""
    source_level: float  # 源头噪声级 dB(A)
    reference_distance: float  # 参考距离（米）


class NoiseModel:
    """噪声传播模型 - 基于A计权声级"""
    
    def __init__(self, env, source_level=85.0, reference_distance=100.0):
        """
        参数:
            env: 城市环境对象（支持CityEnvironment或DynamicEnvironment）
            source_level: eVTOL源头噪声级（dB(A)），默认85 dB(A) @ 100m
            reference_distance: 参考距离（米），默认100m
        """
        self.env = env
        self.source = NoiseSource(
            source_level=source_level,
            reference_distance=reference_distance
        )

        # 获取建筑物列表（兼容两种环境类型）
        if hasattr(env, 'buildings'):
            self.buildings = env.buildings
        elif hasattr(env, 'static_obstacles'):
            self.buildings = env.static_obstacles
        else:
            self.buildings = []

        # 安静时段（夜间和午休时间）
        self.quiet_hours = [(22, 6), (12, 14)]  # (start_hour, end_hour)

        # 噪声阈值
        self.disturbance_threshold = 60.0  # 60dB以上为干扰
        self.severe_threshold = 75.0  # 75dB以上为严重干扰
        
    def calculate_noise_cost(self, node_s: Tuple[int, int, int], 
                             node_s_prime: Tuple[int, int, int],
                             current_hour: int = 10, speed: float = 20.0) -> float:
        """
        计算路径段的噪声影响代价
        
        算法流程:
        1. 采样路径段
        2. 对每个采样点，计算声级衰减
        3. 考虑时间因素（安静时段惩罚加倍）
        4. 累加受影响人口
        5. 返回归一化代价 [0, 1]
        
        参数:
            node_s: 当前节点 (x, y, z)
            node_s_prime: 下一个节点 (x', y', z')
            current_hour: 当前时刻（小时，0-23），默认10点
            speed: 飞行速度（米/秒），默认20m/s
        
        返回:
            noise_cost: 噪声影响代价 [0.0, 1.0]
        """
        # 采样点数
        samples = 10
        total_impact = 0.0
        
        # 时间权重
        time_weight = self._get_time_weight(current_hour)
        
        for i in range(samples + 1):
            t = i / samples
            x = node_s[0] + (node_s_prime[0] - node_s[0]) * t
            y = node_s[1] + (node_s_prime[1] - node_s[1]) * t
            z = node_s[2] + (node_s_prime[2] - node_s[2]) * t
            
            # 计算声级衰减
            sound_level = self._calculate_sound_level(z, y, x)
            
            # 计算影响半径
            impact_radius = self._calculate_impact_radius(sound_level)
            
            if impact_radius <= 0:
                continue  # 无影响
            
            # 估算受影响人口
            affected_population = self._estimate_affected_population(x, y, impact_radius)
            
            # 噪声严重度权重（超过75dB加倍惩罚）
            severity_weight = 1.0
            if sound_level > self.severe_threshold:
                severity_weight = 2.0
            elif sound_level > self.disturbance_threshold:
                severity_weight = 1.5
            
            total_impact += affected_population * time_weight * severity_weight
        
        # 标准化到[0, 1]
        # 假设最大影响：每采样点影响1000人，严重时段3倍权重
        max_possible_impact = 1000.0 * (samples + 1) * 3.0
        normalized_impact = min(total_impact / max_possible_impact, 1.0)
        
        return normalized_impact
    
    def _calculate_sound_level(self, altitude: float, x: float, y: float) -> float:
        """
        计算地面某点的声级
        
        简化模型：假设地面高度为0
        
        参数:
            altitude: 航空器高度（米）
            x, y: 航空器水平坐标
        
        返回:
            sound_level: 地面声级 dB(A)
        """
        # 斜距（航空器到地面正下方的距离）
        slant_distance = altitude
        
        # 1. 几何扩散损失
        # 球面扩散：每增加距离倍数，衰减6dB
        if slant_distance > 0:
            geometric_spreading = 20 * math.log10(slant_distance / self.source.reference_distance)
        else:
            geometric_spreading = 0
        
        # 2. 大气吸收损失
        # 简化：每1000m衰减3dB
        atmospheric_absorption = (slant_distance / 1000.0) * 3.0
        
        # 3. 地面反射修正
        # 简化：高度>50m时考虑地面反射衰减
        if altitude > 50:
            ground_reflection = -3.0
        else:
            ground_reflection = -1.0
        
        # 4. 建筑物遮挡修正（简化）
        # 如果下方有建筑物，声级衰减
        occlusion_loss = self._calculate_occlusion_loss(x, y, altitude)
        
        # 总声级
        sound_level = (
            self.source.source_level 
            - geometric_spreading 
            - atmospheric_absorption 
            + ground_reflection 
            - occlusion_loss
        )
        
        return sound_level

    def calculate_noise_level(self, x: float, y: float, z: float, speed: float = 20.0) -> float:
        """
        计算某点在飞行时产生的噪声水平（公共方法，供噪声约束使用）

        参数:
            x, y, z: 航空器坐标
            speed: 飞行速度（米/秒），默认20m/s

        返回:
            noise_level: 地面接收到的噪声级 dB(A)
        """
        # 基础声级
        sound_level = self._calculate_sound_level(z, y, x)

        # 速度修正（速度越快，噪声越大）
        speed_factor = 1.0 + (speed - 20.0) * 0.01
        sound_level = sound_level * speed_factor

        return sound_level

    def _calculate_occlusion_loss(self, x: float, y: float, altitude: float) -> float:
        """
        计算建筑物遮挡造成的声级衰减

        简化实现：检查下方是否有高大建筑物

        参数:
            x, y: 航空器坐标
            altitude: 高度

        返回:
            occlusion_loss: 遮挡损失（dB）
        """
        occlusion_loss = 0.0

        for building in self.buildings:
            # 兼容字典和对象两种格式
            if isinstance(building, dict):
                bx = building.get('x', 0)
                by = building.get('y', 0)
                blength = building.get('length', 0)
                bwidth = building.get('width', 0)
                bheight = building.get('height', 0)
            else:
                bx = building.x
                by = building.y
                blength = building.length
                bwidth = building.width
                bheight = building.height

            # 检查是否在建筑物上方
            if (bx <= x < bx + blength and
                by <= y < by + bwidth):
                # 如果建筑物高度 > 航空器高度的一半，有遮挡
                if bheight > altitude * 0.5:
                    occlusion_loss = 10.0  # 简化：10dB衰减
                    break

        return occlusion_loss
    
    def _calculate_impact_radius(self, sound_level: float) -> float:
        """
        计算影响半径（60dB阈值）
        
        参数:
            sound_level: 声源声级（dB）
        
        返回:
            impact_radius: 影响半径（米）
        """
        if sound_level < self.disturbance_threshold:
            return 0.0
        
        # 反推距离：假设6dB倍频程衰减
        # L2 = L1 - 20*log10(d2/d1)
        # d2 = d1 * 10^((L1-L2)/20)
        delta_db = sound_level - self.disturbance_threshold
        impact_radius = self.source.reference_distance * (10 ** (delta_db / 20.0))
        
        return impact_radius
    
    def _estimate_affected_population(self, center_x: float, center_y: float, 
                                        radius: float) -> int:
        """
        估算受影响人口
        
        简化实现：基于建筑物密度估算
        
        参数:
            center_x, center_y: 中心坐标
            radius: 影响半径（米）

        返回:
            affected_population: 受影响人口数量
        """
        population = 0

        for building in self.buildings:  # 使用 self.buildings（已经在__init__中处理）
            # 兼容字典和对象两种格式
            if isinstance(building, dict):
                bx = building.get('x', 0)
                by = building.get('y', 0)
                blength = building.get('length', 0)
                bwidth = building.get('width', 0)
                btype = building.get('btype', 1)
            else:
                bx = building.x
                by = building.y
                blength = building.length
                bwidth = building.width
                btype = building.btype

            # 检查建筑物中心是否在影响半径内
            building_center_x = bx + blength / 2
            building_center_y = by + bwidth / 2

            dist = math.sqrt((building_center_x - center_x)**2 +
                           (building_center_y - center_y)**2)

            if dist < radius:
                # 根据建筑类型估算人口
                population_per_building = {
                    1: 150,  # 居民楼
                    2: 80,   # 公共建筑
                    3: 30    # 政府建筑
                }.get(btype, 50)

                # 考虑部分覆盖（距离越远，影响越小）
                coverage_factor = 1.0 - (dist / radius) * 0.5
                population += int(population_per_building * coverage_factor)

        return population
    
    def _get_time_weight(self, hour: int) -> float:
        """
        获取时间权重（安静时段惩罚更大）
        
        参数:
            hour: 当前时刻（0-23）
        
        返回:
            weight: 时间权重
        """
        for (start, end) in self.quiet_hours:
            if start <= hour or hour < end:
                return 3.0  # 安静时段3倍惩罚
        
        return 1.0  # 正常时段
    
    def calculate_noise_map(self, grid_x: int, grid_y: int, altitude: float = 100,
                            current_hour: int = 10) -> np.ndarray:
        """
        计算噪声影响地图（用于可视化）
        
        参数:
            grid_x, grid_y: 地图尺寸
            altitude: 假设的航空器高度
            current_hour: 当前时刻
        
        返回:
            noise_map: 噪声影响地图 (grid_x, grid_y)
        """
        noise_map = np.zeros((grid_x, grid_y), dtype=np.float32)
        
        # 时间权重
        time_weight = self._get_time_weight(current_hour)
        
        for x in range(0, grid_x, 10):  # 降采样，减少计算量
            for y in range(0, grid_y, 10):
                # 计算声级
                sound_level = self._calculate_sound_level(altitude, x, y)
                
                # 计算影响半径
                impact_radius = self._calculate_impact_radius(sound_level)
                
                if impact_radius > 0:
                    # 受影响人口
                    affected_pop = self._estimate_affected_population(x, y, impact_radius)
                    
                    # 严重度权重
                    severity_weight = 1.5 if sound_level > self.disturbance_threshold else 1.0
                    
                    # 填充到噪声地图（填充10x10区域）
                    noise_map[x:x+10, y:y+10] = affected_pop * time_weight * severity_weight
        
        # 标准化到[0, 1]
        if noise_map.max() > 0:
            noise_map = noise_map / noise_map.max()
        
        return noise_map
    
    def get_noise_profile(self, x: float, y: float, altitude: float) -> dict:
        """
        获取指定位置的噪声剖面
        
        参数:
            x, y: 航空器坐标
            altitude: 高度
        
        返回:
            profile: 噪声剖面信息
        """
        sound_level = self._calculate_sound_level(altitude, x, y)
        impact_radius = self._calculate_impact_radius(sound_level)
        affected_pop = self._estimate_affected_population(x, y, impact_radius)
        
        profile = {
            'sound_level': sound_level,
            'impact_radius': impact_radius,
            'affected_population': affected_pop,
            'is_disturbing': sound_level > self.disturbance_threshold,
            'is_severe': sound_level > self.severe_threshold
        }
        
        return profile


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("噪声影响模型测试")
    print("=" * 60)
    
    # 创建模拟环境
    class MockEnvironment:
        def __init__(self):
            self.buildings = []
            
            # 添加测试建筑物（密集区域）
            for i in range(3):
                for j in range(3):
                    self.buildings.append(Building(
                        x=100 + i*60, y=100 + j*60, z=0,
                        length=50, width=50, height=80, btype=1, color='blue'
                    ))
            
            # 添加一栋政府建筑
            self.buildings.append(Building(
                x=400, y=400, z=0,
                length=80, width=80, height=120, btype=3, color='red'
            ))
    
    # 模拟Building类（用于测试）
    from dataclasses import dataclass
    @dataclass
    class Building:
        x: int
        y: int
        z: int
        length: int
        width: int
        height: int
        btype: int
        color: str
    
    env = MockEnvironment()
    print(f"\n模拟环境: {len(env.buildings)} 栋建筑物")
    
    # 创建模型
    model = NoiseModel(env, source_level=85.0, reference_distance=100.0)
    print(f"  源头噪声级: {model.source.source_level} dB(A)")
    print(f"  参考距离: {model.source.reference_distance} m")
    print(f"  干扰阈值: {model.disturbance_threshold} dB(A)")
    print(f"  严重阈值: {model.severe_threshold} dB(A)")
    
    # 测试1：低高度路径段
    print("\n测试1: 低高度路径段 (高度=50m)")
    node_s = (200, 200, 50)
    node_s_prime = (300, 200, 50)
    cost = model.calculate_noise_cost(node_s, node_s_prime, current_hour=10)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  噪声影响代价: {cost:.4f}")
    
    # 测试2：高高度路径段
    print("\n测试2: 高高度路径段 (高度=150m)")
    node_s = (200, 200, 150)
    node_s_prime = (300, 200, 150)
    cost = model.calculate_noise_cost(node_s, node_s_prime, current_hour=10)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  噪声影响代价: {cost:.4f}")
    
    # 测试3：夜间时段
    print("\n测试3: 夜间时段路径段 (23:00)")
    node_s = (200, 200, 100)
    node_s_prime = (300, 200, 100)
    cost = model.calculate_noise_cost(node_s, node_s_prime, current_hour=23)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  噪声影响代价: {cost:.4f}")
    
    # 测试4：获取噪声剖面
    print("\n测试4: 获取指定位置的噪声剖面")
    profile = model.get_noise_profile(x=250, y=200, altitude=100)
    print(f"  位置: (250, 200, 100)")
    print(f"  地面声级: {profile['sound_level']:.2f} dB(A)")
    print(f"  影响半径: {profile['impact_radius']:.2f} m")
    print(f"  受影响人口: {profile['affected_population']} 人")
    print(f"  是否干扰: {profile['is_disturbing']}")
    print(f"  是否严重: {profile['is_severe']}")
    
    # 测试5：生成噪声影响地图（小范围）
    print("\n测试5: 生成噪声影响地图 (500x500)")
    noise_map = model.calculate_noise_map(500, 500, altitude=100, current_hour=10)
    high_impact_count = (noise_map > 0.5).sum()
    avg_impact = noise_map.mean()
    print(f"  高影响区域数量 (>0.5): {high_impact_count}")
    print(f"  平均影响值: {avg_impact:.4f}")
    print(f"  最大影响值: {noise_map.max():.4f}")
    
    # 测试6：不同时间的噪声代价对比
    print("\n测试6: 不同时段噪声代价对比")
    node_s = (200, 200, 100)
    node_s_prime = (300, 200, 100)
    
    hours = [8, 12, 14, 18, 23, 2]
    print(f"  路径段: {node_s} -> {node_s_prime}")
    for hour in hours:
        cost = model.calculate_noise_cost(node_s, node_s_prime, current_hour=hour)
        period = "正常时段" if (8 <= hour < 12 or 14 <= hour < 22) else "安静时段"
        print(f"    {hour:02d}:00 ({period}): {cost:.4f}")
    
    print("\n测试完成!")
