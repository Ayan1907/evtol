"""
地面风险模型 (C_groundRisk)
评估路径下方的城市密集度和人口暴露风险
"""

import math
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Building:
    """建筑物数据类"""
    x: int
    y: int
    z: int
    length: int
    width: int
    height: int
    btype: int
    color: str


@dataclass
class SensitiveZone:
    """敏感区域（学校、医院等）"""
    center: Tuple[float, float]  # (x, y)
    radius: float  # 影响半径（米）
    zone_type: str  # 'school', 'hospital', 'government', etc.


class GroundRiskModel:
    """地面风险模型 - 评估路径段下方的城市密集度和人口暴露"""
    
    def __init__(self, env, map_resolution=50, sensitive_zones=None):
        """
        参数:
            env: 城市环境对象（需包含buildings列表）
            map_resolution: 人口密度栅格分辨率（米），默认50m
            sensitive_zones: 敏感区域列表（可选）
        """
        self.env = env
        self.resolution = map_resolution
        self.population_density_map = None
        self.sensitive_zones = sensitive_zones if sensitive_zones else []
        
        # 初始化模型
        self._initialize()
    
    def _initialize(self):
        """初始化模型：生成人口密度图和识别敏感区域"""
        self.population_density_map = self._generate_population_density()
        if not self.sensitive_zones:
            self.sensitive_zones = self._identify_sensitive_zones()
    
    def _generate_population_density(self) -> Dict[Tuple[int, int], float]:
        """
        生成人口密度图（基于建筑物分布模拟）
        
        返回:
            density_map: {(grid_x, grid_y): density, ...}
        """
        density_map = {}
        
        for building in self.env.buildings:
            # 建筑物中心坐标
            center_x = building.x + building.length / 2
            center_y = building.y + building.width / 2
            
            # 栅格坐标
            grid_x = int(center_x / self.resolution)
            grid_y = int(center_y / self.resolution)
            
            # 根据建筑类型分配人口密度（人/公顷）
            density = {
                1: 80,   # 居民楼：高密度
                2: 40,   # 公共建筑：中密度
                3: 15    # 政府建筑：低密度
            }.get(building.btype, 20)
            
            # 累积密度
            if (grid_x, grid_y) in density_map:
                density_map[(grid_x, grid_y)] += density
            else:
                density_map[(grid_x, grid_y)] = density
        
        return density_map
    
    def _identify_sensitive_zones(self) -> List[SensitiveZone]:
        """
        识别敏感区域（学校、医院、政府机关等）
        
        简化实现：使用建筑物类型3（政府建筑）作为敏感区
        
        返回:
            sensitive_zones: [SensitiveZone, ...]
        """
        zones = []
        
        for building in self.env.buildings:
            if building.btype == 3:  # 政府建筑
                zone = SensitiveZone(
                    center=(building.x + building.length/2, building.y + building.width/2),
                    radius=150.0,  # 影响半径150米
                    zone_type='government'
                )
                zones.append(zone)
        
        return zones
    
    def calculate_ground_risk(self, node_s: Tuple[int, int, int], 
                             node_s_prime: Tuple[int, int, int]) -> float:
        """
        计算路径段的地面风险
        
        算法流程:
        1. 采样路径段下方的地面点
        2. 累加人口密度暴露
        3. 敏感区域加权惩罚
        4. 返回归一化代价 [0, 1]
        
        参数:
            node_s: 当前节点 (x, y, z)
            node_s_prime: 下一个节点 (x', y', z')
        
        返回:
            ground_risk: 地面风险代价 [0.0, 1.0]
        """
        # 采样点数
        samples = 10
        total_exposure = 0.0
        sensitive_penalty = 0.0
        max_density = 150  # 假设最大人口密度（人/公顷）
        
        for i in range(samples + 1):
            t = i / samples
            x = node_s[0] + (node_s_prime[0] - node_s[0]) * t
            y = node_s[1] + (node_s_prime[1] - node_s[1]) * t
            z = node_s[2] + (node_s_prime[2] - node_s[2]) * t
            
            # 1. 人口密度暴露
            grid_x = int(x / self.resolution)
            grid_y = int(y / self.resolution)
            density = self.population_density_map.get((grid_x, grid_y), 5.0)  # 默认5人/公顷
            total_exposure += density
            
            # 2. 敏感区域检查
            for zone in self.sensitive_zones:
                dist = math.sqrt((x - zone.center[0])**2 + (y - zone.center[1])**2)
                if dist < zone.radius:
                    # 越接近中心，惩罚越大
                    penalty_factor = (1 - dist / zone.radius) ** 2  # 二次衰减
                    sensitive_penalty += 100.0 * penalty_factor
        
        # 3. 标准化
        max_possible_exposure = max_density * (samples + 1)
        normalized_exposure = total_exposure / max_possible_exposure
        
        max_possible_penalty = 100.0 * (samples + 1) * 0.5  # 假设最多覆盖一半敏感区
        normalized_penalty = min(sensitive_penalty / max_possible_penalty, 1.0)
        
        # 4. 综合代价（70%人口密度 + 30%敏感区）
        ground_risk = 0.7 * normalized_exposure + 0.3 * normalized_penalty
        
        return ground_risk
    
    def add_sensitive_zone(self, center_x: float, center_y: float, 
                          radius: float, zone_type: str = 'custom'):
        """添加自定义敏感区域"""
        zone = SensitiveZone(
            center=(center_x, center_y),
            radius=radius,
            zone_type=zone_type
        )
        self.sensitive_zones.append(zone)
    
    def clear_sensitive_zones(self):
        """清空敏感区域"""
        self.sensitive_zones.clear()
    
    def calculate_risk_map(self, grid_x: int, grid_y: int) -> np.ndarray:
        """
        计算地面风险地图（用于可视化）
        
        参数:
            grid_x, grid_y: 地图尺寸
        
        返回:
            risk_map: 地面风险地图 (grid_x, grid_y)
        """
        risk_map = np.zeros((grid_x, grid_y), dtype=np.float32)
        
        for x in range(grid_x):
            for y in range(grid_y):
                # 简化计算：只考虑该点正下方
                node_s = (x, y, 0)
                node_s_prime = (x, y, 200)  # 假设从地面到200m
                risk = self.calculate_ground_risk(node_s, node_s_prime)
                risk_map[x, y] = risk
        
        return risk_map
    
    def get_high_risk_areas(self, threshold=0.5) -> List[Tuple[int, int]]:
        """
        获取高风险区域列表
        
        参数:
            threshold: 风险阈值
        
        返回:
            high_risk_areas: [(x, y), ...]
        """
        high_risk_areas = []
        
        for (grid_x, grid_y), density in self.population_density_map.items():
            # 转换为实际坐标
            x = grid_x * self.resolution
            y = grid_y * self.resolution
            
            # 计算风险
            node_s = (x, y, 0)
            node_s_prime = (x, y, 200)
            risk = self.calculate_ground_risk(node_s, node_s_prime)
            
            if risk >= threshold:
                high_risk_areas.append((int(x), int(y)))
        
        return high_risk_areas


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("地面风险模型测试")
    print("=" * 60)
    
    # 创建模拟环境
    class MockEnvironment:
        def __init__(self):
            self.buildings = []
            
            # 添加测试建筑物
            # 居民楼（高密度）
            for i in range(5):
                self.buildings.append(Building(
                    x=100 + i*80, y=100, z=0,
                    length=50, width=50, height=80, btype=1, color='blue'
                ))
            
            # 公共建筑（中密度）
            for i in range(3):
                self.buildings.append(Building(
                    x=600 + i*80, y=300, z=0,
                    length=60, width=60, height=100, btype=2, color='green'
                ))
            
            # 政府建筑（敏感区，低密度）
            self.buildings.append(Building(
                x=1000, y=500, z=0,
                length=80, width=80, height=120, btype=3, color='red'
            ))
    
    env = MockEnvironment()
    print(f"\n模拟环境: {len(env.buildings)} 栋建筑物")
    
    # 创建模型
    model = GroundRiskModel(env, map_resolution=50)
    print(f"  人口密度栅格分辨率: {model.resolution}m")
    print(f"  敏感区域数量: {len(model.sensitive_zones)}")
    
    # 显示敏感区域
    for zone in model.sensitive_zones:
        print(f"    - {zone.zone_type}: 中心={zone.center}, 半径={zone.radius}m")
    
    # 测试1：低风险路径段
    print("\n测试1: 低风险路径段（无建筑物上方）")
    node_s = (50, 50, 100)
    node_s_prime = (150, 50, 100)
    risk = model.calculate_ground_risk(node_s, node_s_prime)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  地面风险: {risk:.4f}")
    
    # 测试2：中等风险路径段（居民楼上方）
    print("\n测试2: 中等风险路径段（居民楼上方）")
    node_s = (100, 100, 120)
    node_s_prime = (200, 100, 120)
    risk = model.calculate_ground_risk(node_s, node_s_prime)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  地面风险: {risk:.4f}")
    
    # 测试3：高风险路径段（敏感区域上方）
    print("\n测试3: 高风险路径段（政府建筑/敏感区上方）")
    node_s = (1000, 500, 150)
    node_s_prime = (1100, 500, 150)
    risk = model.calculate_ground_risk(node_s, node_s_prime)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  地面风险: {risk:.4f}")
    
    # 测试4：添加自定义敏感区域
    print("\n测试4: 添加自定义敏感区域")
    model.add_sensitive_zone(center_x=700, center_y=400, radius=200, zone_type='school')
    print(f"  添加学校敏感区: 中心=(700, 400), 半径=200m")
    
    node_s = (700, 400, 100)
    node_s_prime = (800, 400, 100)
    risk = model.calculate_ground_risk(node_s, node_s_prime)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  地面风险: {risk:.4f}")
    
    # 测试5：生成风险地图（小范围）
    print("\n测试5: 生成地面风险地图 (500x500)")
    risk_map = model.calculate_risk_map(500, 500)
    high_risk_count = (risk_map > 0.5).sum()
    avg_risk = risk_map.mean()
    print(f"  高风险区域数量 (>0.5): {high_risk_count}")
    print(f"  平均风险值: {avg_risk:.4f}")
    
    # 测试6：获取高风险区域列表
    print("\n测试6: 获取高风险区域列表 (阈值=0.3)")
    high_risk_areas = model.get_high_risk_areas(threshold=0.3)
    print(f"  高风险区域数量: {len(high_risk_areas)}")
    if high_risk_areas:
        print(f"  示例区域: {high_risk_areas[:5]}")
    
    print("\n测试完成!")
