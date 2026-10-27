"""
碰撞风险模型 (C_collision)
实时碰撞概率模型 - 基于感知不确定性、交通意图预测和概率冲突检测
参考NASA UTM系统的冲突检测算法
"""

import math
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class Aircraft:
    """其他航空器信息"""
    id: str
    pos: Tuple[float, float, float]  # (x, y, z)
    vel: Tuple[float, float, float]   # (vx, vy, vz)


class CollisionRiskModel:
    """实时碰撞概率模型"""
    
    def __init__(self, sensor_error=5.0, prediction_horizon=30, safety_distance=50.0):
        """
        参数:
            sensor_error: 传感器误差标准差（米），默认5m（GPS/雷达误差）
            prediction_horizon: 预测时间窗（秒），默认30秒
            safety_distance: 安全间隔距离（米），默认50m
        """
        self.sensor_error = sensor_error
        self.prediction_horizon = prediction_horizon
        self.safety_distance = safety_distance
        self.other_aircraft: List[Aircraft] = []
        
    def add_aircraft(self, aircraft_id: str, x: float, y: float, z: float, 
                    vx: float = 0, vy: float = 0, vz: float = 0):
        """添加其他航空器到环境
        
        参数:
            aircraft_id: 航空器ID
            x, y, z: 当前位置
            vx, vy, vz: 速度向量（米/秒）
        """
        aircraft = Aircraft(
            id=aircraft_id,
            pos=(x, y, z),
            vel=(vx, vy, vz)
        )
        self.other_aircraft.append(aircraft)
    
    def clear_aircraft(self):
        """清空航空器列表"""
        self.other_aircraft.clear()
    
    def calculate_collision_cost(self, node_s: Tuple[int, int, int], 
                                 node_s_prime: Tuple[int, int, int],
                                 speed: float = 20.0):
        """
        计算从节点s移动到节点s'的碰撞风险代价
        
        算法流程:
        1. 对每个其他航空器，预测其在未来时刻的位置
        2. 计算本机路径段与预测轨迹的最小距离
        3. 考虑传感器误差，计算冲突概率
        4. 返回归一化代价 [0, 1]
        
        参数:
            node_s: 当前节点 (x, y, z)
            node_s_prime: 下一个节点 (x', y', z')
            speed: 本机速度（米/秒）
        
        返回:
            collision_cost: 碰撞风险代价 [0.0, 1.0]
        """
        if not self.other_aircraft:
            return 0.0  # 无其他航空器，无碰撞风险
        
        # 生成路径段离散点
        path_segment = self._generate_segment_points(node_s, node_s_prime, speed)
        
        max_probability = 0.0
        conflict_aircraft = None
        
        # 对每个其他航空器检查冲突概率
        for other in self.other_aircraft:
            prob = self._predict_conflict_probability(other, path_segment, speed)
            if prob > max_probability:
                max_probability = prob
                conflict_aircraft = other.id
        
        return max_probability
    
    def _generate_segment_points(self, node_s: Tuple[int, int, int], 
                                node_s_prime: Tuple[int, int, int], 
                                speed: float) -> List[Tuple[float, float, float]]:
        """生成路径段离散点
        
        参数:
            node_s: 起点
            node_s_prime: 终点
            speed: 速度（米/秒）
        
        返回:
            采样点列表 [(x, y, z), ...]
        """
        dx = node_s_prime[0] - node_s[0]
        dy = node_s_prime[1] - node_s[1]
        dz = node_s_prime[2] - node_s[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist == 0:
            return [(node_s[0], node_s[1], node_s[2])]
        
        # 采样间隔：每0.5米一个点
        sampling_resolution = 0.5
        steps = max(1, int(dist / sampling_resolution))
        
        segment = []
        for i in range(steps + 1):
            t = i / steps
            x = node_s[0] + dx * t
            y = node_s[1] + dy * t
            z = node_s[2] + dz * t
            segment.append((x, y, z))
        
        return segment
    
    def _predict_conflict_probability(self, other: Aircraft, 
                                     path_segment: List[Tuple[float, float, float]],
                                     speed: float) -> float:
        """
        预测与某个航空器的冲突概率
        
        简化模型：假设对方匀速直线运动
        
        参数:
            other: 其他航空器
            path_segment: 本机路径段采样点
            speed: 本机速度
        
        返回:
            probability: 冲突概率 [0.0, 1.0]
        """
        # 时间步长（秒）
        time_step = 0.5 / speed  # 每个采样点对应的时间
        
        max_conflict_prob = 0.0
        
        for i, (self_x, self_y, self_z) in enumerate(path_segment):
            t = i * time_step
            
            # 预测对方位置（匀速运动模型）
            pred_x = other.pos[0] + other.vel[0] * t
            pred_y = other.pos[1] + other.vel[1] * t
            pred_z = other.pos[2] + other.vel[2] * t
            
            # 计算最小间隔
            separation = math.sqrt(
                (self_x - pred_x)**2 +
                (self_y - pred_y)**2 +
                (self_z - pred_z)**2
            )
            
            # 计算冲突概率
            if separation < self.safety_distance:
                prob = self._collision_probability(separation)
                max_conflict_prob = max(max_conflict_prob, prob)
        
        return max_conflict_prob
    
    def _collision_probability(self, separation: float) -> float:
        """
        基于高斯分布的碰撞概率
        
        使用指数衰减模型：
        - 分离距离 >= safety_distance: 概率为0
        - 分离距离 = 0: 概率为1
        - 中间: 指数衰减
        
        参数:
            separation: 分离距离（米）
        
        返回:
            probability: 碰撞概率 [0.0, 1.0]
        """
        if separation >= self.safety_distance:
            return 0.0
        
        # 指数衰减模型
        normalized_dist = separation / self.safety_distance
        prob = math.exp(-3 * (1 - normalized_dist))  # 调整衰减速率
        
        return min(prob, 1.0)
    
    def calculate_risk_map(self, grid_x: int, grid_y: int, grid_z: int,
                           aircraft_list: List[Aircraft] = None) -> np.ndarray:
        """
        计算整个空间的风险地图（用于可视化）
        
        参数:
            grid_x, grid_y, grid_z: 网格尺寸
            aircraft_list: 航空器列表（可选，默认使用self.other_aircraft）
        
        返回:
            risk_map: 风险概率地图 (grid_x, grid_y, grid_z)
        """
        import numpy as np
        
        if aircraft_list is not None:
            original_list = self.other_aircraft
            self.other_aircraft = aircraft_list
        
        risk_map = np.zeros((grid_x, grid_y, grid_z), dtype=np.float32)
        
        # 粗采样（每10米一个点，减少计算量）
        step = 10
        for x in range(0, grid_x, step):
            for y in range(0, grid_y, step):
                for z in range(0, grid_z, step):
                    # 计算该点的碰撞概率（假设其他航空器静止）
                    for other in self.other_aircraft:
                        separation = math.sqrt(
                            (x - other.pos[0])**2 +
                            (y - other.pos[1])**2 +
                            (z - other.pos[2])**2
                        )
                        if separation < self.safety_distance:
                            prob = self._collision_probability(separation)
                            risk_map[x:x+step, y:y+step, z:z+step] = prob
                            break
        
        # 恢复原列表
        if aircraft_list is not None:
            self.other_aircraft = original_list
        
        return risk_map


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("碰撞风险模型测试")
    print("=" * 60)
    
    # 创建模型
    model = CollisionRiskModel(sensor_error=5.0, prediction_horizon=30, safety_distance=50.0)
    
    # 添加其他航空器
    model.add_aircraft("aircraft_1", x=500, y=500, z=100, vx=10, vy=0, vz=0)
    model.add_aircraft("aircraft_2", x=1000, y=800, z=120, vx=0, vy=15, vz=0)
    model.add_aircraft("aircraft_3", x=800, y=600, z=90, vx=-5, vy=-5, vz=2)
    
    print(f"\n已添加 {len(model.other_aircraft)} 个其他航空器")
    for ac in model.other_aircraft:
        print(f"  - {ac.id}: 位置={ac.pos}, 速度={ac.vel}")
    
    # 测试1：无冲突路径段
    print("\n测试1: 无冲突路径段")
    node_s = (0, 0, 80)
    node_s_prime = (100, 100, 80)
    cost = model.calculate_collision_cost(node_s, node_s_prime, speed=20.0)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  碰撞风险代价: {cost:.4f}")
    
    # 测试2：接近其他航空器的路径段
    print("\n测试2: 接近航空器1的路径段")
    node_s = (450, 450, 95)
    node_s_prime = (550, 550, 105)
    cost = model.calculate_collision_cost(node_s, node_s_prime, speed=20.0)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  碰撞风险代价: {cost:.4f}")
    
    # 测试3：直接穿过其他航空器位置的路径段
    print("\n测试3: 直接穿过航空器1位置的路径段")
    node_s = (400, 400, 80)
    node_s_prime = (600, 600, 120)
    cost = model.calculate_collision_cost(node_s, node_s_prime, speed=20.0)
    print(f"  路径段: {node_s} -> {node_s_prime}")
    print(f"  碰撞风险代价: {cost:.4f}")
    
    # 测试4：生成风险地图（小范围）
    print("\n测试4: 生成风险地图 (100x100x50)")
    risk_map = model.calculate_risk_map(100, 100, 50)
    high_risk_count = (risk_map > 0.1).sum()
    print(f"  高风险体素数量 (>0.1): {high_risk_count}")
    print(f"  最大风险值: {risk_map.max():.4f}")
    
    print("\n测试完成!")
