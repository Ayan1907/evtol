"""
低碳路径节点 - 扩展原有Node类
增加累计指标：高度变化、转向角度
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class LowCarbonNode:
    """低碳路径节点 - 用于多目标优化"""
    x: int
    y: int
    z: int
    
    # 传统A*指标
    g: float = 0.0      # 实际代价
    h: float = 0.0      # 启发式代价
    f: float = 0.0      # 估价函数 f = g + h
    parent: Optional['LowCarbonNode'] = None
    
    # ========== 低碳指标 ==========
    accumulated_length: float = 0.0           # 累计航迹长度（米）
    accumulated_energy: float = 0.0           # 累计基础能耗
    accumulated_vertical_change: float = 0.0  # 累计高度变化绝对值（米）
    accumulated_turn_angle: float = 0.0       # 累计转向角度（弧度）
    
    def __lt__(self, other):
        """用于优先队列比较"""
        return self.f < other.f
    
    def __eq__(self, other):
        """节点相等性判断（基于坐标）"""
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __hash__(self):
        """用于字典键"""
        return hash((self.x, self.y, self.z))


# ============================================================================
# 辅助计算函数
# ============================================================================

def calculate_segment_length(node1: LowCarbonNode, node2: LowCarbonNode) -> float:
    """计算两点间的欧氏距离"""
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    dz = node2.z - node1.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def calculate_vertical_change(node1: LowCarbonNode, node2: LowCarbonNode) -> float:
    """计算高度变化绝对值 |Δh|"""
    return abs(node2.z - node1.z)


def calculate_bearing(node1: LowCarbonNode, node2: LowCarbonNode) -> float:
    """
    计算从node1到node2的水平方位角（弧度）
    
    方位角定义：
    - 0: 东向（+X方向）
    - π/2: 北向（+Y方向）
    - π: 西向（-X方向）
    - 3π/2: 南向（-Y方向）
    
    公式: bearing = atan2(dy, dx)
    """
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    
    if dx == 0 and dy == 0:
        return 0.0  # 无水平移动，返回0
    
    bearing = math.atan2(dy, dx)
    
    # 归一化到 [0, 2π)
    if bearing < 0:
        bearing += 2 * math.pi
    
    return bearing


def calculate_turn_angle(previous: LowCarbonNode, 
                        current: LowCarbonNode, 
                        next_node: LowCarbonNode) -> float:
    """
    计算三节点间的水平转向角 |Δβ|
    
    参数:
        previous: 前一个节点
        current: 当前节点
        next_node: 下一个节点
    
    返回:
        turn_angle: 转向角绝对值（弧度）
    
    示例:
        previous → current → next_node
        如果是直线运动，turn_angle = 0
        如果是直角转弯，turn_angle = π/2
    """
    # 计算incoming和outgoing的方位角
    bearing_in = calculate_bearing(previous, current)
    bearing_out = calculate_bearing(current, next_node)
    
    # 计算转向角
    delta_beta = abs(bearing_out - bearing_in)
    
    # 处理角度环绕（例如从350°转到10°，实际是20°转向）
    if delta_beta > math.pi:
        delta_beta = 2 * math.pi - delta_beta
    
    return delta_beta


def update_node_metrics(current: LowCarbonNode, 
                       neighbor: LowCarbonNode,
                       grandparent: Optional[LowCarbonNode] = None):
    """
    更新邻居节点的累计指标
    
    核心逻辑：
    1. 累计长度 = 当前累计长度 + 本段长度
    2. 累计高度变化 = 当前累计高度变化 + |z_neighbor - z_current|
    3. 累计转向角 = 当前累计转向角 + |Δβ|
    
    参数:
        current: 当前节点
        neighbor: 邻居节点（待更新）
        grandparent: 祖父节点（用于计算转向角）
    """
    # 1. 累计长度
    segment_length = calculate_segment_length(current, neighbor)
    neighbor.accumulated_length = current.accumulated_length + segment_length
    
    # 2. 累计高度变化
    vertical_change = calculate_vertical_change(current, neighbor)
    neighbor.accumulated_vertical_change = (
        current.accumulated_vertical_change + vertical_change
    )
    
    # 3. 累计转向角
    if grandparent:
        # 有祖父节点，可以计算转向角
        turn_angle = calculate_turn_angle(grandparent, current, neighbor)
        neighbor.accumulated_turn_angle = (
            current.accumulated_turn_angle + turn_angle
        )
    else:
        # 无祖父节点（起点的直接邻居），转向角为0
        neighbor.accumulated_turn_angle = current.accumulated_turn_angle


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("低碳路径节点测试")
    print("=" * 60)
    
    # 测试1：创建节点
    print("\n测试1: 创建节点")
    start = LowCarbonNode(0, 0, 0)
    print(f"  起始节点: ({start.x}, {start.y}, {start.z})")
    print(f"    累计长度: {start.accumulated_length}")
    print(f"    累计高度变化: {start.accumulated_vertical_change}")
    print(f"    累计转向角: {start.accumulated_turn_angle:.4f} rad")
    
    # 测试2：直线移动（无转向）
    print("\n测试2: 直线移动 (0,0,0) -> (10,0,0) -> (20,0,0)")
    node1 = LowCarbonNode(0, 0, 0)
    node2 = LowCarbonNode(10, 0, 0)
    node3 = LowCarbonNode(20, 0, 0)
    
    # 更新node2（起点->node1）
    update_node_metrics(node1, node2, None)
    print(f"  节点2: ({node2.x}, {node2.y}, {node2.z})")
    print(f"    累计长度: {node2.accumulated_length:.2f} m")
    print(f"    累计高度变化: {node2.accumulated_vertical_change:.2f} m")
    print(f"    累计转向角: {node2.accumulated_turn_angle:.4f} rad (直线)")
    
    # 更新node3（node1->node2->node3，直线）
    update_node_metrics(node2, node3, node1)
    print(f"  节点3: ({node3.x}, {node3.y}, {node3.z})")
    print(f"    累计长度: {node3.accumulated_length:.2f} m")
    print(f"    累计高度变化: {node3.accumulated_vertical_change:.2f} m")
    print(f"    累计转向角: {node3.accumulated_turn_angle:.4f} rad (直线)")
    
    # 测试3：直角转弯
    print("\n测试3: 直角转弯 (0,0,0) -> (10,0,0) -> (10,10,0)")
    node1 = LowCarbonNode(0, 0, 0)
    node2 = LowCarbonNode(10, 0, 0)
    node3 = LowCarbonNode(10, 10, 0)
    
    update_node_metrics(node1, node2, None)
    update_node_metrics(node2, node3, node1)
    
    print(f"  节点3: ({node3.x}, {node3.y}, {node3.z})")
    print(f"    累计长度: {node3.accumulated_length:.2f} m")
    print(f"    累计高度变化: {node3.accumulated_vertical_change:.2f} m")
    print(f"    累计转向角: {node3.accumulated_turn_angle:.4f} rad ({node3.accumulated_turn_angle*180/math.pi:.1f}°)")
    print(f"    预期转向角: {math.pi/2:.4f} rad (90°)")
    
    # 测试4：高度变化
    print("\n测试4: 高度变化 (0,0,0) -> (10,0,0) -> (10,0,20)")
    node1 = LowCarbonNode(0, 0, 0)
    node2 = LowCarbonNode(10, 0, 0)
    node3 = LowCarbonNode(10, 0, 20)
    
    update_node_metrics(node1, node2, None)
    update_node_metrics(node2, node3, node1)
    
    print(f"  节点3: ({node3.x}, {node3.y}, {node3.z})")
    print(f"    累计长度: {node3.accumulated_length:.2f} m")
    print(f"    累计高度变化: {node3.accumulated_vertical_change:.2f} m")
    print(f"    累计转向角: {node3.accumulated_turn_angle:.4f} rad")
    
    # 测试5：复杂路径
    print("\n测试5: 复杂路径")
    nodes = [
        LowCarbonNode(0, 0, 0),
        LowCarbonNode(10, 0, 10),
        LowCarbonNode(10, 10, 20),
        LowCarbonNode(0, 10, 10),
        LowCarbonNode(0, 0, 0)
    ]
    
    for i in range(1, len(nodes)):
        grandparent = nodes[i-2] if i > 1 else None
        update_node_metrics(nodes[i-1], nodes[i], grandparent)
        
        print(f"  节点{i+1}: ({nodes[i].x}, {nodes[i].y}, {nodes[i].z})")
        print(f"    累计长度: {nodes[i].accumulated_length:.2f} m")
        print(f"    累计高度变化: {nodes[i].accumulated_vertical_change:.2f} m")
        print(f"    累计转向角: {nodes[i].accumulated_turn_angle:.4f} rad ({nodes[i].accumulated_turn_angle*180/math.pi:.1f}°)")
    
    # 测试6：方位角计算
    print("\n测试6: 方位角计算")
    test_cases = [
        ((0, 0), (10, 0), "东向"),
        ((0, 0), (0, 10), "北向"),
        ((0, 0), (-10, 0), "西向"),
        ((0, 0), (0, -10), "南向"),
        ((0, 0), (10, 10), "东北向"),
    ]
    
    for (x1, y1), (x2, y2), desc in test_cases:
        node1 = LowCarbonNode(x1, y1, 0)
        node2 = LowCarbonNode(x2, y2, 0)
        bearing = calculate_bearing(node1, node2)
        print(f"  {desc}: {bearing:.4f} rad ({bearing*180/math.pi:.1f}°)")
    
    print("\n测试完成!")
