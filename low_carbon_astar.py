"""
低碳多目标A*路径规划算法
集成：航迹长度、能耗、垂直机动、水平转向
"""

import numpy as np
import heapq
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 导入低碳节点
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_segment_length


# ============================================================================
# 城市环境类（复用v3环境）
# ============================================================================

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


class CityEnvironment:
    """城市环境类（简化版）"""

    def __init__(self, size_x=2500, size_y=2500, size_z=200):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.buildings: List[Building] = []
        self.grid = np.zeros((size_x, size_y, size_z), dtype=np.uint8)
        
        # 安全裕度参数
        self.safety_margins = {
            1: 16.974,
            2: 17.588,
            3: 18.085
        }
        self.grid_margins_xy = {k: int(v) for k, v in self.safety_margins.items()}
        self.color_map = {1: 'blue', 2: 'green', 3: 'red'}

    def generate_random_buildings(self, n_buildings=50, reserved_zones=None):
        """生成随机建筑物"""
        if reserved_zones is None:
            reserved_zones = []

        self.buildings.clear()
        initial_pool = []
        attempts = 0
        max_attempts = 2000

        while len(initial_pool) < 1000 and attempts < max_attempts:
            attempts += 1
            length = random.randint(30, 60)
            width = random.randint(30, 60)
            x = random.randint(50, self.size_x - length - 50)
            y = random.randint(50, self.size_y - width - 50)

            # 检查保留区域
            in_reserved = False
            for rz_x, rz_y, radius in reserved_zones:
                center_x = x + length / 2
                center_y = y + width / 2
                distance = math.sqrt((center_x - rz_x)**2 + (center_y - rz_y)**2)
                if distance < radius + max(self.grid_margins_xy.values()):
                    in_reserved = True
                    break
            if in_reserved:
                continue

            height = random.choice([30, 54, 72, 90, 120, 150])
            type_choice = random.randint(1, 7)
            btype = 3 if type_choice == 7 else random.choice([1, 2])

            margin_xy = self.grid_margins_xy[btype]
            initial_pool.append({
                'x': x, 'y': y, 'length': length, 'width': width,
                'height': height, 'btype': btype, 'margin_xy': margin_xy
            })

        # 筛选建筑物
        for building_data in initial_pool:
            if len(self.buildings) >= n_buildings:
                break

            x_left = building_data['x'] - building_data['margin_xy']
            x_right = building_data['x'] + building_data['length'] + building_data['margin_xy']
            y_bottom = building_data['y'] - building_data['margin_xy']
            y_top = building_data['y'] + building_data['width'] + building_data['margin_xy']

            overlap = False
            for existing in self.buildings:
                ex_margin = self.grid_margins_xy[existing.btype]
                ex_left = existing.x - ex_margin
                ex_right = existing.x + existing.length + ex_margin
                ey_bottom = existing.y - ex_margin
                ey_top = existing.y + existing.width + ex_margin

                if not (x_right <= ex_left or x_left >= ex_right or
                        y_top <= ey_bottom or y_bottom >= ey_top):
                    overlap = True
                    break

            if not overlap:
                building = Building(
                    x=building_data['x'], y=building_data['y'], z=0,
                    length=building_data['length'], width=building_data['width'],
                    height=building_data['height'], btype=building_data['btype'],
                    color=self.color_map[building_data['btype']]
                )
                self.buildings.append(building)
                self.grid[building.x:building.x+building.length,
                         building.y:building.y+building.width,
                         0:building.height] = building.btype

    def expand_with_safety_margin(self):
        """根据安全裕度扩展建筑物"""
        expanded_grid = self.grid.copy()
        for building in self.buildings:
            margin_xy = self.grid_margins_xy[building.btype]
            x1 = max(0, building.x - margin_xy)
            x2 = min(self.size_x, building.x + building.length + margin_xy)
            y1 = max(0, building.y - margin_xy)
            y2 = min(self.size_y, building.y + building.width + margin_xy)
            z1 = 0
            z2 = self.size_z
            expanded_grid[x1:x2, y1:y2, z1:z2] = 9
        return expanded_grid

    def is_free(self, x, y, z, expanded_grid=None):
        """检查位置是否可通行"""
        if not (0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z):
            return False
        grid_to_check = expanded_grid if expanded_grid is not None else self.grid
        return grid_to_check[x, y, z] == 0


# ============================================================================
# 低碳A*算法
# ============================================================================

class LowCarbonAStar:
    """低碳多目标A*算法
    
    代价函数:
    G_carbon(s) = α₁·L(s) + α₂·E_basic(s) + α₃·Σ|Δh| + α₄·Σ|Δβ|
    F_carbon(s) = ω₁·G_carbon(s) + ω₂·H(s)
    """

    def __init__(self, env: CityEnvironment, 
                 omega1=0.7, omega2=0.3,
                 alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25):
        """
        参数:
            env: 城市环境
            omega1: G_carbon权重（实际代价）
            omega2: H(s)权重（启发式）
            alpha1: 航迹长度权重
            alpha2: 基础能耗权重
            alpha3: 垂直变化权重
            alpha4: 转向角度权重
        """
        self.env = env
        
        # 估价函数权重
        self.omega1 = omega1  # G_carbon权重
        self.omega2 = omega2  # H(s)权重
        
        # G_carbon分量权重
        self.alpha1 = alpha1  # L(s): 航迹长度
        self.alpha2 = alpha2  # E_basic(s): 基础能耗
        self.alpha3 = alpha3  # Σ|Δh|: 垂直变化
        self.alpha4 = alpha4  # Σ|Δβ|: 转向角度
        
        # 能耗参数
        self.climb_energy_cost = 1.0
        self.turn_energy_cost = 0.8
        
        # 移动方向
        self.move_directions = self.get_3d_directions(6)
        
        # 统计信息
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def get_3d_directions(self, n_directions=6):
        """获取3D移动方向"""
        if n_directions == 6:
            return [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        else:
            directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        directions.append((dx, dy, dz))
            return directions

    def heuristic_manhattan(self, node: LowCarbonNode, goal: LowCarbonNode) -> float:
        """曼哈顿距离启发式函数
        
        H(s) = |x_goal - x| + |y_goal - y| + |z_goal - z|
        """
        dx = abs(goal.x - node.x)
        dy = abs(goal.y - node.y)
        dz = abs(goal.z - node.z)
        return dx + dy + dz

    def heuristic_euclidean(self, node: LowCarbonNode, goal: LowCarbonNode) -> float:
        """欧氏距离启发式函数（可选）"""
        dx = goal.x - node.x
        dy = goal.y - node.y
        dz = goal.z - node.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def calculate_basic_energy(self, current: LowCarbonNode, 
                            neighbor: LowCarbonNode, 
                            parent: Optional[LowCarbonNode]) -> float:
        """计算基础能耗代价
        
        考虑:
        1. 垂直加速度变化
        2. 水平转向（角度>10°）
        """
        energy = 0.0

        if parent:
            # 垂直加速度变化
            dz1 = current.z - parent.z
            dz2 = neighbor.z - current.z
            if dz1 != dz2:
                energy += self.climb_energy_cost

            # 水平转向
            dx1 = current.x - parent.x
            dy1 = current.y - parent.y
            dx2 = neighbor.x - current.x
            dy2 = neighbor.y - current.y

            if (dx1 != 0 or dy1 != 0) and (dx2 != 0 or dy2 != 0):
                dot = dx1*dx2 + dy1*dy2
                norm1 = math.sqrt(dx1**2 + dy1**2)
                norm2 = math.sqrt(dx2**2 + dy2**2)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot / (norm1 * norm2)
                    cos_angle = max(-1, min(1, cos_angle))
                    angle = math.acos(cos_angle)
                    if angle > math.pi/18:  # > 10°
                        energy += self.turn_energy_cost

        return energy

    def calculate_G_carbon(self, node: LowCarbonNode) -> float:
        """
        计算低碳实际代价 G_carbon(s)
        
        G_carbon = α₁·L + α₂·E_basic + α₃·Σ|Δh| + α₄·Σ|Δβ|
        """
        G_carbon = (
            self.alpha1 * node.accumulated_length +
            self.alpha2 * node.accumulated_energy +
            self.alpha3 * node.accumulated_vertical_change +
            self.alpha4 * node.accumulated_turn_angle
        )
        return G_carbon

    def calculate_F_carbon(self, node: LowCarbonNode, goal: LowCarbonNode) -> float:
        """
        计算低碳估价函数 F_carbon(s)
        
        F_carbon = ω₁·G_carbon(s) + ω₂·H(s)
        """
        G_carbon = self.calculate_G_carbon(node)
        H = self.heuristic_manhattan(node, goal)
        
        F_carbon = self.omega1 * G_carbon + self.omega2 * H
        return F_carbon

    def find_path(self, start: Tuple[int, int, int], 
                  goal: Tuple[int, int, int],
                  use_safety_margin=True,
                  heuristic_type='manhattan',
                  max_iterations=100000):
        """
        低碳A*路径规划
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            use_safety_margin: 是否使用安全裕度
            heuristic_type: 启发式类型 ('manhattan' or 'euclidean')
            max_iterations: 最大迭代次数
        
        返回:
            path: 路径列表 [(x, y, z), ...]
            metrics: 性能指标字典
        """
        # 获取环境网格
        if use_safety_margin:
            self.expanded_grid = self.env.expand_with_safety_margin()
        else:
            self.expanded_grid = None

        # 初始化起点和终点
        start_node = LowCarbonNode(start[0], start[1], start[2])
        goal_node = LowCarbonNode(goal[0], goal[1], goal[2])

        # 检查起点和终点可行性
        if not self.env.is_free(start[0], start[1], start[2], self.expanded_grid):
            print(f"    警告: 起点 {start} 不可行")
            return None, {}

        if not self.env.is_free(goal[0], goal[1], goal[2], self.expanded_grid):
            print(f"    警告: 终点 {goal} 不可行")
            return None, {}

        # 开放列表和关闭列表
        open_list = []
        open_dict = {}
        closed_set = set()

        # 初始化起点
        start_node.g = 0.0
        start_node.h = self.heuristic_manhattan(start_node, goal_node)
        start_node.f = self.omega1 * start_node.g + self.omega2 * start_node.h
        
        heapq.heappush(open_list, start_node)
        open_dict[(start_node.x, start_node.y, start_node.z)] = start_node

        iterations = 0

        while open_list and iterations < max_iterations:
            iterations += 1

            # 取出f值最小的节点
            current = heapq.heappop(open_list)
            
            # 从字典中移除
            if (current.x, current.y, current.z) in open_dict:
                del open_dict[(current.x, current.y, current.z)]

            # 检查是否到达终点
            if current.x == goal_node.x and current.y == goal_node.y and current.z == goal_node.z:
                # 重构路径
                path = self._reconstruct_path(current)
                
                # 计算性能指标
                metrics = self._calculate_metrics(path, iterations)
                
                print(f"    找到路径! 迭代次数: {iterations}, 路径长度: {len(path)}")
                print(f"    总距离: {metrics['total_length']:.2f}m")
                print(f"    总能耗: {metrics['total_energy']:.2f}")
                print(f"    垂直变化: {metrics['total_vertical_change']:.2f}m")
                print(f"    转向角度: {metrics['total_turn_angle']:.4f}rad ({metrics['total_turn_angle']*180/math.pi:.1f}°)")
                
                return path, metrics

            # 添加到关闭列表
            closed_set.add((current.x, current.y, current.z))

            # 扩展节点
            self._expand_node(current, goal_node, open_list, open_dict, closed_set)

        # 更新统计信息
        self.nodes_expanded = len(closed_set)
        self.nodes_generated = len(closed_set) + len(open_list)

        print(f"    搜索失败: 迭代 {iterations} 次后未找到路径")
        print(f"    扩展节点: {self.nodes_expanded}, 生成节点: {self.nodes_generated}")
        return None, {}

    def _expand_node(self, current: LowCarbonNode, goal: LowCarbonNode,
                    open_list, open_dict, closed_set):
        """扩展单个节点"""
        
        # 获取祖父节点（用于计算转向角）
        grandparent = current.parent if current.parent else None
        
        for dx, dy, dz in self.move_directions:
            nx, ny, nz = current.x + dx, current.y + dy, current.z + dz

            # 检查边界
            if not (0 <= nx < self.env.size_x and 
                    0 <= ny < self.env.size_y and 
                    0 <= nz < self.env.size_z):
                continue

            # 检查障碍物
            if not self.env.is_free(nx, ny, nz, self.expanded_grid):
                continue

            # 检查是否在关闭列表中
            if (nx, ny, nz) in closed_set:
                continue

            # 创建邻居节点
            neighbor = LowCarbonNode(nx, ny, nz)
            
            # ========== 关键：更新低碳指标 ==========
            update_node_metrics(current, neighbor, grandparent)
            
            # 计算基础能耗
            segment_energy = self.calculate_basic_energy(current, neighbor, grandparent)
            neighbor.accumulated_energy = current.accumulated_energy + segment_energy
            
            # ========== 计算G_carbon ==========
            neighbor.g = self.calculate_G_carbon(neighbor)
            
            # 计算H(s)
            neighbor.h = self.heuristic_manhattan(neighbor, goal)
            
            # 计算F_carbon
            neighbor.f = self.calculate_F_carbon(neighbor, goal)
            
            # 设置父节点
            neighbor.parent = current

            # 检查是否在开放列表中
            if (nx, ny, nz) in open_dict:
                # 如果新路径更优，更新
                if neighbor.g < open_dict[(nx, ny, nz)].g:
                    open_dict[(nx, ny, nz)].g = neighbor.g
                    open_dict[(nx, ny, nz)].h = neighbor.h
                    open_dict[(nx, ny, nz)].f = neighbor.f
                    open_dict[(nx, ny, nz)].parent = current
                    open_dict[(nx, ny, nz)].accumulated_length = neighbor.accumulated_length
                    open_dict[(nx, ny, nz)].accumulated_energy = neighbor.accumulated_energy
                    open_dict[(nx, ny, nz)].accumulated_vertical_change = neighbor.accumulated_vertical_change
                    open_dict[(nx, ny, nz)].accumulated_turn_angle = neighbor.accumulated_turn_angle
                    heapq.heapify(open_list)
            else:
                # 添加到开放列表
                heapq.heappush(open_list, neighbor)
                open_dict[(nx, ny, nz)] = neighbor

    def _reconstruct_path(self, node: LowCarbonNode) -> List[Tuple[int, int, int]]:
        """重构路径"""
        path = []
        while node:
            path.insert(0, (node.x, node.y, node.z))
            node = node.parent
        return path

    def _calculate_metrics(self, path: List[Tuple[int, int, int]], 
                        iterations: int) -> dict:
        """计算路径性能指标"""
        if not path:
            return {}

        metrics = {
            'iterations': iterations,
            'path_length': len(path),
            'total_length': 0.0,
            'total_energy': 0.0,
            'total_vertical_change': 0.0,
            'total_turn_angle': 0.0,
            'avg_vertical_change': 0.0,
            'avg_turn_angle': 0.0
        }

        # 计算累计指标
        total_length = 0.0
        total_vertical = 0.0
        total_turn = 0.0
        total_energy = 0.0

        for i in range(1, len(path)):
            # 距离
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            dz = path[i][2] - path[i-1][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += dist
            
            # 垂直变化
            total_vertical += abs(dz)
            
            # 能耗计算
            segment_energy = 0.0
            if i > 1:
                # 垂直加速度变化
                dz1 = path[i-1][2] - path[i-2][2]
                dz2 = path[i][2] - path[i-1][2]
                if dz1 != dz2:
                    segment_energy += self.climb_energy_cost
                
                # 水平转向
                dx1 = path[i-1][0] - path[i-2][0]
                dy1 = path[i-1][1] - path[i-2][1]
                dx2 = path[i][0] - path[i-1][0]
                dy2 = path[i][1] - path[i-1][1]
                
                if (dx1 != 0 or dy1 != 0) and (dx2 != 0 or dy2 != 0):
                    dot = dx1*dx2 + dy1*dy2
                    norm1 = math.sqrt(dx1**2 + dy1**2)
                    norm2 = math.sqrt(dx2**2 + dy2**2)
                    if norm1 > 0 and norm2 > 0:
                        cos_angle = dot / (norm1 * norm2)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = math.acos(cos_angle)
                        if angle > math.pi/18:
                            segment_energy += self.turn_energy_cost
            
            total_energy += segment_energy
            
            # 转向角（需要三个点）
            if i > 1:
                # 计算方位角
                bearing1 = math.atan2(
                    path[i-1][1] - path[i-2][1],
                    path[i-1][0] - path[i-2][0]
                )
                bearing2 = math.atan2(
                    path[i][1] - path[i-1][1],
                    path[i][0] - path[i-1][0]
                )
                
                # 转向角
                delta_beta = abs(bearing2 - bearing1)
                if delta_beta > math.pi:
                    delta_beta = 2 * math.pi - delta_beta
                
                total_turn += delta_beta

        metrics['total_length'] = total_length
        metrics['total_energy'] = total_energy
        metrics['total_vertical_change'] = total_vertical
        metrics['total_turn_angle'] = total_turn
        
        if len(path) > 1:
            metrics['avg_vertical_change'] = total_vertical / (len(path) - 1)
        if len(path) > 2:
            metrics['avg_turn_angle'] = total_turn / (len(path) - 2)

        return metrics


# ============================================================================
# 主程序
# ============================================================================

def test_low_carbon_astar(visualize=True):
    """测试低碳A*算法"""
    print("=" * 60)
    print("低碳多目标A*路径规划测试")
    print("=" * 60)

    # 创建环境
    print("\n1. 创建城市环境...")
    env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)
    
    # 设置起点和终点
    start = (250, 250, 80)
    goal = (2250, 2250, 100)
    
    # 设置保留区域
    reserved_zones = [
        (start[0], start[1], 40),
        (goal[0], goal[1], 40)
    ]
    
    # 生成建筑物
    n_buildings = 50
    env.generate_random_buildings(n_buildings=n_buildings, reserved_zones=reserved_zones)
    
    print(f"  地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    print(f"  建筑物数量: {len(env.buildings)}")
    print(f"  起飞区: {start}, 停靠区: {goal}")

    # 测试不同配置
    print("\n2. 测试不同权重配置...")
    
    configs = [
        {
            'name': '传统A*',
            'omega1': 0.5, 'omega2': 0.5,
            'alpha1': 1.0, 'alpha2': 0.0, 'alpha3': 0.0, 'alpha4': 0.0
        },
        {
            'name': '低碳均衡',
            'omega1': 0.7, 'omega2': 0.3,
            'alpha1': 0.3, 'alpha2': 0.2, 'alpha3': 0.25, 'alpha4': 0.25
        },
        {
            'name': '平稳优先',
            'omega1': 0.8, 'omega2': 0.2,
            'alpha1': 0.2, 'alpha2': 0.2, 'alpha3': 0.3, 'alpha4': 0.3
        },
        {
            'name': '最短路径',
            'omega1': 0.5, 'omega2': 0.5,
            'alpha1': 0.5, 'alpha2': 0.0, 'alpha3': 0.25, 'alpha4': 0.25
        }
    ]

    results = []

    for config in configs:
        print(f"\n  [{config['name']}]")
        print(f"    ω₁={config['omega1']}, ω₂={config['omega2']}")
        print(f"    α₁={config['alpha1']}, α₂={config['alpha2']}, α₃={config['alpha3']}, α₄={config['alpha4']}")
        
        planner = LowCarbonAStar(
            env,
            omega1=config['omega1'], omega2=config['omega2'],
            alpha1=config['alpha1'], alpha2=config['alpha2'],
            alpha3=config['alpha3'], alpha4=config['alpha4']
        )
        
        path, metrics = planner.find_path(start, goal, use_safety_margin=True)
        
        if path:
            results.append({
                'config': config['name'],
                'path': path,
                'metrics': metrics
            })

    # 对比结果
    print("\n3. 算法性能对比:")
    print("-" * 100)
    print(f"{'配置':<15} {'路径长度':<12} {'总距离(m)':<12} {'垂直变化(m)':<14} {'转向角(rad)':<15}")
    print("-" * 100)
    
    for result in results:
        m = result['metrics']
        print(f"{result['config']:<15} {m['path_length']:<12} {m['total_length']:<12.2f} "
              f"{m['total_vertical_change']:<14.2f} {m['total_turn_angle']:<15.4f}")
    
    print("-" * 100)

    # 可视化路径（文本模式）
    if visualize:
        print("\n4. 路径可视化（文本模式）:")
        print("=" * 120)
        for result in results:
            print(f"\n【{result['config']}】")
            print(f"  路径节点数: {result['metrics']['path_length']}")
            print(f"  路径节点 (起点 -> 前5个节点 ... 后5个节点 -> 终点):")

            path = result['path']
            if len(path) > 12:
                # 显示起点 + 前5个 + ... + 后5个 + 终点
                print("    起点:", path[0])
                for i in range(1, min(6, len(path)-6)):
                    print(f"    节点{i:2d}:", path[i])
                print("    ...")
                for i in range(max(6, len(path)-6), len(path)-1):
                    print(f"    节点{i:2d}:", path[i])
                print("    终点:", path[-1])
            else:
                for i, node in enumerate(path):
                    label = "起点" if i == 0 else ("终点" if i == len(path)-1 else f"节点{i}")
                    print(f"    {label}:", node)

            # 计算转向点
            turn_points = []
            for i in range(1, len(path)-1):
                if i > 1:
                    angle = calculate_turn_angle(path[i-1], path[i], path[i+1])
                    if abs(angle) > 0.1:  # 大于约5.7度的转向
                        turn_points.append((i, path[i], angle))
            print(f"\n  转向点: {len(turn_points)}处")
            for i, point, angle in turn_points[:5]:  # 显示前5个转向点
                print(f"    节点{i:2d}: {point} 转向角度={angle*180/np.pi:.1f}°")
            if len(turn_points) > 5:
                print(f"    ... 还有{len(turn_points)-5}处转向")
        print("=" * 120)

    return results


if __name__ == "__main__":
    results = test_low_carbon_astar(visualize=True)
    print("\n测试完成!")
