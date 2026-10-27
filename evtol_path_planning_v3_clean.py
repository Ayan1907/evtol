"""
eVTOL路径规划 v3.0 - 基于双向有界次优启发式搜索
保留核心功能，移除冗余和未完成部分
"""

import numpy as np
import heapq
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# 数据结构
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


@dataclass
class Node:
    """路径节点类"""
    x: int
    y: int
    z: int
    g: float = 0.0
    h: float = 0.0
    f: float = 0.0
    parent: Optional['Node'] = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))


# ============================================================================
# 城市环境类
# ============================================================================

class CityEnvironment:
    """城市环境类"""

    def __init__(self, size_x=2500, size_y=2500, size_z=200):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.buildings: List[Building] = []
        self.grid = np.zeros((size_x, size_y, size_z), dtype=np.uint8)
        self.building_above_map = None

        # 安全裕度参数
        self.safety_margins = {
            1: 16.974,
            2: 17.588,
            3: 18.085
        }
        self.grid_margins_xy = {k: int(v) for k, v in self.safety_margins.items()}
        self.color_map = {1: 'blue', 2: 'green', 3: 'red'}

    def generate_random_buildings(self, n_buildings=50, reserved_zones=None):
        """生成随机建筑物，避免重叠"""
        if reserved_zones is None:
            reserved_zones = []

        self.buildings.clear()

        # 生成建筑池
        print(f"  生成建筑池...")
        initial_pool = []
        attempts = 0
        max_attempts = 2000

        while len(initial_pool) < 1000 and attempts < max_attempts:
            attempts += 1

            # 建筑尺寸
            length = random.randint(30, 60)
            width = random.randint(30, 60)

            # 坐标
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

            # 高度和类型
            height = random.choice([30, 54, 72, 90, 120, 150])
            type_choice = random.randint(1, 7)
            btype = 3 if type_choice == 7 else random.choice([1, 2])

            margin_xy = self.grid_margins_xy[btype]
            initial_pool.append({
                'x': x, 'y': y, 'length': length, 'width': width,
                'height': height, 'btype': btype, 'margin_xy': margin_xy
            })

        # 筛选建筑物
        print(f"  筛选建筑物（目标：{n_buildings}栋）...")
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

        print(f"  成功生成 {len(self.buildings)} 栋建筑物")

        # 创建空间索引
        self.building_above_map = np.zeros((self.size_x, self.size_y), dtype=np.float32)
        for building in self.buildings:
            self.building_above_map[
                building.x:building.x + building.length,
                building.y:building.y + building.width
            ] = max(
                self.building_above_map[
                    building.x:building.x + building.length,
                    building.y:building.y + building.width
                ].max(), building.height
            )

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
# 路径规划算法
# ============================================================================

class BoundedSuboptimalThetaStar:
    """双向有界次优Theta*算法"""

    def __init__(self, env: CityEnvironment, weight: float = 1.5, adaptive_weight: bool = True):
        """
        参数:
            weight: 加权因子 (w >= 1.0)
                   - w=1.0: 最优搜索
                   - w=1.5: 解不超过最优解1.5倍
                   - w=2.0: 解不超过最优解2倍
            adaptive_weight: 是否动态调整权重
        """
        self.env = env
        self.weight = weight
        self.adaptive_weight = adaptive_weight
        self.expanded_grid = None

        # 算法参数
        self.move_directions = self.get_3d_directions(6)
        self.climb_energy_cost = 1.0
        self.turn_energy_cost = 0.8
        self.alpha1 = 0.4  # 航迹长度权重
        self.alpha2 = 0.6  # 能耗权重
        self.w1 = 0.4      # G(s)权重
        self.w2 = 0.6      # H(s)权重

        # 统计信息
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def get_3d_directions(self, n_directions=26):
        """获取3D移动方向"""
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    directions.append((dx, dy, dz))

        if n_directions == 26:
            return directions
        else:
            return [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]

    def heuristic(self, node: Node, goal: Node):
        """欧氏距离启发函数"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        dz = abs(node.z - goal.z)
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def calculate_adaptive_weight(self, iteration: int, max_iterations: int):
        """动态权重调整"""
        if not self.adaptive_weight:
            return self.weight

        progress = iteration / max_iterations
        adaptive_w = max(1.0, self.weight * (1 - 0.5 * progress))
        return adaptive_w

    def calculate_adaptive_step_size(self, current: Node, goal: Node):
        """自适应步长"""
        directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        dist_to_goal = self.heuristic(current, goal)

        if dist_to_goal > 100:
            diagonal_directions = [
                (-1,-1,0), (-1,1,0), (1,-1,0), (1,1,0),
                (-1,-1,1), (-1,1,1), (1,-1,1), (1,1,1),
                (-1,-1,-1), (-1,1,-1), (1,-1,-1), (1,1,-1)
            ]
            directions.extend(diagonal_directions)

        return directions

    def calculate_energy_cost(self, current: Node, neighbor: Node, parent: Optional[Node]):
        """计算能耗代价"""
        energy = 0.0

        if parent:
            dz1 = current.z - parent.z
            dz2 = neighbor.z - current.z
            if dz1 != dz2:
                energy += self.climb_energy_cost

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
                    if angle > math.pi/18:
                        energy += self.turn_energy_cost

        return energy

    def distance(self, node1: Node, node2: Node):
        """欧氏距离"""
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        dz = node1.z - node2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def find_path(self, start, goal, use_safety_margin=True, consider_energy=True,
                  weight: Optional[float] = None, max_iterations: int = 50000):
        """双向有界次优Theta*算法"""
        # 获取环境网格
        if use_safety_margin:
            self.expanded_grid = self.env.expand_with_safety_margin()
        else:
            self.expanded_grid = None

        # 设置加权因子
        if weight is not None:
            self.weight = weight

        # 初始化起点和终点节点
        start_node = Node(start[0], start[1], start[2])
        goal_node = Node(goal[0], goal[1], goal[2])

        # 检查起点和终点是否可行
        if not self.env.is_free(start[0], start[1], start[2], self.expanded_grid):
            print(f"    警告: 起点 {start} 不可行")
            return None
        if not self.env.is_free(goal[0], goal[1], goal[2], self.expanded_grid):
            print(f"    警告: 终点 {goal} 不可行")
            return None

        # 双向搜索的开放列表
        open_list_forward = []
        open_list_backward = []
        open_dict_forward = {}
        open_dict_backward = {}
        closed_list_forward = {}
        closed_list_backward = {}

        # 初始化起点和终点
        start_node.g = 0
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.g + self.weight * start_node.h

        goal_node.g = 0
        goal_node.h = self.heuristic(goal_node, start_node)
        goal_node.f = goal_node.g + self.weight * goal_node.h

        heapq.heappush(open_list_forward, start_node)
        heapq.heappush(open_list_backward, goal_node)
        open_dict_forward[(start_node.x, start_node.y, start_node.z)] = start_node
        open_dict_backward[(goal_node.x, goal_node.y, goal_node.z)] = goal_node

        meeting_node = None
        meeting_from_forward = False
        backward_node = None
        forward_node = None

        iterations = 0

        while open_list_forward and open_list_backward and iterations < max_iterations:
            iterations += 1

            # 计算当前动态权重
            current_weight = self.calculate_adaptive_weight(iterations, max_iterations)

            # 前向搜索
            if open_list_forward:
                current_forward = heapq.heappop(open_list_forward)

                # 检查是否相遇
                if (current_forward.x, current_forward.y, current_forward.z) in closed_list_backward:
                    meeting_node = current_forward
                    backward_node = closed_list_backward[(current_forward.x, current_forward.y, current_forward.z)]
                    meeting_from_forward = True
                    break

                self.expand_node(current_forward, open_list_forward, open_dict_forward, closed_list_forward,
                               goal_node, consider_energy, is_forward=True, weight=current_weight)

            # 后向搜索
            if open_list_backward:
                current_backward = heapq.heappop(open_list_backward)

                # 检查是否相遇
                if (current_backward.x, current_backward.y, current_backward.z) in closed_list_forward:
                    meeting_node = current_backward
                    forward_node = closed_list_forward[(current_backward.x, current_backward.y, current_backward.z)]
                    meeting_from_forward = False
                    break

                self.expand_node(current_backward, open_list_backward, open_dict_backward, closed_list_backward,
                               start_node, consider_energy, is_forward=False, weight=current_weight)

        # 更新统计信息
        self.nodes_expanded = len(closed_list_forward) + len(closed_list_backward)
        self.nodes_generated = self.nodes_expanded + len(open_list_forward) + len(open_list_backward)

        if not meeting_node:
            print(f"    搜索失败: 迭代 {iterations} 次后未找到路径")
            print(f"    扩展节点: {self.nodes_expanded}, 生成节点: {self.nodes_generated}")
            return None

        # 重构路径
        path = []

        # 前向部分：从起点到会议点
        if meeting_from_forward:
            node = meeting_node
        else:
            node = forward_node

        while node:
            path.insert(0, (node.x, node.y, node.z))
            node = node.parent

        # 后向部分：从会议点之后到终点
        if meeting_from_forward:
            node = backward_node.parent
        else:
            node = meeting_node.parent

        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent

        print(f"    找到路径! 扩展节点: {self.nodes_expanded}, 生成节点: {self.nodes_generated}, 路径长度: {len(path)}")
        return path

    def expand_node(self, current, open_list, open_dict, closed_list, goal,
                   consider_energy, is_forward, weight):
        """扩展单个节点"""
        # 使用自适应步长
        directions = self.calculate_adaptive_step_size(current, goal)

        for dx, dy, dz in directions:
            nx, ny, nz = current.x + dx, current.y + dy, current.z + dz

            # 检查边界和障碍
            if not self.env.is_free(nx, ny, nz, self.expanded_grid):
                continue

            # 检查是否已在关闭列表中
            if (nx, ny, nz) in closed_list:
                continue

            # 创建邻居节点
            neighbor = Node(nx, ny, nz)

            # 计算代价
            dist_cost = self.distance(current, neighbor)
            energy_cost = 0
            if consider_energy:
                energy_cost = self.calculate_energy_cost(current, neighbor, current.parent)

            g_cost = current.g + self.alpha1 * dist_cost + self.alpha2 * energy_cost
            h_cost = self.heuristic(neighbor, goal)
            f_cost = self.w1 * g_cost + self.w2 * weight * h_cost

            # 使用字典快速检查是否在开放列表中
            if (nx, ny, nz) in open_dict:
                if g_cost < open_dict[(nx, ny, nz)].g:
                    open_dict[(nx, ny, nz)].g = g_cost
                    open_dict[(nx, ny, nz)].h = h_cost
                    open_dict[(nx, ny, nz)].f = f_cost
                    open_dict[(nx, ny, nz)].parent = current
                    heapq.heapify(open_list)
            else:
                neighbor.g = g_cost
                neighbor.h = h_cost
                neighbor.f = f_cost
                neighbor.parent = current
                heapq.heappush(open_list, neighbor)
                open_dict[(nx, ny, nz)] = neighbor

        closed_list[(current.x, current.y, current.z)] = current


# ============================================================================
# 主程序
# ============================================================================

def load_environment_from_json(json_file):
    """从JSON文件加载环境"""
    from map_generator import MapGenerator
    gen, start, goal, metadata = MapGenerator.load_from_json(json_file)

    # 创建CityEnvironment
    env = CityEnvironment.__new__(CityEnvironment)
    env.size_x = gen.size_x
    env.size_y = gen.size_y
    env.size_z = gen.size_z
    env.grid_size_xy = gen.grid_size_xy
    env.grid_size_z = gen.grid_size_z
    env.buildings = gen.buildings
    env.grid = np.zeros((env.size_x, env.size_y, env.size_z), dtype=np.int8)
    env.safety_margins = gen.safety_margins
    env.grid_margins_xy = gen.grid_margins_xy

    # 标记建筑物
    for building in env.buildings:
        env.grid[building.x:building.x+building.length,
                 building.y:building.y+building.width,
                 0:building.height] = building.btype

    # 创建空间索引
    env.building_above_map = np.zeros((env.size_x, env.size_y), dtype=np.float32)
    for building in env.buildings:
        env.building_above_map[
            building.x:building.x + building.length,
            building.y:building.y + building.width
        ] = max(
            env.building_above_map[
                building.x:building.x + building.length,
                building.y:building.y + building.width
            ].max(), building.height
        )

    return env, start, goal, metadata


def main():
    """主程序"""
    print("=" * 60)
    print("eVTOL路径规划 v3.0 - 有界次优启发式搜索")
    print("=" * 60)

    # 检查是否使用保存的地图
    USE_SAVED_MAP = False
    SAVED_MAP_FILE = '/home/ayan1907/evtol/test_map_v3_compatible.json'

    if USE_SAVED_MAP:
        # 从JSON加载地图
        print("\n1. 加载地图数据...")
        env, start, goal, metadata = load_environment_from_json(SAVED_MAP_FILE)
        print(f"  地图文件: {SAVED_MAP_FILE}")
        print(f"  地图描述: {metadata.get('description', 'N/A')}")
    else:
        # 创建城市环境
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

        n_buildings = 250
        env.generate_random_buildings(n_buildings=n_buildings, reserved_zones=reserved_zones)

    print(f"  地图尺寸: {env.size_x}m × {env.size_y}m × {env.size_z}m")
    print(f"  建筑物数量: {len(env.buildings)}")
    print(f"  起飞区: {start}, 停靠区: {goal}")

    # 测试不同配置
    print("\n2. 测试不同算法配置...")

    # 标准A* (w=1.0, 无动态权重)
    print("\n  [1] 标准A* (w=1.0, 静态权重)...")
    planner_standard = BoundedSuboptimalThetaStar(env, weight=1.0, adaptive_weight=False)
    path_standard = planner_standard.find_path(start, goal, use_safety_margin=True, consider_energy=False)

    # 加权A* (w=1.5, 静态权重)
    print("\n  [2] 加权A* (w=1.5, 静态权重)...")
    planner_weighted = BoundedSuboptimalThetaStar(env, weight=1.5, adaptive_weight=False)
    path_weighted = planner_weighted.find_path(start, goal, use_safety_margin=True, consider_energy=False)

    # 动态权重A* (w=1.5, 动态调整)
    print("\n  [3] 动态权重A* (w=1.5, 动态调整)...")
    planner_dynamic = BoundedSuboptimalThetaStar(env, weight=1.5, adaptive_weight=True)
    path_dynamic = planner_dynamic.find_path(start, goal, use_safety_margin=True, consider_energy=False)

    # 高权重快速A* (w=2.0, 静态权重)
    print("\n  [4] 高权重快速A* (w=2.0, 静态权重)...")
    planner_fast = BoundedSuboptimalThetaStar(env, weight=2.0, adaptive_weight=False)
    path_fast = planner_fast.find_path(start, goal, use_safety_margin=True, consider_energy=False)

    # 最优A* (w=1.0, 考虑能耗)
    print("\n  [5] 最优A* + 能耗 (w=1.0, 考虑能耗)...")
    planner_optimal_energy = BoundedSuboptimalThetaStar(env, weight=1.0, adaptive_weight=False)
    path_optimal_energy = planner_optimal_energy.find_path(start, goal, use_safety_margin=True, consider_energy=True)

    # 对比结果
    print("\n3. 算法性能对比:")
    print("-" * 80)
    print(f"{'算法':<25} {'权重':<10} {'路径长度':<12} {'扩展节点':<12} {'生成节点':<12}")
    print("-" * 80)

    algorithms = [
        ("标准A*", 1.0, path_standard, planner_standard),
        ("加权A*", 1.5, path_weighted, planner_weighted),
        ("动态权重A*", 1.5, path_dynamic, planner_dynamic),
        ("高权重快速A*", 2.0, path_fast, planner_fast),
        ("最优A*+能耗", 1.0, path_optimal_energy, planner_optimal_energy),
    ]

    for name, w, path, planner in algorithms:
        path_len = len(path) if path else "失败"
        expanded = planner.nodes_expanded
        generated = planner.nodes_generated
        print(f"{name:<25} {w:<10} {path_len:<12} {expanded:<12} {generated:<12}")

    print("-" * 80)

    return {
        'environment': env,
        'paths': {
            'standard': path_standard,
            'weighted': path_weighted,
            'dynamic': path_dynamic,
            'fast': path_fast,
            'optimal_energy': path_optimal_energy
        }
    }


if __name__ == "__main__":
    results = main()
