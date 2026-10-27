"""
eVTOL路径规划 v3.0简化版 - 基于双向A*算法
移除复杂特性，保留核心功能
"""

import numpy as np
import heapq
import math
from dataclasses import dataclass
from typing import List, Optional

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
# 环境类
# ============================================================================

class CityEnvironment:
    """城市环境类 - 简化版"""

    def __init__(self, size_x=2500, size_y=2500, size_z=200):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.buildings: List[Building] = []
        self.grid = np.zeros((size_x, size_y, size_z), dtype=np.uint8)

        # 安全裕度
        self.safety_margins = {1: 17, 2: 18, 3: 18}
        self.grid_margins = {k: int(v) for k, v in self.safety_margins.items()}
        self.color_map = {1: 'blue', 2: 'green', 3: 'red'}

    def generate_random_buildings(self, n_buildings=50, reserved_zones=None):
        """生成随机建筑物"""
        if reserved_zones is None:
            reserved_zones = []

        self.buildings.clear()
        attempts = 0
        max_attempts = 5000

        while len(self.buildings) < n_buildings and attempts < max_attempts:
            attempts += 1

            length = 40
            width = 40
            x = np.random.randint(50, self.size_x - length - 50)
            y = np.random.randint(50, self.size_y - width - 50)
            height = np.random.choice([30, 54, 72, 90, 120, 150])
            btype = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])

            # 检查保留区域
            in_reserved = False
            for rz_x, rz_y, radius in reserved_zones:
                center_x = x + length / 2
                center_y = y + width / 2
                if math.sqrt((center_x - rz_x)**2 + (center_y - rz_y)**2) < radius + 30:
                    in_reserved = True
                    break
            if in_reserved:
                continue

            # 检查重叠
            overlap = False
            margin = self.grid_margins[btype]
            for existing in self.buildings:
                ex_margin = self.grid_margins[existing.btype]
                if not (x + length + margin <= existing.x - ex_margin or
                        x - margin >= existing.x + existing.length + ex_margin or
                        y + width + margin <= existing.y - ex_margin or
                        y - margin >= existing.y + existing.width + ex_margin):
                    overlap = True
                    break
            if overlap:
                continue

            building = Building(x, y, 0, length, width, height, btype, self.color_map[btype])
            self.buildings.append(building)
            self.grid[x:x+length, y:y+width, 0:height] = btype

        print(f"  生成 {len(self.buildings)} 栋建筑物")

    def expand_with_safety_margin(self):
        """扩展安全裕度"""
        expanded = self.grid.copy()
        for building in self.buildings:
            margin = self.grid_margins[building.btype]
            x1 = max(0, building.x - margin)
            x2 = min(self.size_x, building.x + building.length + margin)
            y1 = max(0, building.y - margin)
            y2 = min(self.size_y, building.y + building.width + margin)
            expanded[x1:x2, y1:y2, 0:self.size_z] = 9
        return expanded

    def is_free(self, x, y, z, expanded_grid=None):
        """检查位置是否可通行"""
        if not (0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z):
            return False
        grid = expanded_grid if expanded_grid is not None else self.grid
        return grid[x, y, z] == 0


# ============================================================================
# 路径规划算法
# ============================================================================

class BidirectionalAStar:
    """双向A*算法 - 简化版"""

    def __init__(self, env: CityEnvironment):
        self.env = env
        self.nodes_expanded = 0
        self.nodes_generated = 0
        # 6方向移动
        self.directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]

    def heuristic(self, node, goal):
        """欧氏距离启发式"""
        return math.sqrt((node.x-goal.x)**2 + (node.y-goal.y)**2 + (node.z-goal.z)**2)

    def find_path(self, start, goal, use_safety_margin=True, max_iterations=50000):
        """双向A*搜索"""
        expanded_grid = self.env.expand_with_safety_margin() if use_safety_margin else None

        # 检查起点终点
        if not self.env.is_free(*start, expanded_grid):
            print(f"    起点 {start} 不可行")
            return None
        if not self.env.is_free(*goal, expanded_grid):
            print(f"    终点 {goal} 不可行")
            return None

        # 初始化双向搜索
        start_node = Node(*start)
        goal_node = Node(*goal)
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.h
        goal_node.h = self.heuristic(goal_node, start_node)
        goal_node.f = goal_node.h

        open_fwd, open_bwd = [], []
        heapq.heappush(open_fwd, start_node)
        heapq.heappush(open_bwd, goal_node)

        closed_fwd, closed_bwd = {}, {}
        open_dict_fwd, open_dict_bwd = {}, {}
        open_dict_fwd[(start_node.x, start_node.y, start_node.z)] = start_node
        open_dict_bwd[(goal_node.x, goal_node.y, goal_node.z)] = goal_node

        meeting_node = None
        meeting_from_fwd = False
        backward_node = None
        forward_node = None
        iterations = 0

        while open_fwd and open_bwd and iterations < max_iterations:
            iterations += 1

            # 前向搜索
            current_fwd = heapq.heappop(open_fwd)
            if (current_fwd.x, current_fwd.y, current_fwd.z) in closed_bwd:
                meeting_node = current_fwd
                backward_node = closed_bwd[(current_fwd.x, current_fwd.y, current_fwd.z)]
                meeting_from_fwd = True
                break

            closed_fwd[(current_fwd.x, current_fwd.y, current_fwd.z)] = current_fwd
            self._expand(current_fwd, open_fwd, open_dict_fwd, closed_fwd, goal_node, expanded_grid)

            # 后向搜索
            current_bwd = heapq.heappop(open_bwd)
            if (current_bwd.x, current_bwd.y, current_bwd.z) in closed_fwd:
                meeting_node = current_bwd
                forward_node = closed_fwd[(current_bwd.x, current_bwd.y, current_bwd.z)]
                meeting_from_fwd = False
                break

            closed_bwd[(current_bwd.x, current_bwd.y, current_bwd.z)] = current_bwd
            self._expand(current_bwd, open_bwd, open_dict_bwd, closed_bwd, start_node, expanded_grid)

        self.nodes_expanded = len(closed_fwd) + len(closed_bwd)
        self.nodes_generated = self.nodes_expanded + len(open_fwd) + len(open_bwd)

        if not meeting_node:
            print(f"    未找到路径 (迭代{iterations}次)")
            return None

        # 重构路径
        path = []
        node = meeting_node if meeting_from_fwd else forward_node
        while node:
            path.insert(0, (node.x, node.y, node.z))
            node = node.parent

        node = backward_node.parent if meeting_from_fwd else meeting_node.parent
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent

        print(f"    找到路径! 长度:{len(path)}, 扩展:{self.nodes_expanded}")
        return path

    def _expand(self, current, open_list, open_dict, closed, goal, expanded_grid):
        """扩展节点"""
        for dx, dy, dz in self.directions:
            nx, ny, nz = current.x + dx, current.y + dy, current.z + dz

            if not self.env.is_free(nx, ny, nz, expanded_grid):
                continue
            if (nx, ny, nz) in closed:
                continue

            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            new_g = current.g + dist
            new_h = math.sqrt((nx-goal.x)**2 + (ny-goal.y)**2 + (nz-goal.z)**2)
            new_f = new_g + new_h

            if (nx, ny, nz) in open_dict:
                if new_g < open_dict[(nx, ny, nz)].g:
                    node = open_dict[(nx, ny, nz)]
                    node.g = new_g
                    node.h = new_h
                    node.f = new_f
                    node.parent = current
                    heapq.heapify(open_list)
            else:
                neighbor = Node(nx, ny, nz, new_g, new_h, new_f, current)
                heapq.heappush(open_list, neighbor)
                open_dict[(nx, ny, nz)] = neighbor


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 50)
    print("eVTOL路径规划 v3.0简化版")
    print("=" * 50)

    # 创建环境
    print("\n1. 创建环境...")
    env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)

    start = (250, 250, 80)
    goal = (2250, 2250, 100)

    reserved_zones = [(start[0], start[1], 50), (goal[0], goal[1], 50)]
    env.generate_random_buildings(n_buildings=250, reserved_zones=reserved_zones)

    print(f"  地图: {env.size_x}×{env.size_y}×{env.size_z}")
    print(f"  建筑物: {len(env.buildings)}栋")
    print(f"  起点: {start}, 终点: {goal}")

    # 路径规划
    print("\n2. 路径规划...")
    planner = BidirectionalAStar(env)
    path = planner.find_path(start, goal, use_safety_margin=True)

    if path:
        print(f"\n3. 结果:")
        print(f"  路径长度: {len(path)}步")
        print(f"  扩展节点: {planner.nodes_expanded}")
        print(f"  直线距离: {math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2):.1f}米")
    else:
        print("\n  未找到路径")

    return path


if __name__ == "__main__":
    main()
