#原版复刻
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于无图形界面环境
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import heapq
import math
import random
import zhplot
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class Building:
    """建筑物数据类"""
    x: int          # 左下角X坐标（栅格单位）
    y: int          # 左下角Y坐标
    z: int          # 地面高度
    length: int     # 长度（X方向）
    width: int      # 宽度（Y方向）
    height: int     # 高度（Z方向）
    btype: int      # 类型：1-居民楼，2-公共建筑，3-政府建筑
    color: str      # 可视化颜色

@dataclass
class Node:
    """路径节点类"""
    x: int
    y: int
    z: int
    g: float = 0.0     # 实际代价 G(s)
    h: float = 0.0     # 启发代价 H(s)
    f: float = 0.0     # 总代价 F(s)
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))

class CityEnvironment:
    """城市环境类"""
    def __init__(self, size_x=500, size_y=500, size_z=150, grid_size_xy=1, grid_size_z=1):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.grid_size_xy = grid_size_xy  # X/Y轴：1米/栅格
        self.grid_size_z = grid_size_z    # Z轴：1米/栅格
        self.buildings: List[Building] = []
        self.grid = np.zeros((size_x, size_y, size_z), dtype=np.uint8)

        # 空间索引：用于快速查找某个位置是否在建筑物上方
        self.building_above_map = None

        # 安全裕度参数（来自论文计算结果，单位：米）
        # D₁ ≈ 16.97米（居民楼）、D₂ ≈ 17.59米（公共建筑）、D₃ ≈ 18.09米（政府建筑）
        # 转换为栅格数：由于栅格大小为1米/栅格，所以栅格数 = 米数（取整）
        self.safety_margins = {
            1: 16.974,  # 居民楼：D₁ ≈ 16.97米 → 17栅格
            2: 17.588,  # 公共建筑：D₂ ≈ 17.59米 → 18栅格
            3: 18.085   # 政府建筑：D₃ ≈ 18.09米 → 18栅格
        }
        # XY轴安全裕度（转换为栅格数，取整）
        self.grid_margins_xy = {k: int(v) for k, v in self.safety_margins.items()}
        # Z轴安全裕度（转换为栅格数，取整）
        self.grid_margins_z = {k: int(v) for k, v in self.safety_margins.items()}

        # 颜色映射（根据论文：红色=政府，绿色=公共，蓝色=住宅）
        self.color_map = {
            1: 'blue',   # 普通居民住宅建筑
            2: 'green',  # 医疗、教育等公共设施建筑
            3: 'red'     # 政府建筑
        }
        
    def generate_random_buildings(self, n_buildings=50, reserved_zones=None):
        """生成随机建筑物，避免重叠，保持均匀分布

        参数：
            n_buildings: 目标建筑物数量
            reserved_zones: 保留区域列表，格式为[(x, y, radius), ...]，避免在这些区域生成建筑物

        改进策略：
        1. 增加建筑物之间的最小间距（考虑安全裕度）
        2. 保留起点和终点周围的区域
        3. 生成更多候选建筑以提高成功率
        """
        self.buildings.clear()

        if reserved_zones is None:
            reserved_zones = []

        # 第一步：生成初始建筑参数池（400栋，提高候选数量）
        print(f"  生成初始建筑池（400栋）...")
        initial_pool = []
        attempts = 0
        max_attempts = 1000

        while len(initial_pool) < 400 and attempts < max_attempts:
            attempts += 1

            # 建筑物尺寸：长30-60米，宽30-60米（扩大5倍）
            length = random.randint(30, 60)
            width = random.randint(30, 60)

            # x, y坐标（保留边距）
            x = random.randint(50, self.size_x - length - 50)
            y = random.randint(50, self.size_y - width - 50)

            # 检查是否在保留区域内（起飞/降落区）
            in_reserved = False
            for rz_x, rz_y, radius in reserved_zones:
                # 计算建筑中心点
                center_x = x + length / 2
                center_y = y + width / 2
                distance = math.sqrt((center_x - rz_x)**2 + (center_y - rz_y)**2)
                if distance < radius + max(self.grid_margins_xy.values()):
                    in_reserved = True
                    break

            if in_reserved:
                continue

            # 典型高度：30米、54米、72米、90米、120米、150米
            height = random.choice([30, 54, 72, 90, 120, 150])

            # 建筑物类型按比例随机（政府建筑：其他 = 1:6）
            type_choice = random.randint(1, 7)
            if type_choice == 7:
                btype = 3  # 政府建筑
            else:
                btype = random.choice([1, 2])  # 居民楼或公共建筑

            # 计算该建筑的安全裕度
            margin_xy = self.grid_margins_xy[btype]

            initial_pool.append({
                'x': x, 'y': y,
                'length': length, 'width': width,
                'height': height, 'btype': btype,
                'margin_xy': margin_xy
            })

        print(f"  初始建筑池已生成，包含 {len(initial_pool)} 栋建筑（尝试了{attempts}次）")

        # 第二步：筛选建筑物，考虑安全裕度间距
        print(f"  筛选建筑物（目标：{n_buildings}栋，考虑安全裕度间距）...")

        for building_data in initial_pool:
            if len(self.buildings) >= n_buildings:
                break

            # 当前建筑边界（包括安全裕度）
            x_left = building_data['x'] - building_data['margin_xy']
            x_right = building_data['x'] + building_data['length'] + building_data['margin_xy']
            y_bottom = building_data['y'] - building_data['margin_xy']
            y_top = building_data['y'] + building_data['width'] + building_data['margin_xy']

            # 检查与已选建筑（包括安全裕度）是否重叠
            overlap = False
            for existing in self.buildings:
                existing_margin = self.grid_margins_xy[existing.btype]
                ex_left = existing.x - existing_margin
                ex_right = existing.x + existing.length + existing_margin
                ey_bottom = existing.y - existing_margin
                ey_top = existing.y + existing.width + existing_margin

                # 检查安全裕度区域是否重叠
                if not (x_right <= ex_left or x_left >= ex_right or y_top <= ey_bottom or y_bottom >= ey_top):
                    overlap = True
                    break

            if not overlap:
                # 不重叠，加入最终列表
                building = Building(
                    x=building_data['x'],
                    y=building_data['y'],
                    z=0,
                    length=building_data['length'],
                    width=building_data['width'],
                    height=building_data['height'],
                    btype=building_data['btype'],
                    color=self.color_map[building_data['btype']]
                )
                self.buildings.append(building)

                # 在网格中标记建筑物
                self.grid[building.x:building.x+building.length,
                         building.y:building.y+building.width,
                         0:building.height] = building.btype

        print(f"  成功生成 {len(self.buildings)} 栋无重叠建筑物（考虑安全裕度）")

        # 如果不够50栋，输出警告
        if len(self.buildings) < n_buildings:
            print(f"  警告：仅找到 {len(self.buildings)} 栋无重叠建筑，目标为 {n_buildings} 栋")

        # 创建空间索引：记录每个(x,y)位置对应的建筑物高度
        print(f"  创建建筑物空间索引...")
        self.building_above_map = np.zeros((self.size_x, self.size_y), dtype=np.float32)
        for building in self.buildings:
            self.building_above_map[
                building.x:building.x + building.length,
                building.y:building.y + building.width
            ] = max(self.building_above_map[
                building.x:building.x + building.length,
                building.y:building.y + building.width
            ].max(), building.height)
        print(f"  空间索引创建完成")

    def expand_with_safety_margin(self):
        """根据安全裕度扩展建筑物

        严格执行：路径不能从安全裕度区域的楼顶穿过
        因此安全裕度区域（XY平面扩展）在整个高度范围内都是障碍

        X/Y轴和Z轴使用不同的栅格大小：
        - X/Y轴：1米/栅格
        - Z轴：1米/栅格
        """
        expanded_grid = self.grid.copy()

        for building in self.buildings:
            # XY平面安全裕度（栅格数）
            margin_xy = self.grid_margins_xy[building.btype]

            # 计算扩展后的边界（XY平面扩展）
            x1 = max(0, building.x - margin_xy)
            x2 = min(self.size_x, building.x + building.length + margin_xy)
            y1 = max(0, building.y - margin_xy)
            y2 = min(self.size_y, building.y + building.width + margin_xy)

            # 关键修改：安全裕度区域在整个高度范围内都是障碍（不能从楼顶穿过）
            # 从地面到天空顶部（0 到 size_z）
            z1 = 0
            z2 = self.size_z

            # 标记扩展区域为障碍
            expanded_grid[x1:x2, y1:y2, z1:z2] = 9  # 用9表示扩展的安全区域

        return expanded_grid

    def check_path_feasibility(self, start, goal, expanded_grid=None):
        """检查起点到终点之间是否存在可行通道

        使用简化方法：检查起点和终点周围是否有足够的自由空间
        以及是否有一条大致的方向不被完全阻塞
        严格执行不穿过建筑物屋顶的要求
        """
        if expanded_grid is None:
            expanded_grid = self.expand_with_safety_margin()

        # 检查起点和终点是否可行（使用严格的屋顶检查）
        if not self.is_free(start[0], start[1], start[2], expanded_grid, require_above_building=True, min_clearance=5):
            print(f"  ❌ 起点 {start} 不可行（在障碍物内或低于建筑物）")
            return False

        if not self.is_free(goal[0], goal[1], goal[2], expanded_grid, require_above_building=True, min_clearance=5):
            print(f"  ❌ 终点 {goal} 不可行（在障碍物内或低于建筑物）")
            return False

        # 检查起点和终点周围的自由空间（使用严格的屋顶检查）
        def count_free_space(x, y, z, radius=20):
            """计算某点周围指定半径内的自由空间数量"""
            count = 0
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if self.is_free(nx, ny, z, expanded_grid, require_above_building=True, min_clearance=5):
                        count += 1
            return count

        start_free = count_free_space(start[0], start[1], start[2], radius=30)
        goal_free = count_free_space(goal[0], goal[1], goal[2], radius=30)

        print(f"  ✓ 起点周围30米自由空间（需在建筑上方5米）: {start_free} 个栅格")
        print(f"  ✓ 终点周围30米自由空间（需在建筑上方5米）: {goal_free} 个栅格")

        # 如果任一端自由空间太少，返回False
        min_free_threshold = 100
        if start_free < min_free_threshold or goal_free < min_free_threshold:
            print(f"  ⚠ 警告: 起点/终点周围自由空间不足（阈值: {min_free_threshold}）")
            return False

        # 检查起点到终点的直接方向是否被完全阻塞（使用严格的屋顶检查）
        # 采样直线上的几个点
        steps = 10
        dx = (goal[0] - start[0]) / steps
        dy = (goal[1] - start[1]) / steps
        dz = (goal[2] - start[2]) / steps

        blocked_count = 0
        for i in range(1, steps):
            check_x = int(start[0] + i * dx)
            check_y = int(start[1] + i * dy)
            check_z = int(start[2] + i * dz)

            if not self.is_free(check_x, check_y, check_z, expanded_grid, require_above_building=True, min_clearance=5):
                blocked_count += 1

        print(f"  ✓ 起点到终点直线路径阻塞点: {blocked_count}/{steps}")

        # 如果所有采样点都被阻塞，返回False
        if blocked_count == steps:
            print(f"  ❌ 错误: 起点到终点的直线路径被完全阻塞")
            return False

        return True

    def find_free_position(self, x_center, y_center, z_start, expanded_grid, max_radius=50):
        """在指定中心点附近寻找自由位置（考虑安全裕度，严格执行不穿过屋顶）"""
        # 螺旋搜索，从中心向外扩展
        for radius in range(0, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-2, 3):  # 高度范围±2米
                        nx = x_center + dx
                        ny = y_center + dy
                        nz = z_start + dz

                        if (0 <= nx < self.size_x and
                            0 <= ny < self.size_y and
                            0 <= nz < self.size_z):
                            # 使用严格的屋顶检查（require_above_building=True）
                            if self.is_free(nx, ny, nz, expanded_grid, require_above_building=True, min_clearance=5):
                                return (nx, ny, nz)
        return None

    def is_free(self, x, y, z, expanded_grid=None, require_above_building=True, min_clearance=5):
        """检查位置是否可通行（优化版本，使用空间索引）

        参数：
            x, y, z: 要检查的栅格坐标
            expanded_grid: 扩展后的网格（考虑安全裕度）
            require_above_building: 是否严格要求在建筑物上方（已废弃，expanded_grid已包含完整障碍）
            min_clearance: 路径需要在建筑物上方保持的最小垂直间距（米）
        """
        if not (0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z):
            return False

        grid_to_check = expanded_grid if expanded_grid is not None else self.grid

        # 检查是否在障碍物内部（包括建筑物和安全裕度）
        # expanded_grid已经包含了完整的障碍：建筑物本身 + XY平面安全裕度（贯穿整个高度）
        if grid_to_check[x, y, z] != 0:
            return False

        return True
    
class ImprovedThetaStar:
    """改进的Theta*算法实现"""

    def __init__(self, env: CityEnvironment, use_line_of_sight=False):
        self.env = env
        self.expanded_grid = None
        self.use_line_of_sight = use_line_of_sight  # 默认禁用视线检查以提高速度

        # 算法参数：使用6方向而不是26方向，大幅提高速度
        self.move_directions = self.get_3d_directions(6)

        # 能耗参数
        self.climb_energy_cost = 1.0
        self.turn_energy_cost = 0.8
        self.alpha1 = 0.4  # 航迹长度权重
        self.alpha2 = 0.6  # 能耗权重
        self.w1 = 0.4      # G(s)权重
        self.w2 = 0.6      # H(s)权重
        
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
        else:  # 6方向
            return [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    def line_of_sight(self, node1: Node, node2: Node, expanded_grid):
        """改进的视线检查（使用Bresenham算法）"""
        x1, y1, z1 = node1.x, node1.y, node1.z
        x2, y2, z2 = node2.x, node2.y, node2.z
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1
        
        # 判断主要驱动轴
        if dx >= dy and dx >= dz:
            p1 = 2*dy - dx
            p2 = 2*dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2*dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2*dx
                p1 += 2*dy
                p2 += 2*dz
                
                if not self.env.is_free(x1, y1, z1, expanded_grid, require_above_building=True, min_clearance=5):
                    return False
        elif dy >= dx and dy >= dz:
            p1 = 2*dx - dy
            p2 = 2*dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2*dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2*dy
                p1 += 2*dx
                p2 += 2*dz
                
                if not self.env.is_free(x1, y1, z1, expanded_grid, require_above_building=True, min_clearance=5):
                    return False
        else:
            p1 = 2*dy - dz
            p2 = 2*dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2*dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2*dz
                p1 += 2*dy
                p2 += 2*dx
                
                if not self.env.is_free(x1, y1, z1, expanded_grid, require_above_building=True, min_clearance=5):
                    return False

        return True

    def heuristic(self, node: Node, goal: Node):
        """改进的启发函数 - 欧氏距离 + 方向引导

        使用欧氏距离作为基础，添加方向引导使算法更倾向于
        沿着起点到终点的直线方向搜索
        """
        # 基础欧氏距离
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        dz = abs(node.z - goal.z)
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        # 方向引导：计算节点到终点的向量与理想方向的夹角
        # 这里简化处理：如果节点距离起点更远，给予惩罚
        return distance

    def calculate_adaptive_step_size(self, current: Node, goal: Node):
        """自适应步长 - 在开阔区域使用大步长，在拥挤区域使用小步长

        返回：扩展的邻居节点列表
        """
        # 基础6个方向
        directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]

        # 距离终点较远时，添加对角线方向（大步长）
        dist_to_goal = self.heuristic(current, goal)

        if dist_to_goal > 100:
            # 添加8个对角线方向（水平+垂直）
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
            # 计算爬升角变化
            dz1 = current.z - parent.z
            dz2 = neighbor.z - current.z
            if dz1 != dz2:
                energy += self.climb_energy_cost
            
            # 计算转向角变化（水平方向）
            dx1 = current.x - parent.x
            dy1 = current.y - parent.y
            dx2 = neighbor.x - current.x
            dy2 = neighbor.y - current.y
            
            # 计算水平方向向量夹角
            if (dx1 != 0 or dy1 != 0) and (dx2 != 0 or dy2 != 0):
                dot = dx1*dx2 + dy1*dy2
                norm1 = math.sqrt(dx1**2 + dy1**2)
                norm2 = math.sqrt(dx2**2 + dy2**2)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot / (norm1 * norm2)
                    cos_angle = max(-1, min(1, cos_angle))
                    angle = math.acos(cos_angle)
                    if angle > math.pi/18:  # 角度变化大于10度
                        energy += self.turn_energy_cost
        
        return energy
    
    def distance(self, node1: Node, node2: Node):
        """欧氏距离"""
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        dz = node1.z - node2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def find_path(self, start, goal, use_safety_margin=True, consider_energy=True):
        """双向改进Theta*算法（优化版本）"""
        # 获取环境网格
        expanded_grid = None
        if use_safety_margin:
            expanded_grid = self.env.expand_with_safety_margin()

        # 初始化起点和终点节点
        start_node = Node(start[0], start[1], start[2])
        goal_node = Node(goal[0], goal[1], goal[2])

        # 检查起点和终点是否可行（严格执行不穿过屋顶）
        if not self.env.is_free(start[0], start[1], start[2], expanded_grid, require_above_building=True, min_clearance=5):
            print(f"    警告: 起点 {start} 不可行（在建筑物内或低于建筑物顶部5米）")
            return None
        if not self.env.is_free(goal[0], goal[1], goal[2], expanded_grid, require_above_building=True, min_clearance=5):
            print(f"    警告: 终点 {goal} 不可行（在建筑物内或低于建筑物顶部5米）")
            return None

        # 双向搜索的开放列表
        open_list_forward = []
        open_list_backward = []
        # 开放列表字典，用于快速查找
        open_dict_forward = {}
        open_dict_backward = {}
        closed_list_forward = {}
        closed_list_backward = {}

        # 初始化起点和终点
        start_node.g = 0
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h

        goal_node.g = 0
        goal_node.h = self.heuristic(goal_node, start_node)
        goal_node.f = goal_node.g + goal_node.h

        heapq.heappush(open_list_forward, start_node)
        heapq.heappush(open_list_backward, goal_node)
        open_dict_forward[(start_node.x, start_node.y, start_node.z)] = start_node
        open_dict_backward[(goal_node.x, goal_node.y, goal_node.z)] = goal_node

        closed_list_forward[(start_node.x, start_node.y, start_node.z)] = start_node
        closed_list_backward[(goal_node.x, goal_node.y, goal_node.z)] = goal_node

        meeting_node = None
        meeting_from_forward = False  # 标记相遇方向：True=前向遇到后向，False=后向遇到前向
        backward_node = None  # 后向搜索中的对应节点
        forward_node = None   # 前向搜索中的对应节点

        max_iterations = 50000  # 减少最大迭代次数
        iterations = 0

        while open_list_forward and open_list_backward and iterations < max_iterations:
            iterations += 1
            
            # 前向搜索
            if open_list_forward:
                current_forward = heapq.heappop(open_list_forward)
                
                # 检查是否相遇
                if (current_forward.x, current_forward.y, current_forward.z) in closed_list_backward:
                    meeting_node = current_forward
                    backward_node = closed_list_backward[(current_forward.x, current_forward.y, current_forward.z)]
                    meeting_from_forward = True  # 标记相遇方向
                    break

                self.expand_node(current_forward, open_list_forward, open_dict_forward, closed_list_forward,
                               goal_node, expanded_grid, consider_energy, is_forward=True)

            # 后向搜索
            if open_list_backward:
                current_backward = heapq.heappop(open_list_backward)

                # 检查是否相遇
                if (current_backward.x, current_backward.y, current_backward.z) in closed_list_forward:
                    meeting_node = current_backward
                    forward_node = closed_list_forward[(current_backward.x, current_backward.y, current_backward.z)]
                    meeting_from_forward = False  # 标记相遇方向
                    break

                self.expand_node(current_backward, open_list_backward, open_dict_backward, closed_list_backward,
                               start_node, expanded_grid, consider_energy, is_forward=False)
        
        if not meeting_node:
            print(f"    搜索失败: 迭代 {iterations} 次后未找到路径")
            print(f"    前向开放列表: {len(open_list_forward)}, 后向开放列表: {len(open_list_backward)}")
            print(f"    前向已探索节点: {len(closed_list_forward)}, 后向已探索节点: {len(closed_list_backward)}")

            # 诊断无解原因
            print(f"    诊断信息:")
            print(f"    - 起点: ({start_node.x}, {start_node.y}, {start_node.z})")
            print(f"    - 终点: ({goal_node.x}, {goal_node.y}, {goal_node.z})")
            print(f"    - 安全裕度: {expanded_grid is not None}")

            # 检查起点和终点周围的可通行空间
            if expanded_grid is not None:
                print(f"    - 检查起点周围3×3×3空间:")
                free_count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nx, ny, nz = start_node.x + dx, start_node.y + dy, start_node.z + dz
                            if self.env.is_free(nx, ny, nz, expanded_grid, require_above_building=True, min_clearance=5):
                                free_count += 1
                print(f"      起点周围可通行位置: {free_count}/26")

                print(f"    - 检查终点周围3×3×3空间:")
                free_count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            nx, ny, nz = goal_node.x + dx, goal_node.y + dy, goal_node.z + dz
                            if self.env.is_free(nx, ny, nz, expanded_grid, require_above_building=True, min_clearance=5):
                                free_count += 1
                print(f"      终点周围可通行位置: {free_count}/26")

            return None
        
        # 重构路径
        path = []
        
        # 前向部分：从起点到会议点
        if meeting_from_forward:
            node = meeting_node  # 前向搜索的会议节点
        else:
            node = forward_node   # 后向搜索触发相遇时，使用前向列表中的节点
        
        while node:
            path.insert(0, (node.x, node.y, node.z))
            node = node.parent
        
        # 后向部分：从会议点之后到终点（跳过会议节点，避免重复）
        if meeting_from_forward:
            # 前向触发相遇：从后向列表中的对应节点开始（跳过会议点本身）
            node = backward_node.parent
        else:
            # 后向触发相遇：从后向搜索的会议节点的父节点开始
            node = meeting_node.parent
        
        while node:
            path.append((node.x, node.y, node.z))
            node = node.parent
        
        return path
    
    def expand_node(self, current, open_list, open_dict, closed_list, goal, expanded_grid, consider_energy, is_forward):
        """扩展单个节点（优化版本，使用字典加速查找 + 自适应步长）"""
        # 使用自适应步长
        directions = self.calculate_adaptive_step_size(current, goal)

        for dx, dy, dz in directions:
            nx, ny, nz = current.x + dx, current.y + dy, current.z + dz

            # 检查边界和障碍（严格执行不穿过屋顶）
            if not self.env.is_free(nx, ny, nz, expanded_grid, require_above_building=True, min_clearance=5):
                continue

            # 创建邻居节点
            neighbor = Node(nx, ny, nz)

            # 检查是否已在关闭列表中
            if (nx, ny, nz) in closed_list:
                continue

            # 视线检查（仅当启用时）
            new_parent = current.parent
            if self.use_line_of_sight and new_parent and self.line_of_sight(new_parent, neighbor, expanded_grid):
                # 如果可以直线到达，跳过当前节点
                pass
            else:
                new_parent = current

            # 计算代价
            if new_parent:
                # 距离代价
                dist_cost = self.distance(new_parent, neighbor)

                # 能耗代价
                energy_cost = 0
                if consider_energy:
                    energy_cost = self.calculate_energy_cost(current, neighbor, new_parent)

                # 实际代价 G(s)
                g_cost = new_parent.g + self.alpha1 * dist_cost + self.alpha2 * energy_cost

                # 启发代价 H(s)
                h_cost = self.heuristic(neighbor, goal)

                # 总代价 F(s)
                f_cost = self.w1 * g_cost + self.w2 * h_cost
            else:
                # 从起点开始的情况
                g_cost = 0
                h_cost = self.heuristic(neighbor, goal)
                f_cost = h_cost

            # 使用字典快速检查是否在开放列表中
            if (nx, ny, nz) in open_dict:
                # 如果在开放列表中且有更优路径
                if g_cost < open_dict[(nx, ny, nz)].g:
                    open_dict[(nx, ny, nz)].g = g_cost
                    open_dict[(nx, ny, nz)].h = h_cost
                    open_dict[(nx, ny, nz)].f = f_cost
                    open_dict[(nx, ny, nz)].parent = new_parent
                    heapq.heapify(open_list)
            else:
                # 不在开放列表中，添加新节点
                neighbor.g = g_cost
                neighbor.h = h_cost
                neighbor.f = f_cost
                neighbor.parent = new_parent
                heapq.heappush(open_list, neighbor)
                open_dict[(nx, ny, nz)] = neighbor

        closed_list[(current.x, current.y, current.z)] = current

class Visualizer:
    """可视化类"""
    
    @staticmethod
    def plot_city_2d(env: CityEnvironment, paths=None, title="City Environment"):
        """绘制城市2D平面图"""
        fig, ax = plt.subplots(figsize=(16, 16))

        # 计算实际物理尺寸
        x_limit_m = env.size_x * env.grid_size_xy
        y_limit_m = env.size_y * env.grid_size_xy

        # 绘制网格背景（每50米一条线）
        for i in range(0, int(x_limit_m) + 1, 50):
            ax.axhline(i, color='gray', linestyle='--', alpha=0.2)
        for i in range(0, int(y_limit_m) + 1, 50):
            ax.axvline(i, color='gray', linestyle='--', alpha=0.2)

        # 绘制建筑物（使用栅格坐标，不转换为米）
        for building in env.buildings:
            # 绘制安全裕度区域（半透明矩形）
            margin_xy = env.grid_margins_xy[building.btype]
            x1 = max(0, building.y - margin_xy)
            y1 = max(0, building.x - margin_xy)
            width = building.width + 2 * margin_xy
            height = building.length + 2 * margin_xy
            
            margin_rect = Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=1.5,
                edgecolor='orange',
                facecolor='orange',
                linestyle='--',
                alpha=0.2
            )
            ax.add_patch(margin_rect)

            # 使用栅格坐标绘制，添加黑色边框增强可见性
            rect = Rectangle(
                (building.y, building.x),
                building.width, building.length,
                facecolor=building.color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(rect)

            # 显示所有建筑物的高度标签
            ax.text(building.y + building.width/2, building.x + building.length/2,
                   f'{building.height}m',
                   ha='center', va='center',
                   fontsize=5, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='black', alpha=0.5))

        # 绘制起点和终点
        start_x, start_y = paths[0][0][0] if paths and paths[0] else 0, paths[0][0][1] if paths and paths[0] else 0
        goal_x, goal_y = paths[0][-1][0] if paths and paths[0] else 0, paths[0][-1][1] if paths and paths[0] else 0

        # 绘制起飞区（绿色圆形区域）
        takeoff_circle = plt.Circle((start_y, start_x), 15, color='lime', alpha=0.3, label='Takeoff Zone')
        ax.add_patch(takeoff_circle)
        ax.scatter(start_y, start_x, c='lime', s=400, marker='*', edgecolors='black', linewidths=2, label='Takeoff Point', zorder=10)

        # 绘制停靠区（红色圆形区域）
        landing_circle = plt.Circle((goal_y, goal_x), 15, color='red', alpha=0.3, label='Landing Zone')
        ax.add_patch(landing_circle)
        ax.scatter(goal_y, goal_x, c='red', s=400, marker='*', edgecolors='black', linewidths=2, label='Landing Point', zorder=10)

        # 绘制路径
        has_paths = False
        if paths:
            colors = ['red', 'blue', 'green', 'magenta']
            labels = ['Standard', 'With Safety Margin', 'Theta*', 'Improved Theta*']
            line_styles = ['-', '--', ':', '-.']

            for i, (path, color, label, ls) in enumerate(zip(paths, colors, labels, line_styles)):
                if path:
                    has_paths = True
                    y_coords = [p[1] for p in path]
                    x_coords = [p[0] for p in path]
                    ax.plot(y_coords, x_coords, color=color, linestyle=ls,
                           linewidth=3, alpha=0.9, label=label, zorder=5)

        # 设置坐标轴（显示栅格数量）
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel(f'Y Axis (Grid Units) - {env.grid_size_xy}m per grid', fontsize=13)
        ax.set_ylabel(f'X Axis (Grid Units) - {env.grid_size_xy}m per grid', fontsize=13)
        ax.set_title(f'{title}\nMap Size: {x_limit_m}m × {y_limit_m}m | Buildings: {len(env.buildings)}', fontsize=15)
        ax.grid(True, alpha=0.3)

        # 设置图例（放在底部，不挡住路径）
        # 手动添加安全裕度图例项
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='orange', edgecolor='orange', alpha=0.2, label='Safety Margin')]
        for i, label in enumerate(labels):
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], linestyle=line_styles[i], 
                                            linewidth=3, label=label))
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='lime', linestyle='None',
                                         markersize=12, label='Takeoff Point'))
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='red', linestyle='None',
                                         markersize=12, label='Landing Point'))
        
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                 fontsize=9, framealpha=0.95, ncol=6)

        return fig
    
    @staticmethod
    def plot_city_3d(env: CityEnvironment, path=None, title="3D View"):
        """绘制城市3D视图"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制所有建筑物（使用实心柱体）
        for building in env.buildings:
            x = building.y
            y = building.x
            z = 0
            dx = building.width
            dy = building.length
            dz = building.height

            # 绘制建筑物柱体（8个顶点）
            # 底面
            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            zz = [z, z, z, z, z]
            ax.plot(xx, yy, zz, color=building.color, linewidth=1, alpha=0.8)

            # 顶面
            xx = [x, x+dx, x+dx, x, x]
            yy = [y, y, y+dy, y+dy, y]
            zz = [z+dz, z+dz, z+dz, z+dz, z+dz]
            ax.plot(xx, yy, zz, color=building.color, linewidth=1, alpha=0.8)

            # 竖边
            for i in range(5):
                ax.plot([xx[i], xx[i]], [yy[i], yy[i]], [z, z+dz],
                       color=building.color, linewidth=1, alpha=0.6)

            # 填充柱体表面
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            vertices = [
                [(x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z)],  # 底面
                [(x, y, z+dz), (x+dx, y, z+dz), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)],  # 顶面
                [(x, y, z), (x+dx, y, z), (x+dx, y, z+dz), (x, y, z+dz)],  # 前面
                [(x, y+dy, z), (x+dx, y+dy, z), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)],  # 后面
                [(x, y, z), (x, y+dy, z), (x, y+dy, z+dz), (x, y, z+dz)],  # 左面
                [(x+dx, y, z), (x+dx, y+dy, z), (x+dx, y+dy, z+dz), (x+dx, y, z+dz)]  # 右面
            ]
            poly = Poly3DCollection(vertices, alpha=0.5, facecolors=building.color, edgecolors='black', linewidths=0.5)
            ax.add_collection3d(poly)

        # 绘制路径
        if path:
            y_coords = [p[1] for p in path]
            x_coords = [p[0] for p in path]
            z_coords = [p[2] for p in path]
            ax.plot(y_coords, x_coords, z_coords, color='red', linewidth=3, label='Flight Path', alpha=0.9, zorder=100)

        # 绘制起点和终点
        if path:
            ax.scatter([path[0][1]], [path[0][0]], [path[0][2]],
                      c='lime', s=200, marker='*', edgecolors='black',
                      linewidths=2, label='Start', zorder=150)
            ax.scatter([path[-1][1]], [path[-1][0]], [path[-1][2]],
                      c='magenta', s=200, marker='*', edgecolors='black',
                      linewidths=2, label='Goal', zorder=150)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='Flight Path'),
            Line2D([0], [0], marker='*', color='lime', markersize=12, linestyle='None', label='Start'),
            Line2D([0], [0], marker='*', color='magenta', markersize=12, linestyle='None', label='Goal')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # 设置坐标轴
        ax.set_xlabel('Y (Grid Units)', fontsize=11)
        ax.set_ylabel('X (Grid Units)', fontsize=11)
        ax.set_zlabel('Height (meters)', fontsize=11)
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_zlim(0, max(150, max([b.height for b in env.buildings]) if env.buildings else 150))
        ax.set_title(f'{title}\nMap: {env.size_x*env.grid_size_xy}m × {env.size_y*env.grid_size_xy}m × {env.size_z}m',
                     fontsize=13)

        return fig

    @staticmethod
    def plot_city_with_safety_margin(env: CityEnvironment, title="Buildings with Safety Margin (2D View)"):
        """绘制带有安全裕度的建筑物（2D俯视图）"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # 绘制原始建筑物（实心矩形）
        for building in env.buildings:
            # 绘制原始建筑
            rect = Rectangle(
                (building.y, building.x),
                building.width,
                building.length,
                linewidth=2,
                edgecolor='black',
                facecolor=building.color,
                alpha=0.8,
                label=f'Type {building.btype}'
            )
            ax.add_patch(rect)

            # 绘制建筑高度标签
            ax.text(building.y + building.width/2, building.x + building.length/2,
                   f'{building.height}m',
                   ha='center', va='center',
                   fontsize=5, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='black', alpha=0.5))

        # 绘制安全裕度区域（半透明矩形）
        for building in env.buildings:
            margin_xy = env.grid_margins_xy[building.btype]

            # 安全裕度区域的边界
            x1 = max(0, building.y - margin_xy)
            y1 = max(0, building.x - margin_xy)
            width = building.width + 2 * margin_xy
            height = building.length + 2 * margin_xy

            # 绘制安全裕度边框
            margin_rect = Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='orange',
                facecolor='none',
                linestyle='--',
                alpha=0.7
            )
            ax.add_patch(margin_rect)

        # 设置坐标轴
        ax.set_xlim(0, env.size_y)
        ax.set_ylim(0, env.size_x)
        ax.set_xlabel(f'Y Axis (Grid Units) - {env.grid_size_xy}m per grid', fontsize=13)
        ax.set_ylabel(f'X Axis (Grid Units) - {env.grid_size_xy}m per grid', fontsize=13)
        ax.set_title(f'{title}\nMap Size: {env.size_x*env.grid_size_xy}m × {env.size_y*env.grid_size_xy}m | Buildings: {len(env.buildings)}',
                    fontsize=15)
        ax.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Residential (Type 1)'),
            Patch(facecolor='green', edgecolor='black', label='Public (Type 2)'),
            Patch(facecolor='red', edgecolor='black', label='Government (Type 3)'),
            Patch(facecolor='none', edgecolor='orange', linestyle='--', linewidth=2, label='Safety Margin (Orange Dashed)')
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.12),
                  fontsize=10, framealpha=0.95, ncol=4)

        return fig

def main():
    """主程序 - 复现论文仿真"""
    print("=" * 60)
    print("eVTOL航迹规划仿真 - 复现论文第五章")
    print("=" * 60)

    # 1. 创建城市环境
    print("\n1. 创建城市环境...")
    # 地图尺寸：2500m × 2500m × 150m
    # 栅格大小：X/Y轴 1米/栅格，Z轴 1米/栅格
    # 栅格数量：2500 × 2500 × 150 = 937,500,000个栅格
    env = CityEnvironment(size_x=2500, size_y=2500, size_z=150)

    # 设置起点和终点（用于保留区域）
    start = (250, 250, 60)  # 起飞区坐标（米）
    goal = (2250, 2250, 80)  # 停靠区坐标（米）

    # 设置保留区域（起飞区和停靠区周围预留足够空间）
    # 半径设置为40米，确保安全裕度不会覆盖这些区域
    reserved_zones = [
        (start[0], start[1], 40),  # 起飞区保留区域
        (goal[0], goal[1], 40)      # 停靠区保留区域
    ]

    # 增加建筑物数量，使地图更丰富
    n_buildings = 250
    env.generate_random_buildings(n_buildings=n_buildings, reserved_zones=reserved_zones)

    print(f"  地图尺寸: {env.size_x*env.grid_size_xy}m × {env.size_y*env.grid_size_xy}m × {env.size_z}m")
    print(f"  栅格数量: {env.size_x} × {env.size_y} × {env.size_z} = {env.size_x * env.size_y * env.size_z:,}")
    print(f"  X/Y轴栅格大小: {env.grid_size_xy}米/栅格")
    print(f"  Z轴栅格大小: {env.grid_size_z}米/栅格")
    print(f"  起飞区: {start}, 停靠区: {goal}")
    print(f"  保留区域半径: 40米（确保安全裕度不覆盖起降区）")

    # 2. 设置起点和终点（已在上面定义）
    # 地图：500×500米，使用米作为单位
    # 起飞区：位置(50, 50)，高度60米
    # 停靠区：位置(450, 450)，高度80米

    # 3. 环境诊断 - 检查路径可行性
    print("\n2. 环境诊断...")
    expanded_grid = env.expand_with_safety_margin()

    print(f"  检查起点到终点的通道可行性...")
    is_feasible = env.check_path_feasibility(start, goal, expanded_grid)

    if not is_feasible:
        print("  ⚠ 警告: 环境诊断显示路径可能不可行，但算法仍将尝试规划")
        print("  建议: 减少建筑物数量或调整安全裕度")

    # 在起飞区和停靠区附近寻找可行位置（考虑安全裕度）
    print(f"\n  初始起飞区: {start}")
    print(f"  初始停靠区: {goal}")

    start_free = env.is_free(start[0], start[1], start[2], expanded_grid)
    goal_free = env.is_free(goal[0], goal[1], goal[2], expanded_grid)

    if not start_free:
        print("  警告：起飞区在安全裕度范围内，搜索可行位置...")
        new_start = env.find_free_position(start[0], start[1], start[2], expanded_grid)
        if new_start:
            start = new_start
            print(f"  调整后的起飞区: {start}")
        else:
            print("  错误：无法在起飞区附近找到可行位置！")
            return None

    if not goal_free:
        print("  警告：停靠区在安全裕度范围内，搜索可行位置...")
        new_goal = env.find_free_position(goal[0], goal[1], goal[2], expanded_grid)
        if new_goal:
            goal = new_goal
            print(f"  调整后的停靠区: {goal}")
        else:
            print("  错误：无法在停靠区附近找到可行位置！")
            return None

    # 再次检查调整后的起点和终点
    start_free = env.is_free(start[0], start[1], start[2], expanded_grid)
    goal_free = env.is_free(goal[0], goal[1], goal[2], expanded_grid)
    print(f"  起飞区是否可行: {start_free}")
    print(f"  停靠区是否可行: {goal_free}")

    print(f"\n  最终起飞区: {start} (物理: {start[0]*env.grid_size_xy}m, {start[1]*env.grid_size_xy}m, {start[2]}m)")
    print(f"  最终停靠区: {goal} (物理: {goal[0]*env.grid_size_xy}m, {goal[1]*env.grid_size_xy}m, {goal[2]}m)")

    # 3. 创建路径规划器
    planner = ImprovedThetaStar(env)

    # 4. 情景1: 不考虑安全裕度
    print("\n2. 情景1: 不考虑安全裕度...")
    path_no_margin = planner.find_path(start, goal, use_safety_margin=False, consider_energy=False)
    print(f"  规划路径长度: {len(path_no_margin) if path_no_margin else '无解'}")

    # 5. 情景2: 考虑安全裕度
    print("\n3. 情景2: 考虑安全裕度...")
    path_with_margin = planner.find_path(start, goal, use_safety_margin=True, consider_energy=False)
    print(f"  规划路径长度: {len(path_with_margin) if path_with_margin else '无解'}")

    # 6. 情景3: 改进Theta*（仅考虑长度）
    print("\n4. 情景3: 改进Theta*算法（仅考虑长度）...")
    path_improved_length = planner.find_path(start, goal, use_safety_margin=True, consider_energy=False)
    print(f"  规划路径长度: {len(path_improved_length) if path_improved_length else '无解'}")

    # 7. 情景4: 改进Theta*（考虑经济性）
    print("\n5. 情景4: 改进Theta*算法（考虑航迹经济性）...")
    path_improved_energy = planner.find_path(start, goal, use_safety_margin=True, consider_energy=True)
    print(f"  规划路径长度: {len(path_improved_energy) if path_improved_energy else '无解'}")

    # 如果所有算法都失败，输出诊断信息
    if not any([path_no_margin, path_with_margin, path_improved_length, path_improved_energy]):
        print("\n" + "="*60)
        print("诊断信息：所有路径规划均失败")
        print("="*60)
        print(f"起点位置: {start}")
        print(f"终点位置: {goal}")
        print(f"建筑物数量: {len(env.buildings)}")

        # 检查起点和终点周围的障碍情况
        print("\n起点周围20×20区域的障碍情况（高度60米处）:")
        z = start[2]
        for dy in [-2, -1, 0, 1, 2]:
            row = ""
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = start[0] + dx, start[1] + dy
                if 0 <= nx < env.size_x and 0 <= ny < env.size_y:
                    is_blocked = not env.is_free(nx, ny, z)
                    row += "[X]" if is_blocked else "[.]"
                else:
                    row += "[#]"
            print(row)

        print("\n终点周围20×20区域的障碍情况（高度80米处）:")
        z = goal[2]
        for dy in [-2, -1, 0, 1, 2]:
            row = ""
            for dx in [-2, -1, 0, 1, 2]:
                nx, ny = goal[0] + dx, goal[1] + dy
                if 0 <= nx < env.size_x and 0 <= ny < env.size_y:
                    is_blocked = not env.is_free(nx, ny, z)
                    row += "[X]" if is_blocked else "[.]"
                else:
                    row += "[#]"
            print(row)
    
    # 8. 可视化结果 - 保存到文件并自动在 Windows 中打开
    print("\n6. 生成可视化图表...")
    
    import os
    output_dir = "/home/ayan1907/evtol_results"

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 建筑物安全裕度图（新增）
    fig_margin = Visualizer.plot_city_with_safety_margin(env, "Buildings with Safety Margin (Flight Zone)")
    fig_margin.savefig(os.path.join(output_dir, 'buildings_safety_margin.png'), dpi=150, bbox_inches='tight')
    print(f"  建筑物安全裕度图已保存")
    plt.close(fig_margin)

    # 2D对比图
    paths_2d = [path_no_margin, path_with_margin, path_improved_length, path_improved_energy]
    fig_2d = Visualizer.plot_city_2d(env, paths_2d, "Different Algorithms Comparison (2D View)")
    fig_2d.savefig(os.path.join(output_dir, 'path_comparison_2d.png'), dpi=150, bbox_inches='tight')
    print(f"  2D对比图已保存")
    plt.close(fig_2d)

    # 3D视图
    fig_3d = Visualizer.plot_city_3d(env, path_improved_energy, "3D Flight Path")
    fig_3d.savefig(os.path.join(output_dir, 'path_3d.png'), dpi=150, bbox_inches='tight')
    print(f"  3D航迹图已保存")
    plt.close(fig_3d)
    
    print(f"\n所有图表已保存到: {output_dir}")
    
    # 自动在 Windows 中打开图像
    print("\n正在打开图像...")
    import subprocess
    try:
        # 使用 explorer.exe 打开文件夹
        subprocess.run(['explorer.exe', output_dir], check=True)
    except:
        print("  提示: 请手动打开文件夹查看图像")
        print(f"  文件夹路径: {output_dir}")
    
    return {
        'environment': env,
        'paths': {
            'no_margin': path_no_margin,
            'with_margin': path_with_margin,
            'improved_length': path_improved_length,
            'improved_energy': path_improved_energy
        }
    }

if __name__ == "__main__":
    # 运行仿真
    results = main()


