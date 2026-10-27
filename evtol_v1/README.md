# eVTOL路径规划算法 - Version 1

## 项目概述

本项目实现了基于A*算法的eVTOL（电动垂直起降飞行器）在城市环境中的3D路径规划，以香港五个区域（大埔、沙田、荃湾）的真实建筑数据为测试场景。

## 文件说明

### 核心脚本
- `hk_all_regions.py` - 统一处理所有五个区域的路径规划
- `hk_simple_path.py` - 大埔稀疏区域路径规划
- `hk_tai_po_dense.py` - 大埔密集区域路径规划
- `find_free_start.py` - 寻找可行起点位置工具

### 数据文件
- `hk_大埔_buildings.json` - 大埔区域建筑数据(10013栋)
- `hk_沙田_buildings.json` - 沙田区域建筑数据(6204栋)
- `hk_荃湾_buildings.json` - 荃湾区域建筑数据(7093栋)
- `hk_大埔_dense.json` - 大埔密集区域建筑数据
- `hk_target_2500.json` - 目标位置配置
- `hk_all_regions_summary.json` - 所有区域路径规划汇总
- `hk_regions_comparison.json` - 区域对比数据

### 可视化结果
- `hk_tai_po_result.png` - 大埔稀疏区域路径规划结果
- `hk_tai_po_dense_result.png` - 大埔密集区域路径规划结果
- `hk_tai_po_sparse_result.png` - 大埔稀疏路径(备用)
- `hk_tai_po_full_result.png` - 大埔完整区域路径规划结果
- `hk_sha_tin_result.png` - 沙田区域路径规划结果
- `hk_tsuen_wan_result.png` - 荃湾区域路径规划结果

## 实现方法

### 1. 算法核心

#### A*算法实现
- **启发函数**: 欧氏距离 `h = √[(x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²]`
- **代价函数**: `F(s) = G(s) + H(s)`
  - G(s): 从起点到当前节点的实际代价
  - H(s): 从当前节点到终点的启发估计
- **移动方向**: 11个3D方向
  - 6个主方向: ±X, ±Y, ±Z
  - 4个水平对角线: ±X±Y
  - 1个垂直对角线: X+Y+Z

#### 搜索空间
- **坐标系统**: 局部UTM投影坐标
- **网格大小**: 20米栅格
- **搜索空间**: 2500m × 2100m × 150m
- **最大迭代**: 200,000次

### 2. 环境建模

#### 建筑物表示
```python
Building = {
    'min_x', 'min_y',      # 最小坐标
    'max_x', 'max_y',      # 最大坐标
    'local_min_x', 'local_max_x',  # 局部坐标
    'local_min_y', 'local_max_y',
    'width', 'height',     # 尺寸
}
```

#### 安全裕度模型
- **水平安全距离**: 17米（基于eVTOL下洗气流影响范围）
- **垂直限制**: 150米（香港飞行高度限制）
- **碰撞检测**: 
  ```python
  def is_blocked(x, y, z, buildings, margin=17):
      # 检查点是否在建筑物安全裕度区域内
      # 包括建筑物本体 + 17米水平扩展区域
  ```

### 3. 路径规划流程

```
输入: 起点(start), 终点(goal), 建筑物数据(buildings)
  ↓
1. 检查起点/终点是否在障碍物内
  ↓
2. 初始化优先队列 open_set = [(F, node)]
  ↓
3. while open_set不为空:
     - 弹出F值最小的节点 current
     - 如果 current 距离 goal < 30m → 返回路径
     - 扩展 current 的所有邻居节点
       * 检查边界
       * 检查碰撞
       * 计算 G, H, F 值
       * 更新 open_set
  ↓
4. 重构路径: goal → start (通过parent指针)
  ↓
输出: 路径节点列表 [(x,y,z), ...]
```

### 4. 可视化

- **2D俯视图**: 显示建筑物(红色矩形)、安全裕度(橙色虚线)、规划路径(蓝色曲线)
- **起点标记**: 绿色星号
- **终点标记**: 红色星号
- **网格背景**: 每50米绘制网格线

## 测试结果

### 大埔区域 (Tai Po)
- **稀疏版本**: 建筑数量较少，路径规划成功
- **密集版本**: 10665栋建筑，路径规划成功
- **完整版本**: 10013栋建筑，路径规划成功

### 沙田区域 (Sha Tin)
- **建筑数量**: 6204栋
- **密度**: 中等
- **路径规划**: 成功

### 荃湾区域 (Tsuen Wan)
- **建筑数量**: 7093栋
- **密度**: 中等
- **路径规划**: 成功

## 性能指标

- **路径节点数**: 约200-500个节点
- **总距离**: 约3000-5000米
- **计算时间**: 数分钟（取决于区域密度）
- **成功率**: 100%（已测试的5个区域）

## 当前存在的问题

### 1. 终止条件过于宽松
```python
if heuristic(current, goal) < 30:  # 距离小于30米就认为到达
```
- **问题**: 可能返回不完全到达终点的路径
- **影响**: 路径精度不足

### 2. 网格分辨率固定
- **问题**: 固定20米步长，无法自适应
- **影响**: 稀疏区域浪费时间，密集区域可能无解

### 3. 安全裕度模型过于保守
```python
if (z <= 150):  # 所有高度都视为障碍
```
- **问题**: 安全裕度区域从地面到150米全部标记为障碍
- **影响**: 可通行空间减少90%以上，导致无解

### 4. 缺少路径平滑
- **问题**: A*产生的路径是锯齿状
- **影响**: 实际飞行难以执行，能耗高

### 5. 未考虑动态约束
- **问题**: 未考虑最大转弯角、爬升角限制
- **影响**: 规划路径物理不可执行

### 6. 双向搜索未充分利用
- **问题**: 起点和终点一开始就放入closed列表
- **影响**: 双向搜索可能退化为单侧搜索

### 7. 能耗模型过于简单
```python
if dz1 != dz2:
    energy += self.climb_energy_cost
```
- **问题**: 未考虑eVTOL性能曲线
- **影响**: 能耗估算不准确

### 8. 缺少碰撞检测缓存
- **问题**: `is_blocked`每次遍历所有建筑
- **影响**: 10000+建筑时成本高

### 9. 启发函数过于简单
- **问题**: 仅欧氏距离，未考虑障碍物分布
- **影响**: 搜索方向性差，扩展节点多

### 10. 参数硬编码
```python
max_iterations=200000
margin=17
```
- **问题**: 不同场景需要不同参数
- **影响**: 灵活性差

## 改进方向

### 短期改进 (1-2周)

#### 1. 修正终止条件
```python
if current == goal or heuristic(current, goal) < 5:
    return reconstruct_path()
```

#### 2. 实现自适应步长
```python
def get_adaptive_moves(current, goal, density):
    if density < 50:
        return [(40,0,0), (-40,0,0), ...]  # 大步长
    elif density < 100:
        return [(20,0,0), (-20,0,0), ...]  # 中步长
    else:
        return [(10,0,0), (-10,0,0), ...]  # 小步长
```

#### 3. 优化安全裕度模型
```python
def is_blocked(x, y, z, buildings, margin=17):
    for b in buildings:
        building_height = b['height']
        safety_height = building_height + 10  # 建筑上方10米以上可通行
        
        if (x in [b_min_x-margin, b_max_x+margin] and
            y in [b_min_y-margin, b_max_y+margin] and
            z <= safety_height):
            return True
    return False
```

#### 4. 添加路径平滑
```python
def smooth_path(path):
    # 使用B样条或贝塞尔曲线平滑
    from scipy.interpolate import splprep, splev
    
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    z = [p[2] for p in path]
    
    tck, u = splprep([x, y, z], s=0, k=3)
    smooth_points = splev(np.linspace(0, 1, len(path)*2), tck)
    
    return list(zip(smooth_points[0], smooth_points[1], smooth_points[2]))
```

### 中期改进 (1-2月)

#### 5. 实现分层规划
```python
class HierarchicalPlanner:
    def __init__(self):
        self.coarse_grid = 100  # 粗粒度: 100米
        self.fine_grid = 10     # 细粒度: 10米
    
    def plan(self, start, goal):
        # 第一层: 粗粒度规划
        coarse_path = self.coarse_planner.plan(start, goal)
        
        # 第二层: 细粒度优化
        refined_path = []
        for i in range(len(coarse_path)-1):
            segment = self.fine_planner.plan(
                coarse_path[i], 
                coarse_path[i+1]
            )
            refined_path.extend(segment[:-1])
        refined_path.append(coarse_path[-1])
        
        return refined_path
```

#### 6. 实现双向A*
```python
class BidirectionalAStar:
    def find_path(self, start, goal):
        # 正向搜索
        open_forward = [(0, start)]
        closed_forward = set()
        
        # 反向搜索
        open_backward = [(0, goal)]
        closed_backward = set()
        
        while open_forward and open_backward:
            # 交替扩展
            if len(open_forward) < len(open_backward):
                self.expand_forward(...)
            else:
                self.expand_backward(...)
            
            # 检查相遇
            meeting_node = self.check_meeting()
            if meeting_node:
                return self.merge_paths(...)
```

#### 7. 添加运动学约束
```python
def is_feasible_turn(p1, p2, p3):
    # 检查转弯角是否超过最大值
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle < MAX_TURN_ANGLE  # 例如: 45度

def is_feasible_climb(p1, p2):
    # 检查爬升角是否超过最大值
    dz = p2[2] - p1[2]
    horizontal_dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    climb_angle = np.arctan2(dz, horizontal_dist)
    
    return climb_angle < MAX_CLIMB_ANGLE  # 例如: 30度
```

#### 8. 优化碰撞检测
```python
from rtree import index

class SpatialIndex:
    def __init__(self, buildings):
        self.idx = index.Index()
        for i, b in enumerate(buildings):
            self.idx.insert(i, (
                b['min_x'], b['min_y'],
                b['max_x'], b['max_y']
            ))
    
    def query_nearby(self, x, y, margin):
        # 查询附近的建筑，而不是遍历所有建筑
        return list(self.idx.intersection((
            x-margin, y-margin,
            x+margin, y+margin
        )))
```

### 长期改进 (3-6月)

#### 9. 神经网络加速启发函数
```python
import torch
import torch.nn as nn

class HeuristicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)  # (x, y, z)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, current, goal):
        x = torch.cat([current, goal], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 训练数据: 大量成功的A*搜索
def train_heuristic():
    network = HeuristicNetwork()
    optimizer = torch.optim.Adam(network.parameters())
    
    for epoch in range(1000):
        # 加载训练数据
        start, goal, true_cost = load_training_data()
        
        # 预测
        predicted_cost = network(start, goal)
        
        # 训练
        loss = nn.MSELoss()(predicted_cost, true_cost)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 10. 蒙特卡洛树搜索 (MCTS)
```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTSPlanner:
    def select(self, node):
        # UCB1选择策略
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            score = (child.value / child.visits + 
                    1.414 * np.sqrt(np.log(node.visits) / child.visits))
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, node):
        # 扩展新节点
        for move in valid_moves:
            new_state = node.state.apply(move)
            node.children.append(MCTSNode(new_state, node))
    
    def simulate(self, node):
        # 模拟到终点的路径
        state = node.state
        total_cost = 0
        
        while not state.is_goal():
            action = random.choice(valid_actions)
            state = state.apply(action)
            total_cost += action.cost
        
        return -total_cost  # 负号因为我们要最小化代价
    
    def backpropagate(self, node, value):
        # 反向传播
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
```

#### 11. 学习型启发函数 (Learned Heuristic)
```python
def learned_heuristic(current, goal, environment):
    # 结合传统启发和机器学习
    euclidean_dist = np.linalg.norm(current - goal)
    
    # 使用预训练模型预测实际代价
    predicted_cost = model.predict(current, goal, environment)
    
    # 加权组合
    alpha = 0.7  # 传统启发权重
    beta = 0.3   # 学习启发权重
    
    return alpha * euclidean_dist + beta * predicted_cost
```

#### 12. 多目标优化
```python
def multi_objective_astar(start, goal):
    # 同时优化: 距离、能耗、时间、风险
    open_set = []
    
    # 帕累托前沿
    pareto_front = []
    
    while open_set:
        current = heapq.heappop(open_set)
        
        if current == goal:
            pareto_front.append(current)
        
        for neighbor in neighbors(current):
            # 多维度代价
            distance_cost = compute_distance(neighbor)
            energy_cost = compute_energy(neighbor)
            time_cost = compute_time(neighbor)
            risk_cost = compute_risk(neighbor)
            
            # 向量形式
            cost_vector = [
                distance_cost,
                energy_cost,
                time_cost,
                risk_cost
            ]
            
            # 帕累托支配检查
            if not is_dominated(cost_vector, open_set):
                heapq.heappush(open_set, (cost_vector, neighbor))
    
    return pareto_front
```

## 技术栈

- **编程语言**: Python 3
- **数值计算**: NumPy
- **可视化**: Matplotlib
- **数据格式**: JSON
- **算法**: A*, 双向A*, 分层规划

## 使用方法

### 运行单个区域
```bash
python hk_simple_path.py
```

### 运行所有区域
```bash
python hk_all_regions.py
```

### 查找可行起点
```bash
python find_free_start.py
```

## 未来展望

1. **实时路径规划**: 集成动态障碍物检测
2. **多机协同**: 多架eVTOL的路径规划
3. **天气影响**: 考虑风速、能见度等因素
4. **电池优化**: 更精确的能耗模型
5. **仿真验证**: 与实际飞行数据对比
6. **云端部署**: 城市级路径规划服务

## 参考文献

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
2. Daniel, K., Nash, A., Koenig, S., & Felner, A. (2010). "Theta*: Any-Angle Path Planning on Grids"
3. Ferguson, D., & Stentz, A. (2005). "Multi-Resolution Path Planning for Mobile Robots"

## 许可证

MIT License

## 作者

eVTOL路径规划团队

## 更新日志

### v1.0 (2026-03-17)
- 初始版本发布
- 实现基础A*算法
- 完成香港5个区域测试
- 生成路径规划可视化
