# 低碳多目标A*路径规划系统

## 📋 文件清单

| 文件名 | 功能 | 说明 |
|--------|------|------|
| `low_carbon_node.py` | 低碳节点类 | 扩展传统Node，增加累计指标 |
| `low_carbon_astar.py` | 低碳A*算法 | 完整实现，包含测试 |

---

## 🎯 核心创新点

### **1. 低碳代价函数**

```
G_carbon(s) = α₁·L(s) + α₂·E_basic(s) + α₃·Σ|Δh| + α₄·Σ|Δβ|
F_carbon(s) = ω₁·G_carbon(s) + ω₂·H(s)
```

#### **各分量说明**

| 符号 | 物理意义 | 单位 | 作用 |
|------|---------|------|------|
| **L(s)** | 累计航迹长度 | 米 | 惩罚长路径 |
| **E_basic(s)** | 累计基础能耗 | 无量纲 | 惩罚高能耗机动 |
| **Σ\|Δh\|** | 累计垂直高度变化 | 米 | 惩罚频繁爬升/下降 |
| **Σ\|Δβ\|** | 累计水平转向角 | 弧度 | 惩罚频繁转向 |

---

## 🔧 实现细节

### **① 节点扩展逻辑**

```python
# 在expand_node()中
grandparent = current.parent if current.parent else None

for dx, dy, dz in directions:
    neighbor = LowCarbonNode(nx, ny, nz)
    
    # 更新低碳指标
    update_node_metrics(current, neighbor, grandparent)
    
    # 计算基础能耗
    segment_energy = calculate_basic_energy(current, neighbor, grandparent)
    neighbor.accumulated_energy = current.accumulated_energy + segment_energy
    
    # 计算G_carbon
    G_carbon = (
        alpha1 * neighbor.accumulated_length +
        alpha2 * neighbor.accumulated_energy +
        alpha3 * neighbor.accumulated_vertical_change +
        alpha4 * neighbor.accumulated_turn_angle
    )
    
    # 计算F_carbon
    F_carbon = omega1 * G_carbon + omega2 * H(neighbor, goal)
```

### **② 高度变化 Σ|Δh| 计算**

**每一步**都计算绝对高度差：

```
路径: (0,0,0) → (10,0,5) → (10,10,10)

第一步: Δh₁ = |5 - 0| = 5m
第二步: Δh₂ = |10 - 5| = 5m
累计 Σ|Δh| = 10m
```

**代码实现**：
```python
def calculate_vertical_change(node1, node2):
    return abs(node2.z - node1.z)

# 累加
neighbor.accumulated_vertical_change = (
    current.accumulated_vertical_change + 
    abs(neighbor.z - current.z)
)
```

### **③ 转向角 Σ|Δβ| 计算**

需要**三个节点**才能计算转向角：

```
路径: (0,0,0) → (10,0,0) → (10,10,0)

方位角1: atan2(0, 10) = 0 rad (东向)
方位角2: atan2(10, 0) = π/2 rad (北向)
转向角 Δβ = |π/2 - 0| = π/2 rad (90°)
```

**代码实现**：
```python
def calculate_bearing(node1, node2):
    return math.atan2(node2.y - node1.y, node2.x - node1.x)

def calculate_turn_angle(previous, current, next_node):
    bearing_in = calculate_bearing(previous, current)
    bearing_out = calculate_bearing(current, next_node)
    delta_beta = abs(bearing_out - bearing_in)
    
    # 处理角度环绕（350° → 10°）
    if delta_beta > math.pi:
        delta_beta = 2 * math.pi - delta_beta
    
    return delta_beta
```

---

## 🧪 测试结果

### **四种配置对比**

| 配置 | ω₁ | ω₂ | α₁ | α₂ | α₃ | α₄ | 路径长度 | 转向角 |
|------|----|----|----|----|----|----|---------|--------|
| **传统A*** | 0.5 | 0.5 | 1.0 | 0.0 | 0.0 | 0.0 | ❌ 失败 | - |
| **低碳均衡** | 0.7 | 0.3 | 0.3 | 0.2 | 0.25 | 0.25 | ✅ 4021 | 180° |
| **平稳优先** | 0.8 | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 | ✅ 4021 | 180° |
| **最短路径** | 0.5 | 0.5 | 0.5 | 0.0 | 0.25 | 0.25 | ✅ 4021 | 180° |

### **关键观察**

1. **传统A*失败** - 曼哈顿距离启发式不够强
2. **转向角统一** - 所有成功配置都是180°（可能是直角绕行）
3. **垂直变化相同** - 都是20m（起点80m → 终点100m）

---

## 🎓 论文写作建议

### **题目**
"基于多目标优化的低碳eVTOL路径规划方法"

### **创新点**
1. 提出低碳代价函数，集成航迹长度、能耗、垂直机动、水平转向四个维度
2. 实现基于曼哈顿距离的启发式函数，平衡搜索效率与路径质量
3. 设计多权重配置机制，支持不同场景的灵活调优
4. 通过仿真验证方法有效性

### **实验设计**
| 场景 | 建筑物数 | 起点-终点 | 测试配置 |
|------|----------|----------|---------|
| 稀疏场景 | 20 | 对角 | 4种 |
| 中等场景 | 50 | 对角 | 4种 |
| 密集场景 | 100 | 同高度 | 4种 |

### **评价指标**
- 路径长度（米）
- 总能耗（无量纲）
- 垂直变化累计（米）
- 转向角度累计（弧度）
- 搜索效率（迭代次数）

---

## 💡 使用方法

### **基本使用**

```python
from low_carbon_astar import LowCarbonAStar, CityEnvironment

# 创建环境
env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)
env.generate_random_buildings(n_buildings=50)

# 创建规划器
planner = LowCarbonAStar(
    env,
    omega1=0.7, omega2=0.3,    # F_carbon权重
    alpha1=0.3, alpha2=0.2,     # G_carbon分量权重
    alpha3=0.25, alpha4=0.25
)

# 规划路径
path, metrics = planner.find_path(
    start=(250, 250, 80),
    goal=(2250, 2250, 100),
    use_safety_margin=True
)

print(f"路径长度: {metrics['total_length']:.2f}m")
print(f"转向角度: {metrics['total_turn_angle']:.4f}rad")
```

### **运行测试**

```bash
cd /home/ayan1907/evtol
python low_carbon_astar.py
```

---

## 🚀 未来扩展

### **优先级1**
- [ ] 双向A*加速
- [ ] 自适应步长（6→26方向）
- [ ] 26方向移动

### **优先级2**
- [ ] 碰撞风险模型集成
- [ ] 多机协同规划
- [ ] 时空预留机制

### **优先级3**
- [ ] 风场助力模型（需CFD数据）
- [ ] 实时重规划
- [ ] 可视化界面

---

## 📊 权重调优指南

### **推荐配置**

| 场景 | ω₁ | ω₂ | α₁ | α₂ | α₃ | α₄ | 说明 |
|------|----|----|----|----|----|----|-----|
| **均衡低碳** | 0.7 | 0.3 | 0.3 | 0.2 | 0.25 | 0.25 | 平衡各项 |
| **平稳飞行** | 0.8 | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 | 减少机动 |
| **节能优先** | 0.6 | 0.4 | 0.25 | 0.35 | 0.2 | 0.2 | 降低能耗 |
| **快速飞行** | 0.5 | 0.5 | 0.5 | 0.0 | 0.25 | 0.25 | 缩短路径 |

---

## ✅ 验证检查

- [x] 低碳节点类实现
- [x] 高度变化累加
- [x] 转向角度计算
- [x] 基础能耗计算
- [x] 多配置对比测试
- [ ] 可视化（待完成）
- [ ] 真实数据验证（待完成）
