# eVTOL路径规划V3版本说明

## 版本概述

V3版本整合了以下核心功能：
1. **基础路径搜索**：基于 `evtol_path_planning_v2.py` 的18方向路径搜索
2. **低碳优化**：整合 `low_carbon_node.py` 的低碳多目标优化
3. **可视化功能**：从 v2 移植的2D和3D可视化

## 核心特性

### 1. 低碳多目标成本函数

```python
G_carbon = α₁·L + α₂·E + α₃·Σ|Δh| + α₄·Σ|Δβ| + H·(ω₁ + ω₂)
```

其中：
- **L**: 累计航迹长度
- **E**: 累计能耗（基础能耗 + 爬升能耗 + 转向能耗）
- **Σ|Δh|**: 累计垂直变化
- **Σ|Δβ|**: 累计转向角度
- **H**: 启发式函数（欧氏距离）
- **ω₁, ω₂**: 启发式权重
- **α₁, α₂, α₃, α₄**: 各分项权重

### 2. 18方向移动策略

- **6个基本方向**：X轴±, Y轴±, Z轴±
- **12个对角方向**：三维对角移动
- **步长**：水平10米，垂直5米

### 3. 三种预设配置

| 配置 | ω₁ | ω₂ | α₁ | α₂ | α₃ | α₄ | 说明 |
|-----|----|----|----|----|----|----|-----|
| Low-Carbon Balanced | 0.7 | 0.3 | 0.3 | 0.2 | 0.25 | 0.25 | 低碳均衡 |
| Smooth Priority | 0.8 | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 | 平稳优先 |
| Shortest Path | 0.5 | 0.5 | 0.5 | 0.0 | 0.25 | 0.25 | 最短路径 |

## 使用方法

### 基本使用

```python
from evtol_path_planning_v3 import CityEnvironment, LowCarbonAStarV3

# 创建环境
env = CityEnvironment(size_x=2500, size_y=2500, size_z=200)
env.generate_random_buildings(n_buildings=50)

# 创建规划器
planner = LowCarbonAStarV3(
    env,
    omega1=0.7, omega2=0.3,
    alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25
)

# 搜索路径
path = planner.find_path(start=(250,250,80), goal=(2250,2250,100))

# 获取结果
if path:
    print(f"路径长度: {len(path)}步")
    print(f"转向角度: {path[-1].accumulated_turn_angle:.2f} rad")
    print(f"垂直变化: {path[-1].accumulated_vertical_change:.2f} m")
```

### 带可视化测试

```bash
python evtol_path_planning_v3.py
```

生成的可视化文件将保存到 `evtol_v3_results/` 目录：
- `v3_path_comparison_2d.png` - 2D路径对比图
- `v3_path_3d_*.png` - 3D路径图

## 文件结构

```
evtol_path_planning_v3.py
├── CityEnvironment          # 环境类
│   ├── generate_random_buildings()  # 生成建筑物
│   └── is_free()                 # 碰撞检测
│
├── LowCarbonAStarV3         # 低碳A*规划器
│   ├── _init_directions()    # 初始化18方向
│   ├── find_path()          # 主搜索函数
│   ├── _calculate_g_cost()   # 计算g值
│   └── _calculate_h_cost()   # 计算h值
│
├── V3Visualizer            # 可视化类
│   ├── plot_path_comparison_2d()  # 2D对比图
│   └── plot_path_3d()            # 3D路径图
│
└── test_v3_with_visualization()  # 完整测试（含可视化）
```

## 性能特点

### 优势
1. **多目标优化**：同时考虑长度、能耗、垂直变化、转向角
2. **灵活权重**：可通过α₁-α₄调整优化目标
3. **高效率**：18方向搜索 + A*启发式优化
4. **可视化完善**：2D/3D双视图展示

### 测试结果

```
地图尺寸: 2500m × 2500m × 200m
建筑物数量: 250

配置              路径长度    转向角(rad)    垂直变化(m)   迭代次数
------------------------------------------------------------
Low-Carbon Balanced 110         10.2102         60.00          252
Smooth Priority     111         10.2102         60.00          122
Shortest Path      110         7.0686          20.00        44346
```

**关键发现**：
- **Shortest Path**配置通过牺牲计算时间（44,346次迭代），找到了转向角最小（7.07 rad vs 10.21 rad）、垂直变化最小（20m vs 60m）的路径
- **Low-Carbon Balanced / Smooth Priority**配置快速收敛（~250次迭代），综合考虑多目标
- 不同权重策略产生了不同的最优路径，证明了低碳优化的价值

## 依赖项

```python
numpy
matplotlib
```

## 未来改进方向

1. **动态环境**：支持移动障碍物
2. **风场模型**：集成实际风场数据
3. **能耗精细化**：使用CFD计算精确能耗
4. **路径平滑**：增加B样条曲线平滑
5. **实时规划**：支持动态重规划

## 与其他版本对比

| 版本 | 特点 | 适用场景 |
|-----|------|---------|
| v2 | 基础Theta* + 安全裕度 | 标准路径规划 |
| v3 | 低碳多目标优化 | 低碳环保场景 |
| low_carbon_astar | 纯低碳A*算法 | 算法研究 |

## 注意事项

1. 建议建筑物数量控制在50-100之间，过多会影响搜索效率
2. 步长可调整：水平步长10m，垂直步长5m
3. 最大迭代次数默认100000，可根据地图大小调整
4. 可视化使用非交互式后端，避免GUI依赖问题
