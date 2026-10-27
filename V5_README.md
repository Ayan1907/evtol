# V5版本快速开始指南

## 📦 安装与运行

### 1. 环境要求
```bash
Python 3.8+
numpy >= 1.19.0
matplotlib >= 3.3.0
```

### 2. 运行完整测试

```bash
cd /home/ayan1907/evtol
python test_v5.py
```

### 3. 生成可视化

```bash
python v5_visualizer.py evtol_v5_results/v5_path_data.json
```

---

## 📂 文件说明

### 核心模块
- `evtol_aircraft_model.py` - 飞行器动力学与能耗模型
- `evtol_standard_constraints.py` - GB 38031-2022标准约束
- `evtol_path_planning_v5.py` - V5路径规划器

### 工具
- `test_v5.py` - 测试套件
- `v5_visualizer.py` - 可视化工具

### 文档
- `V5版本说明.md` - 详细版本说明
- `V5_README.md` - 本文件(快速开始)

### 结果
- `evtol_v5_results/` - 测试结果目录

---

## 🔬 快速测试

### 测试1: 基础路径规划
```python
from evtol_path_planning_v5 import V5PathPlanner, load_buildings_from_json

buildings = load_buildings_from_json("hk_大埔_buildings.json")
planner = V5PathPlanner(
    start=(100, 100, 80),
    goal=(400, 400, 80),
    buildings=buildings
)
path = planner.find_path()
```

### 测试2: 能耗优化
```python
from evtol_path_planning_v5 import PathPlanningConfig

config = PathPlanningConfig(
    alpha4=0.5,  # 高能耗权重
    alpha3=0.2   # 低安全权重
)
planner = V5PathPlanner(start, goal, buildings, config)
```

---

## 📊 输出结果

测试完成后,`evtol_v5_results/`目录将包含:
- `v5_path_data.json` - 路径数据
- `v5_path_3d.png` - 3D路径图
- `v5_energy_profile.png` - 能耗剖面图
- `v5_path_profile.png` - 路径剖面图

---

## ⚡ 性能优化建议

1. 降低网格分辨率: `grid_resolution=20.0`
2. 减少最大迭代: `max_iterations=50000`
3. 使用硬约束: `strict_constraints=True`

---

## 📞 需要帮助?

查看详细文档: `V5版本说明.md`
