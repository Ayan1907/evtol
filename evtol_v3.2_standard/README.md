# eVTOL路径规划 V3.2 标准版 - 完整备份

## 📦 备份内容

### 核心代码
- `evtol_path_planning_v3_standard.py` - 主程序（1016行）
  - 集成eVTOL标准几何约束
  - 低碳A*算法实现
  - 完整的可视化功能

### 文档
- `V3标准版说明.md` - 详细技术说明文档
- `README.md` - 本文件

### 地图文件
- `test_map_v3_compatible.json` - 测试地图（2500m × 2500m × 200m，250栋建筑）

### 结果文件
- `evtol_v3_standard_results/` - 完整的可视化结果（8个文件）

## 🎯 V3.2版本特性

### 1. eVTOL标准约束集成
基于《客运eVTOL起降场技术要求（征求意见稿）》实现：
- ✅ 起降区尺寸约束（FATO & TLOF）
- ✅ 安全区和无障碍空间约束（OFV）
- ✅ 障碍物限制面约束（进近面、爬升面、过渡面）
- ✅ 机型参数标准化（亿航EH216-S）

### 2. 算法特性
- 低碳A*算法优化
- 自适应方向切换（18方向→6方向）
- eVTOL标准约束分级检查（30m/50m/100m/200m）
- 标准约束权重可调（0.0-1.0）

### 3. 可视化功能（8种）
1. **2D路径对比图** - 建筑物、TLOF、安全区、路径
2. **3D路径图**（3张）- 每个配置一张，含OFV圆柱体
3. **能耗剖面图**（3张）- 高度、能耗分量、累计能耗 ✨新增
4. **性能指标对比图** - 6维度综合分析 ✨新增

## 📊 测试结果

### 三种配置对比

| 配置 | 标准权重 | 路径长度 | 转向角 | 垂直变化 | 迭代次数 |
|------|---------|---------|--------|---------|---------|
| Standard Compliance | 0.8 | 123步 | 720° | 200m | 163,205 |
| Balanced | 0.5 | 123步 | 720° | 200m | 77,877 |
| Performance Priority | 0.3 | 125步 | 540° | 200m | 154,175 |

### 关键参数
- TLOF直径：5.63m（半径2.81m）
- FATO直径：8.45m
- 安全区半径：7.22m
- OFV高度：30m
- 进近面坡度：1:12（8.3%）
- 爬升面坡度：1:10（10%）

## 🚀 运行方式

```bash
cd /home/ayan1907/evtol/evtol_v3.2_standard
python evtol_path_planning_v3_standard.py
```

## 📁 文件结构

```
evtol_v3.2_standard/
├── evtol_path_planning_v3_standard.py    # 主程序
├── V3标准版说明.md                      # 技术文档
├── test_map_v3_compatible.json          # 测试地图
├── README.md                            # 本文件
└── evtol_v3_standard_results/           # 可视化结果
    ├── v3_standard_path_comparison_2d.png
    ├── v3_standard_path_3d_Standard Compliance.png
    ├── v3_standard_path_3d_Balanced.png
    ├── v3_standard_path_3d_Performance Priority.png
    ├── v3_standard_energy_profile_Standard Compliance.png
    ├── v3_standard_energy_profile_Balanced.png
    ├── v3_standard_energy_profile_Performance Priority.png
    └── v3_standard_metrics_comparison.png
```

## 🎓 创新点

1. **规范性**：基于官方技术标准，参数有据可依
2. **完整性**：集成完整的eVTOL约束体系
3. **可视化**：8种可视化，全面展示算法性能
4. **灵活性**：约束权重可调，适应不同场景
5. **实用性**：为实际运营提供数据支撑

## 📝 备份时间
2026-03-18

## 🔜 下一步：V4版本研发

准备开始V4版本的研发工作...
