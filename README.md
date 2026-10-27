# 香港 eVTOL 路径规划项目

## 项目概述
本项目为香港不同密度区域的 eVTOL (电动垂直起降飞行器) 路径规划研究，包含五个不同建筑密度的区域数据。

## 参数设置
- 区域大小: 2500×2100 米
- 安全裕度: 17 米
- 飞行高度: 80-100 米
- 路径算法: A* 搜索（连续坐标）

## 保留文件列表

### 核心代码文件
- `evtol_path_planning_original.py` - 原始路径规划代码
- `evtol_path_planning_v2.py` - v2 版本路径规划代码
- `hk_simple_path.py` - 简化版路径规划代码

### 五个区域数据文件 (JSON)
| 文件 | 区域 | 建筑数量 | 密度 |
|------|------|---------|------|
| `hk_target_2500.json` | 大埔（稀疏） | 155 | 24.8 栋/km² |
| `hk_大埔_dense.json` | 大埔（密集） | 489 | 78.2 栋/km² |
| `hk_大埔_buildings.json` | 大埔（完整） | 10013 | 155.8 栋/km² |
| `hk_沙田_buildings.json` | 沙田 | 6204 | 121.7 栋/km² |
| `hk_荃湾_buildings.json` | 荃湾 | 7093 | 118.0 栋/km² |

### 可视化结果 (PNG)
- `hk_tai_po_sparse_result.png` - 大埔稀疏区域路径图
- `hk_tai_po_dense_result.png` - 大埔密集区域路径图
- `hk_tai_po_full_result.png` - 大埔完整区域路径图
- `hk_sha_tin_result.png` - 沙田区域路径图
- `hk_tsuen_wan_result.png` - 荃湾区域路径图

### 数据摘要文件
- `hk_all_regions_summary.json` - 所有区域的完整数据摘要（含坐标系统、建筑样本等）
- `hk_regions_comparison.json` - 区域对比数据（简化版）

## 数据结构说明

### 区域数据文件 (JSON格式)
每个 JSON 文件包含:
- `region`: 区域信息（名称、大小、建筑数量、密度、UTM坐标范围）
- `buildings`: 建筑物列表，每栋建筑包含:
  - `center_x`, `center_y`: 中心坐标 (UTM)
  - `min_x`, `max_x`, `min_y`, `max_y`: 边界坐标 (UTM)
  - `width`, `height`: 建筑物尺寸 (米)
  - `tags`: OpenStreetMap 标签信息

### 坐标系统
- UTM坐标系统（香港）
- 需要通过 `region.min_x` 和 `region.min_y` 转换为局部坐标（0-2500米范围）

## 图片说明
- 红色矩形: 建筑物
- 橙色区域: 安全裕度区域（17米）
- 蓝色线条: eVTOL 飞行路径
- 绿色圆点: 起点
- 红色圆点: 终点

## 使用方法
```bash
# 查看数据摘要
cat hk_regions_comparison.json

# 查看完整数据
cat hk_all_regions_summary.json

# 查看图片
# （使用图片查看器打开对应的 PNG 文件）
```

## 备注
- 所有数据来源于 OpenStreetMap
- 日期: 2026-03-14
- 项目名称: Hong Kong eVTOL Path Planning
