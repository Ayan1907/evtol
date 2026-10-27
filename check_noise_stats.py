#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查路径噪声统计数据"""
import json
import numpy as np
from evtol_path_planning_v6 import CityEnvironment, FlightProfile
from noise_model import NoiseModel

# 加载环境
env = CityEnvironment(size_x=0, size_y=0, size_z=0)
env.load_from_json('test_map_v3_compatible.json')

# 加载路径
with open('evtol_v6_results/v6_main_result.json', 'r') as f:
    result_data = json.load(f)
path = result_data['path']

# 创建噪声模型
noise_model = NoiseModel(env, source_level=85.0, reference_distance=100.0)

# 计算路径上每个点的噪声级
noise_levels = []
heights = []
for point in path:
    x, y, z = point
    noise_level = noise_model.calculate_noise_level(x, y, z, speed=20.0)
    noise_levels.append(noise_level)
    heights.append(z)

noise_levels = np.array(noise_levels)
print('沿路径噪声统计:')
print(f'  平均噪声: {np.mean(noise_levels):.2f} dB')
print(f'  峰值噪声: {np.max(noise_levels):.2f} dB')
print(f'  最小噪声: {np.min(noise_levels):.2f} dB')
print(f'  噪声范围: [{np.min(noise_levels):.2f}, {np.max(noise_levels):.2f}] dB')
print(f'  超过60dB的点: {np.sum(noise_levels > 60)}')
print(f'  超过75dB的点: {np.sum(noise_levels > 75)}')
print()
print('高度统计:')
print(f'  平均高度: {np.mean(heights):.2f} m')
print(f'  最大高度: {np.max(heights):.2f} m')
print(f'  最小高度: {np.min(heights):.2f} m')
