#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 加载路径数据
with open('evtol_v7_results/v7_path_data.json') as f:
    data = json.load(f)

nodes = data['nodes']
x = [n[0] for n in nodes]
y = [n[1] for n in nodes]
z = [n[2] for n in nodes]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 俯视图 (X-Y)
ax1 = axes[0]
ax1.plot(x, y, 'b-o', linewidth=2, markersize=8)
ax1.plot(x[0], y[0], 'go', markersize=15, label='Start')
ax1.plot(x[-1], y[-1], 'ro', markersize=15, label='Goal')
for i, (xi, yi) in enumerate(zip(x, y)):
    ax1.text(xi, yi+50, f'{i}', fontsize=9, ha='center', va='bottom')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('V7 Path - Top View (X-Y)')
ax1.legend()
ax1.grid(True)

# 侧视图 (X-Z)
ax2 = axes[1]
ax2.plot(x, z, 'b-o', linewidth=2, markersize=8)
ax2.plot(x[0], z[0], 'go', markersize=15, label='Start')
ax2.plot(x[-1], z[-1], 'ro', markersize=15, label='Goal')
for i, (xi, zi) in enumerate(zip(x, z)):
    ax2.text(xi, zi+3, f'{i}', fontsize=9, ha='center', va='bottom')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('V7 Path - Side View (X-Z)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('evtol_v7_results/v7_path_2d.png', dpi=150, bbox_inches='tight')
print('已保存: evtol_v7_results/v7_path_2d.png')

# 打印节点信息
print('\n节点列表:')
for i, n in enumerate(nodes):
    print(f'  {i}: ({n[0]:.0f}, {n[1]:.0f}, {n[2]:.0f})')
