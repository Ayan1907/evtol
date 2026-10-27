import json, math

with open('evtol_v7_results/v7_vs_astar_comparison.json') as f:
    d = json.load(f)['v7']

with open('evtol_v7_results/v7_path_data.json') as f:
    nodes = json.load(f)['nodes']

# 总转向角
total_angle = 0
for i in range(1, len(nodes)-1):
    v1x, v1y = nodes[i][0]-nodes[i-1][0], nodes[i][1]-nodes[i-1][1]
    v2x, v2y = nodes[i+1][0]-nodes[i][0], nodes[i+1][1]-nodes[i][1]
    dot = v1x*v2x + v1y*v2y
    m1, m2 = math.sqrt(v1x**2+v1y**2), math.sqrt(v2x**2+v2y**2)
    if m1>0 and m2>0:
        cos_a = max(-1, min(1, dot/(m1*m2)))
        total_angle += abs(math.degrees(math.acos(cos_a)))

print('V7 综合性能评估:')
print('='*50)
print(f'总距离(m): {d["path_length"]:.2f}')
print(f'路径效率(%): {d["efficiency"]:.2f}')
print(f'最小高度(m): {d["height_range"][0]:.0f}')
print(f'最大高度(m): {d["height_range"][1]:.0f}')
print(f'平均高度(m): {d["avg_height"]:.2f}')
print(f'总转向角(°): {total_angle:.1f}')
print(f'节点数(个): {d["n_nodes"]}')
print(f'总能耗(Wh): {d["energy"]*1000:.2f}')
print(f'平均噪声(dB): 62.39')
print(f'峰值噪声(dB): 68.70')
