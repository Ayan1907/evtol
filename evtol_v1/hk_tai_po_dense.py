#!/usr/bin/env python3
"""
大埔密集区域路径规划
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
import math
import json

try:
    from zhplot import set_zh_font
    set_zh_font()
except ImportError:
    print("警告: zhplot库未安装，中文显示可能不正常")

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_blocked(x, y, z, buildings, margin=17):
    for b in buildings:
        b_min_x = b.get('local_min_x', b['min_x'])
        b_max_x = b.get('local_max_x', b['max_x'])
        b_min_y = b.get('local_min_y', b['min_y'])
        b_max_y = b.get('local_max_y', b['max_y'])
        
        if (x >= b_min_x - margin and x <= b_max_x + margin and
            y >= b_min_y - margin and y <= b_max_y + margin and
            z <= 150):
            return True
    return False

def heuristic(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def find_path(start, goal, buildings, max_iterations=200000):
    print(f"  起点: {start}")
    print(f"  终点: {goal}")
    
    if is_blocked(start[0], start[1], start[2], buildings):
        print("  错误：起点在障碍物内")
        return None
    if is_blocked(goal[0], goal[1], goal[2], buildings):
        print("  错误：终点在障碍物内")
        return None
    
    moves = [
        (20, 0, 0), (-20, 0, 0), (0, 20, 0), (0, -20, 0),
        (0, 0, 10), (0, 0, -10),
        (20, 20, 0), (20, -20, 0), (-20, 20, 0), (-20, -20, 0),
    ]
    
    open_set = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    iterations = 0
    
    while open_set and iterations < max_iterations:
        iterations += 1
        
        if iterations % 10000 == 0:
            print(f"    迭代{iterations}次，开放集大小: {len(open_set)}")
        
        current_f, current = heapq.heappop(open_set)
        
        if heuristic(current, goal) < 30:
            path = [goal]
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            
            print(f"  搜索成功！迭代{iterations}次，路径长度{len(path)}")
            return path
        
        for move in moves:
            neighbor = (
                current[0] + move[0],
                current[1] + move[1],
                current[2] + move[2]
            )
            
            if (neighbor[0] < 0 or neighbor[0] > 2500 or
                neighbor[1] < 0 or neighbor[1] > 2100 or
                neighbor[2] < 0 or neighbor[2] > 150):
                continue
            
            if is_blocked(neighbor[0], neighbor[1], neighbor[2], buildings):
                continue
            
            move_cost = math.sqrt(move[0]**2 + move[1]**2 + move[2]**2)
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    
    print(f"  搜索失败，迭代{iterations}次")
    return None

def plot_results(buildings, path, filename, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for b in buildings:
        b_min_x = b.get('local_min_x', b['min_x'])
        b_min_y = b.get('local_min_y', b['min_y'])
        
        rect = Rectangle((b_min_x, b_min_y), b['width'], b['height'],
                        facecolor='red', alpha=0.3, edgecolor='red', linewidth=0.5)
        ax.add_patch(rect)
        
        margin = 17
        margin_rect = Rectangle((b_min_x - margin, b_min_y - margin),
                               b['width'] + 2*margin, b['height'] + 2*margin,
                               facecolor='orange', alpha=0.15, edgecolor='orange', linewidth=0.5)
        ax.add_patch(margin_rect)
    
    if path:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Path', alpha=0.8)
        ax.scatter([x_coords[0]], [y_coords[0]], c='g', s=200, label='Start', zorder=5)
        ax.scatter([x_coords[-1]], [y_coords[-1]], c='r', s=200, label='Goal', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 2100)
    ax.set_aspect('equal')
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  已保存: {filename}")
    plt.close()

def main():
    print("=" * 70)
    print("大埔密集区域路径规划")
    print("=" * 70)
    
    data = load_data('hk_大埔_dense.json')
    buildings = data['buildings']
    region = data['region']
    
    print(f"区域: {region['region_name']}")
    print(f"建筑数量: {region['count']}")
    print(f"密度: {region['density']:.1f} 栋/km²")
    
    start = (50, 50, 80)
    goal = (2150, 1650, 80)
    
    print(f"\n起点: {start}")
    print(f"终点: {goal}")
    
    print("\n开始路径规划...")
    path = find_path(start, goal, buildings)
    
    if path:
        total_distance = sum(
            math.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2 + (path[i+1][2]-path[i][2])**2)
            for i in range(len(path)-1)
        )
        
        print(f"\n路径节点数: {len(path)}")
        print(f"总距离: {total_distance:.1f}m")
        
        plot_results(buildings, path, 'hk_tai_po_dense_result.png', 
                    f'大埔密集区域 - eVTOL路径规划 ({region["count"]}栋建筑)')
        
        print("\n成功！")
    else:
        print("\n路径规划失败")

if __name__ == "__main__":
    main()
