#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5.1完整测试 - 集成改进风场
"""

import json
import sys
import math
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from evtol_path_planning_v5_1 import LowCarbonAStarV51, create_v51_planner
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment, DynamicObstacle
from improved_wind_field import ImprovedWindField


def test_v51_basic():
    """V5.1基础测试"""
    print("=" * 70)
    print("V5.1基础测试 - 改进风场")
    print("=" * 70)
    
    # 测试地图
    test_map = 'test_map_v3_compatible.json'
    
    if not os.path.exists(test_map):
        print(f"地图文件不存在: {test_map}")
        return None
    
    # 创建V5.1规划器
    planner = create_v51_planner(
        test_map,
        use_improved_wind=True,
        wind_speed=8.0,
        wind_direction=315.0,
        cruise_min=120.0,
        cruise_max=180.0,
        omega1=0.7, omega2=0.3,
        alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25,
        alpha5=0.25,  # 风场权重
        alpha6=0.3,   # 国标高度约束权重
        standard_weight=0.5
    )
    
    if not planner:
        print("创建规划器失败")
        return None
    
    # 打印风场信息
    planner.print_wind_field_info()
    
    # 运行规划
    print(f"\n开始路径规划...")
    start_time = time.time()
    path = planner.find_path(max_iterations=200000, current_time=0.0)
    planning_time = time.time() - start_time
    
    if path:
        print(f"\n规划成功!")
        print(f"  路径长度: {len(path)} 个节点")
        print(f"  规划时间: {planning_time:.2f}s")
        
        # 统计信息
        if len(path) > 1:
            total_dist = 0
            total_energy = 0
            min_z = float('inf')
            max_z = float('-inf')
            
            for i in range(len(path) - 1):
                dist = math.sqrt(
                    (path[i+1].x - path[i].x)**2 +
                    (path[i+1].y - path[i].y)**2 +
                    (path[i+1].z - path[i].z)**2
                )
                total_dist += dist
                total_energy += path[i+1].accumulated_energy
                min_z = min(min_z, path[i].z, path[i+1].z)
                max_z = max(max_z, path[i].z, path[i+1].z)
            
            print(f"  总距离: {total_dist:.1f}m")
            print(f"  总能耗: {total_energy:.2f}")
            print(f"  高度范围: {min_z:.1f}-{max_z:.1f}m")
        
        return path
    else:
        print(f"\n规划失败")
        return None


def test_v51_wind_comparison():
    """V5.1风场对比测试 - 改进风场 vs 标准风场"""
    print("\n" + "=" * 70)
    print("V5.1风场对比测试")
    print("=" * 70)
    
    test_map = 'test_map_v3_compatible.json'
    
    if not os.path.exists(test_map):
        print(f"地图文件不存在: {test_map}")
        return
    
    results = {}
    
    # 测试1: 改进风场
    print("\n[1/2] 测试改进风场...")
    planner_improved = create_v51_planner(
        test_map,
        use_improved_wind=True,
        wind_speed=8.0,
        wind_direction=315.0,
        cruise_min=120.0,
        cruise_max=180.0,
        alpha5=0.25,
        alpha6=0.3
    )
    
    if planner_improved:
        start_time = time.time()
        path_improved = planner_improved.find_path(max_iterations=200000, current_time=0.0)
        time_improved = time.time() - start_time
        
        if path_improved:
            results['improved'] = {
                'path': path_improved,
                'time': time_improved,
                'nodes': len(path_improved)
            }
            print(f"  改进风场规划成功: {len(path_improved)}节点, {time_improved:.2f}s")
        else:
            print(f"  改进风场规划失败")
    
    # 测试2: 标准风场
    print("\n[2/2] 测试标准风场...")
    planner_standard = create_v51_planner(
        test_map,
        use_improved_wind=False,
        wind_speed=8.0,
        wind_direction=315.0,
        cruise_min=120.0,
        cruise_max=180.0,
        alpha5=0.25,
        alpha6=0.3
    )
    
    if planner_standard:
        start_time = time.time()
        path_standard = planner_standard.find_path(max_iterations=200000, current_time=0.0)
        time_standard = time.time() - start_time
        
        if path_standard:
            results['standard'] = {
                'path': path_standard,
                'time': time_standard,
                'nodes': len(path_standard)
            }
            print(f"  标准风场规划成功: {len(path_standard)}节点, {time_standard:.2f}s")
        else:
            print(f"  标准风场规划失败")
    
    # 对比分析
    print("\n" + "=" * 70)
    print("对比分析")
    print("=" * 70)
    
    if 'improved' in results and 'standard' in results:
        print(f"\n路径节点数:")
        print(f"  改进风场: {results['improved']['nodes']}")
        print(f"  标准风场: {results['standard']['nodes']}")
        print(f"  差异: {results['improved']['nodes'] - results['standard']['nodes']:+d}")
        
        print(f"\n规划时间:")
        print(f"  改进风场: {results['improved']['time']:.2f}s")
        print(f"  标准风场: {results['standard']['time']:.2f}s")
        print(f"  比例: {results['improved']['time'] / results['standard']['time']:.2f}x")
        
        # 计算路径指标
        def calculate_path_metrics(path):
            if len(path) < 2:
                return 0, 0
            
            total_dist = 0
            total_energy = 0
            total_turn = 0
            
            for i in range(len(path) - 1):
                dist = math.sqrt(
                    (path[i+1].x - path[i].x)**2 +
                    (path[i+1].y - path[i].y)**2 +
                    (path[i+1].z - path[i].z)**2
                )
                total_dist += dist
                total_energy += path[i+1].accumulated_energy
                
                if i < len(path) - 2:
                    # 计算转向角
                    v1 = np.array([path[i+1].x - path[i].x,
                                   path[i+1].y - path[i].y,
                                   path[i+1].z - path[i].z])
                    v2 = np.array([path[i+2].x - path[i+1].x,
                                   path[i+2].y - path[i+1].y,
                                   path[i+2].z - path[i+1].z])
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        turn = math.degrees(math.acos(cos_angle))
                        total_turn += turn
            
            avg_turn = total_turn / (len(path) - 2) if len(path) > 2 else 0
            return total_dist, total_energy, avg_turn
        
        dist_imp, energy_imp, turn_imp = calculate_path_metrics(results['improved']['path'])
        dist_std, energy_std, turn_std = calculate_path_metrics(results['standard']['path'])
        
        print(f"\n路径距离:")
        print(f"  改进风场: {dist_imp:.1f}m")
        print(f"  标准风场: {dist_std:.1f}m")
        print(f"  差异: {dist_imp - dist_std:+.1f}m ({(dist_imp/dist_std-1)*100:+.1f}%)")
        
        print(f"\n累计能耗指标:")
        print(f"  改进风场: {energy_imp:.2f}")
        print(f"  标准风场: {energy_std:.2f}")
        if energy_std > 0:
            print(f"  差异: {energy_imp - energy_std:+.2f} ({(energy_imp/energy_std-1)*100:+.1f}%)")
        else:
            print(f"  差异: {energy_imp - energy_std:+.2f}")
        
        print(f"\n平均转向角:")
        print(f"  改进风场: {turn_imp:.1f}°")
        print(f"  标准风场: {turn_std:.1f}°")
        print(f"  差异: {turn_imp - turn_std:+.1f}°")


def test_v51_multiple_maps():
    """V5.1多地图测试"""
    print("\n" + "=" * 70)
    print("V5.1多地图测试")
    print("=" * 70)
    
    maps = ['map_standard.json', 'map_dense.json', 'map_sparse.json', 'map_zigzag.json']
    results = {}
    
    for map_file in maps:
        if not os.path.exists(map_file):
            print(f"\n跳过不存在的地图: {map_file}")
            continue
        
        print(f"\n测试地图: {map_file}")
        
        planner = create_v51_planner(
            map_file,
            use_improved_wind=True,
            wind_speed=8.0,
            wind_direction=315.0,
            cruise_min=120.0,
            cruise_max=180.0,
            alpha5=0.25,
            alpha6=0.3
        )
        
        if planner:
            start_time = time.time()
            path = planner.find_path(max_iterations=200000, current_time=0.0)
            planning_time = time.time() - start_time
            
            if path:
                # 计算指标
                total_dist = 0
                total_energy = 0
                for i in range(len(path) - 1):
                    dist = math.sqrt(
                        (path[i+1].x - path[i].x)**2 +
                        (path[i+1].y - path[i].y)**2 +
                        (path[i+1].z - path[i].z)**2
                    )
                    total_dist += dist
                    total_energy += path[i+1].accumulated_energy
                
                results[map_file] = {
                    'success': True,
                    'nodes': len(path),
                    'distance': total_dist,
                    'energy': total_energy,
                    'time': planning_time
                }
                
                print(f"  ✓ 成功: {len(path)}节点, {total_dist:.1f}m, {planning_time:.2f}s")
            else:
                results[map_file] = {
                    'success': False,
                    'nodes': 0,
                    'distance': 0,
                    'energy': 0,
                    'time': planning_time
                }
                print(f"  ✗ 失败: {planning_time:.2f}s")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    if total_count > 0:
        print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)")
    else:
        print(f"\n成功数: {success_count}/{total_count}")
    
    if success_count > 0:
        avg_nodes = np.mean([r['nodes'] for r in results.values() if r['success']])
        avg_dist = np.mean([r['distance'] for r in results.values() if r['success']])
        avg_time = np.mean([r['time'] for r in results.values() if r['success']])
        
        print(f"平均节点数: {avg_nodes:.0f}")
        print(f"平均距离: {avg_dist:.1f}m")
        print(f"平均时间: {avg_time:.2f}s")


def main():
    """主函数"""
    print("=" * 70)
    print("V5.1完整测试套件")
    print("=" * 70)
    
    # 1. 基础测试
    print("\n\n[测试1/3] 基础测试")
    path = test_v51_basic()
    
    # 2. 风场对比测试
    print("\n\n[测试2/3] 风场对比测试")
    test_v51_wind_comparison()
    
    # 3. 多地图测试
    print("\n\n[测试3/3] 多地图测试")
    test_v51_multiple_maps()
    
    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
