#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5完整测试 - 不依赖zhplot可视化
"""

import json
import sys
import math
import time
import os
import zhplot
import numpy as np
from evtol_path_planning_v5 import LowCarbonAStarV5, FlightProfile
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment, DynamicObstacle


def test_v5_with_map():
    """使用地图测试V5"""
    print("="*70)
    print("V5完整测试 - 国标合规路径规划")
    print("="*70)

    # 1. 加载地图
    print("\n[1/6] 加载地图...")
    env = CityEnvironment(size_x=0, size_y=0, size_z=0)
    map_data = env.load_from_json('test_map_v3_compatible.json')
    
    start_pos = tuple(map_data['start'])
    goal_pos = tuple(map_data['goal'])
    
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")
    straight_dist = np.linalg.norm(np.array(goal_pos) - np.array(start_pos))
    print(f"  直线距离: {straight_dist:.1f}m")

    # 2. 创建标准参数
    print("\n[2/6] 创建国标参数...")
    standard = EVTOLStandard()
    start_vertiport = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
    goal_vertiport = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)
    print(f"  ✓ 起降场已创建")

    # 3. 创建动态环境
    print("\n[3/6] 创建动态环境...")
    dynamic_env = DynamicEnvironment(env)
    
    # 添加风场
    wind_field = WindField(
        env, base_wind_speed=5.0, base_wind_direction=0.0,
        turbulence_intensity=0.1, time_variation=True
    )
    dynamic_env.set_wind_field(wind_field)
    print(f"  风速: {wind_field.base_wind_speed} m/s")
    
    # 添加动态障碍物
    dynamic_env.add_dynamic_obstacle(
        DynamicObstacle(
            id=1, type='drone', x=1000, y=1000, z=150,
            vx=8.0, vy=5.0, vz=0.0, size=10.0,
            valid_time_start=0.0, valid_time_end=300.0
        )
    )
    dynamic_env.add_dynamic_obstacle(
        DynamicObstacle(
            id=2, type='helicopter', x=1500, y=1500, z=120,
            vx=-4.0, vy=-3.0, vz=0.5, size=25.0,
            valid_time_start=30.0, valid_time_end=200.0
        )
    )
    print(f"  动态障碍物: {len(dynamic_env.dynamic_obstacles)} 个")

    # 4. 创建飞行剖面配置 (V5核心)
    print("\n[4/6] 配置飞行剖面...")
    flight_profile = FlightProfile(
        ofv_height=30.0,
        cruise_min=120.0,      # 降低到120m
        cruise_max=180.0,      # 降低到180m
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    print(f"  OFV高度: {flight_profile.ofv_height}m")
    print(f"  巡航层: {flight_profile.cruise_min}-{flight_profile.cruise_max}m")
    print(f"  起升阈值: {flight_profile.climb_distance_threshold}m")
    print(f"  下降阈值: {flight_profile.approach_distance_threshold}m")

    # 5. 创建规划器
    print("\n[5/6] 创建V5规划器...")
    planner = LowCarbonAStarV5(
        env, standard, start_vertiport, goal_vertiport, dynamic_env, flight_profile,
        omega1=0.7, omega2=0.3,
        alpha1=0.3, alpha2=0.2, alpha3=0.25, alpha4=0.25,
        alpha5=0.2,  # 风场权重
        alpha6=0.3,  # 国标高度约束权重
        standard_weight=0.5
    )
    print(f"  ✓ 规划器已创建")
    print(f"  权重: α5={planner.weights['alpha5']}(风场), α6={planner.weights['alpha6']}(国标)")

    # 6. 搜索路径
    print("\n[6/6] 开始路径搜索...")
    start_time = time.time()
    path = planner.find_path(max_iterations=200000, current_time=0.0)
    search_time = time.time() - start_time

    if path:
        print(f"\n{'='*70}")
        print(f"✓ 搜索成功! 用时: {search_time:.1f}s")
        print(f"{'='*70}")
        
        # 基本统计
        total_dist = path[-1].accumulated_length
        min_height = min(n.z for n in path)
        max_height = max(n.z for n in path)
        height_change = max_height - min_height
        turn_angle = path[-1].accumulated_turn_angle
        
        print(f"\n路径统计:")
        print(f"  节点数: {len(path)}")
        print(f"  总距离: {total_dist:.1f}m")
        print(f"  路径效率: {(straight_dist/total_dist)*100:.1f}%")
        print(f"  高度范围: {min_height:.1f} - {max_height:.1f}m (变化: {height_change:.1f}m)")
        print(f"  总转向角: {turn_angle:.2f} rad ({turn_angle*180/math.pi:.1f}°)")

        # 飞行剖面分析 (V5核心功能)
        print(f"\n{'='*70}")
        print(f"飞行剖面分析 (V5国标合规)")
        print(f"{'='*70}")
        profile = planner.analyze_path_profile(path)
        
        print(f"\n【起升阶段】 {profile['climb']['count']} 个节点")
        print(f"  高度: {profile['climb']['min_height']:.1f} - {profile['climb']['max_height']:.1f}m")
        print(f"  平均: {profile['climb']['avg_height']:.1f}m")
        
        print(f"\n【巡航阶段】 {profile['cruise']['count']} 个节点")
        print(f"  高度: {profile['cruise']['min_height']:.1f} - {profile['cruise']['max_height']:.1f}m")
        print(f"  平均: {profile['cruise']['avg_height']:.1f}m")
        cruise_compliance = (flight_profile.cruise_min <= profile['cruise']['avg_height'] <= 
                             flight_profile.cruise_max)
        print(f"  符合巡航层: {'✓ 是' if cruise_compliance else '✗ 否'}")
        
        print(f"\n【下降阶段】 {profile['approach']['count']} 个节点")
        print(f"  高度: {profile['approach']['min_height']:.1f} - {profile['approach']['max_height']:.1f}m")
        print(f"  平均: {profile['approach']['avg_height']:.1f}m")

        # 保存结果
        output_dir = "evtol_v5_results"
        os.makedirs(output_dir, exist_ok=True)
        
        result = {
            'test_name': 'v5_main_test',
            'start': start_pos,
            'goal': goal_pos,
            'straight_line_distance': float(straight_dist),
            'path': [(float(n.x), float(n.y), float(n.z)) for n in path],
            'stats': {
                'num_nodes': len(path),
                'total_distance': float(total_dist),
                'path_efficiency': float(straight_dist/total_dist),
                'min_height': float(min_height),
                'max_height': float(max_height),
                'height_change': float(height_change),
                'turn_angle_rad': float(turn_angle),
                'turn_angle_deg': float(turn_angle * 180 / math.pi),
                'search_time': float(search_time)
            },
            'flight_profile': profile,
            'config': {
                'cruise_layer': f"{flight_profile.cruise_min}-{flight_profile.cruise_max}m",
                'cruise_compliance': bool(cruise_compliance),
                'wind_speed': wind_field.base_wind_speed,
                'alpha5': planner.weights['alpha5'],
                'alpha6': planner.weights['alpha6']
            }
        }
        
        output_file = os.path.join(output_dir, 'v5_main_result.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存: {output_file}")

        # 路径细节
        print(f"\n路径细节 (每15个节点显示1个):")
        for i in range(0, min(60, len(path)), 15):
            n = path[i]
            dist_to_start = math.sqrt((n.x - start_pos[0])**2 + (n.y - start_pos[1])**2)
            phase = flight_profile.get_flight_phase((n.x, n.y, n.z), start_pos, goal_pos)
            print(f"  [{i:3d}] ({n.x:6.0f}, {n.y:6.0f}, {n.z:6.0f}) | "
                  f"距起点{dist_to_start:5.0f}m | {phase}")

        # 生成可视化
        print(f"\n生成V5可视化...")
        from v5_visualizer import V5Visualizer

        # 转换路径格式
        path_list = [(float(n.x), float(n.y), float(n.z)) for n in path]

        # 1. 路径与建筑物俯视图
        V5Visualizer.plot_path_with_buildings(
            env, path_list, start_pos, goal_pos, flight_profile,
            filename=os.path.join(output_dir, 'v5_path_with_buildings.png')
        )

        # 2. 3D路径剖面
        V5Visualizer.plot_3d_path_profile(
            env, path_list, start_pos, goal_pos, flight_profile,
            filename=os.path.join(output_dir, 'v5_3d_path_profile.png')
        )

        # 3. 高度剖面图
        V5Visualizer.plot_height_profile(
            path_list, start_pos, goal_pos, flight_profile,
            filename=os.path.join(output_dir, 'v5_height_profile.png')
        )

        # 4. 飞行阶段统计
        V5Visualizer.plot_phase_statistics(
            profile,
            filename=os.path.join(output_dir, 'v5_phase_statistics.png')
        )

        # 5. V4动态环境可视化 - 风场2D图
        V5Visualizer.plot_wind_field_2d(
            env, wind_field, path_list,
            filename=os.path.join(output_dir, 'v5_wind_field_2d.png'),
            grid_spacing=100, time_sec=0.0
        )

        # 6. V4动态环境可视化 - 动态障碍物3D图
        V5Visualizer.plot_dynamic_obstacles_3d(
            env, dynamic_env, path_list,
            time_sec=0.0,
            filename=os.path.join(output_dir, 'v5_dynamic_obstacles_3d.png')
        )

        # 7. V4动态环境可视化 - 风场代价热力图
        V5Visualizer.plot_wind_cost_heatmap(
            env, wind_field, path_list,
            filename=os.path.join(output_dir, 'v5_wind_cost_heatmap.png'),
            height=120, time_sec=0.0
        )

        # 8. V4动态环境可视化 - 路径与风场3D图
        V5Visualizer.plot_path_with_wind_3d(
            env, wind_field, path_list, start_pos, goal_pos,
            filename=os.path.join(output_dir, 'v5_path_with_wind_3d.png'),
            time_sec=0.0
        )

        print(f"\n✓ 所有可视化已保存到: {output_dir}/")

        print(f"\n{'='*70}")
        print(f"✓ V5测试完成!")
        print(f"{'='*70}")
        
        return True
    else:
        print(f"\n{'='*70}")
        print(f"✗ 搜索失败! 用时: {search_time:.1f}s")
        print(f"{'='*70}")
        return False


if __name__ == "__main__":
    success = test_v5_with_map()
    sys.exit(0 if success else 1)
