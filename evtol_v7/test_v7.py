#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7版本完整测试脚本
测试内容:
1. 标准路径规划
2. 多目标优化（HV固定参考点修复验证）
3. 不确定性分析
4. 可视化输出
"""

import sys
import os
import json
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evtol_path_planning_v7 import EVTOLV7, FlightProfile
from evtol_path_planning_v3_clean import load_environment_from_json
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport
from evtol_path_planning_v4 import DynamicEnvironment, WindField, DynamicObstacle
from v7_visualizer import V7Visualizer, visualize_optimization_results


def create_test_environment():
    """创建测试环境"""
    print("="*60)
    print("【V7】创建测试环境")
    print("="*60)
    
    # 加载地图
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_file = os.path.join(project_root, 'data/test_maps/test_map_standard.json')
    if not os.path.exists(map_file):
        map_file = os.path.join(project_root, 'data/test_maps/test_map_v3_compatible.json')
    
    print(f"  加载地图: {map_file}")
    env, start_coords, goal_coords, metadata = load_environment_from_json(map_file)
    
    # 创建Vertiport对象
    standard = EVTOLStandard()
    start_vp = Vertiport(x=start_coords[0], y=start_coords[1], z=start_coords[2], standard=standard)
    goal_vp = Vertiport(x=goal_coords[0], y=goal_coords[1], z=goal_coords[2], standard=standard)
    
    # 创建动态环境
    wind_field = WindField(
        env=env,
        base_wind_speed=5.0,
        base_wind_direction=315.0,
        turbulence_intensity=0.1
    )
    dynamic_env = DynamicEnvironment(env)
    dynamic_env.wind_field = wind_field
    
    # 添加动态障碍物
    drone = DynamicObstacle(
        id=1, type='drone', x=1200, y=1200, z=150,
        vx=10, vy=5, vz=0, size=50,
        valid_time_start=0.0, valid_time_end=999999.0, predictability=1.0
    )
    helicopter = DynamicObstacle(
        id=2, type='helicopter', x=1500, y=800, z=120,
        vx=-8, vy=3, vz=0, size=40,
        valid_time_start=0.0, valid_time_end=999999.0, predictability=1.0
    )
    dynamic_env.add_dynamic_obstacle(drone)
    dynamic_env.add_dynamic_obstacle(helicopter)
    
    print(f"  地图尺寸: {env.size_x} x {env.size_y}")
    print(f"  建筑物数量: {len(env.buildings)}")
    print(f"  动态障碍物数量: {len(dynamic_env.dynamic_obstacles)}")
    
    return env, standard, start_vp, goal_vp, dynamic_env


def test_standard_planning(planner, output_dir='results/v7'):
    """测试1: 标准路径规划"""
    print("\n" + "="*60)
    print("【V7测试1】标准路径规划")
    print("="*60)
    
    start_time = time.time()
    path = planner.plan(max_iterations=200000, current_time=0.0)
    elapsed = time.time() - start_time
    
    if path and len(path) >= 2:
        print(f"  ✓ 路径规划成功!")
        print(f"    路径节点数: {len(path)}")
        print(f"    规划时间: {elapsed:.2f}s")
        
        # 计算路径统计
        coords = [(p.x, p.y, p.z) if hasattr(p, 'x') else p for p in path]
        total_length = 0
        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            dz = coords[i+1][2] - coords[i][2]
            total_length += np.sqrt(dx**2 + dy**2 + dz**2)
        
        heights = [c[2] for c in coords]
        print(f"    总距离: {total_length:.1f}m")
        print(f"    高度范围: {min(heights):.0f} - {max(heights):.0f}m")
        
        # 可视化
        vis = V7Visualizer()
        start_pos = planner.start_vertiport.get_TLOF_center()
        goal_pos = planner.goal_vertiport.get_TLOF_center()
        
        vis.plot_path_with_buildings(
            planner.env, coords, start_pos, goal_pos,
            planner.flight_profile,
            filename=f'{output_dir}/v7_path_with_buildings.png'
        )
        vis.plot_3d_path_profile(
            planner.env, coords, start_pos, goal_pos,
            planner.flight_profile,
            filename=f'{output_dir}/v7_3d_path_profile.png'
        )
        vis.plot_height_profile(
            coords, start_pos, goal_pos,
            planner.flight_profile,
            filename=f'{output_dir}/v7_height_profile.png'
        )
        
        return path, coords
    else:
        print(f"  ✗ 路径规划失败!")
        return None, None


def test_multi_objective_optimization(planner, n_generations=30, population_size=30, output_dir='results/v7'):
    """测试2: 多目标优化（验证HV固定参考点修复）"""
    print("\n" + "="*60)
    print("【V7测试2】多目标优化 (HV固定参考点修复验证)")
    print("="*60)
    
    start_time = time.time()
    pareto_front = planner.optimize_multi_objective(
        n_generations=n_generations,
        population_size=population_size,
        enable_adaptive=True
    )
    elapsed = time.time() - start_time
    
    print(f"\n  优化完成! 用时: {elapsed:.2f}s")
    print(f"  帕累托解数量: {len(pareto_front.individuals)}")
    print(f"  超体积(HV): {pareto_front.hypervolume:.2f}")
    print(f"  间距指标: {pareto_front.spacing:.4f}")
    
    # 获取固定参考点
    reference_point = planner.evaluator.get_fixed_reference_point()
    print(f"\n  【V7修复】固定参考点:")
    print(f"    能耗: {reference_point[0]:.3f} kWh")
    print(f"    时间: {reference_point[1]:.1f} s")
    print(f"    噪声: {reference_point[2]:.4f}")
    print(f"    安全: {reference_point[3]:.4f}")
    print(f"    路径: {reference_point[4]:.1f} m")
    
    # 选择最优解
    best = None
    for ind in pareto_front.individuals:
        if hasattr(ind, 'objectives') and ind.objectives is not None and len(ind.objectives) > 0:
            best = ind
            break
    
    if best is None and pareto_front.individuals:
        best = pareto_front.individuals[0]
    
    if best and hasattr(best, 'objectives') and best.objectives is not None and len(best.objectives) > 0:
        print(f"\n  最优解:")
        print(f"    能耗: {best.objectives[0]:.3f} kWh")
        print(f"    时间: {best.objectives[1]:.1f} s")
        print(f"    噪声: {best.objectives[2]:.4f}")
        print(f"    安全: {best.objectives[3]:.4f}")
        print(f"    路径: {best.objectives[4]:.1f} m")
        
        # 目标值雷达图
        vis = V7Visualizer()
        vis.plot_objective_radar({
            'energy': best.objectives[0],
            'time': best.objectives[1],
            'noise': best.objectives[2],
            'safety': best.objectives[3],
            'length': best.objectives[4]
        }, filename=f'{output_dir}/v7_best_solution_radar.png')
    else:
        print("\n  (帕累托解目标值未完整计算，跳过详细信息)")
    
    # 可视化
    vis = V7Visualizer()
    vis.plot_pareto_front_2d(
        pareto_front,
        filename=f'{output_dir}/v7_pareto_front_2d.png'
    )
    vis.plot_pareto_front_3d(
        pareto_front,
        filename=f'{output_dir}/v7_pareto_front_3d.png'
    )
    
    return pareto_front, reference_point


def test_uncertainty_analysis(planner, n_samples=200, output_dir='results/v7'):
    """测试3: 不确定性分析"""
    print("\n" + "="*60)
    print("【V7测试3】不确定性分析 (蒙特卡洛方法)")
    print("="*60)
    
    start_time = time.time()
    results = planner.analyze_uncertainty(n_samples=n_samples)
    elapsed = time.time() - start_time
    
    print(f"\n  分析完成! 用时: {elapsed:.2f}s")
    print(f"  成功采样: {results['n_successful']}/{n_samples}")
    
    print(f"\n  能耗统计:")
    print(f"    均值: {results['energy']['mean']:.3f} kWh")
    print(f"    标准差: {results['energy']['std']:.3f} kWh")
    print(f"    范围: [{results['energy']['min']:.3f}, {results['energy']['max']:.3f}]")
    print(f"    5-95分位: [{results['energy']['p5']:.3f}, {results['energy']['p95']:.3f}]")
    
    print(f"\n  时间统计:")
    print(f"    均值: {results['time']['mean']:.2f} s")
    print(f"    标准差: {results['time']['std']:.2f} s")
    
    print(f"\n  风险评估:")
    print(f"    能耗超标概率: {results['risk_assessment']['energy_exceed_prob']*100:.1f}%")
    print(f"    时间超标概率: {results['risk_assessment']['time_exceed_prob']*100:.1f}%")
    print(f"    能耗变异系数: {results['risk_assessment']['energy_coefficient_variation']*100:.1f}%")
    
    return results


def save_detailed_results(pareto_front, planner, output_dir, uncertainty_results, reference_point):
    """保存详细的优化结果数据"""
    print("\n" + "="*60)
    print("【V7】保存详细结果数据")
    print("="*60)
    
    # 提取个体目标值的辅助函数
    def extract_objectives(ind):
        if hasattr(ind, 'objectives') and ind.objectives is not None and len(ind.objectives) >= 5:
            return {
                'energy': float(ind.objectives[0]),
                'time': float(ind.objectives[1]),
                'noise': float(ind.objectives[2]),
                'safety': float(ind.objectives[3]),
                'length': float(ind.objectives[4])
            }
        return None
    
    # 提取路径坐标的辅助函数（从Individual的genes字段）
    def extract_path_coords(ind):
        """从个体的genes字段提取路径坐标"""
        path_data = {'nodes': [], 'segments': [], 'total_length': 0}
        
        # Individual使用genes存储路径
        genes = getattr(ind, 'genes', None)
        if genes is not None and len(genes) >= 2:
            coords = []
            for p in genes:
                if hasattr(p, 'x'):
                    coord = {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)}
                else:
                    coord = {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
                coords.append(coord)
                path_data['nodes'].append(coord)
            
            # 计算分段和总长度
            total_length = 0
            for i in range(len(coords) - 1):
                dx = coords[i+1]['x'] - coords[i]['x']
                dy = coords[i+1]['y'] - coords[i]['y']
                dz = coords[i+1]['z'] - coords[i]['z']
                seg_length = np.sqrt(dx**2 + dy**2 + dz**2)
                path_data['segments'].append({
                    'start_idx': i,
                    'end_idx': i + 1,
                    'length': float(seg_length),
                    'dx': float(dx),
                    'dy': float(dy),
                    'dz': float(dz)
                })
                total_length += seg_length
            path_data['total_length'] = float(total_length)
            
            # 高度统计
            heights = [c['z'] for c in coords]
            path_data['height_stats'] = {
                'min': float(min(heights)),
                'max': float(max(heights)),
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)) if len(heights) > 1 else 0.0
            }
        
        return path_data
    
    # 1. 保存详细帕累托解数据（包含路径）
    solutions_detailed = []
    for i, ind in enumerate(pareto_front.individuals):
        obj = extract_objectives(ind)
        if obj is not None:
            solution = {
                'id': i,
                'objectives': obj,
                'fitness': float(ind.fitness) if hasattr(ind, 'fitness') and ind.fitness is not None else None,
                'rank': int(ind.rank) if hasattr(ind, 'rank') and ind.rank is not None else None,
                'crowding_distance': float(ind.crowding_distance) if hasattr(ind, 'crowding_distance') and ind.crowding_distance is not None else None
            }
            
            # 添加路径数据（如果可用）
            path_data = extract_path_coords(ind)
            if path_data['nodes']:
                solution['path'] = path_data
            else:
                solution['path'] = None
            
            solutions_detailed.append(solution)
    
    # 按能耗排序
    solutions_detailed.sort(key=lambda x: x['objectives']['energy'])
    
    pareto_detailed_data = {
        'metadata': {
            'version': 'V7',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_solutions': len(solutions_detailed),
            'hypervolume': float(pareto_front.hypervolume),
            'spacing': float(pareto_front.spacing),
            'reference_point': reference_point.tolist() if isinstance(reference_point, np.ndarray) else list(reference_point)
        },
        'objective_ranges': {
            'energy': {
                'min': min(s['objectives']['energy'] for s in solutions_detailed),
                'max': max(s['objectives']['energy'] for s in solutions_detailed)
            },
            'time': {
                'min': min(s['objectives']['time'] for s in solutions_detailed),
                'max': max(s['objectives']['time'] for s in solutions_detailed)
            },
            'noise': {
                'min': min(s['objectives']['noise'] for s in solutions_detailed),
                'max': max(s['objectives']['noise'] for s in solutions_detailed)
            },
            'safety': {
                'min': min(s['objectives']['safety'] for s in solutions_detailed),
                'max': max(s['objectives']['safety'] for s in solutions_detailed)
            },
            'length': {
                'min': min(s['objectives']['length'] for s in solutions_detailed),
                'max': max(s['objectives']['length'] for s in solutions_detailed)
            }
        },
        'solutions': solutions_detailed
    }
    
    with open(f'{output_dir}/v7_pareto_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(pareto_detailed_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 详细帕累托解数据已保存: {output_dir}/v7_pareto_detailed.json")
    print(f"    - 解数量: {len(solutions_detailed)}")
    
    # 2. 保存完整结果摘要
    # 找到各维度最优解
    best_energy = min(solutions_detailed, key=lambda x: x['objectives']['energy'])
    best_time = min(solutions_detailed, key=lambda x: x['objectives']['time'])
    best_noise = min(solutions_detailed, key=lambda x: x['objectives']['noise'])
    best_safety = min(solutions_detailed, key=lambda x: x['objectives']['safety'])
    best_length = min(solutions_detailed, key=lambda x: x['objectives']['length'])
    
    results_summary = {
        'version': 'V7',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'optimization': {
            'n_solutions': len(solutions_detailed),
            'hypervolume': float(pareto_front.hypervolume),
            'spacing': float(pareto_front.spacing),
            'reference_point': reference_point.tolist() if isinstance(reference_point, np.ndarray) else list(reference_point)
        },
        'best_by_objective': {
            'best_energy': {
                'id': best_energy['id'],
                'objectives': best_energy['objectives'],
                'path_length_nodes': len(best_energy['path']['nodes']) if best_energy.get('path') else None
            },
            'best_time': {
                'id': best_time['id'],
                'objectives': best_time['objectives'],
                'path_length_nodes': len(best_time['path']['nodes']) if best_time.get('path') else None
            },
            'best_noise': {
                'id': best_noise['id'],
                'objectives': best_noise['objectives'],
                'path_length_nodes': len(best_noise['path']['nodes']) if best_noise.get('path') else None
            },
            'best_safety': {
                'id': best_safety['id'],
                'objectives': best_safety['objectives'],
                'path_length_nodes': len(best_safety['path']['nodes']) if best_safety.get('path') else None
            },
            'best_length': {
                'id': best_length['id'],
                'objectives': best_length['objectives'],
                'path_length_nodes': len(best_length['path']['nodes']) if best_length.get('path') else None
            }
        },
        'uncertainty': {
            'energy': uncertainty_results['energy'],
            'time': uncertainty_results['time'],
            'n_successful': uncertainty_results['n_successful'],
            'n_samples': uncertainty_results.get('n_samples', len(solutions_detailed)),
            'risk_assessment': uncertainty_results['risk_assessment']
        }
    }
    
    with open(f'{output_dir}/v7_results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 结果摘要已保存: {output_dir}/v7_results_summary.json")
    
    # 3. 保存简洁的帕累托前沿数据（用于快速可视化）
    pareto_simple = {
        'objectives': [s['objectives'] for s in solutions_detailed],
        'ids': [s['id'] for s in solutions_detailed]
    }
    
    with open(f'{output_dir}/v7_pareto_front.json', 'w', encoding='utf-8') as f:
        json.dump(pareto_simple, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Pareto前沿已保存: {output_dir}/v7_pareto_front.json")
    
    # 4. 保存每个解的路径为单独文件（方便后续可视化）
    os.makedirs(f'{output_dir}/paths', exist_ok=True)
    for sol in solutions_detailed:
        if sol.get('path') and sol['path']['nodes']:
            path_file = f'{output_dir}/paths/solution_{sol["id"]:03d}.json'
            path_data = {
                'solution_id': sol['id'],
                'objectives': sol['objectives'],
                'path': sol['path']
            }
            with open(path_file, 'w', encoding='utf-8') as f:
                json.dump(path_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 各解路径已保存: {output_dir}/paths/")
    
    # 5. 保存各解列表（用于可视化选择）
    solutions_list = []
    for sol in solutions_detailed:
        sol_copy = dict(sol)
        # 不包含完整路径，只包含概览
        if 'path' in sol_copy and sol_copy['path']:
            sol_copy['path_summary'] = {
                'n_nodes': len(sol_copy['path']['nodes']),
                'total_length': sol_copy['path']['total_length'],
                'height_stats': sol_copy['path'].get('height_stats', {})
            }
            del sol_copy['path']
        solutions_list.append(sol_copy)
    
    with open(f'{output_dir}/v7_solutions_list.json', 'w', encoding='utf-8') as f:
        json.dump(solutions_list, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 解列表已保存: {output_dir}/v7_solutions_list.json")
    
    return pareto_detailed_data, results_summary


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  eVTOL路径规划V7版本测试")
    print("  【核心改进】HV固定参考点修复 - 解决跨代数HV不可比问题")
    print("="*70)
    
    # 创建输出目录
    output_dir = 'results/v7'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/paths', exist_ok=True)
    
    # 创建环境
    env, standard, start_vp, goal_vp, dynamic_env = create_test_environment()
    
    # 创建V7规划器
    print("\n  创建V7规划器...")
    flight_profile = FlightProfile(
        cruise_min=120.0,
        cruise_max=180.0,
        climb_distance_threshold=400.0,
        approach_distance_threshold=400.0
    )
    
    planner = EVTOLV7(
        env, standard, start_vp, goal_vp, dynamic_env,
        flight_profile=flight_profile,
        use_improved_wind=True,
        use_high_fidelity_energy=True
    )
    print("  ✓ V7规划器创建成功")
    
    # 测试1: 标准路径规划
    path, coords = test_standard_planning(planner, output_dir)
    
    # 测试2: 多目标优化（HV修复验证）
    pareto_front, reference_point = test_multi_objective_optimization(
        planner, 
        n_generations=30,
        population_size=30,
        output_dir=output_dir
    )
    
    # 测试3: 不确定性分析
    uncertainty_results = test_uncertainty_analysis(planner, n_samples=200, output_dir=output_dir)
    
    # 保存详细结果数据
    save_detailed_results(pareto_front, planner, output_dir, uncertainty_results, reference_point)
    
    print("\n" + "="*60)
    print("【V7】测试总结")
    print("="*60)
    print(f"  帕累托解数量: {len(pareto_front.individuals)}")
    print(f"  超体积(HV): {pareto_front.hypervolume:.2f}")
    print(f"  固定参考点: {reference_point}")
    print(f"  能耗不确定性: {uncertainty_results['energy']['mean']:.3f}±{uncertainty_results['energy']['std']:.3f} kWh")
    print(f"\n  输出文件目录: {output_dir}/")
    print("  数据文件:")
    print("    - v7_pareto_detailed.json (完整帕累托解数据)")
    print("    - v7_results_summary.json (结果摘要)")
    print("    - v7_pareto_front.json (简洁Pareto前沿)")
    print("    - v7_solutions_list.json (解列表)")
    print("    - paths/solution_*.json (各解路径)")
    print("  可视化文件:")
    print("    - v7_path_with_buildings.png (俯视图)")
    print("    - v7_3d_path_profile.png (3D剖面)")
    print("    - v7_pareto_front_2d.png (Pareto前沿2D)")
    print("    - v7_pareto_front_3d.png (Pareto前沿3D)")
    print("    - v7_best_solution_radar.png (最优解雷达图)")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
