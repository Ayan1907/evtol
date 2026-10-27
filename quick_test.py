#!/usr/bin/env python3
"""快速测试V6+"""
from evtol_v6_plus import EVTOLV6Plus, FlightProfile
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment
from evtol_path_planning_v4 import WindField, DynamicEnvironment

# 设置环境
env = CityEnvironment(size_x=0, size_y=0, size_z=0)
map_data = env.load_from_json('test_map_v3_compatible.json')
start_pos = tuple(map_data['start'])
goal_pos = tuple(map_data['goal'])

standard = EVTOLStandard()
start_vp = Vertiport(start_pos[0], start_pos[1], start_pos[2], standard)
goal_vp = Vertiport(goal_pos[0], goal_pos[1], goal_pos[2], standard)

dynamic_env = DynamicEnvironment(env)
wind_field = WindField(env, base_wind_speed=8.0, base_wind_direction=135.0)
dynamic_env.set_wind_field(wind_field)

flight_profile = FlightProfile(
    ofv_height=30.0,
    cruise_min=120.0,
    cruise_max=180.0,
    climb_distance_threshold=400.0,
    approach_distance_threshold=400.0
)

# 创建V6+规划器
planner = EVTOLV6Plus(
    env, standard, start_vp, goal_vp, dynamic_env,
    flight_profile=flight_profile,
    use_high_fidelity_energy=True
)

print('='*60)
print('V6+ 整合版测试成功!')
print('='*60)
print(f'起点: {start_vp.get_TLOF_center()}')
print(f'终点: {goal_vp.get_TLOF_center()}')

# 测试规划
path = planner.plan()
if path:
    print(f'路径规划成功! 节点数: {len(path)}')
    metrics = planner.evaluate_path(path)
    print(f'\n路径指标:')
    print(f'  能耗: {metrics["energy"]:.3f} kWh')
    print(f'  时间: {metrics["time"]:.1f} s')
    print(f'  噪声: {metrics["noise"]:.3f}')
    print(f'  安全: {metrics["safety"]:.4f}')
    print(f'  长度: {metrics["length"]:.1f} m')
else:
    print('路径规划失败')
