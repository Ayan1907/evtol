import sys
sys.path.insert(0, '.')
from evtol_v7.evtol_path_planning_v7 import EVTOLV7, MultiObjectiveOptimizerV7
from evtol_path_planning_v3_clean import load_environment_from_json
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport
from evtol_path_planning_v4 import DynamicEnvironment, WindField

env, start_coords, goal_coords, metadata = load_environment_from_json('test_map_v3_compatible.json')
standard = EVTOLStandard()
start_vp = Vertiport(x=start_coords[0], y=start_coords[1], z=start_coords[2], standard=standard)
goal_vp = Vertiport(x=goal_coords[0], y=goal_coords[1], z=goal_coords[2], standard=standard)
wind_field = WindField(env=env, base_wind_speed=5.0, base_wind_direction=315.0, turbulence_intensity=0.1)
dynamic_env = DynamicEnvironment(env)
dynamic_env.wind_field = wind_field

planner = EVTOLV7(env, standard, start_vp, goal_vp, dynamic_env)
result = planner.optimize_multi_objective(n_generations=30, population_size=30)

if result.individuals:
    optimizer = MultiObjectiveOptimizerV7(planner, planner.evaluator)
    path = optimizer._genes_to_path(result.individuals[0].genes)
    
    print('路径节点高度:')
    for i, p in enumerate(path):
        flag = '✓' if 120 <= p[2] <= 180 else '✗'
        print(f'  {i}: X={p[0]:.0f}, Z={p[2]:.0f}m {flag}')
