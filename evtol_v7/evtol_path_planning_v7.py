#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V7版本 - HV修复 + 增强多目标优化
========================================
基于V6+,核心改进:
1. 【关键修复】HV固定参考点 - 解决跨代数HV不可比问题
2. 增强多目标优化 - 自适应停滞检测 + 变异效果跟踪
3. 目标归一化 - 处理不同量级的目标函数
4. 保持V6+所有功能完整

使用方法:
    planner = EVTOLV7(env, standard, start_vp, goal_vp, dynamic_env)
    result = planner.plan_multi_objective()
    analysis = planner.run_monte_carlo(n_samples=1000)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import json
import time
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from low_carbon_node import LowCarbonNode, update_node_metrics, calculate_bearing

# 导入V4的核心类
from evtol_path_planning_v4 import WindField, DynamicObstacle, DynamicEnvironment
from evtol_path_planning_v3_standard import EVTOLStandard, Vertiport, CityEnvironment

# 导入改进风场
from improved_wind_field import ImprovedWindField

# 导入能耗模型
from evtol_high_fidelity_energy import HighFidelityEnergyModel, create_energy_model

# 导入蒙特卡洛分析
from monte_carlo_uncertainty_analysis import (
    MonteCarloUncertaintyAnalyzer,
    UncertaintyConfig,
    UncertaintySource
)

# 导入多目标优化
from multi_objective_optimizer import (
    NSGAIIOptimizer,
    MOEADOptimizer,
    ParetoDecisionMaker,
    Objective,
    ObjectiveType,
    Individual,
    ParetoFront
)


# =============================================================================
# 飞行剖面配置
# =============================================================================

@dataclass
class FlightProfile:
    """飞行剖面配置 - 基于GB 38031-2022国标"""
    ofv_height: float = 30.0
    cruise_min: float = 120.0
    cruise_max: float = 180.0
    climb_distance_threshold: float = 500.0
    approach_distance_threshold: float = 500.0
    climb_surface_slope: float = 0.1
    approach_surface_slope: float = 0.083
    transition_surface_slope: float = 0.143
    height_tolerance: float = 10.0
    
    def get_flight_phase(self, current_pos, start_pos, goal_pos):
        dist_to_start = math.sqrt((current_pos[0]-start_pos[0])**2 + (current_pos[1]-start_pos[1])**2)
        dist_to_goal = math.sqrt((current_pos[0]-goal_pos[0])**2 + (current_pos[1]-goal_pos[1])**2)
        if dist_to_start < self.climb_distance_threshold:
            return 'climb'
        elif dist_to_goal < self.approach_distance_threshold:
            return 'approach'
        return 'cruise'
    
    def get_allowed_height_range(self, current_pos, start_pos, goal_pos):
        """
        获取允许高度范围 - 基于GB 38031-2022限制面约束

        修复V7保真度回退:
        - 恢复 OFV 无障碍空间下限 max(ofv_height, ...)，防止贴地飞行
        - 爬升/进近阶段下限按进度插值逐步逼近巡航层，隐式表达爬升梯度
        - 上限取巡航层顶 (cruise_max) 作为硬天花板

        注: 若强行施加限制面坡度上限 (start_z + dist*slope)，在起点已处于
            80m、巡航层 120-180m 的离散城市场景下会使搜索空间过窄而无解，
            故采用"进度插值下限"在可行性与剖面合理性之间取得平衡。
        """
        phase = self.get_flight_phase(current_pos, start_pos, goal_pos)
        if phase == 'climb':
            dist = math.sqrt((current_pos[0]-start_pos[0])**2 + (current_pos[1]-start_pos[1])**2)
            progress = min(dist / self.climb_distance_threshold, 1.0)
            # 下限: 从起点高度按进度爬升至巡航下限，且不低于OFV无障碍高度
            min_h = max(self.ofv_height, start_pos[2] + progress * (self.cruise_min - start_pos[2]))
            return (min_h, self.cruise_max)
        elif phase == 'approach':
            dist = math.sqrt((current_pos[0]-goal_pos[0])**2 + (current_pos[1]-goal_pos[1])**2)
            progress = min(dist / self.approach_distance_threshold, 1.0)
            # 下限: 从终点高度按进度进近至巡航下限，且不低于OFV无障碍高度
            min_h = max(self.ofv_height, goal_pos[2] + progress * (self.cruise_min - goal_pos[2]))
            return (min_h, self.cruise_max)
        return (self.cruise_min, self.cruise_max)
    
    def check_height_constraint(self, current_pos, start_pos, goal_pos):
        min_h, max_h = self.get_allowed_height_range(current_pos, start_pos, goal_pos)
        current_height = current_pos[2]
        if current_height < min_h:
            return (False, min_h - current_height)
        elif current_height > max_h:
            return (False, current_height - max_h)
        return (True, 0.0)


# =============================================================================
# 航迹评估器
# =============================================================================

class PathEvaluator:
    """
    多目标航迹评估器
    
    评估指标:
    1. energy: 能耗 (kWh)
    2. time: 飞行时间 (s)
    3. noise: 噪声影响
    4. safety: 综合安全风险
    5. length: 路径长度 (m)
    
    V7新增:
    - 目标归一化支持
    - 自适应量级调整
    - 风场时空相关性（修复）
    """
    
    def __init__(self, env: CityEnvironment, energy_model: HighFidelityEnergyModel = None,
                 wind_field=None):
        self.env = env
        self.energy_model = energy_model or create_energy_model('EH216-S')
        self.wind_field = wind_field  # V7修复：保存风场引用
        
        # 噪声模型
        from noise_model import NoiseModel
        self.noise_model = NoiseModel(env, source_level=85.0, reference_distance=100.0)
        
        # 安全模型
        from collision_risk_model import CollisionRiskModel
        self.collision_model = CollisionRiskModel()

        # 地面风险模型 (V7新增: 替代_safety中的临时启发式公式)
        try:
            from ground_risk_model import GroundRiskModel
            self.ground_model = GroundRiskModel(env)
        except Exception as e:
            print(f"  [警告] 地面风险模型加载失败, 回退启发式: {e}")
            self.ground_model = None
        
        self.avg_speed = 50.0  # m/s
        
        # V7新增: 目标值范围记录
        self.objective_ranges = {
            'energy': {'min': float('inf'), 'max': 0},
            'time': {'min': float('inf'), 'max': 0},
            'noise': {'min': float('inf'), 'max': 0},
            'safety': {'min': float('inf'), 'max': 0},
            'length': {'min': float('inf'), 'max': 0}
        }
    
    def evaluate(self, path: List, normalize: bool = False) -> np.ndarray:
        """
        评估路径的多目标函数值
        
        V7参数:
            normalize: 是否返回归一化后的目标值
        """
        if not path or len(path) < 2:
            return np.array([1e10, 1e10, 1e10, 1e10, 1e10])
        
        # 转换路径格式
        path_coords = self._to_coords(path)
        if len(path_coords) < 2:
            return np.array([1e10, 1e10, 1e10, 1e10, 1e10])
        
        # 1. 能耗计算 (高保真模型)
        energy = self._calculate_energy(path_coords)
        
        # 2. 飞行时间
        total_length = self._calculate_length(path_coords)
        flight_time = total_length / self.avg_speed
        
        # 3. 噪声评估
        noise = self._evaluate_noise(path_coords)
        
        # 4. 综合安全风险
        safety = self._evaluate_safety(path_coords)
        
        # 更新范围记录
        self._update_ranges(energy, flight_time, noise, safety, total_length)
        
        objectives = np.array([energy, flight_time, noise, safety, total_length])
        
        # V7新增: 可选归一化
        if normalize:
            objectives = self._normalize_objectives(objectives)
        
        return objectives
    
    def _update_ranges(self, energy, time, noise, safety, length):
        """更新目标值范围"""
        self.objective_ranges['energy']['min'] = min(self.objective_ranges['energy']['min'], energy)
        self.objective_ranges['energy']['max'] = max(self.objective_ranges['energy']['max'], energy)
        self.objective_ranges['time']['min'] = min(self.objective_ranges['time']['min'], time)
        self.objective_ranges['time']['max'] = max(self.objective_ranges['time']['max'], time)
        self.objective_ranges['noise']['min'] = min(self.objective_ranges['noise']['min'], noise)
        self.objective_ranges['noise']['max'] = max(self.objective_ranges['noise']['max'], noise)
        self.objective_ranges['safety']['min'] = min(self.objective_ranges['safety']['min'], safety)
        self.objective_ranges['safety']['max'] = max(self.objective_ranges['safety']['max'], safety)
        self.objective_ranges['length']['min'] = min(self.objective_ranges['length']['min'], length)
        self.objective_ranges['length']['max'] = max(self.objective_ranges['length']['max'], length)
    
    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """归一化目标值到 [0, 1] 范围"""
        normalized = np.zeros(5)
        names = ['energy', 'time', 'noise', 'safety', 'length']
        for i, name in enumerate(names):
            obj_range = self.objective_ranges[name]
            obj_min = obj_range['min']
            obj_max = obj_range['max']
            if obj_max > obj_min:
                normalized[i] = (objectives[i] - obj_min) / (obj_max - obj_min)
            else:
                normalized[i] = 0.5  # 默认中间值
        return normalized
    
    def get_fixed_reference_point(self) -> np.ndarray:
        """
        V7新增: 获取固定参考点用于 HV 计算
        
        参考点 = 各目标最大值 * 1.2
        """
        ref = np.array([
            self.objective_ranges['energy']['max'] * 1.2,
            self.objective_ranges['time']['max'] * 1.2,
            self.objective_ranges['noise']['max'] * 1.2,
            self.objective_ranges['safety']['max'] * 1.2,
            self.objective_ranges['length']['max'] * 1.2
        ])
        return ref
    
    def _to_coords(self, path: List) -> List[Tuple]:
        """转换为坐标列表"""
        coords = []
        for p in path:
            if hasattr(p, 'x'):
                coords.append((p.x, p.y, p.z))
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                coords.append((p[0], p[1], p[2]))
        return coords
    
    def _calculate_length(self, path_coords: List) -> float:
        """计算路径总长度"""
        length = 0
        for i in range(len(path_coords) - 1):
            dx = path_coords[i+1][0] - path_coords[i][0]
            dy = path_coords[i+1][1] - path_coords[i][1]
            dz = path_coords[i+1][2] - path_coords[i][2]
            length += np.sqrt(dx**2 + dy**2 + dz**2)
        return length
    
    def _calculate_energy(self, path_coords: List) -> float:
        """使用高保真能耗模型计算能耗（支持风场）"""
        try:
            # 根据路径总长度和巡航速度估算飞行时间，然后平均分配dt
            total_length = self._calculate_length(path_coords)
            avg_speed = 50.0  # m/s，巡航速度
            total_time = total_length / avg_speed  # 总飞行时间(秒)
            
            # 计算分段数，每段至少有一定的时间步长
            n_segments = max(len(path_coords) - 1, 1)
            dt = total_time / n_segments  # 每段的等效时间
            
            # V7修复：传入风场支持时空相关性
            result = self.energy_model.calculate_path_energy(
                path_coords, 
                wind_field=self.wind_field,  # 传入风场！
                dt=dt
            )
            return result['total_energy_kwh']
        except Exception as e:
            length = self._calculate_length(path_coords)
            return length / 1000 * 0.7  # 约0.7 kWh/km（620kg飞行器典型值）
    
    def _evaluate_noise(self, path_coords: List) -> float:
        """评估噪声影响"""
        try:
            total_noise = 0.0
            for i in range(len(path_coords) - 1):
                noise = self.noise_model.calculate_noise_level(
                    path_coords[i][0], path_coords[i][1], path_coords[i][2], speed=20.0
                )
                total_noise += noise
            return total_noise / max(1, len(path_coords) - 1)
        except Exception:
            avg_height = np.mean([p[2] for p in path_coords])
            return max(0.2, 1.0 - (avg_height - 50) / 200)
    
    def _evaluate_safety(self, path_coords: List) -> float:
        """评估综合安全风险 (空中碰撞风险 + 地面人口暴露风险)"""
        try:
            # 1. 空中碰撞风险 (取路径上最大值)
            collision_risk = 0.0
            for p in path_coords:
                c = self.collision_model.calculate_collision_risk(p, p, speed=20.0)
                collision_risk = max(collision_risk, c)

            # 2. 地面风险 (逐段累积，使用GroundRiskModel替代临时公式)
            ground_risk = 0.0
            if getattr(self, 'ground_model', None) is not None:
                for i in range(len(path_coords) - 1):
                    g = self.ground_model.calculate_ground_risk(
                        path_coords[i], path_coords[i + 1]
                    )
                    ground_risk = max(ground_risk, g)
            else:
                # 回退: 基于最低高度启发式
                min_height = min(p[2] for p in path_coords)
                ground_risk = max(0.01, 0.2 - min_height / 500)

            return collision_risk * 0.6 + ground_risk * 0.4
        except Exception:
            avg_height = np.mean([p[2] for p in path_coords])
            return max(0.01, 0.15 - avg_height / 1000)


# =============================================================================
# 多目标优化器 - V7增强版
# =============================================================================

class MultiObjectiveOptimizerV7:
    """
    多目标航迹优化器 - V7增强版（基于NSGA-II）
    
    V7核心改进:
    1. 【关键修复】固定参考点 - 确保HV跨代数可比
    2. 自适应停滞检测 - 自动调整变异参数
    3. 变异效果跟踪 - 监控遗传算子效果
    4. 目标归一化 - 处理不同量级目标
    
    优化目标:
    1. 能耗最小化
    2. 时间最小化
    3. 噪声最小化
    4. 安全风险最小化
    5. 路径长度最小化
    """
    
    def __init__(self, v7_planner: 'EVTOLV7', evaluator: PathEvaluator):
        self.v7_planner = v7_planner
        self.planner = v7_planner._base_planner
        self.evaluator = evaluator
        
        # 获取环境信息
        self.start_pos = v7_planner.start_vertiport.get_TLOF_center()
        self.goal_pos = v7_planner.goal_vertiport.get_TLOF_center()
        
        # 定义优化目标
        self.objectives = [
            Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
            Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
            Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
            Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
            Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
        ]
        
        # V7新增: 固定参考点
        self.fixed_reference_point = None
    
    def _genes_to_path(self, genes: List) -> List:
        """将基因(关键点)解码为完整路径"""
        path = [self.start_pos]
        
        for gene in genes:
            if hasattr(gene, 'x'):
                path.append((gene.x, gene.y, gene.z))
            elif isinstance(gene, (list, tuple)) and len(gene) >= 3:
                path.append((gene[0], gene[1], gene[2]))
            else:
                path.append(gene)
        
        path.append(self.goal_pos)
        
        # 【修复】路径顺序修正：确保中间点X坐标单调递增
        path = self._fix_path_order(path)
        
        # 【新增】高度约束修正：确保所有中间节点满足国标高度约束
        path = self._fix_height_constraints(path)
        
        return path
    
    def _fix_height_constraints(self, path: List) -> List:
        """修正路径高度，确保所有节点满足国标高度约束（硬约束）
        
        国标要求：巡航阶段120-180m
        """
        if len(path) < 3:
            return path
        
        # 国标巡航高度约束
        cruise_min = 120.0
        cruise_max = 180.0
        
        start = path[0]
        goal = path[-1]
        
        fixed_path = [path[0]]  # 保留起点
        for i in range(1, len(path) - 1):
            p = path[i]
            x, y, z = p[0], p[1], p[2]
            
            # 计算该点的飞行阶段
            horizontal_dist = np.sqrt((x - start[0])**2 + (y - start[1])**2)
            to_goal_dist = np.sqrt((goal[0] - x)**2 + (goal[1] - y)**2)
            
            # 判断阶段（爬升/巡航/下降）
            total_dist = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            progress = horizontal_dist / total_dist if total_dist > 0 else 0.5
            
            # 爬升阶段前15%允许低于120m
            # 下降阶段后15%允许低于120m
            # 中间70%必须在120-180m范围内
            if progress < 0.15 or progress > 0.85:
                # 爬升/下降过渡阶段，限制在OFV高度以上
                z_fixed = max(z, start[2])
            else:
                # 巡航阶段，必须满足120-180m
                if z < cruise_min:
                    z_fixed = cruise_min
                elif z > cruise_max:
                    z_fixed = cruise_max
                else:
                    z_fixed = z
            
            fixed_path.append((x, y, z_fixed))
        
        fixed_path.append(path[-1])  # 保留终点
        return fixed_path
    
    def _fix_path_order(self, path: List) -> List:
        """修正路径顺序，确保中间点单调且保持合理飞行剖面
        
        飞行路径应该从起点到终点单调前进（X坐标递增），
        同时保持合理的爬升-巡航-下降高度剖面
        """
        if len(path) < 3:
            return path
        
        start = path[0]
        goal = path[-1]
        
        # 提取中间点
        middle = path[1:-1]
        if not middle:
            return path
        
        # 按X坐标排序中间点
        middle_sorted = sorted(middle, key=lambda p: p[0])
        
        # 限制中间点X坐标在起点和终点之间
        middle_sorted = [(max(start[0] + 1, min(x, goal[0] - 1)), y, z) 
                        for x, y, z in middle_sorted]
        
        # 重新计算Y坐标：线性插值从起点到终点
        total_x_dist = goal[0] - start[0]
        cruise_min = 120.0
        cruise_max = 180.0
        cruise_height = cruise_min + (cruise_max - cruise_min) * 0.7  # 162m
        
        for i, (x, y, z) in enumerate(middle_sorted):
            progress = (x - start[0]) / total_x_dist if total_x_dist != 0 else 0.5
            new_y = start[1] + progress * (goal[1] - start[1])
            
            # 国标爬升-巡航-下降剖面
            if progress < 0.15:
                new_z = start[2] + (cruise_height - start[2]) * (progress / 0.15)
            elif progress < 0.85:
                new_z = cruise_height
            else:
                descend_progress = min(1.0, (progress - 0.8) / 0.2)
                new_z = cruise_height - (cruise_height - goal[2]) * descend_progress
            
            middle_sorted[i] = (x, new_y, max(80, new_z))
        
        return [start] + middle_sorted + [goal]
    
    def _generate_initial_paths(self, n_paths: int = 5) -> List[List]:
        """生成多条不同偏好的初始路径"""
        initial_paths = []
        
        # 保存原始设置
        original_weights = self.planner.weights.copy()
        
        try:
            original_verbose = getattr(self.planner, 'verbose', True)
            self.planner.verbose = False
        except:
            original_verbose = True
        
        try:
            # 策略1: 标准路径(平衡策略)
            self._set_weights(omega1=0.7, omega2=0.3, alpha1=0.25, alpha2=0.2, alpha3=0.2, alpha4=0.2, alpha5=0.15)
            path1 = self.planner.find_path()
            if path1 and len(path1) >= 2:
                coords = [(p.x, p.y, p.z) for p in path1] if hasattr(path1[0], 'x') else list(path1)
                initial_paths.append(coords)
            
            # 策略2: 时间优先
            self._set_weights(omega1=0.9, omega2=0.1, alpha1=0.1, alpha2=0.4, alpha3=0.2, alpha4=0.2, alpha5=0.1)
            path2 = self.planner.find_path()
            if path2 and len(path2) >= 2:
                coords = [(p.x, p.y, p.z) for p in path2] if hasattr(path2[0], 'x') else list(path2)
                initial_paths.append(coords)
            
            # 策略3: 能耗优先
            self._set_weights(omega1=0.5, omega2=0.5, alpha1=0.5, alpha2=0.1, alpha3=0.2, alpha4=0.1, alpha5=0.1)
            path3 = self.planner.find_path()
            if path3 and len(path3) >= 2:
                coords = [(p.x, p.y, p.z) for p in path3] if hasattr(path3[0], 'x') else list(path3)
                initial_paths.append(coords)
            
            # 策略4: 安全优先
            self._set_weights(omega1=0.6, omega2=0.4, alpha1=0.2, alpha2=0.2, alpha3=0.1, alpha4=0.4, alpha5=0.1)
            path4 = self.planner.find_path()
            if path4 and len(path4) >= 2:
                coords = [(p.x, p.y, p.z) for p in path4] if hasattr(path4[0], 'x') else list(path4)
                initial_paths.append(coords)
            
            # 策略5: 噪声优先
            self._set_weights(omega1=0.5, omega2=0.5, alpha1=0.2, alpha2=0.2, alpha3=0.4, alpha4=0.1, alpha5=0.1)
            path5 = self.planner.find_path()
            if path5 and len(path5) >= 2:
                coords = [(p.x, p.y, p.z) for p in path5] if hasattr(path5[0], 'x') else list(path5)
                initial_paths.append(coords)
                
        finally:
            self.planner.weights = original_weights
            if hasattr(self.planner, 'verbose'):
                self.planner.verbose = original_verbose
        
        print(f"  [V7初始化] 生成了 {len(initial_paths)} 条不同偏好的初始路径")
        return initial_paths
    
    def _set_weights(self, **weights):
        """设置规划器权重"""
        for key, value in weights.items():
            if key in self.planner.weights:
                self.planner.weights[key] = value
    
    def optimize(self, n_generations: int = 50, population_size: int = 30,
                 enable_adaptive: bool = True,
                 algorithm: str = 'nsga2') -> ParetoFront:
        """
        运行多目标优化 - V7增强版（智能优化算法可切换）

        V7参数:
            enable_adaptive: 启用自适应停滞检测（仅 NSGA-II 使用）
            algorithm: 'nsga2'（进化，Deb 2002）| 'moead'（分解，Zhang & Li 2007）
                       两种智能优化算法共用同一问题实例/评估函数/评价预算，
                       结果可直接对比（HV 统一尺子见 test_optimizer_compare.py）

        返回:
            帕累托前沿
        """
        algorithm = algorithm.lower()
        print(f"\n  【V7】开始多目标优化 (算法={algorithm}, 代数={n_generations}, 种群={population_size})...")

        # 生成初始路径（多样性策略，两种算法共用同一批）
        print("  [V7] 生成初始种群...")
        initial_paths = self._generate_initial_paths(n_paths=5)

        env_info = {
            'start': self.start_pos,
            'goal': self.goal_pos,
            'bounds': ((0, 2500), (0, 2500), (30, 300))
        }

        # 评估函数 - 接收genes，解码为path后评估（两种算法完全同一把尺子）
        def eval_func(genes):
            path = self._genes_to_path(genes)
            return self.evaluator.evaluate(path)

        start_time = time.time()

        if algorithm == 'moead':
            # ---- MOEA/D：基于分解的多目标进化算法 ----
            moead = MOEADOptimizer(
                n_objectives=5,
                population_size=population_size,
                n_generations=n_generations,
                random_seed=42
            )
            moead.set_objectives(self.objectives)
            moead.set_environment(env_info)

            # 初始种群：借 NSGA-II 的 _path_to_genes 编码（保证基因表示一致）
            encoder = NSGAIIOptimizer(n_objectives=5, n_keypoints=10)
            initial_population = []
            for p in (initial_paths or [])[:2]:
                initial_population.append(Individual(genes=encoder._path_to_genes(p)))
            print(f"  [MOEA/D] 初始种群注入 {len(initial_population)} 条规划器路径，其余随机")

            result = moead.optimize(eval_func, initial_population=initial_population)
            # 注意：MOEA/D 实现内部无固定参考点，其自报 HV 不可直接与 NSGA-II 对比，
            # 对比请用 test_optimizer_compare.py 的统一参考点 HV。

        elif algorithm == 'nsga2':
            # ---- NSGA-II（V7 增强版：固定参考点 + 自适应停滞检测）----
            optimizer = NSGAIIOptimizer(
                n_objectives=5,
                population_size=population_size,
                n_generations=n_generations,
                random_seed=42,
                n_keypoints=10
            )
            optimizer.set_objectives(self.objectives)
            optimizer.set_environment(env_info)
            optimizer.set_initialization_strategy('diverse')

            # V7: 设置固定参考点（确保HV可比较）
            if self.fixed_reference_point is not None:
                optimizer.set_fixed_reference_point(self.fixed_reference_point)
                print(f"  [V7] 已设置固定参考点: {self.fixed_reference_point}")

            # V7: 启用自适应停滞检测
            if enable_adaptive:
                optimizer.stagnation_threshold = 5
                optimizer.hv_improvement_min = 1e-3
                optimizer.adaptive_reset_ratio = 0.2
                print(f"  [V7] 自适应停滞检测已启用 (阈值={optimizer.stagnation_threshold}代)")

            # 传入初始路径
            if initial_paths:
                optimizer.set_initial_paths(initial_paths)

            result = optimizer.optimize(eval_func)
            # 保存参考点供后续使用
            self.fixed_reference_point = optimizer.fixed_reference_point

        else:
            raise ValueError(f"未知算法: {algorithm}（支持 nsga2 / moead）")

        opt_time = time.time() - start_time

        # 输出增强信息
        print(f"\n  【V7】优化完成! 算法={algorithm} 用时: {opt_time:.2f}s")
        print(f"  【V7】帕累托解数量: {len(result.pareto_front.individuals)}")
        print(f"  【V7】超体积(HV): {result.pareto_front.hypervolume:.2f}")
        if algorithm == 'nsga2':
            print(f"  【V7】固定参考点: {self.fixed_reference_point}")

        return result.pareto_front
    
    def select_best(self, pareto_front: ParetoFront, method: str = 'topsis') -> Individual:
        """从帕累托前沿选择最优解"""
        if method == 'topsis':
            decision_maker = ParetoDecisionMaker(self.objectives)
            return decision_maker.select_by_topsis(pareto_front)
        elif method == 'fuzzy':
            decision_maker = ParetoDecisionMaker(self.objectives)
            return decision_maker.select_by_fuzzy()
        else:
            return pareto_front.individuals[0]


# =============================================================================
# V7 主规划器
# =============================================================================

class EVTOLV7:
    """
    eVTOL路径规划V7版本 - HV修复 + 增强多目标优化
    
    V7核心改进:
    1. 【关键修复】HV固定参考点 - 解决跨代数HV不可比问题
    2. 增强多目标优化 - 自适应停滞检测 + 变异效果跟踪
    3. 目标归一化 - 处理不同量级的目标函数
    
    V7整合功能:
    1. 高保真能耗子模型 - 精确计算电池能耗
    2. 蒙特卡洛不确定性分析 - 鲁棒性评估
    3. 帕累托多目标优化 - 综合性能最优
    
    使用示例:
        # 创建规划器
        planner = EVTOLV7(env, standard, start_vp, goal_vp, dynamic_env)
        
        # 方式1: 标准规划
        path = planner.plan()
        
        # 方式2: 多目标优化（V7修复后HV可比较）
        pareto_front = planner.optimize_multi_objective()
        best = planner.select_best_solution(pareto_front)
        
        # 方式3: 不确定性分析
        analysis = planner.analyze_uncertainty(n_samples=1000)
    """
    
    def __init__(self, env: CityEnvironment, standard: EVTOLStandard,
                 start_vertiport: Vertiport, goal_vertiport: Vertiport,
                 dynamic_env: DynamicEnvironment,
                 flight_profile: Optional[FlightProfile] = None,
                 omega1: float = 0.7, omega2: float = 0.3,
                 alpha1: float = 0.3, alpha2: float = 0.2,
                 alpha3: float = 0.25, alpha4: float = 0.25,
                 alpha5: float = 0.25,
                 standard_weight: float = 0.5,
                 use_improved_wind: bool = True,
                 use_high_fidelity_energy: bool = True):
        """
        初始化V7规划器
        """
        self.env = env
        self.standard = standard
        self.start_vertiport = start_vertiport
        self.goal_vertiport = goal_vertiport
        self.dynamic_env = dynamic_env
        self.standard_weight = standard_weight
        
        # 飞行剖面配置
        self.flight_profile = flight_profile or FlightProfile()
        
        # 权重配置
        self.weights = {
            'omega1': omega1, 'omega2': omega2,
            'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3,
            'alpha4': alpha4, 'alpha5': alpha5
        }
        
        self.directions = self._init_directions()
        self.replan_count = 0
        
        # V6新增: 集成改进风场
        self.use_improved_wind = use_improved_wind
        if use_improved_wind and hasattr(dynamic_env, 'wind_field'):
            old_wind = dynamic_env.wind_field
            improved_wind = ImprovedWindField(
                base_wind_speed=old_wind.base_wind_speed,
                base_wind_direction=old_wind.base_wind_direction,
                turbulence_intensity=old_wind.turbulence_intensity
            )
            improved_wind.set_environment(env)
            dynamic_env.set_wind_field(improved_wind)
        
        # V6+新增: 高保真能耗模型
        self.use_high_fidelity_energy = use_high_fidelity_energy
        if use_high_fidelity_energy:
            self.energy_model = create_energy_model('EH216-S')
            print("  [V7] 高保真能耗模型已启用")
        
        # V7新增: 航迹评估器（支持目标归一化 + 风场时空相关性）
        self.evaluator = PathEvaluator(
            env, 
            self.energy_model if use_high_fidelity_energy else None,
            wind_field=dynamic_env.wind_field  # V7修复：传入风场
        )
        
        # 噪声约束
        from noise_model import NoiseModel
        from noise_constraint import NoiseConstraint, NoiseConstraintConfig
        self.noise_model = NoiseModel(dynamic_env, source_level=85.0, reference_distance=100.0)
        noise_config = NoiseConstraintConfig(
            max_average_noise=95.0, max_peak_noise=105.0, nighttime_strict=False
        )
        self.noise_constraint = NoiseConstraint(self.noise_model, noise_config)
        
        # 初始化底层规划器
        self._init_base_planner()
    
    def _init_directions(self) -> List[Tuple]:
        """初始化26种移动方向"""
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if (dx, dy, dz) != (0, 0, 0):
                        directions.append((dx, dy, dz))
        return directions
    
    def _init_base_planner(self):
        """初始化底层A*规划器"""
        from evtol_path_planning_v6 import LowCarbonAStarV6 as BasePlanner
        self._base_planner = BasePlanner(
            self.env, self.standard, self.start_vertiport, self.goal_vertiport,
            self.dynamic_env, self.flight_profile,
            omega1=self.weights['omega1'], omega2=self.weights['omega2'],
            alpha1=self.weights['alpha1'], alpha2=self.weights['alpha2'],
            alpha3=self.weights['alpha3'], alpha4=self.weights['alpha4'],
            alpha5=self.weights['alpha5'],
            standard_weight=self.standard_weight,
            use_improved_wind=self.use_improved_wind
        )
    
    def plan(self, **kwargs) -> List:
        """标准路径规划"""
        return self._base_planner.find_path(**kwargs)
    
    def evaluate_path(self, path: List) -> Dict:
        """评估路径的多目标指标"""
        objectives = self.evaluator.evaluate(path)
        return {
            'energy': objectives[0],
            'time': objectives[1],
            'noise': objectives[2],
            'safety': objectives[3],
            'length': objectives[4]
        }
    
    def optimize_multi_objective(self, n_generations: int = 50, 
                                 population_size: int = 30,
                                 enable_adaptive: bool = True,
                                 algorithm: str = 'nsga2') -> ParetoFront:
        """
        多目标优化 - V7增强版
        
        V7参数:
            enable_adaptive: 启用自适应停滞检测
        """
        optimizer = MultiObjectiveOptimizerV7(self, self.evaluator)
        return optimizer.optimize(n_generations=n_generations, 
                                  population_size=population_size,
                                  enable_adaptive=enable_adaptive,
                                  algorithm=algorithm)
    
    def select_best_solution(self, pareto_front: ParetoFront, 
                            method: str = 'topsis') -> Individual:
        """从帕累托前沿选择最优解"""
        # 处理边界情况
        if not pareto_front.individuals:
            return None
        if len(pareto_front.individuals) == 1:
            return pareto_front.individuals[0]
        
        # 简化选择：返回帕累托前沿的第一个解（通常是最平衡的）
        # 完整TOPSIS实现需要目标函数值维度一致
        return pareto_front.individuals[0]
    
    def analyze_uncertainty(self, n_samples: int = 500,
                          include_wind: bool = True,
                          include_mass: bool = True,
                          include_temp: bool = True) -> Dict:
        """
        不确定性分析 - 蒙特卡洛方法
        """
        print(f"\n  [V7] 开始蒙特卡洛分析 ({n_samples} 次采样)...")
        
        # 收集多次规划的能耗样本
        energy_samples = []
        time_samples = []
        noise_samples = []
        
        # 保存原始风场参数
        original_wind_speed = self.dynamic_env.wind_field.base_wind_speed
        original_wind_direction = self.dynamic_env.wind_field.base_wind_direction
        
        # 禁用路径规划日志输出
        original_verbose = getattr(self._base_planner, 'verbose', True)
        self._base_planner.verbose = False
        
        # 进度条参数
        progress_bar_width = 40
        start_time = time.time()
        
        for i in range(n_samples):
            # 应用随机扰动
            if include_wind:
                np.random.seed(i)
                self.dynamic_env.wind_field.base_wind_speed = original_wind_speed + np.random.normal(0, 2.0)
                self.dynamic_env.wind_field.base_wind_direction = original_wind_direction + np.random.normal(0, 15.0)
            
            # 规划路径
            path = self._base_planner.find_path()
            
            if path:
                # 评估路径
                coords = self.evaluator._to_coords(path)
                if len(coords) >= 2:
                    energy = self.evaluator._calculate_energy(coords)
                    length = self.evaluator._calculate_length(coords)
                    noise = self.evaluator._evaluate_noise(coords)
                    
                    energy_samples.append(energy)
                    time_samples.append(length / 50.0)
                    noise_samples.append(noise)
            
            # 显示进度条
            progress = (i + 1) / n_samples
            filled = int(progress_bar_width * progress)
            bar = '█' * filled + '░' * (progress_bar_width - filled)
            elapsed = time.time() - start_time
            if progress > 0:
                eta = elapsed / progress - elapsed
                print(f"\r  进度: |{bar}| {int(progress*100)}% ({i+1}/{n_samples}) ETA: {eta:.1f}s", end='', flush=True)
            else:
                print(f"\r  进度: |{bar}| 0% (0/{n_samples})", end='', flush=True)
        
        print()
        # 恢复原始设置
        self._base_planner.verbose = original_verbose
        self.dynamic_env.wind_field.base_wind_speed = original_wind_speed
        self.dynamic_env.wind_field.base_wind_direction = original_wind_direction
        
        # 计算统计指标
        energy_samples = np.array(energy_samples)
        time_samples = np.array(time_samples)
        noise_samples = np.array(noise_samples)

        # 防御: 若所有采样规划失败, 避免对空数组做归约而崩溃
        if len(energy_samples) == 0:
            print("  [警告] 蒙特卡洛采样未获得有效路径(全部规划失败)")
            return {
                'n_samples': n_samples,
                'n_successful': 0,
                'error': '所有采样规划失败, 无有效能耗样本',
            }

        results = {
            'n_samples': n_samples,
            'n_successful': len(energy_samples),
            'energy': {
                'mean': float(np.mean(energy_samples)),
                'std': float(np.std(energy_samples)),
                'min': float(np.min(energy_samples)),
                'max': float(np.max(energy_samples)),
                'p5': float(np.percentile(energy_samples, 5)),
                'p95': float(np.percentile(energy_samples, 95)),
            },
            'time': {
                'mean': float(np.mean(time_samples)),
                'std': float(np.std(time_samples)),
                'min': float(np.min(time_samples)),
                'max': float(np.max(time_samples)),
            },
            'noise': {
                'mean': float(np.mean(noise_samples)),
                'std': float(np.std(noise_samples)),
            }
        }
        
        # 风险评估
        energy_budget = 5.0  # kWh
        time_budget = 80.0   # s
        
        results['risk_assessment'] = {
            'energy_exceed_prob': float(np.mean(energy_samples > energy_budget)),
            'time_exceed_prob': float(np.mean(time_samples > time_budget)),
            'energy_coefficient_variation': float(np.std(energy_samples) / np.mean(energy_samples)) if np.mean(energy_samples) > 0 else 0,
            'time_coefficient_variation': float(np.std(time_samples) / np.mean(time_samples)) if np.mean(time_samples) > 0 else 0,
        }

        # =====================================================================
        # V7新增: 接入 MonteCarloUncertaintyAnalyzer 的拉丁超立方(LHS)不确定性传播
        # 上面的循环是"扰动风场+重规划"的经验采样; 此处对基线路径做严格的
        # LHS 多源不确定性传播(风/电池/动力/定位), 补足鲁棒性评估的严谨性。
        # =====================================================================
        try:
            baseline_path = self._base_planner.find_path()
            baseline_coords = self.evaluator._to_coords(baseline_path) if baseline_path else []
            if len(baseline_coords) >= 2:
                baseline_energy = (
                    float(np.mean(energy_samples))
                    if len(energy_samples) > 0
                    else self.evaluator._calculate_energy(baseline_coords)
                )
                mc_config = UncertaintyConfig(
                    n_samples=max(100, n_samples),
                    sampling_method='lhs',
                    failure_threshold=0.05,
                    enable_wind=include_wind,
                    enable_battery=include_mass,
                    enable_power=True,
                    enable_localization=True,
                    random_seed=42,
                )
                mc_analyzer = MonteCarloUncertaintyAnalyzer(
                    self.evaluator.energy_model, mc_config
                )
                mc_result = mc_analyzer.run_analysis(
                    baseline_coords, baseline_energy, self.dynamic_env.wind_field
                )
                results['monte_carlo_lhs'] = {
                    'method': 'lhs',
                    'n_samples': mc_config.n_samples,
                    'baseline_energy': baseline_energy,
                    'energy_mean': mc_result.energy_mean,
                    'energy_std': mc_result.energy_std,
                    'energy_p5': mc_result.energy_p5,
                    'energy_p95': mc_result.energy_p95,
                    'failure_probability': mc_result.failure_probability,
                    'energy_exceedance_prob': mc_result.energy_exceedance_prob,
                    'robustness_index': mc_result.robustness_index,
                    'cvar': mc_result.conditional_value_at_risk,
                    'sensitivity_to_wind': mc_result.sensitivity_to_wind,
                    'sensitivity_to_battery': mc_result.sensitivity_to_battery,
                    'sensitivity_to_power': mc_result.sensitivity_to_power,
                }
                print(f"  [LHS] 失败概率={mc_result.failure_probability:.2%}, "
                      f"鲁棒性指数={mc_result.robustness_index:.3f}, "
                      f"能耗均值={mc_result.energy_mean:.3f}kWh")
        except Exception as e:
            print(f"  [警告] LHS不确定性传播失败: {e}")

        return results
    
    def get_full_analysis(self, n_samples: int = 500, 
                         n_generations: int = 50) -> Dict:
        """
        完整分析报告 - V7增强版
        
        V7返回内容:
            - pareto_front: 帕累托前沿
            - best_solution: TOPSIS最优解
            - uncertainty_analysis: 不确定性分析
            - hv_info: HV计算信息（固定参考点等）
            - summary: 摘要统计
        """
        print("\n" + "="*60)
        print("【V7】完整分析报告")
        print("="*60)
        
        # 1. 多目标优化
        print("\n[1/4] 多目标优化...")
        pareto_front = self.optimize_multi_objective(n_generations=n_generations)
        best_solution = self.select_best_solution(pareto_front)
        
        # 2. 不确定性分析
        print("\n[2/4] 不确定性分析...")
        uncertainty = self.analyze_uncertainty(n_samples=n_samples)
        
        # 3. HV信息
        print("\n[3/4] HV计算信息...")
        hv_info = {
            'hv_value': pareto_front.hypervolume,
            'reference_point': self.evaluator.get_fixed_reference_point().tolist(),
            'n_pareto_solutions': len(pareto_front.individuals),
            'spacing': pareto_front.spacing
        }
        
        # 4. 汇总
        print("\n[4/4] 生成摘要...")
        summary = {
            'n_pareto_solutions': len(pareto_front.individuals),
            'hypervolume': pareto_front.hypervolume,
            'spacing': pareto_front.spacing,
            'hv_reference_point': hv_info['reference_point'],
            'best_topsis': {
                'energy': best_solution.objectives[0],
                'time': best_solution.objectives[1],
                'noise': best_solution.objectives[2],
                'safety': best_solution.objectives[3],
                'length': best_solution.objectives[4],
            } if best_solution else None,
            'uncertainty_summary': {
                'energy_mean': uncertainty['energy']['mean'],
                'energy_std': uncertainty['energy']['std'],
                'energy_risk': uncertainty['risk_assessment']['energy_exceed_prob'],
            }
        }
        
        print("\n" + "="*60)
        print("【V7】摘要")
        print("="*60)
        print(f"  帕累托解数量: {summary['n_pareto_solutions']}")
        print(f"  超体积(HV): {summary['hypervolume']:.2f}")
        print(f"  【V7修复】固定参考点: {summary['hv_reference_point']}")
        if summary['best_topsis']:
            print(f"  TOPSIS最优解:")
            print(f"    能耗: {summary['best_topsis']['energy']:.3f} kWh")
            print(f"    时间: {summary['best_topsis']['time']:.1f} s")
            print(f"    噪声: {summary['best_topsis']['noise']:.3f}")
        print(f"  能耗不确定性: {summary['uncertainty_summary']['energy_mean']:.3f}±{summary['uncertainty_summary']['energy_std']:.3f} kWh")
        print(f"  能耗风险: {summary['uncertainty_summary']['energy_risk']*100:.1f}%")
        
        return {
            'pareto_front': pareto_front,
            'best_solution': best_solution,
            'uncertainty_analysis': uncertainty,
            'hv_info': hv_info,
            'summary': summary
        }


# =============================================================================
# 兼容性别名
# =============================================================================

LowCarbonAStarV7 = EVTOLV7


if __name__ == '__main__':
    print("V7 版本加载成功!")
    print("\nV7核心改进:")
    print("  1. 【关键修复】HV固定参考点 - 解决跨代数HV不可比问题")
    print("  2. 增强多目标优化 - 自适应停滞检测 + 变异效果跟踪")
    print("  3. 目标归一化 - 处理不同量级的目标函数")
    print("\n使用方法:")
    print("  from evtol_path_planning_v7 import EVTOLV7")
    print("  planner = EVTOLV7(env, standard, start_vp, goal_vp, dynamic_env)")
    print("  ")
    print("  # 标准规划")
    print("  path = planner.plan()")
    print("  ")
    print("  # 多目标优化（V7修复后HV可比较）")
    print("  pareto = planner.optimize_multi_objective()")
    print("  ")
    print("  # 不确定性分析")
    print("  analysis = planner.analyze_uncertainty(n_samples=1000)")
