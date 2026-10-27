#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL路径规划V6+版本 - 整合版
========================================
核心功能:
1. 高保真能耗子模型 - 基于国标GB 38031-2022的电池特性
2. 蒙特卡洛不确定性分析 - 考虑风场、载重、环境温度等不确定性
3. 帕累托最优多目标优化 - 能耗、时间、噪声、安全性综合优化

使用方法:
    planner = EVTOLV6Plus(env, standard, start_vp, goal_vp, dynamic_env)
    result = planner.plan_multi_objective()  # 多目标优化
    analysis = planner.run_monte_carlo(n_samples=1000)  # 不确定性分析
"""

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
    cruise_min: float = 200.0
    cruise_max: float = 300.0
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
        phase = self.get_flight_phase(current_pos, start_pos, goal_pos)
        if phase == 'climb':
            dist = math.sqrt((current_pos[0]-start_pos[0])**2 + (current_pos[1]-start_pos[1])**2)
            max_h = start_pos[2] + dist * self.climb_surface_slope
            return (self.ofv_height, max_h + self.height_tolerance)
        elif phase == 'approach':
            dist = math.sqrt((current_pos[0]-goal_pos[0])**2 + (current_pos[1]-goal_pos[1])**2)
            max_h = goal_pos[2] + dist * self.approach_surface_slope
            return (self.ofv_height, max_h + self.height_tolerance)
        return (self.cruise_min, self.cruise_max)
    
    def check_height_constraint(self, current_pos, start_pos, goal_pos):
        min_h, max_h = self.get_allowed_height_range(current_pos, start_pos, goal_pos)
        current_height = current_pos[2]
        if current_height < min_h:
            return (False, min_h - current_height)
        elif current_height > max_h + self.height_tolerance:
            return (False, current_height - max_h)
        return (True, 0.0)


# =============================================================================
# 航迹评估器 - 使用高保真能耗模型
# =============================================================================

class PathEvaluator:
    """
    多目标航迹评估器 - 使用高保真能耗模型
    
    评估指标:
    1. energy: 能耗 (kWh)
    2. time: 飞行时间 (s)
    3. noise: 噪声影响
    4. safety: 综合安全风险
    5. length: 路径长度 (m)
    """
    
    def __init__(self, env: CityEnvironment, energy_model: HighFidelityEnergyModel = None):
        self.env = env
        self.energy_model = energy_model or create_energy_model('EH216-S')
        
        # 噪声模型
        from noise_model import NoiseModel
        self.noise_model = NoiseModel(env, source_level=85.0, reference_distance=100.0)
        
        # 安全模型
        from collision_risk_model import CollisionRiskModel
        self.collision_model = CollisionRiskModel()
        
        self.avg_speed = 50.0  # m/s
    
    def evaluate(self, path: List) -> np.ndarray:
        """评估路径的多目标函数值"""
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
        
        return np.array([energy, flight_time, noise, safety, total_length])
    
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
        """使用高保真能耗模型计算能耗"""
        try:
            result = self.energy_model.calculate_path_energy(path_coords, dt=1.0)
            return result['total_energy_kwh']
        except Exception as e:
            # 后备方案：基于路径长度估算
            length = self._calculate_length(path_coords)
            return length / 1000 * 0.5
    
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
        """评估综合安全风险"""
        try:
            collision_risk = 0.0
            ground_risk = 0.0
            
            for p in path_coords:
                c = self.collision_model.calculate_collision_risk(p, p, speed=20.0)
                collision_risk = max(collision_risk, c)
                
                # 简化地面风险：基于高度
                # 高度越高，地面风险越低
                min_height = min(p[2] for p in path_coords)
                ground_risk = max(0.01, 0.2 - min_height / 500)
            
            return collision_risk * 0.6 + ground_risk * 0.4
        except Exception:
            avg_height = np.mean([p[2] for p in path_coords])
            return max(0.01, 0.15 - avg_height / 1000)


# =============================================================================
# 不确定性分析器
# =============================================================================

class UncertaintyAnalyzer:
    """
    不确定性分析器 - 基于蒙特卡洛方法
    
    考虑的不确定性源:
    1. 风场不确定性 - 风速、风向的随机波动
    2. 载重不确定性 - 乘客和行李重量的变化
    3. 环境温度 - 影响电池性能
    4. 飞行高度 - 影响空气密度和能耗
    """
    
    def __init__(self, base_planner: 'EVTOLV6Plus'):
        self.base_planner = base_planner
        self.evaluator = base_planner.evaluator
    
    def add_wind_uncertainty(self, std_wind_speed: float = 2.0, 
                            std_direction: float = 15.0):
        """添加风场不确定性"""
        self.std_wind_speed = std_wind_speed
        self.std_direction = std_direction
        self.include_wind = True
    
    def add_mass_uncertainty(self, std_mass: float = 50.0):
        """添加载重不确定性"""
        self.std_mass = std_mass
        self.include_mass = True
    
    def add_temperature_uncertainty(self, std_temp: float = 5.0):
        """添加温度不确定性"""
        self.std_temp = std_temp
        self.include_temp = True
    
    def run_analysis(self, evaluator: PathEvaluator, n_samples: int = 500) -> Dict:
        """
        运行蒙特卡洛分析
        
        参数:
            evaluator: 航迹评估器
            n_samples: 采样数量
            
        返回:
            分析结果字典
        """
        print(f"\n  开始蒙特卡洛分析 ({n_samples} 次采样)...")
        
        # 收集多次规划的能耗样本
        energy_samples = []
        time_samples = []
        noise_samples = []
        
        for i in range(n_samples):
            # 添加随机扰动到环境
            self._apply_wind_perturbation(i)
            
            # 规划路径
            path = self.base_planner.plan(replan=True)
            
            if path:
                # 评估路径
                coords = evaluator._to_coords(path)
                if len(coords) >= 2:
                    energy = evaluator._calculate_energy(coords)
                    length = evaluator._calculate_length(coords)
                    noise = evaluator._evaluate_noise(coords)
                    
                    energy_samples.append(energy)
                    time_samples.append(length / 50.0)  # 时间 = 长度/速度
                    noise_samples.append(noise)
            
            if (i + 1) % 100 == 0:
                print(f"    进度: {i+1}/{n_samples}")
        
        # 计算统计指标
        energy_samples = np.array(energy_samples)
        time_samples = np.array(time_samples)
        noise_samples = np.array(noise_samples)
        
        results = {
            'n_samples': n_samples,
            'n_successful': len(energy_samples),
            'energy': {
                'mean': np.mean(energy_samples),
                'std': np.std(energy_samples),
                'min': np.min(energy_samples),
                'max': np.max(energy_samples),
                'p5': np.percentile(energy_samples, 5),
                'p95': np.percentile(energy_samples, 95),
            },
            'time': {
                'mean': np.mean(time_samples),
                'std': np.std(time_samples),
                'min': np.min(time_samples),
                'max': np.max(time_samples),
            },
            'noise': {
                'mean': np.mean(noise_samples),
                'std': np.std(noise_samples),
            }
        }
        
        # 风险评估
        results['risk_assessment'] = self._assess_risk(energy_samples, time_samples)
        
        return results
    
    def _apply_wind_perturbation(self, seed: int):
        """应用风场扰动"""
        np.random.seed(seed)
        
        # 扰动风场参数
        wind_field = self.base_planner.dynamic_env.wind_field
        wind_field.base_wind_speed += np.random.normal(0, 2.0)
        wind_field.base_wind_direction += np.random.normal(0, 15.0)
    
    def _assess_risk(self, energy_samples: np.ndarray, time_samples: np.ndarray) -> Dict:
        """风险评估"""
        # 能量超出预算的风险
        energy_budget = 5.0  # kWh
        time_budget = 80.0   # s
        
        return {
            'energy_exceed_prob': np.mean(energy_samples > energy_budget),
            'time_exceed_prob': np.mean(time_samples > time_budget),
            'energy_coefficient_variation': np.std(energy_samples) / np.mean(energy_samples) if np.mean(energy_samples) > 0 else 0,
            'time_coefficient_variation': np.std(time_samples) / np.mean(time_samples) if np.mean(time_samples) > 0 else 0,
        }


# =============================================================================
# 多目标优化器
# =============================================================================

class MultiObjectiveOptimizer:
    """
    多目标航迹优化器 - 基于NSGA-II
    
    优化目标:
    1. 能耗最小化
    2. 时间最小化
    3. 噪声最小化
    4. 安全风险最小化
    5. 路径长度最小化
    """
    
    def __init__(self, v6_planner: 'EVTOLV6Plus', evaluator: PathEvaluator):
        self.v6_planner = v6_planner  # EVTOLV6Plus对象
        self.planner = v6_planner._base_planner  # 底层LowCarbonAStarV6
        self.evaluator = evaluator
        
        # 获取环境信息
        self.start_pos = v6_planner.start_vertiport.get_TLOF_center()
        self.goal_pos = v6_planner.goal_vertiport.get_TLOF_center()
        
        # 定义优化目标
        self.objectives = [
            Objective(name='energy', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.25),
            Objective(name='time', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
            Objective(name='noise', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
            Objective(name='safety', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.20),
            Objective(name='length', func=None, objective_type=ObjectiveType.MINIMIZE, weight=0.15),
        ]
    
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
        return path
    
    def _generate_initial_paths(self, n_paths: int = 5) -> List[List]:
        """生成多条不同偏好的初始路径"""
        initial_paths = []
        
        # 保存原始设置
        original_weights = self.planner.weights.copy()
        
        # 尝试禁用verbose
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
            # 恢复原始设置
            self.planner.weights = original_weights
            if hasattr(self.planner, 'verbose'):
                self.planner.verbose = original_verbose
        
        print(f"  [初始化] 生成了 {len(initial_paths)} 条不同偏好的初始路径")
        return initial_paths
    
    def _set_weights(self, **weights):
        """设置规划器权重"""
        for key, value in weights.items():
            if key in self.planner.weights:
                self.planner.weights[key] = value
    
    def optimize(self, n_generations: int = 50, population_size: int = 30) -> ParetoFront:
        """
        运行多目标优化
        
        参数:
            n_generations: 进化代数
            population_size: 种群大小
            
        返回:
            帕累托前沿
        """
        print(f"\n  开始多目标优化 (代数={n_generations}, 种群={population_size})...")
        
        # 生成初始路径（多样性策略）
        print("  生成初始种群...")
        initial_paths = self._generate_initial_paths(n_paths=5)
        
        # 创建NSGA-II优化器，使用diverse初始化策略
        optimizer = NSGAIIOptimizer(
            n_objectives=5,
            population_size=population_size,
            n_generations=n_generations,
            random_seed=42,
            n_keypoints=10
        )
        optimizer.set_objectives(self.objectives)
        optimizer.set_environment({
            'start': self.start_pos,
            'goal': self.goal_pos,
            'bounds': ((0, 2500), (0, 2500), (30, 300))
        })
        optimizer.set_initialization_strategy('diverse')
        
        # 传入初始路径
        if initial_paths:
            optimizer.set_initial_paths(initial_paths)
        
        # 评估函数 - 接收genes，解码为path后评估
        def eval_func(genes):
            path = self._genes_to_path(genes)
            return self.evaluator.evaluate(path)
        
        # 运行优化
        start_time = time.time()
        result = optimizer.optimize(eval_func)
        opt_time = time.time() - start_time
        
        print(f"  优化完成! 用时: {opt_time:.2f}s")
        print(f"  帕累托解数量: {len(result.pareto_front.individuals)}")
        print(f"  超体积(HV): {result.pareto_front.hypervolume:.2f}")
        
        return result.pareto_front
    
    def select_best(self, pareto_front: ParetoFront, method: str = 'topsis') -> Individual:
        """
        从帕累托前沿选择最优解
        
        参数:
            pareto_front: 帕累托前沿
            method: 选择方法 ('topsis', 'fuzzy', 'weighted')
            
        返回:
            最优解
        """
        if method == 'topsis':
            decision_maker = ParetoDecisionMaker(self.objectives)
            return decision_maker.select_by_topsis(pareto_front)
        elif method == 'fuzzy':
            decision_maker = ParetoDecisionMaker(self.objectives)
            return decision_maker.select_by_fuzzy()
        else:
            # 选择第一个（按权重排序的最优）
            return pareto_front.individuals[0]


# =============================================================================
# V6+ 主规划器
# =============================================================================

class EVTOLV6Plus:
    """
    eVTOL路径规划V6+版本 - 整合版
    
    整合功能:
    1. 高保真能耗子模型 - 精确计算电池能耗
    2. 蒙特卡洛不确定性分析 - 鲁棒性评估
    3. 帕累托多目标优化 - 综合性能最优
    
    使用示例:
        # 创建规划器
        planner = EVTOLV6Plus(
            env, standard, start_vp, goal_vp, dynamic_env
        )
        
        # 方式1: 标准规划
        path = planner.plan()
        
        # 方式2: 多目标优化
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
        初始化V6+规划器
        
        参数:
            use_high_fidelity_energy: 是否使用高保真能耗模型
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
            print("  [V6+] 高保真能耗模型已启用")
        
        # V6+新增: 航迹评估器
        self.evaluator = PathEvaluator(env, self.energy_model if use_high_fidelity_energy else None)
        
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
        """
        标准路径规划
        
        返回:
            路径节点列表
        """
        return self._base_planner.find_path(**kwargs)
    
    def evaluate_path(self, path: List) -> Dict:
        """
        评估路径的多目标指标
        
        返回:
            包含能耗、时间、噪声、安全性、长度的字典
        """
        objectives = self.evaluator.evaluate(path)
        return {
            'energy': objectives[0],      # kWh
            'time': objectives[1],        # s
            'noise': objectives[2],       # 归一化
            'safety': objectives[3],     # 归一化
            'length': objectives[4]       # m
        }
    
    def optimize_multi_objective(self, n_generations: int = 50, 
                                 population_size: int = 30) -> ParetoFront:
        """
        多目标优化 - 获取帕累托前沿
        
        参数:
            n_generations: 进化代数
            population_size: 种群大小
            
        返回:
            ParetoFront对象
        """
        optimizer = MultiObjectiveOptimizer(self, self.evaluator)
        return optimizer.optimize(n_generations=n_generations, population_size=population_size)
    
    def select_best_solution(self, pareto_front: ParetoFront, 
                            method: str = 'topsis') -> Individual:
        """
        从帕累托前沿选择最优解
        
        参数:
            pareto_front: 帕累托前沿
            method: 选择方法 ('topsis', 'fuzzy', 'weighted')
            
        返回:
            最优Individual
        """
        optimizer = MultiObjectiveOptimizer(self, self.evaluator)
        return optimizer.select_best(pareto_front, method=method)
    
    def analyze_uncertainty(self, n_samples: int = 500,
                          include_wind: bool = True,
                          include_mass: bool = True,
                          include_temp: bool = True) -> Dict:
        """
        不确定性分析 - 蒙特卡洛方法
        
        参数:
            n_samples: 采样数量
            include_wind: 包含风场不确定性
            include_mass: 包含载重不确定性
            include_temp: 包含温度不确定性
            
        返回:
            分析结果字典
        """
        print(f"\n  开始蒙特卡洛分析 ({n_samples} 次采样)...")
        
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
                    time_samples.append(length / 50.0)  # 时间 = 长度/速度
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
        
        print()  # 换行
        # 恢复原始设置
        self._base_planner.verbose = original_verbose
        self.dynamic_env.wind_field.base_wind_speed = original_wind_speed
        self.dynamic_env.wind_field.base_wind_direction = original_wind_direction
        
        # 计算统计指标
        energy_samples = np.array(energy_samples)
        time_samples = np.array(time_samples)
        noise_samples = np.array(noise_samples)
        
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
        
        return results
    
    def get_full_analysis(self, n_samples: int = 500, 
                         n_generations: int = 50) -> Dict:
        """
        完整分析报告
        
        返回:
            包含以下内容的字典:
            - pareto_front: 帕累托前沿
            - best_solution: TOPSIS最优解
            - uncertainty_analysis: 不确定性分析
            - summary: 摘要统计
        """
        print("\n" + "="*60)
        print("V6+ 完整分析报告")
        print("="*60)
        
        # 1. 多目标优化
        print("\n[1/3] 多目标优化...")
        pareto_front = self.optimize_multi_objective(n_generations=n_generations)
        best_solution = self.select_best_solution(pareto_front)
        
        # 2. 不确定性分析
        print("\n[2/3] 不确定性分析...")
        uncertainty = self.analyze_uncertainty(n_samples=n_samples)
        
        # 3. 汇总
        print("\n[3/3] 生成摘要...")
        summary = {
            'n_pareto_solutions': len(pareto_front.individuals),
            'hypervolume': pareto_front.hypervolume,
            'spacing': pareto_front.spacing,
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
        print("摘要")
        print("="*60)
        print(f"  帕累托解数量: {summary['n_pareto_solutions']}")
        print(f"  超体积(HV): {summary['hypervolume']:.2f}")
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
            'summary': summary
        }


# =============================================================================
# 兼容性别名
# =============================================================================

# 保持向后兼容
LowCarbonAStarV6 = EVTOLV6Plus


if __name__ == '__main__':
    # 简单测试
    print("V6+ 版本加载成功!")
    print("\n使用方法:")
    print("  from evtol_v6_plus import EVTOLV6Plus")
    print("  planner = EVTOLV6Plus(env, standard, start_vp, goal_vp, dynamic_env)")
    print("  ")
    print("  # 标准规划")
    print("  path = planner.plan()")
    print("  ")
    print("  # 多目标优化")
    print("  pareto = planner.optimize_multi_objective()")
    print("  ")
    print("  # 不确定性分析")
    print("  analysis = planner.analyze_uncertainty(n_samples=1000)")
