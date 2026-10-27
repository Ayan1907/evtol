#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL航迹规划多目标优化模块 - V6版本
=====================================

本模块实现基于NSGA-II和MOEA/D的多目标航迹规划优化算法，
生成帕累托最优前沿，支持多维度权衡决策。

## 一、方法论框架

### 1.1 多目标优化问题形式化

eVTOL航迹规划多目标优化问题可形式化为:

    minimize: F(x) = [f₁(x), f₂(x), ..., fₖ(x)]
    subject to: g(x) ≤ 0  (不等式约束)
                h(x) = 0    (等式约束)
                x ∈ Ω      (决策空间)

其中目标函数包括:
- f₁: 能耗 (kWh) - 最小化
- f₂: 飞行时间 (s) - 最小化
- f₃: 噪声影响 (dB·m²) - 最小化
- f₄: 碰撞风险 (概率) - 最小化
- f₅: 路径长度 (m) - 最小化

### 1.2 参考文献

[M1] Deb K, et al. A Fast and Elitist Multiobjective Genetic Algorithm: 
     NSGA-II [J]. IEEE TEVC, 2002.
     - NSGA-II核心算法框架

[M2] Zhang Q, Li H. MOEA/D: A Multiobjective Evolutionary Algorithm 
     Based on Decomposition [J]. IEEE TEVC, 2007.
     - MOEA/D分解方法

[M3] Zitzler E, Thiele L. Multiobjective Evolutionary Algorithms: A 
     Comparative Case Study and the Strength Pareto Approach [J]. 
     IEEE TEVC, 1999.
     - 超体积指标和SPEA2

[M4] Coello C A C, et al. Evolutionary Algorithms for Solving 
     Multi-Objective Problems [M]. Springer, 2007.
     - 多目标进化算法综述

---

**作者**: eVTOL研究团队
**版本**: V7.0 (多目标优化)
**更新**: 2026-04-16
**参考**: Deb(2002), Zhang(2007), Zitzler(1999), Coello(2007)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
from collections import defaultdict
import random
import heapq


# ============================================================================
# 数据来源常量定义
# ============================================================================

# 多目标权重参考 (可调整)
DEFAULT_OBJECTIVE_WEIGHTS = {
    'energy': 0.30,      # 能耗权重
    'time': 0.20,        # 时间权重
    'noise': 0.20,      # 噪声权重
    'collision': 0.20,  # 碰撞风险权重
    'length': 0.10,     # 路径长度权重
}

# 帕累托前沿收敛判定阈值
CONVERGENCE_THRESHOLDS = {
    'hv_improvement': 0.001,  # 超体积改进阈值
    'igd_improvement': 0.01,  # 世代距离改进阈值
    'max_stagnation': 50,     # 最大停滞代数
}


# ============================================================================
# 数据结构定义
# ============================================================================

# ============================================================================
# 修改说明: 解决变异概率失效和早熟收敛问题
# 1. 新增自适应停滞检测和处理机制
# 2. 新增多样性指标计算
# 3. 新增目标函数归一化支持
# 4. eta_mutation改为可配置参数
# ============================================================================

class ObjectiveType(Enum):
    """目标类型枚举"""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """目标函数定义"""
    name: str
    func: Callable
    objective_type: ObjectiveType = ObjectiveType.MINIMIZE
    weight: float = 1.0
    enabled: bool = True
    
    def __repr__(self):
        return f"Objective({self.name}, type={self.objective_type.value}, weight={self.weight})"


@dataclass
class Individual:
    """
    多目标优化个体
    
    Attributes:
        genes: 路径编码 (List of (x, y, z) tuples)
        objectives: 目标函数值列表
        crowding_distance: 拥挤距离 (NSGA-II)
        rank: 帕累托秩 (非支配等级)
        dominated: 被支配标志
        dominates_count: 支配该个体的个体数量
    """
    genes: List[Tuple[int, int, int]]
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    crowding_distance: float = 0.0
    rank: int = 0
    dominated: bool = False
    dominates_count: int = 0
    
    def __lt__(self, other):
        """比较运算符 - 用于堆排序"""
        return self.crowding_distance > other.crowding_distance


@dataclass
class ParetoFront:
    """
    帕累托前沿
    
    Attributes:
        individuals: 非支配解集合
        hypervolume: 超体积指标
        spacing: 间距指标
        generational_distance: 世代距离
        inverted_generational_distance: 反转世代距离
    """
    individuals: List[Individual] = field(default_factory=list)
    hypervolume: float = 0.0
    spacing: float = 0.0
    generational_distance: float = 0.0
    igd: float = 0.0
    
    def add(self, ind: Individual):
        """添加个体到前沿"""
        self.individuals.append(ind)
    
    def clear(self):
        """清空前沿"""
        self.individuals = []
        self.hypervolume = 0.0
        self.spacing = 0.0


@dataclass
class OptimizationResult:
    """
    优化结果
    
    Attributes:
        pareto_front: 最终帕累托前沿
        history: 历代帕累托前沿记录
        convergence_history: 收敛历史
        final_objectives: 各目标统计
    """
    pareto_front: ParetoFront
    history: List[ParetoFront] = field(default_factory=list)
    convergence_history: Dict[str, List[float]] = field(default_factory=dict)
    final_objectives: Dict[str, Dict] = field(default_factory=dict)
    n_iterations: int = 0
    n_function_evaluations: int = 0


# ============================================================================
# 核心算法类
# ============================================================================

class PathEncoder:
    """
    路径编码器 - 将航迹编码为进化算法个体
    
    编码策略:
    - 使用关键点序列表示路径
    - 每个关键点包含 (x, y, z) 坐标
    - 支持多种编码长度
    """
    
    def __init__(self, n_keypoints: int = 10, grid_resolution: int = 50):
        """
        初始化路径编码器
        
        参数:
            n_keypoints: 关键点数量
            grid_resolution: 网格分辨率 (m)
        """
        self.n_keypoints = n_keypoints
        self.grid_resolution = grid_resolution
    
    def encode(self, path: List) -> List[Tuple[int, int, int]]:
        """
        将路径编码为关键点序列
        
        参数:
            path: 原始路径节点列表
            
        返回:
            genes: 关键点基因序列
        """
        if len(path) <= self.n_keypoints:
            return [(p.x if hasattr(p, 'x') else p[0],
                    p.y if hasattr(p, 'y') else p[1],
                    p.z if hasattr(p, 'z') else p[2]) for p in path]
        
        # 均匀采样关键点
        indices = np.linspace(0, len(path) - 1, self.n_keypoints, dtype=int)
        genes = []
        for i in indices:
            node = path[i]
            genes.append((
                int(node.x) if hasattr(node, 'x') else int(node[0]),
                int(node.y) if hasattr(node, 'y') else int(node[1]),
                int(node.z) if hasattr(node, 'z') else int(node[2]),
            ))
        return genes
    
    def decode(self, genes: List[Tuple[int, int, int]], 
               start: Tuple, goal: Tuple) -> List[Tuple[int, int, int]]:
        """
        将基因解码为完整路径
        
        参数:
            genes: 关键点基因序列
            start: 起点坐标
            goal: 终点坐标
            
        返回:
            path: 完整路径节点列表
        """
        path = [start]
        
        for gene in genes:
            # 在相邻关键点之间进行线性插值
            prev = path[-1]
            
            # 简单的直线插值 (实际应用中应使用A*或RRT*)
            dx = gene[0] - prev[0]
            dy = gene[1] - prev[1]
            dz = gene[2] - prev[2]
            
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist > self.grid_resolution * 2:
                # 需要插值
                n_steps = int(dist / self.grid_resolution)
                for step in range(1, n_steps + 1):
                    alpha = step / n_steps
                    x = int(prev[0] + dx * alpha)
                    y = int(prev[1] + dy * alpha)
                    z = int(prev[2] + dz * alpha)
                    path.append((x, y, z))
            else:
                path.append(gene)
        
        path.append(goal)
        return path


class NSGAIIOptimizer:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II)
    
    多目标进化算法，用于求解eVTOL航迹规划问题。
    
    算法特点:
    - 快速非支配排序 (O(MN²) → O(MN))
    - 拥挤距离机制保持解的多样性
    - 精英保留策略
    - 自适应停滞检测和处理机制 (新增)
    - 目标函数归一化支持 (新增)
    
    参考: Deb K, et al. IEEE TEVC, 2002 [M1]
    
    【修改说明】
    - eta_mutation: 从固定20改为可配置参数，测试范围[10, 20, 30]
    - 新增自适应停滞检测: 连续5代HV无提升时自动调整策略
    - 新增多样性指标输出: 每代计算种群基因型多样性
    """
    
    def __init__(self, 
                 n_objectives: int = 5,
                 population_size: int = 100,
                 n_generations: int = 200,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 eta_crossover: float = 20,
                 eta_mutation: float = 20,
                 random_seed: Optional[int] = None,
                 n_keypoints: int = 10):
        """
        初始化NSGA-II优化器
        
        参数:
            n_objectives: 目标函数数量
            population_size: 种群大小 (建议60以增强多样性)
            n_generations: 进化代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            eta_crossover: 模拟二进制交叉分布指数
            eta_mutation: 多项式变异分布指数 (可配置: 10, 20, 30)
            random_seed: 随机种子
            n_keypoints: 路径关键点数量
        """
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation  # 【修改】现在可配置
        self.n_keypoints = n_keypoints
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # 内部状态
        self.population: List[Individual] = []
        self.pareto_front: ParetoFront = ParetoFront()
        self.history: List[ParetoFront] = []
        self.n_evaluations = 0
        
        # 目标函数列表
        self.objectives: List[Objective] = []
        
        # 环境信息
        self.env_info: Optional[Dict] = None
        
        # 初始化策略
        self.init_strategy: str = 'diverse'  # 'random', 'guided', 'diverse'
        
        # 【新增】自适应停滞处理参数
        self.stagnation_threshold = 5  # 连续5代无提升触发自适应
        self.hv_improvement_min = 1e-3  # HV最小提升量
        self.adaptive_mutation_prob = 0.3  # 停滞时临时变异概率
        self.adaptive_reset_ratio = 0.2  # 停滞时重置20%种群
        
        # 【新增】当前使用的参数（可能因自适应调整而变化）
        self.current_mutation_prob = mutation_prob
        self.current_eta_mutation = eta_mutation
        
        # 【新增】归一化参数
        self.normalize_objectives = True
        self.objective_ranges = None  # 存储每个目标的范围
        self.initial_objective_values = None  # 存储初始目标值用于归一化
        
        # 【新增】固定参考点参数 - 确保 HV 计算可比较
        self.fixed_reference_point = None  # 固定参考点
        self.use_fixed_reference = True  # 是否使用固定参考点计算 HV
        self.objective_ranges_history = []  # 记录各代的目标范围，用于确定固定参考点
    
    def set_objectives(self, objectives: List[Objective]):
        """
        设置目标函数列表
        
        参数:
            objectives: 目标函数定义列表
        """
        self.objectives = objectives
        self.n_objectives = len(objectives)
    
    def set_environment(self, env_info: Dict):
        """
        设置环境信息
        
        参数:
            env_info: 包含起终点、障碍物等信息的字典
        """
        self.env_info = env_info
    
    def set_initial_paths(self, paths: List[List[Tuple]]):
        """
        设置初始路径（由外部规划器生成）
        
        参数:
            paths: 初始路径列表，每条路径是坐标点列表
        """
        self.initial_paths = paths
    
    def set_initialization_strategy(self, strategy: str):
        """
        设置初始化策略
        
        参数:
            strategy: 'random', 'guided', 'diverse'
        """
        self.init_strategy = strategy
    
    def set_fixed_reference_point(self, reference_point: np.ndarray):
        """
        【新增】设置固定参考点 - 确保 HV 计算跨代数可比较
        
        固定参考点应该基于问题域的先验知识设定，
        确保整个优化过程中使用相同的参考点计算 HV。
        
        参数:
            reference_point: 固定参考点数组 (长度等于目标数量)
        """
        self.fixed_reference_point = np.array(reference_point, dtype=float)
        self.use_fixed_reference = True
    
    def set_objective_bounds(self, obj_max: np.ndarray, obj_min: np.ndarray = None):
        """
        【新增】设置目标函数边界 - 用于 HV 固定参考点计算
        
        参数:
            obj_max: 各目标的最大值上限
            obj_min: 各目标的最小值下限 (可选)
        """
        self.objective_ranges = {
            'max': np.array(obj_max, dtype=float),
            'min': np.array(obj_min, dtype=float) if obj_min is not None else None
        }
    
    def optimize(self, 
                 evaluate_func: Callable[[List], np.ndarray],
                 initial_population: Optional[List[Individual]] = None,
                 callback: Optional[Callable] = None) -> OptimizationResult:
        """
        执行NSGA-II优化
        
        【修改说明】
        1. 新增自适应停滞检测和处理机制
        2. 新增多样性指标输出
        3. 新增目标函数归一化
        4. 验证变异算子生效
        
        参数:
            evaluate_func: 目标函数评估函数，输入个体基因，输出目标值数组
            initial_population: 初始种群 (可选)
            callback: 每代结束回调函数
            
        返回:
            OptimizationResult: 优化结果
        """
        print(f"\n{'='*60}")
        print(f"NSGA-II 多目标优化 (增强版)")
        print(f"{'='*60}")
        print(f"  种群大小: {self.population_size}")
        print(f"  进化代数: {self.n_generations}")
        print(f"  目标数量: {self.n_objectives}")
        print(f"  交叉概率: {self.crossover_prob}")
        print(f"  变异概率: {self.current_mutation_prob}")
        print(f"  eta_mutation: {self.current_eta_mutation}")
        print(f"  目标归一化: {self.normalize_objectives}")
        print(f"  自适应停滞处理: 开启")
        
        # 初始化种群
        if initial_population:
            self.population = list(initial_population)
            # 【修复】注入的初始种群若不足 population_size，必须用随机个体补齐，
            # 否则主循环按 population_size 索引会越界 (IndexError)。
            # 注入的只是种子路径，补齐的随机个体承担多样性。
            if len(self.population) < self.population_size:
                extra = self._initialize_population()
                need = self.population_size - len(self.population)
                self.population.extend(extra[:need])
        else:
            self.population = self._initialize_population()
        
        # 评估初始种群
        self._evaluate_population(evaluate_func)
        
        # 【新增】计算初始目标值范围，用于归一化
        self._compute_objective_ranges()
        
        # 进化主循环
        convergence_history = defaultdict(list)
        best_hv = 0.0
        stagnation_counter = 0
        stagnation_recovery_counter = 0
        is_adaptive_mode = False
        original_mutation_prob = self.current_mutation_prob
        
        # 【新增】调试信息
        mutation_effectiveness_history = []
        diversity_history = []
        
        for gen in range(self.n_generations):
            # 记录当前参数状态
            if gen == 0:
                print(f"\n  【调试】初始参数: pm={self.current_mutation_prob}, eta_m={self.current_eta_mutation}")
            
            # 计算当前种群多样性
            current_diversity = self._calculate_genotypic_diversity()
            diversity_history.append(current_diversity)
            
            # 快速非支配排序
            fronts = self._fast_non_dominated_sort()
            
            # 计算拥挤距离
            self._calculate_crowding_distance(fronts)
            
            # 【新增】创建子代并统计变异效果
            offspring, mutation_count = self._create_offspring_with_mutation_tracking()
            
            # 评估子代
            self._evaluate_population(evaluate_func, offspring)
            
            # 精英保留: 合并父代和子代
            self.population = self._elitist_selection(
                self.population + offspring
            )
            
            # 更新帕累托前沿
            fronts = self._fast_non_dominated_sort()
            self.pareto_front = ParetoFront(fronts[0] if fronts else [])
            
            # 计算性能指标
            hv = self._calculate_hypervolume(self.pareto_front)
            spacing = self._calculate_spacing(self.pareto_front)
            
            self.pareto_front.hypervolume = hv
            self.pareto_front.spacing = spacing
            
            convergence_history['hypervolume'].append(hv)
            convergence_history['spacing'].append(spacing)
            convergence_history['pareto_size'].append(len(self.pareto_front.individuals))
            convergence_history['diversity'].append(current_diversity)
            convergence_history['mutation_count'].append(mutation_count)
            
            # 【新增】自适应停滞检测
            hv_improvement = hv - best_hv
            if hv_improvement > self.hv_improvement_min:
                best_hv = hv
                stagnation_counter = 0
                stagnation_recovery_counter += 1
                
                # 【新增】如果之前处于自适应模式且连续5代有提升，恢复原参数
                if is_adaptive_mode and stagnation_recovery_counter >= 5:
                    self.current_mutation_prob = original_mutation_prob
                    is_adaptive_mode = False
                    print(f"\n  【自适应】停滞解除，恢复原参数: pm={self.current_mutation_prob}")
            else:
                stagnation_counter += 1
                stagnation_recovery_counter = 0
            
            # 【新增】停滞处理
            if stagnation_counter >= self.stagnation_threshold and not is_adaptive_mode:
                print(f"\n  【自适应】检测到停滞 (连续{stagnation_counter}代HV无提升)")
                print(f"  【自适应】临时调整: pm={self.adaptive_mutation_prob}, 重置20%种群")
                
                # 提升变异概率
                self.current_mutation_prob = self.adaptive_mutation_prob
                is_adaptive_mode = True
                stagnation_recovery_counter = 0
                
                # 重置20%种群
                self._reset_random_individuals(self.adaptive_reset_ratio)
            
            # 记录历史
            self.history.append(copy.deepcopy(self.pareto_front))
            
            # 【新增】记录变异效果
            mutation_effectiveness_history.append(mutation_count)
            
            # 打印进度
            if (gen + 1) % 5 == 0 or gen == 0:
                adaptive_marker = " [自适应]" if is_adaptive_mode else ""
                print(f"  代 {gen+1:3d}: HV={hv:.4f}, Spacing={spacing:.4f}, "
                      f"帕累托解={len(self.pareto_front.individuals)}, "
                      f"多样性={current_diversity:.4f}, "
                      f"变异生效={mutation_count}{adaptive_marker}")
            
            # 回调
            if callback:
                callback(gen, self.pareto_front, convergence_history)
            
            # 早停检查 (不再使用，因为我们已经实现了自适应停滞)
            # if stagnation_counter >= CONVERGENCE_THRESHOLDS['max_stagnation']:
            #     print(f"  检测到收敛，提前终止于第 {gen+1} 代")
            #     break
        
        # 【新增】打印汇总统计
        print(f"\n  【汇总】变异算子效果:")
        avg_mutation = np.mean(mutation_effectiveness_history)
        print(f"    平均每代变异生效个体数: {avg_mutation:.1f} / {self.population_size}")
        print(f"    多样性变化: {diversity_history[0]:.4f} -> {diversity_history[-1]:.4f}")
        
        # 最终结果
        fronts = self._fast_non_dominated_sort()
        final_front = ParetoFront(fronts[0] if fronts else [])
        final_front.hypervolume = self._calculate_hypervolume(final_front)
        final_front.spacing = self._calculate_spacing(final_front)
        
        result = OptimizationResult(
            pareto_front=final_front,
            history=self.history,
            convergence_history=dict(convergence_history),
            n_iterations=len(self.history),
            n_function_evaluations=self.n_evaluations,
        )
        
        # 【新增】添加额外信息到结果
        result.diversity_history = diversity_history
        result.mutation_effectiveness = mutation_effectiveness_history
        
        print(f"\n优化完成!")
        print(f"  总迭代次数: {len(self.history)}")
        print(f"  函数评估次数: {self.n_evaluations}")
        print(f"  最终帕累托解数量: {len(final_front.individuals)}")
        print(f"  最终超体积: {final_front.hypervolume:.4f}")
        print(f"  最终多样性: {diversity_history[-1]:.4f}")
        print(f"{'='*60}")
        
        return result
    
    def _compute_objective_ranges(self):
        """【新增】计算目标函数值范围，用于归一化和固定参考点"""
        if not self.normalize_objectives or not self.population:
            return
            
        obj_matrix = np.array([ind.objectives for ind in self.population])
        
        # 检查各目标量级差异
        obj_mins = np.min(obj_matrix, axis=0)
        obj_maxs = np.max(obj_matrix, axis=0)
        obj_ranges = obj_maxs - obj_mins
        
        self.objective_ranges = {
            'min': obj_mins,
            'max': obj_maxs,
            'range': obj_ranges
        }
        
        # 【关键修复】记录历史参考点，并在第一代时固定参考点
        if self.use_fixed_reference:
            self.objective_ranges_history.append(obj_maxs.copy())
            
            # 如果是第一代（或还没有固定参考点），设置固定参考点
            if len(self.objective_ranges_history) == 1:
                # 第一代: 使用当前最大值 * 1.2 作为固定参考点
                self.fixed_reference_point = obj_maxs * 1.2
                print(f"\n  【HV修复】第1代固定参考点已设定: {self.fixed_reference_point}")
            elif len(self.objective_ranges_history) <= 5:
                # 前5代: 取各目标的最大值作为扩展范围
                historical_max = np.max(self.objective_ranges_history, axis=0)
                self.fixed_reference_point = historical_max * 1.2
        
        # 检查是否存在超过100倍的量级差异
        if len(obj_ranges) > 1:
            max_ratio = np.max(obj_maxs) / (np.min(obj_mins[np.min(obj_mins) > 0]) + 1e-10)
            if max_ratio > 100:
                print(f"\n  【归一化】检测到目标量级差异 (最大比例: {max_ratio:.1f}x), 将启用归一化")
    
    def _calculate_genotypic_diversity(self) -> float:
        """【新增】计算种群基因型多样性指标
        
        使用个体间基因的平均欧氏距离来衡量多样性
        """
        if len(self.population) < 2:
            return 0.0
        
        # 将基因转换为数值矩阵
        genes_matrix = []
        for ind in self.population:
            # 将每个基因(x,y,z)展平
            flat_genes = []
            for gene in ind.genes:
                flat_genes.extend([gene[0], gene[1], gene[2]])
            genes_matrix.append(flat_genes)
        
        genes_matrix = np.array(genes_matrix)
        
        # 计算两两之间的距离
        n = len(genes_matrix)
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((genes_matrix[i] - genes_matrix[j]) ** 2))
                total_dist += dist
                count += 1
        
        avg_distance = total_dist / count if count > 0 else 0.0
        
        # 归一化 (相对于坐标范围)
        if self.env_info:
            bounds = self.env_info.get('bounds', ((0, 3000), (0, 3000), (30, 300)))
            max_range = max(bounds[0][1] - bounds[0][0],
                          bounds[1][1] - bounds[1][0],
                          bounds[2][1] - bounds[2][0])
            return avg_distance / max_range
        
        return avg_distance
    
    def _reset_random_individuals(self, ratio: float):
        """【新增】随机重置部分种群个体
        
        【修复】确保重置后多样性增加，使用高度多样化的初始化
        """
        if not self.env_info:
            return
        
        n_reset = int(len(self.population) * ratio)
        indices = random.sample(range(len(self.population)), n_reset)
        
        start = self.env_info['start']
        goal = self.env_info['goal']
        bounds = self.env_info.get('bounds', ((0, 3000), (0, 3000), (30, 300)))
        
        for idx in indices:
            # 【修改】使用高度多样化的初始化，而非简单随机
            genes = self._generate_highly_diverse_genes(start, goal, bounds)
            self.population[idx].genes = genes
            self.population[idx].objectives = np.array([])  # 标记需要重新评估
            self.population[idx].rank = float('inf')  # 降低优先级
    
    def _initialize_population(self) -> List[Individual]:
        """
        初始化种群 - 支持多种策略
        
        【修改说明】减少初始路径依赖，增强初始种群多样性
        - 原有策略：使用5条外部初始路径，其余随机
        - 新策略：仅使用1条最优初始路径，其余29条改为随机初始化+轻度扰动
        
        改进策略:
        1. 如果有外部初始路径，只使用第一条（最优路径）
        2. 其余个体使用随机初始化 + 轻度扰动
        3. 结合多种生成方式增加种群多样性
        """
        population = []
        
        # 【修改】只使用1条最优初始路径，而非5条
        if hasattr(self, 'initial_paths') and self.initial_paths:
            # 只使用第一条最优路径
            best_path = self.initial_paths[0]
            genes = self._path_to_genes(best_path)
            population.append(Individual(genes=genes))
            print(f"  [初始化] 使用 1 条最优初始路径")
            
            # 如果有多条路径，对第二条进行轻度扰动后作为第二个个体
            if len(self.initial_paths) >= 2:
                perturbed_path = self._perturb_path(self.initial_paths[1], perturbation_ratio=0.05)
                genes = self._path_to_genes(perturbed_path)
                population.append(Individual(genes=genes))
                print(f"  [初始化] 对第2条路径进行5%扰动")
        
        # 计算还需要多少个体
        remaining = self.population_size - len(population)
        
        if remaining > 0 and self.env_info:
            start = self.env_info['start']
            goal = self.env_info['goal']
            bounds = self.env_info.get('bounds', ((0, 3000), (0, 3000), (0, 300)))
            
            if self.init_strategy == 'diverse':
                # 【增强】多样性初始化策略 - 增强版
                population.extend(self._generate_enhanced_diverse_population(start, goal, bounds, remaining))
            elif self.init_strategy == 'guided':
                for _ in range(remaining):
                    genes = self._generate_guided_genes(start, goal, bounds)
                    population.append(Individual(genes=genes))
            else:  # random
                for _ in range(remaining):
                    genes = self._generate_random_genes(bounds)
                    population.append(Individual(genes=genes))
        
        # 如果仍然不足，补充随机个体
        while len(population) < self.population_size:
            genes = self._generate_random_genes(bounds if self.env_info else None)
            population.append(Individual(genes=genes))
        
        print(f"  [初始化] 总共生成 {len(population)} 个体 (减少初始路径依赖)")
        return population
    
    def _perturb_path(self, path: List, perturbation_ratio: float = 0.05) -> List:
        """【新增】对路径进行轻度扰动
        
        参数:
            path: 原始路径
            perturbation_ratio: 扰动比例 (决策变量范围的5-10%)
        """
        if not path or len(path) == 0:
            return path
        
        perturbed = []
        for point in path:
            if hasattr(point, 'x'):
                x, y, z = point.x, point.y, point.z
            elif isinstance(point, (list, tuple)) and len(point) >= 3:
                x, y, z = point[0], point[1], point[2]
            else:
                continue
            
            # 扰动幅度为坐标范围的5%
            dx = int(x * perturbation_ratio * random.uniform(-1, 1))
            dy = int(y * perturbation_ratio * random.uniform(-1, 1))
            dz = int(z * perturbation_ratio * random.uniform(-1, 1))
            
            perturbed.append((x + dx, y + dy, z + dz))
        
        return perturbed
    
    def _generate_enhanced_diverse_population(self, start: Tuple, goal: Tuple,
                                               bounds: Tuple, n: int) -> List[Individual]:
        """
        【增强】生成高度多样性的种群
        
        策略:
        1. 直线解 (最短路径)
        2. 多种绕行解
        3. 不同高度策略
        4. S形曲线解
        5. 锯齿形解
        6. 高度随机化解 (新增)
        7. 大范围随机解 (新增)
        """
        population = []
        n_keypoints = self.n_keypoints
        
        # 1. 直线解 (最短路径)
        genes_straight = self._generate_straight_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_straight))
        
        # 2. 不同绕行策略
        offsets = [
            (200, 200, 0),   # 右上绕行
            (-200, 200, 0),  # 左上绕行
            (200, -200, 0),  # 右下绕行
            (-200, -200, 0), # 左下绕行
        ]
        
        for offset in offsets:
            detour_goal = (goal[0] + offset[0], goal[1] + offset[1], goal[2])
            detour_goal = self._clamp_point(detour_goal, bounds)
            genes = self._generate_straight_path(start, detour_goal, n_keypoints)
            population.append(Individual(genes=genes))
        
        # 3. 不同高度策略
        heights = [50, 100, 150, 200, 250]
        for h in heights[:3]:  # 取3个高度
            mid_goal = (goal[0], goal[1], h)
            genes = self._generate_vertical_strategy(start, mid_goal, n_keypoints)
            population.append(Individual(genes=genes))
        
        # 4. S形曲线解
        genes_s = self._generate_s_curve_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_s))
        
        # 5. 锯齿形解
        genes_zigzag = self._generate_zigzag_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_zigzag))
        
        # 【新增】6. 高度随机化解 - 大幅变化高度以增加多样性
        genes_high_var = self._generate_high_variation_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_high_var))
        
        # 【新增】7. 螺旋式解
        genes_spiral = self._generate_spiral_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_spiral))
        
        # 填充高度随机解
        while len(population) < n:
            genes = self._generate_highly_diverse_genes(start, goal, bounds)
            population.append(Individual(genes=genes))
        
        return population[:n]
    
    def _generate_high_variation_path(self, start: Tuple, goal: Tuple,
                                       n_keypoints: int) -> List[Tuple[int, int, int]]:
        """【新增】生成高度变化的路径"""
        genes = []
        for i in range(n_keypoints):
            alpha = (i + 1) / n_keypoints
            x = int(start[0] + (goal[0] - start[0]) * alpha)
            y = int(start[1] + (goal[1] - start[1]) * alpha)
            # 高度大幅变化
            z = int(30 + 270 * (np.sin(alpha * 3 * np.pi) + 1) / 2)
            genes.append((x, y, z))
        return genes
    
    def _generate_spiral_path(self, start: Tuple, goal: Tuple,
                              n_keypoints: int) -> List[Tuple[int, int, int]]:
        """【新增】生成螺旋式路径"""
        genes = []
        for i in range(n_keypoints):
            alpha = (i + 1) / n_keypoints
            # 螺旋半径
            radius = 200 * alpha
            angle = alpha * 2 * np.pi
            
            x = int(start[0] + (goal[0] - start[0]) * alpha + radius * np.cos(angle))
            y = int(start[1] + (goal[1] - start[1]) * alpha + radius * np.sin(angle))
            z = int(start[2] + (goal[2] - start[2]) * alpha)
            genes.append((x, y, z))
        return genes
    
    def _path_to_genes(self, path: List) -> List[Tuple[int, int, int]]:
        """将路径转换为固定长度的基因序列"""
        n = len(path)
        target_len = self.n_keypoints
        
        if n == 0:
            # 空路径，生成默认基因
            genes = []
            for i in range(target_len):
                genes.append((0, 0, 100))
            return genes
        
        # 提取路径点
        points = []
        for p in path:
            if hasattr(p, 'x'):
                points.append((p.x, p.y, p.z))
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                points.append((p[0], p[1], p[2]))
            else:
                points.append(p)
        
        if len(points) <= target_len:
            # 路径太短，复制中间点以达到目标长度
            genes = []
            for i in range(target_len):
                idx = min(i * (len(points) - 1) // max(1, target_len - 1), len(points) - 1)
                pt = points[idx]
                genes.append((int(pt[0]), int(pt[1]), int(pt[2])))
            return genes
        else:
            # 均匀采样关键点
            indices = np.linspace(0, len(points) - 1, target_len, dtype=int)
            genes = []
            for i in indices:
                pt = points[i]
                genes.append((int(pt[0]), int(pt[1]), int(pt[2])))
            return genes
    
    def _generate_diverse_population(self, start: Tuple, goal: Tuple,
                                      bounds: Tuple, n: int) -> List[Individual]:
        """
        生成多样性种群 - 覆盖不同偏好方向
        
        策略:
        1. 均匀采样生成直线解
        2. 绕行解(增加距离但可能更安全/更低噪声)
        3. 高度变化解(不同巡航高度策略)
        4. 极端随机解(探索边界)
        """
        population = []
        n_keypoints = self.n_keypoints
        
        # 1. 直线解 (最短路径)
        genes_straight = self._generate_straight_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_straight))
        
        # 2. 不同绕行策略
        offsets = [
            (200, 200, 0),   # 右上绕行
            (-200, 200, 0),  # 左上绕行
            (200, -200, 0),  # 右下绕行
            (-200, -200, 0), # 左下绕行
        ]
        
        for offset in offsets:
            detour_goal = (goal[0] + offset[0], goal[1] + offset[1], goal[2])
            detour_goal = self._clamp_point(detour_goal, bounds)
            genes = self._generate_straight_path(start, detour_goal, n_keypoints)
            population.append(Individual(genes=genes))
        
        # 3. 不同高度策略
        heights = [50, 100, 150, 200, 250]
        for h in heights[:3]:  # 取3个高度
            mid_goal = (goal[0], goal[1], h)
            genes = self._generate_vertical_strategy(start, mid_goal, n_keypoints)
            population.append(Individual(genes=genes))
        
        # 4. S形曲线解
        genes_s = self._generate_s_curve_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_s))
        
        # 5. 锯齿形解
        genes_zigzag = self._generate_zigzag_path(start, goal, n_keypoints)
        population.append(Individual(genes=genes_zigzag))
        
        # 6. 填充随机解
        while len(population) < n:
            genes = self._generate_highly_diverse_genes(start, goal, bounds)
            population.append(Individual(genes=genes))
        
        return population[:n]
    
    def _generate_straight_path(self, start: Tuple, goal: Tuple, 
                                n_keypoints: int) -> List[Tuple[int, int, int]]:
        """生成直线路径（不包括起点和终点）"""
        genes = []
        for i in range(n_keypoints):
            alpha = (i + 1) / n_keypoints  # 从0到1的比例
            x = int(start[0] + (goal[0] - start[0]) * alpha)
            y = int(start[1] + (goal[1] - start[1]) * alpha)
            z = int(start[2] + (goal[2] - start[2]) * alpha)
            genes.append((x, y, z))
        return genes
    
    def _generate_vertical_strategy(self, start: Tuple, goal: Tuple,
                                    n_keypoints: int) -> List[Tuple[int, int, int]]:
        """生成垂直爬升策略路径"""
        genes = []
        for i in range(n_keypoints):
            alpha = (i + 1) / n_keypoints
            # 前30%爬升，后70%巡航
            if alpha < 0.3:
                z_ratio = alpha / 0.3
                z = int(start[2] + (goal[2] - start[2]) * z_ratio)
            else:
                z = goal[2]
            
            x = int(start[0] + (goal[0] - start[0]) * alpha)
            y = int(start[1] + (goal[1] - start[1]) * alpha)
            genes.append((x, y, z))
        return genes
    
    def _generate_s_curve_path(self, start: Tuple, goal: Tuple,
                               n_keypoints: int) -> List[Tuple[int, int, int]]:
        """生成S形曲线路径"""
        genes = []
        for i in range(n_keypoints):
            alpha = (i + 1) / n_keypoints
            # S曲线偏移
            dx = 300 * np.sin(alpha * 2 * np.pi)
            x = int(start[0] + (goal[0] - start[0]) * alpha + dx)
            y = int(start[1] + (goal[1] - start[1]) * alpha)
            z = int(start[2] + (goal[2] - start[2]) * alpha + 50 * np.sin(alpha * np.pi))
            genes.append((x, y, z))
        return genes
    
    def _generate_zigzag_path(self, start: Tuple, goal: Tuple,
                              n_keypoints: int) -> List[Tuple[int, int, int]]:
        """生成锯齿形路径"""
        genes = []
        dx = (goal[0] - start[0]) / n_keypoints
        dy = (goal[1] - start[1]) / n_keypoints
        
        for i in range(n_keypoints):
            # 交替左右偏移
            offset = 200 if i % 2 == 0 else -200
            x = int(start[0] + dx * (i + 1))
            y = int(start[1] + dy * (i + 1) + offset)
            z = int(start[2] + (goal[2] - start[2]) * (i + 1) / n_keypoints)
            genes.append((x, y, z))
        return genes
    
    def _generate_highly_diverse_genes(self, start: Tuple, goal: Tuple,
                                       bounds: Tuple) -> List[Tuple[int, int, int]]:
        """生成高度多样化的随机基因"""
        genes = []
        for _ in range(self.n_keypoints):
            # 更大的扰动范围
            perturbation_x = random.uniform(-400, 400)
            perturbation_y = random.uniform(-400, 400)
            perturbation_z = random.uniform(-100, 100)
            
            # 在目标方向上生成点
            alpha = random.uniform(0.1, 0.9)
            x = int(start[0] + (goal[0] - start[0]) * alpha + perturbation_x)
            y = int(start[1] + (goal[1] - start[1]) * alpha + perturbation_y)
            z = int(start[2] + (goal[2] - start[2]) * alpha + perturbation_z)
            
            x, y, z = self._clamp_point((x, y, z), bounds)
            genes.append((x, y, z))
        
        return genes
    
    def _clamp_point(self, point: Tuple, bounds: Tuple) -> Tuple:
        """将点限制在边界内"""
        x = max(bounds[0][0], min(point[0], bounds[0][1]))
        y = max(bounds[1][0], min(point[1], bounds[1][1]))
        z = max(bounds[2][0], min(point[2], bounds[2][1]))
        return (x, y, z)
    
    def _generate_random_genes(self, bounds: Optional[Tuple] = None) -> List[Tuple[int, int, int]]:
        """生成随机基因序列"""
        if bounds is None:
            bounds = ((0, 3000), (0, 3000), (0, 300))
        
        genes = []
        for _ in range(self.n_keypoints):
            x = random.randint(bounds[0][0], bounds[0][1])
            y = random.randint(bounds[1][0], bounds[1][1])
            z = random.randint(bounds[2][0], bounds[2][1])
            genes.append((x, y, z))
        return genes
    
    def _generate_guided_genes(self, start: Tuple, goal: Tuple,
                               bounds: Tuple) -> List[Tuple[int, int, int]]:
        """生成有引导的基因序列 - 国标飞行剖面"""
        genes = [start]
        
        # 国标巡航高度配置
        cruise_min = 120.0
        cruise_max = 180.0
        cruise_height = cruise_min + (cruise_max - cruise_min) * 0.7  # 162m
        
        # 插值中间点
        n_interp = 8
        for i in range(1, n_interp):
            alpha = i / n_interp
            # 添加随机扰动
            perturbation = random.uniform(-200, 200)
            x = int(start[0] + (goal[0] - start[0]) * alpha + perturbation)
            y = int(start[1] + (goal[1] - start[1]) * alpha + perturbation)
            
            # 国标爬升-巡航-下降剖面
            if alpha < 0.2:
                # 爬升阶段: 80m -> cruise_height
                z = int(start[2] + (cruise_height - start[2]) * (alpha / 0.2))
            elif alpha < 0.8:
                # 巡航阶段: 保持cruise_height
                z = int(cruise_height)
            else:
                # 下降阶段: cruise_height -> goal[2]
                descend_alpha = (alpha - 0.8) / 0.2
                z = int(cruise_height - (cruise_height - goal[2]) * descend_alpha)
            
            # 限制在边界内
            x = max(bounds[0][0], min(x, bounds[0][1]))
            y = max(bounds[1][0], min(y, bounds[1][1]))
            z = max(bounds[2][0], min(z, bounds[2][1]))
            
            genes.append((x, y, z))
        
        genes.append(goal)
        return genes
    
    def _evaluate_population(self, evaluate_func: Callable,
                            population: Optional[List[Individual]] = None):
        """评估种群目标函数值"""
        pop = population if population else self.population
        
        for ind in pop:
            if len(ind.objectives) == 0:
                ind.objectives = evaluate_func(ind.genes)
                self.n_evaluations += 1
    
    def _fast_non_dominated_sort(self) -> List[List[Individual]]:
        """
        快速非支配排序 - O(MN²) → O(MN log N)
        
        参考: Deb(2002) Algorithm 1
        """
        fronts = [[]]
        
        # 初始化
        for p in self.population:
            p.dominated = False
            p.dominates_count = 0
            p.rank = 0
        
        # 遍历所有个体对
        for p in self.population:
            for q in self.population:
                if p is q:
                    continue
                
                # 检查支配关系
                if self._dominates(p, q):
                    p.dominates_count += 1
                elif self._dominates(q, p):
                    p.dominated = True
        
        # 第一个前沿: 非支配解
        for p in self.population:
            if not p.dominated:
                p.rank = 1
                fronts[0].append(p)
        
        # 后续前沿
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in self.population:
                    if q.rank == 0 and p.dominates_count > 0:
                        q.dominates_count -= 1
                        if q.dominates_count == 0:
                            q.rank = i + 2
                            next_front.append(q)
            fronts.append(next_front)
            i += 1
        
        return [f for f in fronts if f]  # 移除空前沿
    
    def _dominates(self, p: Individual, q: Individual) -> bool:
        """
        检查个体p是否支配个体q
        
        支配定义: p在所有目标上不差于q，且至少在一个目标上严格优于q
        """
        if len(p.objectives) == 0 or len(q.objectives) == 0:
            return False
        
        p_dominates_q = False
        
        for i in range(len(p.objectives)):
            obj_type = self.objectives[i].objective_type if self.objectives else ObjectiveType.MINIMIZE
            
            if obj_type == ObjectiveType.MINIMIZE:
                if p.objectives[i] > q.objectives[i]:
                    return False
                if p.objectives[i] < q.objectives[i]:
                    p_dominates_q = True
            else:  # MAXIMIZE
                if p.objectives[i] < q.objectives[i]:
                    return False
                if p.objectives[i] > q.objectives[i]:
                    p_dominates_q = True
        
        return p_dominates_q
    
    def _calculate_crowding_distance(self, fronts: List[List[Individual]]):
        """
        计算拥挤距离 - 保持解的多样性
        
        参考: Deb(2002) Algorithm 2
        
        【修复】增加空解检查，防止IndexError
        """
        for front in fronts:
            if not front:
                continue
            
            # 检查所有个体的目标值是否有效
            valid_front = [ind for ind in front if ind.objectives is not None and len(ind.objectives) > 0]
            if len(valid_front) < 2:
                # 如果有效个体不足，跳过拥挤距离计算
                for ind in front:
                    ind.crowding_distance = float('inf')
                continue
            
            n = len(valid_front)
            
            # 初始化
            for p in valid_front:
                p.crowding_distance = 0.0
            
            # 对每个目标计算
            for m in range(self.n_objectives):
                obj_type = self.objectives[m].objective_type if self.objectives else ObjectiveType.MINIMIZE
                
                # 按目标值排序
                if obj_type == ObjectiveType.MINIMIZE:
                    sorted_front = sorted(valid_front, key=lambda x: x.objectives[m])
                else:
                    sorted_front = sorted(valid_front, key=lambda x: x.objectives[m], reverse=True)
                
                # 边界解给予无穷大距离
                sorted_front[0].crowding_distance = float('inf')
                sorted_front[-1].crowding_distance = float('inf')
                
                # 计算其他解的距离
                obj_range = sorted_front[-1].objectives[m] - sorted_front[0].objectives[m]
                if obj_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    distance = (sorted_front[i + 1].objectives[m] - 
                               sorted_front[i - 1].objectives[m]) / obj_range
                    sorted_front[i].crowding_distance += distance
    
    def _create_offspring(self) -> List[Individual]:
        """创建子代个体"""
        offspring, _ = self._create_offspring_with_mutation_tracking()
        return offspring
    
    def _create_offspring_with_mutation_tracking(self) -> Tuple[List[Individual], int]:
        """【新增】创建子代个体并跟踪变异效果
        
        返回:
            Tuple: (子代列表, 变异生效的个体数量)
        """
        offspring = []
        mutation_affected_count = 0
        
        while len(offspring) < self.population_size:
            # 二元锦标赛选择
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交叉
            if random.random() < self.crossover_prob:
                child1_genes, child2_genes = self._sbx_crossover(
                    parent1.genes, parent2.genes
                )
            else:
                child1_genes = copy.deepcopy(parent1.genes)
                child2_genes = copy.deepcopy(parent2.genes)
            
            # 变异 - 使用当前变异概率(可能因自适应调整而变化)
            child1_genes, affected1 = self._polynomial_mutation_with_tracking(child1_genes)
            child2_genes, affected2 = self._polynomial_mutation_with_tracking(child2_genes)
            
            mutation_affected_count += affected1 + affected2
            
            offspring.append(Individual(genes=child1_genes))
            if len(offspring) < self.population_size:
                offspring.append(Individual(genes=child2_genes))
        
        return offspring[:self.population_size], mutation_affected_count
    
    def _tournament_selection(self) -> Individual:
        """二元锦标赛选择"""
        tournament_size = 2
        candidates = random.sample(self.population, tournament_size)
        
        # 选择帕累托秩小(优)的，如果相同则选择拥挤距离大(好)的
        best = min(candidates, key=lambda x: (x.rank, -x.crowding_distance))
        return best
    
    def _sbx_crossover(self, parent1: List, parent2: List
                      ) -> Tuple[List, List]:
        """
        模拟二进制交叉 (SBX)
        
        参考: Deb(2002) Algorithm 3
        """
        n_points = len(parent1)
        child1 = []
        child2 = []
        
        # 随机选择交叉点
        crossover_point = random.randint(0, n_points - 1)
        
        for i in range(n_points):
            if i <= crossover_point:
                child1.append(copy.deepcopy(parent1[i]))
                child2.append(copy.deepcopy(parent2[i]))
            else:
                child1.append(copy.deepcopy(parent2[i]))
                child2.append(copy.deepcopy(parent1[i]))
        
        return child1, child2
    
    def _polynomial_mutation(self, genes: List) -> List:
        """多项式变异 - 使用当前变异概率"""
        mutated, _ = self._polynomial_mutation_with_tracking(genes)
        return mutated
    
    def _polynomial_mutation_with_tracking(self, genes: List) -> Tuple[List, int]:
        """
        【新增】多项式变异 - 带变异效果跟踪
        
        【修改说明】
        - 使用 self.current_mutation_prob 而非固定的 self.mutation_prob
        - 使用 self.current_eta_mutation 而非固定的 self.eta_mutation
        - 返回变异是否在个体层面生效的标志
        - 【修复】对变异的X坐标进行修正，确保路径单调递增
        
        参考: Deb(2002) Algorithm 4
        """
        mutated = []
        mutation_affected = False
        
        # 第一遍：正常变异
        for gene in genes:
            x, y, z = gene
            orig_x, orig_y, orig_z = x, y, z
            
            # 对每个维度进行变异 - 使用当前eta_mutation
            new_x = self._mutate_dimension(x, 0, 3000, self.current_mutation_prob, self.current_eta_mutation)
            new_y = self._mutate_dimension(y, 0, 3000, self.current_mutation_prob, self.current_eta_mutation)
            new_z = self._mutate_dimension(z, 30, 300, self.current_mutation_prob, self.current_eta_mutation)
            
            # 检查是否有任何维度发生变化
            if new_x != orig_x or new_y != orig_y or new_z != orig_z:
                mutation_affected = True
            
            mutated.append((new_x, new_y, new_z))
        
        # 【修复】第二遍：修正X坐标，确保单调递增
        # X坐标应在[前一个X+最小间隔, 下一个X-最小间隔]范围内
        min_spacing = 100  # 最小节点间距
        prev_x = 0  # 起点X（假设起点在0附近，实际会从start_pos开始）
        
        for i in range(len(mutated)):
            x, y, z = mutated[i]
            
            # 确保X不小于前一个X + 最小间距
            if x < prev_x + min_spacing:
                x = prev_x + min_spacing
            
            # 确保X不超过3000 - 剩余空间
            remaining = 3000 - (len(mutated) - i - 1) * min_spacing
            if x > remaining:
                x = remaining
            
            mutated[i] = (x, y, z)
            prev_x = x
        
        return mutated, int(mutation_affected)
    
    def _mutate_dimension(self, x: int, low: int, high: int, 
                          mutation_prob: float = None, 
                          eta_mutation: float = None) -> int:
        """
        对单维度进行多项式变异
        
        【修改说明】允许动态传入 mutation_prob 和 eta_mutation 参数
        """
        if mutation_prob is None:
            mutation_prob = self.current_mutation_prob
        if eta_mutation is None:
            eta_mutation = self.current_eta_mutation
            
        if random.random() < mutation_prob:
            delta_l = (x - low) / (high - low) if high > low else 0
            delta_r = (high - x) / (high - low) if high > low else 0
            
            u = random.random()
            
            if u < 0.5:
                delta_q = (2 * u + (1 - 2 * u) * 
                          ((1 - delta_l) ** (eta_mutation + 1))) ** \
                          (1 / (eta_mutation + 1)) - 1
                x_new = x + delta_q * (high - low)
            else:
                delta_q = (2 * (1 - u) + 2 * (u - 0.5) * 
                          ((1 - delta_r) ** (eta_mutation + 1))) ** \
                          (1 / (eta_mutation + 1)) - 1
                x_new = x + delta_q * (high - low)
            
            x_new = max(low, min(x_new, high))
            return int(x_new)
        
        return x
    
    def _elitist_selection(self, combined: List[Individual]) -> List[Individual]:
        """
        精英保留选择
        
        从父代和子代的合并种群中选择下一代种群
        """
        # 快速非支配排序
        fronts = self._fast_non_dominated_sort()
        
        # 计算拥挤距离
        self._calculate_crowding_distance(fronts)
        
        # 选择帕累托秩小且拥挤距离大的个体
        new_population = []
        
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # 需要选择部分个体
                remaining = self.population_size - len(new_population)
                sorted_front = sorted(front, key=lambda x: -x.crowding_distance)
                new_population.extend(sorted_front[:remaining])
                break
        
        return new_population
    
    def _calculate_hypervolume(self, front: ParetoFront,
                               reference_point: Optional[np.ndarray] = None
                               ) -> float:
        """
        计算超体积指标 (Hypervolume Indicator)
        
        超体积越大，表示帕累托前沿越好地覆盖目标空间。
        
        参考: Zitzler & Thrun (1998)
        
        【修复】使用固定参考点计算 HV，确保跨代数可比性
        【修复】增加目标值有效性检查
        
        参数:
            front: 帕累托前沿
            reference_point: 参考点 (默认使用固定参考点)
            
        返回:
            hv: 超体积值
        """
        if not front.individuals:
            return 0.0
        
        # 【修复】获取有效个体的目标值
        valid_individuals = []
        for ind in front.individuals:
            if ind.objectives is not None and len(ind.objectives) > 0:
                # 确保所有目标值都是有效数字
                if not np.any(np.isnan(ind.objectives)) and not np.any(np.isinf(ind.objectives)):
                    valid_individuals.append(ind)
        
        if not valid_individuals:
            return 0.0
        
        # 获取目标值矩阵
        obj_matrix = np.array([ind.objectives for ind in valid_individuals])
        
        # 检查矩阵是否有效
        if obj_matrix.size == 0 or len(obj_matrix.shape) != 2:
            return 0.0
        
        if reference_point is None:
            # 【关键修复】使用固定参考点而非动态参考点
            # 优先级: 1) 预设固定参考点 2) 预设目标范围 3) 第一代计算的范围
            if self.fixed_reference_point is not None:
                # 使用预设的固定参考点
                reference_point = self.fixed_reference_point
            elif self.objective_ranges is not None and self.objective_ranges.get('max') is not None:
                # 使用预设的目标范围
                reference_point = self.objective_ranges['max'] * 1.2
            elif self.use_fixed_reference and len(self.objective_ranges_history or []) > 0:
                # 使用历史上第一个参考点作为固定参考点
                reference_point = self.objective_ranges_history[0] * 1.2
            else:
                # 最后手段: 使用当前 Pareto 前沿计算但标记为动态
                reference_point = np.max(obj_matrix, axis=0) * 1.2
        
        # 简单的二维/三维超体积计算
        n = len(valid_individuals)
        if n == 1:
            # 单点超体积
            return np.prod(reference_point - valid_individuals[0].objectives)
        
        # 对于高维问题，使用近似方法
        try:
            # 排序并计算
            hv = 0.0
            sorted_inds = sorted(valid_individuals, 
                                key=lambda x: x.objectives[0])
            
            if self.n_objectives == 2:
                # 二维精确计算
                prev_obj = 0.0
                for ind in sorted_inds:
                    width = reference_point[0] - ind.objectives[0]
                    height = reference_point[1] - ind.objectives[1]
                    hv += width * height
                    prev_obj = ind.objectives[0]
            else:
                # 高维近似 (使用蒙特卡洛)
                hv = self._hv_monte_carlo(obj_matrix, reference_point, n_samples=10000)
            
            return max(0, hv)
            
        except Exception as e:
            print(f"    [警告] 超体积计算异常: {e}")
            return 0.0
    
    def _hv_monte_carlo(self, points: np.ndarray, ref: np.ndarray,
                       n_samples: int = 10000) -> float:
        """蒙特卡洛方法计算超体积"""
        # 生成随机点
        samples = np.random.uniform(
            low=np.min(points, axis=0),
            high=ref,
            size=(n_samples, points.shape[1])
        )
        
        # 计算有多少点在帕累托前沿下方
        dominated_count = 0
        for sample in samples:
            # 检查是否被某个点支配
            is_dominated = False
            for point in points:
                if np.all(point <= sample) and np.any(point < sample):
                    is_dominated = True
                    break
            if is_dominated:
                dominated_count += 1
        
        # 超体积估计
        box_volume = np.prod(ref - np.min(points, axis=0))
        return box_volume * dominated_count / n_samples
    
    def _calculate_spacing(self, front: ParetoFront) -> float:
        """
        计算间距指标 (Spacing Metric)
        
        间距越小，表示帕累托解分布越均匀。
        
        参考: Schott (1995)
        
        【修复】增加目标值有效性检查
        
        参数:
            front: 帕累托前沿
            
        返回:
            spacing: 间距指标
        """
        # 【修复】获取有效个体
        valid_individuals = []
        for ind in front.individuals:
            if ind.objectives is not None and len(ind.objectives) > 0:
                if not np.any(np.isnan(ind.objectives)) and not np.any(np.isinf(ind.objectives)):
                    valid_individuals.append(ind)
        
        if len(valid_individuals) < 2:
            return 0.0
        
        # 计算相邻解之间的欧氏距离
        try:
            obj_matrix = np.array([ind.objectives for ind in valid_individuals])
            if len(obj_matrix.shape) != 2:
                return 0.0
            n = len(obj_matrix)
        except Exception:
            return 0.0
        
        distances = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(obj_matrix[i] - obj_matrix[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        # 计算间距
        mean_dist = np.mean(distances)
        if mean_dist == 0:
            return 0.0
        
        variance = np.sum([(d - mean_dist) ** 2 for d in distances]) / n
        return np.sqrt(variance)
    
    def _calculate_generational_distance(self, front: ParetoFront,
                                        reference_front: ParetoFront
                                        ) -> float:
        """
        计算世代距离 (Generational Distance)
        
        衡量帕累托前沿与参考前沿的接近程度。
        
        参考: Van Veldhuizen & Lamont (1998)
        """
        if not front.individuals or not reference_front.individuals:
            return float('inf')
        
        # 计算每个帕累托解到最近参考解的距离
        front_objs = np.array([ind.objectives for ind in front.individuals])
        ref_objs = np.array([ind.objectives for ind in reference_front.individuals])
        
        distances = []
        for p in front_objs:
            min_dist = min(np.linalg.norm(p - q) for q in ref_objs)
            distances.append(min_dist)
        
        # GD = (1/n) * sqrt(sum(d_i^2))
        return np.sqrt(np.sum(np.array(distances) ** 2)) / len(front.individuals)


class MOEADOptimizer:
    """
    MOEA/D (Multiobjective Evolutionary Algorithm based on Decomposition)
    
    基于分解的多目标进化算法。
    
    算法特点:
    - 将多目标问题分解为多个标量子问题
    - 使用权重向量均匀分布
    - 邻居关系促进信息交换
    
    参考: Zhang & Li (2007) [M2]
    """
    
    def __init__(self,
                 n_objectives: int = 5,
                 population_size: int = 100,
                 n_generations: int = 200,
                 neighborhood_size: int = 20,
                 random_seed: Optional[int] = None):
        """
        初始化MOEA/D优化器
        """
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.neighborhood_size = neighborhood_size
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # 内部状态
        self.population: List[Individual] = []
        self.weights: np.ndarray = None
        self.neighborhoods: Dict[int, List[int]] = {}
        self.pareto_front: ParetoFront = ParetoFront()
        self.n_evaluations = 0
        
        # 目标函数列表
        self.objectives: List[Objective] = []
        
        # 环境信息
        self.env_info: Optional[Dict] = None
    
    def set_objectives(self, objectives: List[Objective]):
        """设置目标函数"""
        self.objectives = objectives
    
    def set_environment(self, env_info: Dict):
        """设置环境信息"""
        self.env_info = env_info
    
    def optimize(self,
                evaluate_func: Callable[[List], np.ndarray],
                initial_population: Optional[List[Individual]] = None,
                callback: Optional[Callable] = None) -> OptimizationResult:
        """
        执行MOEA/D优化
        """
        print(f"\n{'='*60}")
        print(f"MOEA/D 多目标优化 (基于分解)")
        print(f"{'='*60}")
        print(f"  种群大小: {self.population_size}")
        print(f"  进化代数: {self.n_generations}")
        print(f"  邻居大小: {self.neighborhood_size}")
        
        # 生成权重向量
        self._generate_weight_vectors()
        
        # 计算邻居关系
        self._compute_neighborhoods()
        
        # 初始化种群
        if initial_population:
            self.population = list(initial_population)
            # 【修复】注入的初始种群若不足 population_size，必须用随机个体补齐，
            # 否则主循环按 population_size 索引会越界 (IndexError)。
            # 注入的只是种子路径，补齐的随机个体承担多样性。
            if len(self.population) < self.population_size:
                extra = self._initialize_population()
                need = self.population_size - len(self.population)
                self.population.extend(extra[:need])
        else:
            self.population = self._initialize_population()
        
        # 评估种群
        for ind in self.population:
            if len(ind.objectives) == 0:
                ind.objectives = evaluate_func(ind.genes)
                self.n_evaluations += 1
        
        # 进化主循环
        convergence_history = defaultdict(list)
        
        for gen in range(self.n_generations):
            # 对每个子问题进行优化
            for i in range(self.population_size):
                # 从邻居中选择
                neighbors = self.neighborhoods[i]
                if len(neighbors) < 2:
                    continue
                
                # 随机选择两个邻居
                k, l = random.sample(neighbors, 2)
                
                # 生成新个体 (差分进化风格)
                new_genes = self._differential_evolution(
                    self.population[i].genes,
                    self.population[k].genes,
                    self.population[l].genes
                )
                
                new_ind = Individual(genes=new_genes)
                new_ind.objectives = evaluate_func(new_genes)
                self.n_evaluations += 1
                
                # 更新邻居
                self._update_neighbors(i, new_ind, neighbors)
            
            # 更新帕累托前沿
            self._update_pareto_front()
            
            # 记录历史
            hv = self._calculate_hypervolume(self.pareto_front)
            convergence_history['hypervolume'].append(hv)
            convergence_history['pareto_size'].append(len(self.pareto_front.individuals))
            
            if (gen + 1) % 20 == 0 or gen == 0:
                print(f"  代 {gen+1:3d}: HV={hv:.4f}, 帕累托解={len(self.pareto_front.individuals)}")
            
            if callback:
                callback(gen, self.pareto_front, convergence_history)
        
        result = OptimizationResult(
            pareto_front=self.pareto_front,
            history=[self.pareto_front],
            convergence_history=dict(convergence_history),
            n_iterations=self.n_generations,
            n_function_evaluations=self.n_evaluations,
        )
        
        print(f"\n优化完成! 帕累托解数量: {len(self.pareto_front.individuals)}")
        print(f"{'='*60}")
        
        return result
    
    def _generate_weight_vectors(self):
        """生成均匀分布的权重向量"""
        # 使用简单方法: 对每个目标分配相等权重
        self.weights = np.eye(self.population_size, self.n_objectives)
        
        # 如果目标数少于种群数，添加组合权重
        if self.n_objectives < self.population_size:
            additional_weights = []
            for i in range(self.n_objectives):
                for j in range(i + 1, self.n_objectives):
                    w = np.zeros(self.n_objectives)
                    w[i] = 0.5
                    w[j] = 0.5
                    additional_weights.append(w)
                    if len(additional_weights) >= self.population_size - self.n_objectives:
                        break
                if len(additional_weights) >= self.population_size - self.n_objectives:
                    break
            
            if additional_weights:
                self.weights = np.vstack([self.weights, 
                                         np.array(additional_weights[:self.population_size - self.n_objectives])])
    
    def _compute_neighborhoods(self):
        """计算邻居关系 - 基于权重向量相似度"""
        # 计算权重向量之间的欧氏距离
        distances = np.zeros((self.population_size, self.population_size))
        
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                dist = np.linalg.norm(self.weights[i] - self.weights[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 为每个个体选择最近的邻居
        for i in range(self.population_size):
            sorted_indices = np.argsort(distances[i])
            self.neighborhoods[i] = sorted_indices[1:self.neighborhood_size + 1].tolist()
    
    def _initialize_population(self) -> List[Individual]:
        """初始化种群"""
        population = []
        # 类似NSGA-II
        nsga2_like = NSGAIIOptimizer(
            n_objectives=self.n_objectives,
            population_size=self.population_size,
            random_seed=None
        )
        nsga2_like.set_environment(self.env_info)
        nsga2_like.set_objectives(self.objectives)
        
        # 生成初始种群
        for _ in range(self.population_size):
            genes = nsga2_like._generate_random_genes()
            population.append(Individual(genes=genes))
        
        return population
    
    def _differential_evolution(self, x_i: List, x_k: List, x_l: List) -> List:
        """差分进化算子"""
        if len(x_i) != len(x_k) or len(x_k) != len(x_l):
            return x_i
        
        F = 0.5  # 缩放因子
        new_genes = []
        
        for i in range(len(x_i)):
            gene_i = x_i[i]
            gene_k = x_k[i]
            gene_l = x_l[i]
            
            # 差分变异
            if hasattr(gene_i, '__getitem__') and hasattr(gene_k, '__getitem__'):
                new_x = tuple(int(g1 + F * (g2 - g3)) for g1, g2, g3 
                            in zip(gene_i, gene_k, gene_l))
            else:
                new_x = int(gene_i + F * (gene_k - gene_l))
            
            # 边界检查
            new_genes.append(new_x)
        
        return new_genes
    
    def _update_neighbors(self, idx: int, new_ind: Individual, neighbors: List[int]):
        """更新邻居解"""
        for n_idx in neighbors:
            # 使用Tchebycheff分解方法更新
            if self._is_better(new_ind, self.population[n_idx], self.weights[idx]):
                self.population[n_idx] = new_ind
    
    def _is_better(self, ind1: Individual, ind2: Individual, weight: np.ndarray) -> bool:
        """使用Tchebycheff方法比较"""
        if len(ind1.objectives) == 0 or len(ind2.objectives) == 0:
            return False
        
        obj1 = ind1.objectives
        obj2 = ind2.objectives
        
        # 归一化 (假设已知目标范围)
        ref = np.ones(self.n_objectives) * 2.0  # 参考点
        
        tche1 = max([weight[i] * abs(obj1[i] - ref[i]) for i in range(self.n_objectives)])
        tche2 = max([weight[i] * abs(obj2[i] - ref[i]) for i in range(self.n_objectives)])
        
        return tche1 < tche2
    
    def _update_pareto_front(self):
        """更新帕累托前沿"""
        # 简单收集所有非支配解
        fronts = [[]]
        
        for p in self.population:
            p.dominated = False
            p.dominates_count = 0
        
        for i, p in enumerate(self.population):
            for j, q in enumerate(self.population):
                if i != j and self._dominates(p, q):
                    p.dominates_count += 1
            
            if p.dominates_count == 0:
                fronts[0].append(p)
        
        self.pareto_front = ParetoFront(fronts[0] if fronts else [])
    
    def _dominates(self, p: Individual, q: Individual) -> bool:
        """检查支配关系"""
        if len(p.objectives) == 0 or len(q.objectives) == 0:
            return False
        
        better = False
        for i in range(len(p.objectives)):
            obj_type = self.objectives[i].objective_type if self.objectives else ObjectiveType.MINIMIZE
            
            if obj_type == ObjectiveType.MINIMIZE:
                if p.objectives[i] > q.objectives[i]:
                    return False
                if p.objectives[i] < q.objectives[i]:
                    better = True
            else:
                if p.objectives[i] < q.objectives[i]:
                    return False
                if p.objectives[i] > q.objectives[i]:
                    better = True
        
        return better
    
    def _calculate_hypervolume(self, front: ParetoFront) -> float:
        """计算超体积"""
        if not front.individuals:
            return 0.0
        
        hv = 0.0
        obj_matrix = np.array([ind.objectives for ind in front.individuals])
        
        if self.n_objectives == 2:
            sorted_inds = sorted(front.individuals, key=lambda x: x.objectives[0])
            ref_y = max(ind.objectives[1] for ind in front.individuals) * 1.1
            
            prev_x = 0
            for ind in sorted_inds:
                width = ind.objectives[0] - prev_x
                height = ref_y - ind.objectives[1]
                hv += width * height
                prev_x = ind.objectives[0]
        
        return hv


# ============================================================================
# 帕累托前沿决策支持
# ============================================================================

class ParetoDecisionMaker:
    """
    帕累托前沿决策器
    
    提供多种方法从帕累托前沿选择最终解:
    1. 折衷规划 (TOPSIS)
    2. 模糊决策
    3. 基于偏好的选择
    4. 交互式选择
    """
    
    def __init__(self, objectives: List[Objective]):
        """
        初始化决策器
        
        参数:
            objectives: 目标函数列表
        """
        self.objectives = objectives
        self.n_objectives = len(objectives)
    
    def select_by_weights(self, front: ParetoFront, 
                          weights: Dict[str, float]) -> Optional[Individual]:
        """
        基于权重选择最终解
        
        使用加权求和方法，选择使加权和最小的解
        
        参数:
            front: 帕累托前沿
            weights: 各目标权重字典
            
        返回:
            选中的个体
        """
        if not front.individuals:
            return None
        
        best_ind = None
        best_score = float('inf')
        
        for ind in front.individuals:
            score = 0.0
            for i, obj in enumerate(self.objectives):
                if obj.name in weights:
                    w = weights[obj.name]
                else:
                    w = weights.get(obj.name.lower(), 1.0)
                
                # 归一化目标值
                if obj.objective_type == ObjectiveType.MINIMIZE:
                    score += w * ind.objectives[i]
                else:
                    score -= w * ind.objectives[i]
            
            if score < best_score:
                best_score = score
                best_ind = ind
        
        return best_ind
    
    def select_by_topsis(self, front: ParetoFront) -> Optional[Individual]:
        """
        使用TOPSIS方法选择最终解
        
        TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
        
        参数:
            front: 帕累托前沿
            
        返回:
            选中的个体
        """
        if not front.individuals:
            return None
        
        obj_matrix = np.array([ind.objectives for ind in front.individuals])
        n = len(front.individuals)
        
        # 归一化
        norm_matrix = obj_matrix / (np.linalg.norm(obj_matrix, axis=0) + 1e-10)
        
        # 确定正理想解和负理想解
        ideal_pos = np.zeros(self.n_objectives)
        ideal_neg = np.zeros(self.n_objectives)
        
        for i, obj in enumerate(self.objectives):
            if obj.objective_type == ObjectiveType.MINIMIZE:
                ideal_pos[i] = np.min(norm_matrix[:, i])
                ideal_neg[i] = np.max(norm_matrix[:, i])
            else:
                ideal_pos[i] = np.max(norm_matrix[:, i])
                ideal_neg[i] = np.min(norm_matrix[:, i])
        
        # 计算到正负理想解的距离
        d_pos = np.sqrt(np.sum((norm_matrix - ideal_pos) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((norm_matrix - ideal_neg) ** 2, axis=1))
        
        # 计算相对接近度
        closeness = d_neg / (d_pos + d_neg + 1e-10)
        
        # 选择最大接近度对应的解
        best_idx = np.argmax(closeness)
        return front.individuals[best_idx]
    
    def select_by_fuzzy(self, front: ParetoFront) -> Optional[Individual]:
        """
        使用模糊决策方法选择最终解
        
        模糊决策通过隶属度函数将每个目标转换为[0,1]范围内的值，
        然后选择使总隶属度最大的解。
        
        参数:
            front: 帕累托前沿
            
        返回:
            选中的个体
        """
        if not front.individuals:
            return None
        
        obj_matrix = np.array([ind.objectives for ind in front.individuals])
        
        # 计算每个目标的隶属度
        membership = np.zeros_like(obj_matrix)
        
        for i, obj in enumerate(self.objectives):
            obj_values = obj_matrix[:, i]
            f_min = np.min(obj_values)
            f_max = np.max(obj_values)
            
            if f_max == f_min:
                membership[:, i] = 1.0
            else:
                if obj.objective_type == ObjectiveType.MINIMIZE:
                    membership[:, i] = (f_max - obj_values) / (f_max - f_min)
                else:
                    membership[:, i] = (obj_values - f_min) / (f_max - f_min)
        
        # 计算总隶属度
        total_membership = np.mean(membership, axis=1)
        
        # 选择最大隶属度对应的解
        best_idx = np.argmax(total_membership)
        return front.individuals[best_idx]
    
    def select_by_ideal_point(self, front: ParetoFront,
                            ideal_point: np.ndarray) -> Optional[Individual]:
        """
        基于理想点选择最终解
        
        选择距离理想点最近的帕累托解
        
        参数:
            front: 帕累托前沿
            ideal_point: 理想点坐标数组
            
        返回:
            选中的个体
        """
        if not front.individuals:
            return None
        
        obj_matrix = np.array([ind.objectives for ind in front.individuals])
        
        # 计算到理想点的距离
        distances = np.sqrt(np.sum((obj_matrix - ideal_point) ** 2, axis=1))
        
        # 选择距离最近的解
        best_idx = np.argmin(distances)
        return front.individuals[best_idx]


# ============================================================================
# 工具函数
# ============================================================================

def create_default_objectives() -> List[Objective]:
    """
    创建默认的多目标函数列表
    
    默认目标:
    1. 能耗 (最小化)
    2. 时间 (最小化)
    3. 噪声 (最小化)
    4. 碰撞风险 (最小化)
    5. 路径长度 (最小化)
    """
    objectives = [
        Objective(
            name='energy',
            func=None,  # 将由外部设置
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.30
        ),
        Objective(
            name='time',
            func=None,
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.20
        ),
        Objective(
            name='noise',
            func=None,
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.20
        ),
        Objective(
            name='collision',
            func=None,
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.20
        ),
        Objective(
            name='length',
            func=None,
            objective_type=ObjectiveType.MINIMIZE,
            weight=0.10
        ),
    ]
    return objectives


def print_pareto_summary(result: OptimizationResult, 
                        objective_names: List[str] = None):
    """
    打印帕累托优化结果摘要
    
    参数:
        result: 优化结果
        objective_names: 目标名称列表
    """
    if objective_names is None:
        objective_names = [f'obj_{i}' for i in range(len(result.pareto_front.individuals[0].objectives) 
                                                        if result.pareto_front.individuals else 0)]
    
    print(f"\n{'='*60}")
    print(f"帕累托优化结果摘要")
    print(f"{'='*60}")
    
    print(f"\n【算法性能】")
    print(f"  迭代次数: {result.n_iterations}")
    print(f"  函数评估次数: {result.n_function_evaluations}")
    print(f"  最终帕累托解数量: {len(result.pareto_front.individuals)}")
    
    print(f"\n【帕累托前沿质量】")
    print(f"  超体积指标 (HV): {result.pareto_front.hypervolume:.4f}")
    print(f"  间距指标 (Spacing): {result.pareto_front.spacing:.4f}")
    
    print(f"\n【目标统计】")
    for i, name in enumerate(objective_names):
        values = [ind.objectives[i] for ind in result.pareto_front.individuals]
        if values:
            print(f"  {name}: [{min(values):.2f}, {max(values):.2f}], 均值={np.mean(values):.2f}")
    
    print(f"\n【代表性解】")
    # 选择边界解
    fronts = result.pareto_front.individuals
    if fronts:
        # 最小化每个目标的解
        for i, name in enumerate(objective_names):
            best_idx = np.argmin([ind.objectives[i] for ind in fronts])
            best = fronts[best_idx]
            print(f"  {name}最优: {[f'{v:.2f}' for v in best.objectives]}")
    
    print(f"{'='*60}")


def compare_algorithms(results: Dict[str, OptimizationResult]):
    """
    比较不同算法的性能
    
    参数:
        results: 算法名称 -> 优化结果的字典
    """
    print(f"\n{'='*60}")
    print(f"算法性能对比")
    print(f"{'='*60}")
    
    print(f"\n{'算法':<15} {'HV':<12} {'Spacing':<12} {'解数量':<10} {'评估次数':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result.pareto_front.hypervolume:<12.4f} "
              f"{result.pareto_front.spacing:<12.4f} "
              f"{len(result.pareto_front.individuals):<10} "
              f"{result.n_function_evaluations:<12}")
    
    print("-" * 60)
    
    # 找出最优算法
    best_hv = max(results.items(), key=lambda x: x[1].pareto_front.hypervolume)
    print(f"\n超体积最优: {best_hv[0]} (HV={best_hv[1].pareto_front.hypervolume:.4f})")
    
    print(f"{'='*60}")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("eVTOL航迹规划多目标优化模块 - V7版本")
    print("=" * 60)
    
    print("\n参考文献:")
    print("  [M1] Deb K, et al. NSGA-II. IEEE TEVC, 2002.")
    print("  [M2] Zhang Q, Li H. MOEA/D. IEEE TEVC, 2007.")
    print("  [M3] Zitzler E, Thiele L. SPEA2. IEEE TEVC, 1999.")
    print("  [M4] Coello C A C, et al. Evolutionary Algorithms for MOPs. Springer, 2007.")
    
    print("\n" + "=" * 60)
    print("测试: 帕累托前沿生成")
    print("=" * 60)
    
    # 创建简单的测试目标函数
    def test_objectives(genes):
        """测试目标函数: f1 = 路径长度, f2 = 能耗估计"""
        length = sum(np.sqrt((genes[i][0] - genes[i-1][0])**2 + 
                            (genes[i][1] - genes[i-1][1])**2 + 
                            (genes[i][2] - genes[i-1][2])**2) 
                    for i in range(1, len(genes)))
        energy = length * 0.0008  # 简化能耗模型
        noise = length * 0.001
        time = length / 50.0
        collision = np.random.uniform(0, 0.1)
        
        return np.array([energy, time, noise, collision, length])
    
    # 创建优化器
    objectives = create_default_objectives()
    
    optimizer = NSGAIIOptimizer(
        n_objectives=5,
        population_size=50,
        n_generations=30,
        crossover_prob=0.9,
        mutation_prob=0.1,
        random_seed=42
    )
    
    optimizer.set_objectives(objectives)
    optimizer.set_environment({
        'start': (0, 0, 50),
        'goal': (2000, 2000, 50),
        'bounds': ((0, 3000), (0, 3000), (30, 300))
    })
    
    # 运行优化
    result = optimizer.optimize(test_objectives)
    
    # 打印结果
    print_pareto_summary(result, ['能耗', '时间', '噪声', '碰撞风险', '长度'])
    
    # 决策支持
    print("\n【决策支持】")
    decision_maker = ParetoDecisionMaker(objectives)
    
    # 使用TOPSIS选择
    best_topsis = decision_maker.select_by_topsis(result.pareto_front)
    if best_topsis:
        print(f"TOPSIS推荐解: {[f'{v:.2f}' for v in best_topsis.objectives]}")
    
    # 使用模糊决策
    best_fuzzy = decision_maker.select_by_fuzzy(result.pareto_front)
    if best_fuzzy:
        print(f"模糊决策推荐: {[f'{v:.2f}' for v in best_fuzzy.objectives]}")
    
    print("\n" + "=" * 60)
    print("多目标优化模块测试完成")
    print("=" * 60)
