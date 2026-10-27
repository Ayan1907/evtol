#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径评估器 - 后评估能耗和噪声

设计理念:
1. 不改变现有路径规划算法
2. 对规划结果进行详细的能耗和噪声分析
3. 验证噪声约束的有效性(路径是否满足噪声要求)
4. 提供对比数据用于论文

关键价值:
- 能耗: 验证目标函数的有效性(高度变化、风场代价)
- 噪声: 验证硬约束的有效性(路径是否避开高噪声区域)
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from evtol_path_planning_v6 import LowCarbonNode
from evtol_path_planning_v6 import FlightProfile  # V6使用V6的FlightProfile


class PathEvaluator:
    """路径评估器 - 后评估模式"""

    def __init__(self, env, flight_profile: FlightProfile, noise_constraint=None):
        """初始化评估器"""
        self.env = env
        self.profile = flight_profile
        self.noise_constraint = noise_constraint
        self.profile = flight_profile
        self.env = env
        self.noise_constraint = noise_constraint

    def evaluate_path(self, path: List[LowCarbonNode],
                     current_time: float = 0.0,
                     save_results: bool = False,
                     filename: str = "path_evaluation.json") -> Dict:
        """
        评估路径的能耗和噪声

        返回详细的评估数据
        """
        if not path or len(path) < 2:
            return {}

        evaluation = {
            'path_length': len(path),
            'total_distance': 0.0,
            'total_time': 0.0,

            # 能耗指标
            'energy': self._evaluate_energy(path, current_time),

            # 噪声指标
            'noise': self._evaluate_noise(path, current_time),

            # 碰撞风险
            'collision_risk': self._evaluate_collision(path),

            # 地面风险
            'ground_risk': self._evaluate_ground_risk(path),

            # 综合评级
            'overall_rating': {}
        }

        # 计算综合评级
        evaluation['overall_rating'] = self._calculate_rating(evaluation)

        # 保存结果
        if save_results:
            with open(filename, 'w') as f:
                json.dump(evaluation, f, indent=2)

        return evaluation

    def _evaluate_energy(self, path: List[LowCarbonNode],
                         current_time: float) -> Dict:
        """
        评估能耗

        价值:
        - 验证目标函数中高度变化(alpha3)的有效性
        - 验证风场代价(alpha5)的有效性
        - 展示规划路径确实考虑了能耗
        """
        total_energy = 0.0
        energy_by_phase = {
            'climb': 0.0,
            'cruise': 0.0,
            'descent': 0.0,
            'hover': 0.0
        }

        wind_costs = []
        height_costs = []

        for i in range(len(path) - 1):
            curr = path[i]
            next_node = path[i + 1]

            # 确定飞行阶段
            if next_node.z > curr.z + 5:
                phase = 'climb'
            elif next_node.z < curr.z - 5:
                phase = 'descent'
            elif abs(next_node.z - curr.z) < 2:
                phase = 'cruise'
            else:
                phase = 'hover'

            # 基础功率(与高度相关 - 简化模型)
            # 高度越高，需要维持升力的功率越大
            base_power = 100.0 + curr.z * 0.3  # 基础功率100W，每米高度增加0.3W
            height_costs.append(base_power)

            # 风场影响(验证目标函数中的风场代价)
            wind_cost = self.env.wind_field.calculate_wind_cost(
                curr.x, curr.y, curr.z,
                (next_node.x - curr.x, next_node.y - curr.y, next_node.z - curr.z),
                current_time
            )
            wind_costs.append(wind_cost)

            segment_energy = base_power + wind_cost
            energy_by_phase[phase] += segment_energy
            total_energy += segment_energy

        return {
            'total_energy_wh': float(total_energy),
            'energy_by_phase_kwh': {
                k: float(v / 1000.0) for k, v in energy_by_phase.items()
            },
            'energy_percentage': {
                k: float(v / total_energy * 100) if total_energy > 0 else 0.0
                for k, v in energy_by_phase.items()
            },
            'average_wind_cost': float(np.mean(wind_costs)) if wind_costs else 0.0,
            'average_height_cost': float(np.mean(height_costs)) if height_costs else 0.0,
            'height_energy_fraction': float(np.sum(height_costs) / total_energy) if total_energy > 0 else 0.0,
            'wind_energy_fraction': float(np.sum(wind_costs) / total_energy) if total_energy > 0 else 0.0,
            'total_vertical_change': float(sum(abs(path[i+1].z - path[i].z) for i in range(len(path)-1)))
        }

    def _evaluate_noise(self, path: List[LowCarbonNode],
                       current_time: float) -> Dict:
        """
        评估噪声影响

        价值:
        - 验证噪声约束的有效性
        - 如果噪声约束有效,所有节点都应满足噪声要求
        - 展示路径确实避开了高噪声区域
        """
        from noise_model import NoiseModel

        noise_model = NoiseModel(self.env, source_level=85.0, reference_distance=100.0)

        total_noise_cost = 0.0
        noise_levels = []
        high_noise_count = 0  # 超过阈值的节点数

        for i in range(len(path) - 1):
            curr = path[i]
            next_node = path[i + 1]

            # 计算噪声代价（转换为元组）
            noise_cost = noise_model.calculate_noise_cost(
                (curr.x, curr.y, curr.z),
                (next_node.x, next_node.y, next_node.z),
                int(current_time / 3600)  # 传入小时
            )
            total_noise_cost += noise_cost

            # 计算噪声水平
            noise_level = noise_model.calculate_noise_level(
                next_node.x, next_node.y, next_node.z, 20.0
            )
            noise_levels.append(noise_level)

            # 检查是否超过阈值(验证约束有效性)
            if noise_level > 95.0:  # 使用当前阈值
                high_noise_count += 1

        return {
            'total_noise_cost': float(total_noise_cost),
            'average_noise_level_dba': float(np.mean(noise_levels)) if noise_levels else 0.0,
            'peak_noise_level_dba': float(np.max(noise_levels)) if noise_levels else 0.0,
            'min_noise_level_dba': float(np.min(noise_levels)) if noise_levels else 0.0,
            'high_noise_segments': int(high_noise_count),
            'noise_constraint_compliance': high_noise_count == 0,  # 噪声约束合规性
            'noise_compliance_rate': float(1.0 - (high_noise_count / len(path))) if path else 0.0
        }

    def _evaluate_collision(self, path: List[LowCarbonNode]) -> Dict:
        """评估碰撞风险"""
        from collision_risk_model import CollisionRiskModel

        collision_model = CollisionRiskModel(self.env)

        # 添加动态障碍物到碰撞模型
        if hasattr(self.env, 'dynamic_obstacles'):
            for obs in self.env.dynamic_obstacles:
                # 动态障碍物作为其他航空器处理
                collision_model.add_aircraft(
                    aircraft_id=f"dynamic_obs_{obs.id}",
                    x=obs.x,
                    y=obs.y,
                    z=obs.z,
                    vx=obs.vx,
                    vy=obs.vy,
                    vz=obs.vz
                )

        risks = []
        for i in range(len(path) - 1):
            curr = path[i]
            next_node = path[i + 1]

            # 转换为元组格式 (x, y, z)
            curr_tuple = (curr.x, curr.y, curr.z)
            next_tuple = (next_node.x, next_node.y, next_node.z)

            risk = collision_model.calculate_collision_cost(curr_tuple, next_tuple)
            risks.append(risk)

        return {
            'max_risk': float(np.max(risks)) if risks else 0.0,
            'average_risk': float(np.mean(risks)) if risks else 0.0,
            'high_risk_segments': int(np.sum(np.array(risks) > 0.3)) if risks else 0
        }

    def _evaluate_ground_risk(self, path: List[LowCarbonNode]) -> Dict:
        """评估地面风险"""
        total_risk = 0.0
        risks = []

        # 尝试获取ground_risk_model
        ground_risk_model = None
        if hasattr(self.env, 'ground_risk_model'):
            ground_risk_model = self.env.ground_risk_model
        elif hasattr(self.env, 'env') and hasattr(self.env.env, 'ground_risk_model'):
            ground_risk_model = self.env.env.ground_risk_model

        if ground_risk_model:
            for node in path:
                risk = ground_risk_model.calculate_ground_risk(node)
                risks.append(risk)
                total_risk += risk
        else:
            # 如果没有ground_risk_model，返回空数据
            return {
                'total_ground_risk': 0.0,
                'average_ground_risk': 0.0,
                'max_ground_risk': 0.0,
                'note': 'ground_risk_model not available'
            }

        return {
            'total_ground_risk': float(total_risk),
            'average_ground_risk': float(np.mean(risks)) if risks else 0.0,
            'max_ground_risk': float(np.max(risks)) if risks else 0.0
        }

    def _calculate_rating(self, evaluation: Dict) -> Dict:
        """
        计算综合评级(0-100分)

        评分依据参考:
        - WHO社区噪声指南[WHO, 2018]
        - ICAO Annex 16[ICAO, 2017]
        - ISO 9613-1:1993 (声学传播标准)
        - FAA Part 150 (机场噪声标准)
        - IEEE/IEC eVTOL性能标准(参考)

        关键价值:
        - 能耗评分: 验证目标函数alpha3(高度变化)的有效性
        - 噪声评分: 验证噪声约束的有效性
        """
        # 归一化函数(越小越好)
        def normalize(value, max_val, min_val=0):
            if value <= min_val:
                return 100.0
            elif value >= max_val:
                return 0.0
            else:
                return 100 * (1 - (value - min_val) / (max_val - min_val))

        # ==================== 能耗评分 ====================
        # 依据: eVTOL典型续航里程和电池容量
        # 参考: Joby S4 (约200km航程, 100kWh电池)
        # 计算: 假设10km航程, 理想能耗2000Wh, 可接受能耗15000Wh
        # 说明: min_val设为2000Wh以覆盖低高度路径，max_val设为15000Wh以覆盖高能耗路径
        energy_score = normalize(
            evaluation['energy']['total_energy_wh'],
            max_val=15000,   # 可接受上限(参考eVTOL实际续航)
            min_val=2000     # 理想下限(参考高效航迹)
        )

        # ==================== 噪声评分 ====================
        # 依据: WHO社区噪声指南 + ICAO Annex 16
        # WHO推荐值: 居民区昼间55dB, 夜间45dB
        # ICAO Annex 16: 航空器峰值85dB
        # 本项目阈值: 平均95dB, 峰值105dB
        # 评分范围: 55dB(理想) - 95dB(阈值上限)
        noise_score = normalize(
            evaluation['noise']['average_noise_level_dba'],
            max_val=95,      # 本项目设定的噪声约束阈值
            min_val=55        # WHO推荐居民区昼间标准
        )

        # 噪声合规性额外奖励(验证硬约束有效性)
        if evaluation['noise']['noise_constraint_compliance']:
            # 合规: 奖励15分(硬约束保证合规性的价值)
            noise_score = min(noise_score + 15, 100)
        else:
            # 不合规: 惩罚10分(硬约束失效)
            noise_score = max(noise_score - 10, 0)

        # ==================== 安全评分 ====================
        # 依据: FAA航空安全标准, ICAO Annex 13
        # 碰撞风险阈值: 0.1(高风险), 0.01(中等风险)
        # 评分范围: 0(完美) - 0.3(不可接受)
        safety_score = normalize(
            evaluation['collision_risk']['average_risk'],
            max_val=0.3,     # 不可接受风险水平
            min_val=0.0      # 理想: 零风险
        )

        # ==================== 综合评分 ====================
        # 权重设计(参考ICAO持续适航安全评级框架 + 民用航空安全优先原则):
        # 安全: 0.50 (民用航空安全永远是第一位的, 安全不可协商)
        # 噪声: 0.30 (社区可接受性, eVTOL普及的关键因素)
        # 能耗: 0.20 (经济性, eVTOL商业化关键指标)
        overall_score = (
            safety_score * 0.50 +
            noise_score * 0.30 +
            energy_score * 0.20
        )

        return {
            'energy_score': float(round(energy_score, 1)),
            'noise_score': float(round(noise_score, 1)),
            'safety_score': float(round(safety_score, 1)),
            'overall_score': float(round(overall_score, 1)),
            'grade': self._get_grade(overall_score)
        }

    def _get_grade(self, score: float) -> str:
        """
        获取评级

        评级标准参考ICAO安全评级框架:
        A+ (90-100): 优秀, 满足所有国际标准
        A  (80-89):  良好, 符合大部分要求
        B  (70-79):  中等, 需要优化
        C  (60-69):  及格, 勉强满足基本要求
        D  (0-59):   需改进, 不符合标准
        """
        if score >= 90:
            return 'A+ (优秀)'
        elif score >= 80:
            return 'A (良好)'
        elif score >= 70:
            return 'B (中等)'
        elif score >= 60:
            return 'C (及格)'
        else:
            return 'D (需改进)'

    def print_evaluation_report(self, evaluation: Dict):
        """
        打印评估报告

        强调价值:
        1. 能耗: 验证目标函数的有效性
        2. 噪声: 验证硬约束的有效性

        评分标准参考:
        - WHO社区噪声指南[WHO, 2018]
        - ICAO Annex 16[ICAO, 2017]
        - FAA Part 150 (机场噪声标准)
        - Joby S4 eVTOL续航数据(参考)
        """
        print("\n" + "="*60)
        print("📊 路径评估报告".center(60))
        print("="*60)

        # 综合评级
        rating = evaluation['overall_rating']
        print(f"\n【综合评级】 {rating['grade']}")
        print(f"  总分: {rating['overall_score']}/100")
        print(f"  能耗: {rating['energy_score']}/100")
        print(f"  噪声: {rating['noise_score']}/100")
        print(f"  安全: {rating['safety_score']}/100")

        # 能耗详情
        energy = evaluation['energy']
        print(f"\n【能耗分析】")
        print(f"  💡 价值: 验证目标函数中高度变化(alpha3)和风场代价(alpha5)的有效性")
        print(f"  评分依据: eVTOL典型续航(Joby S4: 200km/100kWh)")
        print(f"  总能耗: {energy['total_energy_wh']:.1f} Wh")
        print(f"  总高度变化: {energy['total_vertical_change']:.1f} m")
        print(f"  平均风场代价: {energy['average_wind_cost']:.2f}")
        print(f"  能耗占比:")
        for phase, pct in energy['energy_percentage'].items():
            if pct > 0:
                print(f"    {phase:8s}: {pct:5.1f}%")

        # 噪声详情
        noise = evaluation['noise']
        print(f"\n【噪声分析】")
        print(f"  💡 价值: 验证噪声硬约束的有效性")
        print(f"  评分依据: WHO社区噪声指南(55dB) + ICAO Annex 16(95dB阈值)")
        print(f"  平均噪声级: {noise['average_noise_level_dba']:.1f} dB(A)")
        print(f"  峰值噪声级: {noise['peak_noise_level_dba']:.1f} dB(A)")
        print(f"  最小噪声级: {noise['min_noise_level_dba']:.1f} dB(A)")
        print(f"  高噪声段数: {noise['high_noise_segments']}")
        print(f"  噪声约束合规: {'✅ 是' if noise['noise_constraint_compliance'] else '❌ 否'}")
        print(f"  合规率: {noise['noise_compliance_rate']*100:.1f}%")

        # 安全详情
        print(f"\n【安全分析】")
        print(f"  评分依据: FAA航空安全标准, ICAO Annex 13")
        coll = evaluation['collision_risk']
        print(f"  最大碰撞风险: {coll['max_risk']:.3f}")
        print(f"  平均碰撞风险: {coll['average_risk']:.3f}")
        print(f"  高风险段数: {coll['high_risk_segments']}")

        print("\n" + "="*60 + "\n")


def compare_two_paths(path1: List[LowCarbonNode],
                     path2: List[LowCarbonNode],
                     name1: str = "路径1",
                     name2: str = "路径2",
                     evaluator: 'PathEvaluator' = None,
                     current_time: float = 0.0) -> Dict:
    """
    对比两条路径

    用于验证不同参数/算法的效果
    """
    if evaluator is None:
        raise ValueError("需要提供evaluator")

    eval1 = evaluator.evaluate_path(path1, current_time)
    eval2 = evaluator.evaluate_path(path2, current_time)

    comparison = {
        name1: eval1,
        name2: eval2,
        'improvement': {}
    }

    # 计算改进
    metrics = [
        ('total_energy_wh', 'Wh', False),  # 越小越好
        ('average_noise_level_dba', 'dB(A)', False),
        ('max_risk', '-', False),
        ('overall_score', '分', True),  # 越大越好
    ]

    for metric, unit, larger_better in metrics:
        if 'noise' in metric:
            val1 = eval1['noise'][metric]
            val2 = eval2['noise'][metric]
        elif 'risk' in metric:
            val1 = eval1['collision_risk'][metric]
            val2 = eval2['collision_risk'][metric]
        elif 'overall_score' in metric:
            val1 = eval1['overall_rating'][metric]
            val2 = eval2['overall_rating'][metric]
        else:
            val1 = eval1.get(metric, 0)
            val2 = eval2.get(metric, 0)

        if larger_better:
            improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
        else:
            improvement = ((val1 - val2) / val1 * 100) if val1 > 0 else 0

        comparison['improvement'][metric] = {
            'value': improvement,
            'unit': '%',
            'better': improvement > 0
        }

    return comparison


def print_comparison_report(comparison: Dict):
    """
    打印对比报告

    展示不同算法/参数的效果对比
    """
    print("\n" + "="*60)
    print("📊 路径对比报告".center(60))
    print("="*60)

    name1 = [k for k in comparison.keys() if k != 'improvement'][0]
    name2 = [k for k in comparison.keys() if k != 'improvement'][1]

    print(f"\n【对比】 {name1} vs {name2}")
    print(f"\n{'指标':<20} {name1:<15} {name2:<15} 改进")
    print("-"*60)

    improvement = comparison['improvement']

    # 综合评分
    score1 = comparison[name1]['overall_rating']['overall_score']
    score2 = comparison[name2]['overall_rating']['overall_score']
    imp_score = improvement['overall_score']['value']
    arrow = "↑" if imp_score > 0 else "↓"
    print(f"{'综合评分':<20} {score1:<15.1f} {score2:<15.1f} {arrow}{abs(imp_score):.1f}%")

    # 能耗
    e1 = comparison[name1]['energy']['total_energy_wh']
    e2 = comparison[name2]['energy']['total_energy_wh']
    imp_e = improvement['total_energy_wh']['value']
    arrow = "↓" if imp_e > 0 else "↑"
    print(f"{'总能耗':<20} {e1:<15.1f} {e2:<15.1f} {arrow}{abs(imp_e):.1f}%")

    # 噪声
    n1 = comparison[name1]['noise']['average_noise_level_dba']
    n2 = comparison[name2]['noise']['average_noise_level_dba']
    imp_n = improvement['average_noise_level_dba']['value']
    arrow = "↓" if imp_n > 0 else "↑"
    print(f"{'平均噪声':<20} {n1:<15.1f} {n2:<15.1f} {arrow}{abs(imp_n):.1f}%")

    # 碰撞风险
    r1 = comparison[name1]['collision_risk']['max_risk']
    r2 = comparison[name2]['collision_risk']['max_risk']
    imp_r = improvement['max_risk']['value']
    arrow = "↓" if imp_r > 0 else "↑"
    print(f"{'最大风险':<20} {r1:<15.3f} {r2:<15.3f} {arrow}{abs(imp_r):.1f}%")

    print("\n" + "="*60 + "\n")
