#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eVTOL飞行器动力学与能耗模型 - V5核心模块
完成以下功能:
1. 飞行器六自由度运动学模型
2. 气动力计算(升力/阻力)
3. 电机-电池系统模型
4. 实时功率与能耗计算
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class AircraftParameters:
    """eVTOL飞行器物理参数 - 以亿航EH216-S为例"""
    
    # 基础参数
    name: str = "EH216-S"
    empty_weight: float = 400.0  # 空机重量 (kg) - 亿航官方技术规格
    max_payload: float = 220.0  # 最大有效载荷 (kg) - 2名乘客约180-220kg
    MTOW: float = 620.0  # 最大起飞重量 (kg) = 空机重量 + 最大有效载荷
    # 飞行计算使用实际质量（含载荷），默认为MTOW的典型载荷（乘客约180kg）
    mass: float = 580.0  # 飞行质量 (kg) - 空机+典型载荷(180kg)
    
    # 几何尺寸
    length: float = 5.73  # 全长 (m)
    width: float = 5.61  # 全宽 (m)
    height: float = 1.93  # 高度 (m)
    diameter: float = 5.63  # 旋翼直径 (m)
    rotor_radius: float = 2.805  # 旋翼半径 (m)
    rotor_area: float = field(init=False)  # 旋翼盘面积 (m²)
    
    # 惯性参数
    I_xx: float = 450.0  # 滚转转动惯量 (kg·m²)
    I_yy: float = 450.0  # 俯仰转动惯量 (kg·m²)
    I_zz: float = 850.0  # 偏航转动惯量 (kg·m²)
    
    # 推进系统参数
    num_rotors: int = 16  # 旋翼数量
    k_T: float = 1.2e-6  # 推力系数 (N/(rad/s)²)
    k_Q: float = 2.5e-8  # 扭矩系数 (N·m/(rad/s)²)
    max_rotor_speed: float = 5000.0  # 最大旋翼转速 (RPM)
    max_thrust: float = 6500.0  # 单旋翼最大推力 (N)
    max_total_thrust: float = 96000.0  # 总最大推力 (N)
    
    # 电机参数
    motor_efficiency_max: float = 0.92  # 最大电机效率
    motor_rated_power: float = 15.0  # 额定功率 (kW)
    motor_efficiency_loss: float = 0.15  # 效率损失系数
    
    # 螺旋桨参数
    propeller_efficiency: float = 0.80  # 螺旋桨效率
    
    # 电池参数
    battery_capacity: float = 120.0  # 电池容量 (kWh)
    battery_voltage_nom: float = 400.0  # 标称电压 (V)
    battery_voltage_max: float = 460.0  # 最大电压 (V)
    battery_voltage_min: float = 300.0  # 最小电压 (V)
    internal_resistance: float = 0.05  # 内阻 (Ω)
    
    # 气动参数
    drag_coefficient_body: float = 0.8  # 机身零升阻力系数
    lift_slope: float = 2.0 * np.pi  # 升力线斜率 (1/rad)
    aspect_ratio: float = 5.0  # 展弦比
    oswald_efficiency: float = 0.8  # Oswald效率因子
    
    # 飞行性能参数
    max_forward_speed: float = 60.0  # 最大前飞速度 (m/s)
    cruise_speed: float = 50.0  # 巡航速度 (m/s)
    max_climb_rate: float = 8.0  # 最大爬升率 (m/s)
    max_turn_rate: float = 30.0  # 最大转弯率 (deg/s)
    max_climb_angle: float = 30.0  # 最大爬升角 (deg)
    max_turn_angle: float = 45.0  # 最大转向角 (deg)
    
    def __post_init__(self):
        """计算派生参数"""
        self.rotor_area = np.pi * self.rotor_radius**2
        self.battery_capacity_ah = self.battery_capacity * 1000 / self.battery_voltage_nom
        
        # 计算诱导阻力因子
        self.induced_drag_factor = 1.0 / (np.pi * self.oswald_efficiency * self.aspect_ratio)


@dataclass
class AircraftState:
    """飞行器状态向量"""
    # 位置 (m)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # 姿态角 (rad)
    phi: float = 0.0  # 滚转角
    theta: float = 0.0  # 俯仰角
    psi: float = 0.0  # 偏航角
    
    # 线速度 (本体系) (m/s)
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    
    # 角速度 (本体系) (rad/s)
    p: float = 0.0  # 滚转角速度
    q: float = 0.0  # 俯仰角速度
    r: float = 0.0  # 偏航角速度
    
    def to_array(self) -> np.ndarray:
        """转换为状态数组"""
        return np.array([
            self.x, self.y, self.z,
            self.phi, self.theta, self.psi,
            self.u, self.v, self.w,
            self.p, self.q, self.r
        ])


@dataclass
class WindVector:
    """风场向量"""
    vx: float = 0.0  # X方向风速 (m/s)
    vy: float = 0.0  # Y方向风速 (m/s)
    vz: float = 0.0  # Z方向风速 (m/s)
    
    @property
    def magnitude(self) -> float:
        """风速大小"""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def to_array(self) -> np.ndarray:
        """转换为向量"""
        return np.array([self.vx, self.vy, self.vz])


class AircraftKinematics:
    """飞行器运动学模型"""
    
    def __init__(self, params: AircraftParameters):
        self.params = params
    
    def rotation_matrix(self, state: AircraftState) -> np.ndarray:
        """
        计算从本体系到惯性系的旋转矩阵
        
        R = R_z(ψ) * R_y(θ) * R_x(φ)
        """
        c_phi, s_phi = np.cos(state.phi), np.sin(state.phi)
        c_theta, s_theta = np.cos(state.theta), np.sin(state.theta)
        c_psi, s_psi = np.cos(state.psi), np.sin(state.psi)
        
        R = np.array([
            [c_theta * c_psi,
             s_phi * s_theta * c_psi - c_phi * s_psi,
             c_phi * s_theta * c_psi + s_phi * s_psi],
            [c_theta * s_psi,
             s_phi * s_theta * s_psi + c_phi * c_psi,
             c_phi * s_theta * s_psi - s_phi * c_psi],
            [-s_theta,
             s_phi * c_theta,
             c_phi * c_theta]
        ])
        
        return R
    
    def body_to_inertial_velocity(self, state: AircraftState) -> np.ndarray:
        """
        将本体系速度转换到惯性系
        v_inertial = R * v_body
        """
        R = self.rotation_matrix(state)
        v_body = np.array([state.u, state.v, state.w])
        v_inertial = R @ v_body
        return v_inertial
    
    def calculate_turn_angle(self, v1: Tuple[float, float, float], 
                            v2: Tuple[float, float, float]) -> float:
        """
        计算两个速度向量之间的转向角
        
        参数:
            v1, v2: 速度向量 (vx, vy, vz)
            
        返回:
            转向角 (弧度)
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # 避免除零
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算夹角
        dot_product = np.dot(v1, v2)
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        
        angle = np.arccos(cos_angle)
        return angle
    
    def calculate_climb_angle(self, v: Tuple[float, float, float]) -> float:
        """
        计算爬升角 (与水平面的夹角)
        
        参数:
            v: 速度向量 (vx, vy, vz)
            
        返回:
            爬升角 (弧度)
        """
        v = np.array(v)
        horizontal_speed = np.sqrt(v[0]**2 + v[1]**2)
        
        if horizontal_speed == 0:
            return np.pi / 2 if v[2] > 0 else -np.pi / 2
        
        return np.arctan(v[2] / horizontal_speed)


class AerodynamicsModel:
    """气动模型"""
    
    def __init__(self, params: AircraftParameters):
        self.params = params
        self.rho = 1.225  # 海平面空气密度 (kg/m³)
    
    def calculate_airspeed(self, ground_velocity: np.ndarray, 
                          wind: WindVector) -> np.ndarray:
        """
        计算空速向量
        V_air = V_ground - V_wind
        """
        return ground_velocity - wind.to_array()
    
    def calculate_lift_drag(self, airspeed: np.ndarray, 
                           alpha: float = 0.0) -> Tuple[float, float]:
        """
        计算升力和阻力
        
        参数:
            airspeed: 空速向量 (m/s)
            alpha: 攻角 (rad)
            
        返回:
            (升力, 阻力) (N)
        """
        V = np.linalg.norm(airspeed)
        
        if V == 0:
            return 0.0, 0.0
        
        # 计算升力系数
        C_L = self.params.lift_slope * alpha
        
        # 计算阻力系数 (零升阻力 + 诱导阻力)
        C_D0 = self.params.drag_coefficient_body
        C_D = C_D0 + self.params.induced_drag_factor * C_L**2
        
        # 计算参考面积 (使用旋翼盘面积)
        S = self.params.rotor_area * self.params.num_rotors
        
        # 计算升力和阻力
        q = 0.5 * self.rho * V**2 * S  # 动压
        L = q * C_L
        D = q * C_D
        
        return L, D
    
    def calculate_angle_of_attack(self, body_velocity: np.ndarray) -> float:
        """
        根据本体系速度计算攻角
        
        参数:
            body_velocity: 本体系速度向量 (u, v, w) (m/s)
            
        返回:
            攻角 (rad)
        """
        u, v, w = body_velocity
        horizontal_speed = np.sqrt(u**2 + v**2)
        
        if horizontal_speed == 0:
            return 0.0
        
        # 攻角是速度向量与机体X轴的夹角
        alpha = np.arctan(w / horizontal_speed)
        return alpha


class BatteryModel:
    """电池模型"""
    
    def __init__(self, params: AircraftParameters):
        self.params = params
        self.soc = 1.0  # 荷电状态 (0-1)
        self.energy_used = 0.0  # 已用能量 (kWh)
    
    def get_ocv(self, soc: float) -> float:
        """
        计算开路电压 (Open Circuit Voltage)
        
        简化模型: 线性变化
        """
        return self.params.battery_voltage_min + \
               (self.params.battery_voltage_max - self.params.battery_voltage_min) * soc
    
    def calculate_power_loss(self, current: float) -> float:
        """
        计算内阻功率损失
        
        P_loss = I² * R
        """
        return current**2 * self.params.internal_resistance
    
    def calculate_efficiency(self, current: float) -> float:
        """
        计算电池效率
        
        η = V_terminal / V_ocv = 1 - I * R / V_ocv
        """
        ocv = self.get_ocv(self.soc)
        voltage_drop = current * self.params.internal_resistance
        efficiency = 1.0 - voltage_drop / ocv
        return max(0.5, min(1.0, efficiency))  # 限制在合理范围
    
    def update_soc(self, power: float, dt: float) -> float:
        """
        更新荷电状态
        
        参数:
            power: 电功率 (W)
            dt: 时间步长 (s)
            
        返回:
            更新后的SOC
        """
        # 计算消耗的能量 (kWh)
        energy_kwh = power * dt / 3600.0
        
        self.energy_used += energy_kwh
        self.soc = max(0.0, 1.0 - self.energy_used / self.params.battery_capacity)
        
        return self.soc


class MotorModel:
    """电机模型"""
    
    def __init__(self, params: AircraftParameters):
        self.params = params
    
    def calculate_mechanical_power(self, thrust: float, rpm: float) -> float:
        """
        计算机械输出功率
        
        P_mech = T * Ω
        其中 Ω = 2π * rpm / 60
        """
        omega = 2.0 * np.pi * rpm / 60.0
        return thrust * omega  # (N * m/s) = W
    
    def thrust_to_rpm(self, thrust: float) -> float:
        """
        根据推力计算所需转速
        
        T = k_T * ω²
        ω = sqrt(T / k_T)
        """
        omega = np.sqrt(thrust / self.params.k_T)
        rpm = omega * 60.0 / (2.0 * np.pi)
        return min(rpm, self.params.max_rotor_speed)
    
    def calculate_efficiency(self, mechanical_power: float) -> float:
        """
        计算电机效率
        
        简化模型: 二次损失模型
        """
        p_ratio = mechanical_power / (self.params.motor_rated_power * 1000.0)
        
        # 额定工况附近效率最高
        efficiency = self.params.motor_efficiency_max * \
                     (1.0 - self.params.motor_efficiency_loss * (1.0 - p_ratio)**2)
        
        return max(0.5, min(0.95, efficiency))


class PowerConsumptionModel:
    """完整的功率消耗模型"""
    
    def __init__(self, params: AircraftParameters):
        self.params = params
        self.aero = AerodynamicsModel(params)
        self.motor = MotorModel(params)
        self.battery = BatteryModel(params)
        
        # 功率分量缓存
        self.power_hover = self._calculate_hover_power()
        self.power_cruise = self._calculate_cruise_power()
    
    def _calculate_hover_power(self) -> float:
        """计算悬停功率"""
        # 悬停推力 = mg
        thrust_hover = self.params.mass * 9.81
        
        # 计算转速
        rpm_hover = self.motor.thrust_to_rpm(thrust_hover / self.params.num_rotors)
        
        # 单旋翼机械功率
        power_per_rotor_mech = self.motor.calculate_mechanical_power(
            thrust_hover / self.params.num_rotors, rpm_hover
        )
        
        # 总机械功率
        total_mech_power = power_per_rotor_mech * self.params.num_rotors
        
        # 考虑电机和螺旋桨效率
        power_hover = total_mech_power / (self.params.motor_efficiency_max * 
                                          self.params.propeller_efficiency)
        
        return power_hover  # (W)
    
    def _calculate_cruise_power(self) -> float:
        """计算巡航功率"""
        # 水平巡航速度
        V_cruise = self.params.cruise_speed
        
        # 估算巡航气动阻力
        q = 0.5 * self.aero.rho * V_cruise**2
        S = self.params.rotor_area * self.params.num_rotors
        D = q * S * self.params.drag_coefficient_body
        
        # 阻力功率
        power_drag = D * V_cruise
        
        # 悬停功率 (基础)
        power_hover = self.power_hover
        
        # 总巡航功率 (悬停功率 + 阻力功率，但由于倾旋可利用升力)
        power_cruise = power_hover * 0.7 + power_drag * 0.3
        
        return power_cruise  # (W)
    
    def calculate_segment_power(self, 
                                position_from: Tuple[float, float, float],
                                position_to: Tuple[float, float, float],
                                wind: WindVector,
                                dt: float = 1.0) -> dict:
        """
        计算一段路径的平均功率分解
        
        参数:
            position_from: 起点 (x, y, z) (m)
            position_to: 终点 (x, y, z) (m)
            wind: 风场向量
            dt: 时间段 (s)
            
        返回:
            功率分量字典
        """
        # 计算位移向量
        dx = position_to[0] - position_from[0]
        dy = position_to[1] - position_from[1]
        dz = position_to[2] - position_from[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if dt == 0:
            return {k: 0.0 for k in ['total', 'hover', 'translational', 'climb', 'turn', 'wind']}
        
        # 地速
        v_ground = np.array([dx, dy, dz]) / dt
        ground_speed = np.linalg.norm(v_ground)
        
        if ground_speed == 0:
            return {
                'total': self.power_hover,
                'hover': self.power_hover,
                'translational': 0.0,
                'climb': 0.0,
                'turn': 0.0,
                'wind': 0.0
            }
        
        # 计算空速
        v_air = self.aero.calculate_airspeed(v_ground, wind)
        airspeed = np.linalg.norm(v_air)
        
        # 1. 悬停功率 (基础)
        power_hover = self.power_hover * 0.6  # 巡航时悬停功率降低
        
        # 2. 平飞功率 (气动阻力)
        if airspeed > 0:
            q = 0.5 * self.aero.rho * airspeed**2
            S = self.params.rotor_area * self.params.num_rotors
            
            # 计算攻角
            alpha = self.aero.calculate_angle_of_attack(v_air)
            L, D = self.aero.calculate_lift_drag(v_air, alpha)
            
            power_translational = D * airspeed
        else:
            power_translational = 0.0
        
        # 3. 爬升功率
        power_climb = abs(dz) / dt * self.params.mass * 9.81 if dt > 0 else 0.0
        
        # 4. 转向功率 (简化模型)
        # 转向角通过水平方向变化估算
        v_horizontal = np.array([dx, dy, 0]) / dt
        if np.linalg.norm(v_horizontal) > 0:
            # 假设转向产生额外的惯性力矩
            power_turn = 1000.0 * np.abs(dx * dy) / (dist + 1.0)  # 简化模型
        else:
            power_turn = 0.0
        
        # 5. 风场影响功率
        wind_magnitude = wind.magnitude
        if wind_magnitude > 0 and ground_speed > 0:
            # 计算逆风/顺风比例
            v_ground_norm = v_ground / ground_speed
            wind_norm = wind.to_array() / wind_magnitude
            dot_product = np.dot(v_ground_norm, wind_norm)
            
            if dot_product < 0:  # 逆风
                wind_penalty = abs(dot_product) * 0.3
            else:  # 顺风
                wind_penalty = 0.0
            
            power_wind = power_hover * wind_penalty
        else:
            power_wind = 0.0
        
        # 总功率
        power_total = power_hover + power_translational + power_climb + power_turn + power_wind
        
        return {
            'total': power_total,
            'hover': power_hover,
            'translational': power_translational,
            'climb': power_climb,
            'turn': power_turn,
            'wind': power_wind
        }
    
    def calculate_path_energy(self, 
                             path: list,
                             wind_field: object = None,
                             dt_per_segment: float = 1.0) -> dict:
        """
        计算整条路径的能耗
        
        参数:
            path: 路径节点列表 [(x,y,z), ...]
            wind_field: 风场对象 (可选)
            dt_per_segment: 每段的时间 (s)
            
        返回:
            能耗统计字典
        """
        if len(path) < 2:
            return {'total_energy': 0.0, 'total_power_avg': 0.0}
        
        total_energy = 0.0
        power_components = {
            'hover': 0.0,
            'translational': 0.0,
            'climb': 0.0,
            'turn': 0.0,
            'wind': 0.0
        }
        
        for i in range(len(path) - 1):
            pos_from = path[i]
            pos_to = path[i+1]
            
            # 获取该位置的风场 (如果有风场对象)
            wind = WindVector()
            if wind_field is not None:
                vx, vy, vz = wind_field.get_wind_at(
                    pos_to[0], pos_to[1], pos_to[2], 
                    time_sec=i * dt_per_segment
                )
                wind = WindVector(vx, vy, vz)
            
            # 计算功率
            power_dict = self.calculate_segment_power(pos_from, pos_to, wind, dt_per_segment)
            
            # 累加能量
            energy_segment = power_dict['total'] * dt_per_segment
            total_energy += energy_segment
            
            # 累加分量
            for key in power_components:
                power_components[key] += power_dict[key] * dt_per_segment
        
        total_power_avg = total_energy / (len(path) * dt_per_segment)
        
        return {
            'total_energy': total_energy,  # (J)
            'total_energy_kwh': total_energy / 3.6e6,  # (kWh)
            'total_power_avg': total_power_avg,  # (W)
            'energy_components': {k: v / 3.6e6 for k, v in power_components.items()},  # (kWh)
            'power_components_avg': {k: v / (len(path) * dt_per_segment) for k, v in power_components.items()}  # (W)
        }
    
    def get_max_flight_time(self) -> float:
        """
        计算最大飞行时间 (基于电池容量和平均功率)
        """
        energy_total = self.params.battery_capacity * 3.6e6  # (J)
        power_avg = (self.power_hover + self.power_cruise) / 2.0  # (W)
        return energy_total / power_avg  # (s)
    
    def get_max_range(self) -> float:
        """
        计算最大航程 (km)
        """
        flight_time = self.get_max_flight_time()  # (s)
        distance = flight_time * self.params.cruise_speed  # (m)
        return distance / 1000.0  # (km)
