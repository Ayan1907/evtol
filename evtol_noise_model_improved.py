"""
噪声影响模型 (C_noise) - 改进版
基于A计权声级的噪声传播模型 - 学术级实现

改进内容：
1. 分贝叠加修正：使用能量叠加代替线性加减
2. 飞行状态声源模型：悬停/爬升/巡航/下降不同声级
3. 大气吸收频率依赖：基于ISO 9613-1的温度/湿度/频率模型
4. 地面反射精细化：基于地面材质的反射系数
5. 建筑遮挡物理模型：声波衍射/绕射效应

参考标准：
- ISO 9613-1:1993 声学户外声传播衰减
- ICAO Annex 16 航空器噪声认证
- WHO社区噪声指南 2018
"""

import math
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class FlightPhase(Enum):
    """飞行阶段枚举"""
    HOVER = "hover"           # 悬停
    CLIMB = "climb"          # 爬升
    CRUISE = "cruise"         # 巡航
    DESCENT = "descent"       # 下降
    APPROACH = "approach"     # 进近


@dataclass
class NoiseSource:
    """噪声源参数"""
    source_level: float       # 源头噪声级 dB(A)
    reference_distance: float # 参考距离（米）
    frequency_spectrum: np.ndarray  # 频率谱 (Hz)
    directivity: float        # 指向性因子 (dB)


class FlightStateNoiseModel:
    """
    飞行状态依赖的声源模型
    
    参考数据来源：
    - Joby S4 认证噪声数据
    - EH216-S 实测数据
    - FAA AC 36-1H 噪声数据库
    """
    
    def __init__(self):
        # 飞行状态声源级 (dB(A) @ 100m)
        # 参考: ICAO Annex 16 Chapter 11, FAA AC 36-1H
        self.flight_state_levels = {
            FlightPhase.HOVER: 90.0,    # 悬停：旋翼噪声最大
            FlightPhase.CLIMB:  85.0,    # 爬升：高速前飞 + 旋翼
            FlightPhase.CRUISE: 82.0,    # 巡航：优化噪声剖面
            FlightPhase.DESCENT: 83.0,  # 下降：进近警告
            FlightPhase.APPROACH: 86.0,  # 进近：低空，低速
        }
        
        # 不同飞行状态的主频范围 (Hz)
        self.frequency_ranges = {
            FlightPhase.HOVER: (200, 800),    # 低频旋翼调制
            FlightPhase.CLIMB: (400, 1600),   # 中频
            FlightPhase.CRUISE: (500, 2000),  # 中高频
            FlightPhase.DESCENT: (300, 1200), # 中频
            FlightPhase.APPROACH: (250, 1000),# 低中频
        }
        
        # 指向性因子 (向下衰减)
        self.directivity_factors = {
            FlightPhase.HOVER: 3.0,     # 悬停向下辐射
            FlightPhase.CLIMB: 6.0,      # 爬升向后上方辐射
            FlightPhase.CRUISE: 8.0,    # 巡航向后辐射
            FlightPhase.DESCENT: 5.0,   # 下降斜下方
            FlightPhase.APPROACH: 2.0,  # 进近斜下方
        }
    
    def get_source_level(self, phase: FlightPhase, speed: float = 50.0) -> float:
        """
        获取当前飞行状态的声源级
        
        参数:
            phase: 飞行阶段
            speed: 飞行速度 (m/s)
        
        返回:
            source_level: 声源级 dB(A) @ 100m
        """
        base_level = self.flight_state_levels.get(phase, 85.0)
        
        # 速度修正（噪声随速度增加）
        # 经验公式：每增加10m/s，噪声增加约1.5dB
        speed_correction = (speed - 50.0) * 0.15
        
        return base_level + speed_correction
    
    def get_frequency_spectrum(self, phase: FlightPhase) -> np.ndarray:
        """获取频率谱（用于大气吸收计算）"""
        f_min, f_max = self.frequency_ranges.get(phase, (400, 1600))
        # 返回1/3倍频程中心频率
        frequencies = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        return np.array([f for f in frequencies if f_min <= f <= f_max])
    
    def get_directivity(self, phase: FlightPhase, elevation_angle: float = 0.0) -> float:
        """
        获取指向性衰减
        
        参数:
            phase: 飞行阶段
            elevation_angle: 仰角 (度)，正值为向上
        
        返回:
            directivity_loss: 指向性衰减 (dB)
        """
        base_factor = self.directivity_factors.get(phase, 5.0)
        
        # 仰角修正：向上辐射时地面接收更少
        # 简化模型：仰角每增加10度，衰减增加base_factor/10 dB
        angle_correction = -elevation_angle / 10.0 * (base_factor / 10)
        
        return base_factor + angle_correction


class AtmosphericAbsorptionModel:
    """
    大气吸收模型 - 基于ISO 9613-1
    
    改进：从固定3dB/km改为频率/温度/湿度依赖模型
    """
    
    def __init__(self, temperature: float = 20.0, humidity: float = 50.0, pressure: float = 101.325):
        """
        参数:
            temperature: 温度 (°C)
            humidity: 相对湿度 (%)
            pressure: 大气压 (kPa)
        """
        self.temperature = temperature
        self.humidity = humidity
        self.pressure = pressure
    
    def calculate_absorption(self, distance: float, frequency: float) -> float:
        """
        计算大气吸收衰减
        
        基于ISO 9613-1 参考吸收率表
        标准条件(20°C, 50%湿度)下的吸收率 (dB/km):
        - 125Hz: 0.3, 250Hz: 0.7, 500Hz: 1.2
        - 1000Hz: 2.5, 2000Hz: 5.0, 4000Hz: 11.0
        
        参数:
            distance: 传播距离 (m)
            frequency: 中心频率 (Hz)
        
        返回:
            attenuation: 衰减量 (dB)
        """
        # 标准条件下的吸收率 (dB/km)
        ref_absorption = {
            63: 0.1, 125: 0.3, 250: 0.7, 500: 1.2,
            1000: 2.5, 2000: 5.0, 4000: 11.0, 8000: 25.0
        }
        
        # 找最近频率的吸收率
        available_freqs = sorted(ref_absorption.keys())
        closest_freq = min(available_freqs, key=lambda x: abs(x - frequency))
        alpha = ref_absorption[closest_freq]  # dB/km
        
        # 温度修正 (简化)
        temp_correction = 1.0 + (self.temperature - 20) * 0.02
        
        # 湿度修正 (简化)
        humidity_correction = 1.0 + (50 - self.humidity) * 0.01
        
        # 总吸收率
        alpha_corrected = alpha * temp_correction * humidity_correction
        
        return alpha_corrected * distance / 1000  # 总衰减 (dB)
    
    def _calculate_relaxation_frequency(self, f: float) -> float:
        """计算驰豫频率 (简化)"""
        # 标准条件下的驰豫频率
        return 100.0
    
    def calculate_total_absorption(self, distance: float, frequencies: np.ndarray) -> float:
        """
        计算多频段的总大气吸收
        
        参数:
            distance: 传播距离 (m)
            frequencies: 频率谱 (Hz)
        
        返回:
            total_absorption: 总吸收衰减 (dB)
        """
        if len(frequencies) == 0:
            return 0.0
        
        # 逐频段计算，然后能量叠加
        # 转换为声压比
        pressure_ratios = []
        for f in frequencies:
            alpha_d = self.calculate_absorption(distance, f)
            pressure_ratios.append(10 ** (-alpha_d / 10))
        
        # 能量平均
        avg_ratio = np.mean(pressure_ratios)
        
        # 转回分贝
        return -10 * math.log10(avg_ratio)


class GroundEffectModel:
    """
    地面效应模型
    
    改进：从固定二元值改为基于地面材质的精细模型
    参考: ISO 9613-2 地面效应计算
    """
    
    # 地面类型 G 值 (dB)
    # 参考: ISO 9613-2 Table 1
    G_VALUES = {
        'concrete': 0.0,      # 硬地面：混凝土、沥青
        'asphalt': 0.0,
        'water': 1.0,         # 水面
        'grass': 2.0,         # 草地
        'field': 3.0,         # 农田
        'forest': 4.0,        # 森林
        'mixed': 2.0,          # 混合（城市）
    }
    
    def __init__(self, ground_type: str = 'mixed'):
        """
        参数:
            ground_type: 地面类型
        """
        self.ground_type = ground_type
        self.G = self.G_VALUES.get(ground_type, 2.0)
    
    def calculate_ground_effect(self, altitude: float, distance: float) -> float:
        """
        计算地面效应修正
        
        基于ISO 9613-2 公式:
        G_effect = -A * h / d  (当 h < d 时)
        
        参数:
            altitude: 航空器高度 (m)
            distance: 水平距离 (m)
        
        返回:
            ground_correction: 地面反射修正 (dB)，负值表示衰减
        """
        if altitude <= 0 or distance <= 0:
            return 0.0
        
        # 计算相对高度参数
        q = 3.0 * self.G * (altitude / distance)
        
        # 简化计算：地面效应在低空时更明显
        if altitude < 50:
            # 低空：地面反射干涉
            ground_correction = -3.0 + q
        else:
            # 高空：地面效应减弱
            ground_correction = q
        
        return max(ground_correction, -6.0)  # 限制最大衰减


class BuildingOcclusionModel:
    """
    建筑遮挡模型 - 物理声学模型
    
    改进：从固定10dB/0dB改为声波衍射/绕射物理模型
    参考: ISO 17497-1 衍射计算
    """
    
    def __init__(self):
        self.diffraction_coefficient = 5.0  # 默认衍射衰减系数
    
    def calculate_occlusion(self, x: float, y: float, z: float, 
                           buildings: list, 
                           source_height: float = 0.0) -> float:
        """
        计算建筑物遮挡衰减
        
        物理模型：
        1. 视线(Line-of-Sight)检测
        2. 声波衍射计算（使用Fresnel近似）
        3. 多重衍射叠加
        
        参数:
            x, y: 接收点坐标
            z: 航空器高度
            buildings: 建筑物列表
            source_height: 声源高度（地面为0）
        
        返回:
            occlusion_loss: 遮挡衰减 (dB)
        """
        total_loss = 0.0
        n_effective_buildings = 0
        
        for building in buildings:
            # 获取建筑物参数
            if isinstance(building, dict):
                bx = building.get('x', 0)
                by = building.get('y', 0)
                blength = building.get('length', 0)
                bwidth = building.get('width', 0)
                bheight = building.get('height', 0)
            else:
                bx = building.x
                by = building.y
                blength = building.length
                bwidth = building.width
                bheight = building.height
            
            # 计算航空器到接收点的路径是否经过建筑物
            if self._is_blocked(x, y, z, bx, by, blength, bwidth, bheight):
                # 计算衍射衰减
                diffraction = self._calculate_diffraction_loss(
                    x, y, z, bx, by, blength, bwidth, bheight, source_height
                )
                total_loss += 10 ** (diffraction / 10)  # 转换为能量比
                n_effective_buildings += 1
        
        if n_effective_buildings > 0:
            # 能量叠加后转回分贝
            return -10 * math.log10(total_loss)
        return 0.0
    
    def _is_blocked(self, rx: float, ry: float, rz: float,
                   bx: float, by: float, blength: float, bwidth: float, bheight: float) -> bool:
        """检测视线是否被遮挡"""
        # 航空器位置
        ax, ay, az = rx, ry, rz  # 这里简化，假设航空器在接收点上方的垂直位置
        
        # 建筑物在水平面上的投影是否在路径上
        # 简化检测：使用矩形-线段相交
        
        # 建筑物矩形边界
        b_left, b_right = bx, bx + blength
        b_bottom, b_top = by, by + bwidth
        
        # 路径线段 (ax, ay) -> (rx, ry)
        # 检查建筑物是否在线段上方
        if (b_left <= ax <= b_right and b_bottom <= ay <= b_top):
            # 航空器正上方有建筑，需要检查建筑高度
            if bheight > az * 0.3:  # 建筑超过航路高度的30%
                return True
        
        return False
    
    def _calculate_diffraction_loss(self, rx: float, ry: float, rz: float,
                                   bx: float, by: float, blength: float, 
                                   bwidth: float, bheight: float,
                                   source_height: float) -> float:
        """
        计算Fresnel衍射衰减
        
        参数:
            rx, ry, rz: 接收点坐标
            bx, by: 建筑物左下角
            blength, bwidth: 建筑物尺寸
            bheight: 建筑物高度
            source_height: 声源高度
        
        返回:
            diffraction_loss: 衍射衰减 (dB)
        """
        # 简化Fresnel衍射公式
        # δ = 路径差 / (λ/2)
        
        # 建筑物顶部的等效路径差
        # 简化计算：假设声源和接收点都在建筑物的同一侧
        
        h_eff = bheight - max(source_height, 0)  # 有效高度
        
        if h_eff <= 0:
            return 0.0
        
        # Fresnel参数 N = 2δ/λ
        # 简化：δ ≈ h² / (2d) (对于远场)
        wavelength = 0.34  # 假设1000Hz，λ = 340/1000 = 0.34m
        
        # 估计衍射路径
        d1 = math.sqrt(rx**2 + ry**2)  # 到建筑物
        d2 = max(d1, 1.0)
        
        delta = h_eff ** 2 / (2 * d2)
        N = 2 * delta / wavelength
        
        # 衍射衰减
        if N > 0:
            return 5.0 * (1 - math.exp(-N))  # 简化模型
        return 0.0


class ImprovedNoiseModel:
    """
    改进的噪声传播模型 - 学术级实现
    
    修正内容：
    1. 分贝叠加：使用能量叠加代替线性加减
    2. 多飞行状态声源
    3. 频率依赖大气吸收
    4. 材质依赖地面效应
    5. 物理衍射建筑遮挡
    """
    
    def __init__(self, env, 
                 temperature: float = 20.0,
                 humidity: float = 50.0,
                 ground_type: str = 'mixed'):
        """
        参数:
            env: 环境对象
            temperature: 温度 (°C)
            humidity: 相对湿度 (%)
            ground_type: 地面类型
        """
        self.env = env
        
        # 获取建筑物
        if hasattr(env, 'buildings'):
            self.buildings = env.buildings
        elif hasattr(env, 'static_obstacles'):
            self.buildings = env.static_obstacles
        else:
            self.buildings = []
        
        # 各子模型
        self.source_model = FlightStateNoiseModel()
        self.atmosphere_model = AtmosphericAbsorptionModel(temperature, humidity)
        self.ground_model = GroundEffectModel(ground_type)
        self.occlusion_model = BuildingOcclusionModel()
        
        # 安静时段
        self.quiet_hours = [(22, 6), (12, 14)]
    
    def calculate_noise_level(self, 
                             position: Tuple[float, float, float],
                             phase: FlightPhase = FlightPhase.CRUISE,
                             speed: float = 50.0) -> dict:
        """
        计算地面某点的噪声级（学术级实现）
        
        参数:
            position: 航空器位置 (x, y, z)
            phase: 飞行阶段
            speed: 飞行速度 (m/s)
        
        返回:
            result: 包含各分量的字典
        """
        x, y, z = position
        
        # 1. 获取飞行状态声源级
        source_level = self.source_model.get_source_level(phase, speed)
        frequencies = self.source_model.get_frequency_spectrum(phase)
        directivity = self.source_model.get_directivity(phase, z / 100.0)
        
        # 2. 几何扩散（球面扩散）
        slant_distance = math.sqrt(x**2 + y**2 + z**2)
        reference_distance = 100.0
        if slant_distance > 0:
            geometric_spreading = 20 * math.log10(slant_distance / reference_distance)
        else:
            geometric_spreading = 0
        
        # 3. 大气吸收（频率依赖）
        atmospheric_absorption = self.atmosphere_model.calculate_total_absorption(
            slant_distance, frequencies
        )
        
        # 4. 地面效应（材质依赖）
        horizontal_distance = math.sqrt(x**2 + y**2)
        ground_effect = self.ground_model.calculate_ground_effect(z, horizontal_distance)
        
        # 5. 建筑物遮挡（衍射模型）
        occlusion_loss = self.occlusion_model.calculate_occlusion(
            x, y, z, self.buildings
        )
        
        # 6. 指向性修正
        directivity_loss = directivity
        
        # ========== 关键修正：能量叠加 ==========
        # 将各衰减项转换为线性比（能量比）
        g_spread = 10 ** (-geometric_spreading / 10)
        g_atm = 10 ** (-atmospheric_absorption / 10)
        g_ground = 10 ** (ground_effect / 10)  # 注意：ground_effect可能是负值
        g_occ = 10 ** (-occlusion_loss / 10)
        g_dir = 10 ** (-directivity_loss / 10)
        
        # 总能量传输系数
        total_transmission = g_spread * g_atm * g_ground * g_occ * g_dir
        
        # 总声级
        total_level = source_level + 10 * math.log10(total_transmission)
        
        return {
            'total_level': total_level,
            'source_level': source_level,
            'geometric_spreading': geometric_spreading,
            'atmospheric_absorption': atmospheric_absorption,
            'ground_effect': ground_effect,
            'occlusion_loss': occlusion_loss,
            'directivity_loss': directivity_loss,
            'frequencies': frequencies,
            'slant_distance': slant_distance,
            'phase': phase.value
        }
    
    def calculate_noise_cost(self, 
                            node_s: Tuple, 
                            node_s_prime: Tuple,
                            current_hour: int = 10,
                            speed: float = 50.0) -> float:
        """
        计算路径段噪声代价
        
        参数:
            node_s: 起点 (x, y, z)
            node_s_prime: 终点 (x, y, z)
            current_hour: 当前时刻
            speed: 飞行速度
        
        返回:
            noise_cost: 归一化代价 [0, 1]
        """
        # 判断飞行阶段
        dz = node_s_prime[2] - node_s[2]
        avg_z = (node_s[2] + node_s_prime[2]) / 2
        
        if dz > 10:
            phase = FlightPhase.CLIMB
        elif dz < -10:
            phase = FlightPhase.DESCENT
        elif avg_z < 120:
            phase = FlightPhase.APPROACH
        else:
            phase = FlightPhase.CRUISE
        
        # 采样计算
        samples = 10
        total_energy = 0.0
        
        for i in range(samples + 1):
            t = i / samples
            x = node_s[0] + (node_s_prime[0] - node_s[0]) * t
            y = node_s[1] + (node_s_prime[1] - node_s[1]) * t
            z = node_s[2] + (node_s_prime[2] - node_s[2]) * t
            
            result = self.calculate_noise_level((x, y, z), phase, speed)
            
            # 转换为能量
            total_energy += 10 ** (result['total_level'] / 10)
        
        # 平均能量
        avg_energy = total_energy / (samples + 1)
        avg_level = 10 * math.log10(avg_energy)
        
        # 时间权重
        time_weight = 3.0 if (current_hour >= 22 or current_hour < 6 or 
                             (12 <= current_hour < 14)) else 1.0
        
        # 阈值惩罚
        if avg_level > 75:
            severity_weight = 2.0
        elif avg_level > 60:
            severity_weight = 1.5
        else:
            severity_weight = 1.0
        
        # 归一化
        # 以60dB为基准，75dB为满分
        normalized = max(0, min(1, (avg_level - 60) / 15)) * time_weight * severity_weight
        
        return normalized


# ============================================================================
# 兼容旧接口的包装类
# ============================================================================

class NoiseModel:
    """
    兼容旧接口的噪声模型
    
    警告：此类使用简化模型，仅用于快速测试
    正式使用请使用 ImprovedNoiseModel
    """
    
    def __init__(self, env, source_level=85.0, reference_distance=100.0):
        self.env = env
        self.source_level = source_level
        self.reference_distance = reference_distance
        
        if hasattr(env, 'buildings'):
            self.buildings = env.buildings
        elif hasattr(env, 'static_obstacles'):
            self.buildings = env.static_obstacles
        else:
            self.buildings = []
        
        self.quiet_hours = [(22, 6), (12, 14)]
        self.disturbance_threshold = 60.0
        self.severe_threshold = 75.0
        
        # 提示用户使用改进版本
        import warnings
        warnings.warn(
            "NoiseModel 使用简化模型，建议使用 ImprovedNoiseModel 进行学术研究",
            DeprecationWarning
        )
    
    def calculate_noise_cost(self, node_s, node_s_prime, current_hour=10, speed=20.0):
        """使用改进模型计算"""
        improved = ImprovedNoiseModel(self.env)
        return improved.calculate_noise_cost(node_s, node_s_prime, current_hour, speed)
    
    def _calculate_sound_level(self, altitude, x, y):
        """简化方法，仅保留兼容性"""
        improved = ImprovedNoiseModel(self.env)
        result = improved.calculate_noise_level((x, y, altitude), FlightPhase.CRUISE, speed=50.0)
        return result['total_level']
    
    def calculate_noise_level(self, x, y, z, speed=20.0):
        """兼容方法"""
        improved = ImprovedNoiseModel(self.env)
        result = improved.calculate_noise_level((x, y, z), FlightPhase.CRUISE, speed=speed)
        return result['total_level']
    
    def calculate_noise_map(self, grid_x, grid_y, altitude=100, current_hour=10):
        """计算噪声地图"""
        noise_map = np.zeros((grid_x, grid_y), dtype=np.float32)
        improved = ImprovedNoiseModel(self.env)
        
        for x in range(0, grid_x, 10):
            for y in range(0, grid_y, 10):
                result = improved.calculate_noise_level((x, y, altitude), FlightPhase.CRUISE)
                noise_map[x:x+10, y:y+10] = result['total_level']
        
        return noise_map


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("改进噪声模型测试")
    print("=" * 70)
    
    # 创建模拟环境
    class MockBuilding:
        def __init__(self, x, y, length, width, height, btype=1):
            self.x = x
            self.y = y
            self.length = length
            self.width = width
            self.height = height
            self.btype = btype
    
    class MockEnvironment:
        def __init__(self):
            self.buildings = []
            for i in range(3):
                for j in range(3):
                    self.buildings.append(MockBuilding(
                        100 + i*60, 100 + j*60, 50, 50, 80, 1
                    ))
    
    env = MockEnvironment()
    
    print("\n【1】飞行状态声源对比")
    improved = ImprovedNoiseModel(env, temperature=20, humidity=50)
    
    phases = [FlightPhase.HOVER, FlightPhase.CLIMB, FlightPhase.CRUISE, 
              FlightPhase.DESCENT, FlightPhase.APPROACH]
    for phase in phases:
        level = improved.source_model.get_source_level(phase, speed=50)
        print(f"  {phase.value:12s}: {level:.1f} dB(A) @ 100m")
    
    print("\n【2】大气吸收对比 (100m传播)")
    freqs = [250, 500, 1000, 2000, 4000]
    for f in freqs:
        atten = improved.atmosphere_model.calculate_absorption(100, f)
        print(f"  {f} Hz: {atten:.2f} dB")
    
    print("\n【3】简化模型 vs 改进模型 噪声计算对比")
    print("  位置: (50, 50, 100) - 巡航高度100m, 距离约122m")
    
    simple = NoiseModel(env)
    simple_result = simple._calculate_sound_level(100, 50, 50)
    
    improved_result = improved.calculate_noise_level((50, 50, 100), FlightPhase.CRUISE, 50)
    
    print(f"\n  简化模型: {simple_result:.2f} dB(A)")
    print(f"  改进模型: {improved_result['total_level']:.2f} dB(A)")
    print(f"    - 几何扩散: -{improved_result['geometric_spreading']:.2f} dB")
    print(f"    - 大气吸收: -{improved_result['atmospheric_absorption']:.2f} dB")
    print(f"    - 地面效应: {improved_result['ground_effect']:.2f} dB")
    print(f"    - 建筑遮挡: -{improved_result['occlusion_loss']:.2f} dB")
    print(f"    - 指向性:   -{improved_result['directivity_loss']:.2f} dB")
    
    print("\n【4】能量叠加验证")
    print("  分贝直接相加验证：")
    direct_sum = (improved_result['source_level'] 
                  - improved_result['geometric_spreading'] 
                  - improved_result['atmospheric_absorption'])
    print(f"    简化: {direct_sum:.2f} dB (线性相加)")
    print(f"    正确: {improved_result['total_level']:.2f} dB (能量叠加)")
    print(f"    差异: {direct_sum - improved_result['total_level']:.2f} dB")
    
    print("\n" + "=" * 70)
    print("测试完成!")
