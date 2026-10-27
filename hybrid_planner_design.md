# 混合路径规划系统设计方案

## 架构概述

```
┌─────────────────────────────────────────────────────────┐
│                   用户输入                              │
│  起点/终点/时间约束/性能要求                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          阶段1: 几何路径搜索 (快速)                      │
│  ┌────────────────────────────────────────────────┐   │
│  │ V5.1简化规划器                                │   │
│  │ - 静态障碍避障                                │   │
│  │ - 安全距离约束                                 │   │
│  │ - 多路径生成(3-5条候选)                        │   │
│  │ - 搜索时间: <1秒                              │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │ 输出: 候选路径集合
                     │   [(path1, metadata1), ...]
                     ▼
┌─────────────────────────────────────────────────────────┐
│          阶段2: 物理评估与优化 (精细)                     │
│  ┌────────────────────────────────────────────────┐   │
│  │ 物理评估器                                    │   │
│  │ ├─ 六自由度运动学                              │   │
│  │ ├─ 气动力计算                                  │   │
│  │ ├─ 功率消耗模型                                │   │
│  │ ├─ 电池SOC预测                                 │   │
│  │ ├─ 电机效率分析                                │   │
│  │ └─ 飞行可行性检查                              │   │
│  └────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────┐   │
│  │ 优化器                                        │   │
│  │ - 能耗最优路径选择                              │   │
│  │ - 参数敏感度分析                                │   │
│  │ - 路径平滑建议                                 │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │ 输出: 详细评估报告
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   可视化与决策支持                       │
│  - 路径3D显示                                       │
│  - 能耗剖面图                                        │
│  - 效率分析报告                                       │
│  - 推荐路径                                          │
└─────────────────────────────────────────────────────────┘
```

## 阶段1: 几何路径搜索器

### 核心功能
1. **多路径生成**: 返回3-5条不同的候选路径
2. **元数据保存**: 每条路径记录几何信息
3. **快速搜索**: 保持V5.1的高效性

### 候选路径策略
```python
# 策略1: 不同代价权重组合
paths = []
for alpha_safety in [0.2, 0.35, 0.5]:
    for alpha_height in [0.15, 0.25, 0.35]:
        config = PathPlanningConfig(
            alpha1=0.3,  # 转向
            alpha2=alpha_height,  # 高度
            alpha3=alpha_safety,  # 安全
            alpha4=0.15  # 距离
        )
        path = planner.find_path(start, goal)
        if path:
            paths.append({
                'path': path,
                'config': config,
                'type': 'weighted'
            })

# 策略2: 不同巡航高度
for cruise_height in [150, 200, 250]:
    path = planner.find_path_with_height_constraint(start, goal, cruise_height)
    paths.append({'path': path, 'height': cruise_height, 'type': 'cruise'})

# 策略3: 不同避障方式
path_strict = planner.find_path_strict(start, goal)   # 严格避障
path_relaxed = planner.find_path_relaxed(start, goal) # 宽松避障
paths.extend([path_strict, path_relaxed])
```

### 路径元数据
```python
path_metadata = {
    'waypoints': [(x1,y1,z1), (x2,y2,z2), ...],  # 路径点
    'total_distance': 2817.0,                     # 总距离
    'path_length': 25,                            # 节点数
    'height_profile': [80, 80, 160, 240, ...],   # 高度序列
    'min_safety': 23.4,                          # 最小安全距离
    'max_turn_angle': 35.3,                      # 最大转向角
    'turn_count': 8,                             # 转向次数
    'climb_segments': 3,                         # 爬升段数
    'segment_lengths': [640, 640, ...],         # 段长度
    'curvature': 0.23,                           # 平均曲率
    'config_used': {...},                         # 使用的配置
    'search_time': 0.025                         # 搜索时间
}
```

---

## 阶段2: 物理评估器

### 评估流程
```python
class PhysicalEvaluator:
    def evaluate_path(self, path: List, metadata: Dict) -> Dict:
        """
        对单条路径进行完整物理评估
        """
        results = {}

        # 1. 轨迹生成 (插值到1m精度)
        trajectory = self._interpolate_trajectory(path, dt=0.1)

        # 2. 运动学仿真
        kinematics = self._simulate_kinematics(trajectory)

        # 3. 气动力计算
        aerodynamics = self._calculate_aerodynamics(trajectory, kinematics)

        # 4. 功率消耗分析
        power_profile = self._calculate_power_consumption(
            trajectory, kinematics, aerodynamics
        )

        # 5. 电池SOC预测
        battery_state = self._predict_battery_soc(power_profile)

        # 6. 电机效率分析
        motor_efficiency = self._analyze_motor_efficiency(power_profile)

        # 7. 可行性检查
        feasibility = self._check_feasibility(
            kinematics, power_profile, battery_state
        )

        # 8. 汇总报告
        results.update({
            'trajectory': trajectory,
            'kinematics': kinematics,
            'aerodynamics': aerodynamics,
            'power_profile': power_profile,
            'battery': battery_state,
            'motor_efficiency': motor_efficiency,
            'feasibility': feasibility,
            'total_energy': sum(p['total'] for p in power_profile),
            'flight_time': trajectory[-1]['time']
        })

        return results
```

### 详细评估模块

#### 2.1 轨迹插值
```python
def _interpolate_trajectory(self, waypoints: List, dt: float = 0.1) -> List:
    """
    将稀疏路点插值为密集轨迹 (0.1秒间隔)

    输出:
        [
            {'time': 0.0, 'pos': (x,y,z), 'vel': (vx,vy,vz)},
            {'time': 0.1, 'pos': (x,y,z), 'vel': (vx,vy,vz)},
            ...
        ]
    """
    trajectory = []
    t = 0.0

    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]

        # 计算距离和时间
        dist = np.linalg.norm(p2 - p1)
        segment_time = dist / cruise_speed
        num_steps = int(segment_time / dt)

        # 线性插值
        for step in range(num_steps):
            ratio = step / num_steps
            pos = p1 + ratio * (p2 - p1)
            vel = (p2 - p1) / segment_time

            trajectory.append({
                'time': t + step * dt,
                'pos': pos,
                'vel': vel
            })

        t += segment_time

    return trajectory
```

#### 2.2 六自由度运动学
```python
def _simulate_kinematics(self, trajectory: List) -> List:
    """
    模拟六自由度运动

    输出: 每个时刻的姿态和角速度
        [
            {'time': 0.0,
             'position': (x,y,z),
             'velocity': (u,v,w),
             'attitude': (φ,θ,ψ),
             'angular_rate': (p,q,r)},
            ...
        ]
    """
    kinematics = []

    for i, point in enumerate(trajectory):
        vel = point['vel']  # (vx,vy,vz)

        # 计算姿态角
        if i > 0:
            # 俯仰角 θ = arctan(vz / sqrt(vx²+vy²))
            horizontal_speed = np.sqrt(vel[0]**2 + vel[1]**2)
            theta = np.arctan(vel[2] / horizontal_speed)

            # 偏航角 ψ = atan2(vy, vx)
            psi = np.arctan2(vel[1], vel[0])

            # 滚转角 φ: 转向时滚转补偿
            if i >= 2:
                # 简化模型: 滚转角与转向率相关
                prev_vel = trajectory[i-1]['vel']
                turn_rate = (psi - kinematics[-1]['attitude'][2]) / dt
                phi = -0.1 * turn_rate  # 滚转补偿系数
            else:
                phi = 0.0
        else:
            phi = theta = psi = 0.0

        # 计算角速度 (简化)
        if i >= 1:
            prev_att = kinematics[-1]['attitude']
            p = (phi - prev_att[0]) / dt
            q = (theta - prev_att[1]) / dt
            r = (psi - prev_att[2]) / dt
        else:
            p = q = r = 0.0

        kinematics.append({
            'time': point['time'],
            'position': point['pos'],
            'velocity': vel,
            'attitude': (phi, theta, psi),
            'angular_rate': (p, q, r)
        })

    return kinematics
```

#### 2.3 气动力计算
```python
def _calculate_aerodynamics(self, trajectory: List, kinematics: List) -> List:
    """
    计算每个时刻的气动力

    输出: 升力、阻力、力矩
    """
    aero = []

    for i, (traj, kin) in enumerate(zip(trajectory, kinematics)):
        vel = traj['vel']
        airspeed = np.linalg.norm(vel)

        if airspeed < 0.1:
            aero.append({'lift': 0, 'drag': 0, 'moment': (0,0,0)})
            continue

        # 攻角 α (简化: 与俯仰角相关)
        theta = kin['attitude'][1]
        alpha = theta  # 小角度近似

        # 升力: L = ½·ρ·v²·S·CL
        CL = self.lift_slope * alpha
        lift = 0.5 * self.rho * airspeed**2 * self.wing_area * CL

        # 阻力: D = ½·ρ·v²·S·(CD0 + k·CL²)
        CD = self.CD0 + self.induced_drag_factor * CL**2
        drag = 0.5 * self.rho * airspeed**2 * self.wing_area * CD

        # 力矩 (简化)
        moment = (0, 0, 0)

        aero.append({
            'lift': lift,
            'drag': drag,
            'lift_coefficient': CL,
            'drag_coefficient': CD,
            'moment': moment
        })

    return aero
```

#### 2.4 功率消耗分析
```python
def _calculate_power_consumption(self, trajectory: List,
                                 kinematics: List,
                                 aero: List) -> List:
    """
    计算每个时刻的功率消耗

    输出: 各组件的功率消耗
        [
            {'hover': 5000, 'climb': 2000, 'drag': 500, 'total': 7500},
            ...
        ]
    """
    power = []

    for i, (traj, kin, a) in enumerate(zip(trajectory, kinematics, aero)):
        vel = traj['vel']
        airspeed = np.linalg.norm(vel)

        # 1. 悬停功率 (克服重力)
        hover_power = (aircraft_mass * g) * hover_power_per_kg

        # 2. 爬升功率 (克服重力分量)
        if airspeed > 0.1:
            climb_power = aircraft_mass * g * vel[2]
        else:
            climb_power = 0.0

        # 3. 前飞功率 (克服阻力)
        forward_power = a['drag'] * airspeed

        # 4. 总功率
        total_power = hover_power + climb_power + forward_power

        power.append({
            'hover': hover_power,
            'climb': climb_power,
            'drag': forward_power,
            'total': total_power
        })

    return power
```

#### 2.5 电池SOC预测
```python
def _predict_battery_soc(self, power_profile: List) -> Dict:
    """
    预测电池SOC变化

    输出: SOC时间序列、最终SOC、总能耗
    """
    soc = [1.0]  # 初始SOC = 100%
    energy_used = 0.0

    for i, p in enumerate(power_profile):
        dt = 0.1  # 时间步长

        # 计算当前SOC下的效率
        current_soc = soc[-1]
        efficiency = self.battery_efficiency(current_soc)

        # 计算消耗能量
        energy = p['total'] * dt * efficiency
        energy_used += energy

        # 更新SOC
        new_soc = 1.0 - energy_used / battery_capacity
        soc.append(max(0.0, new_soc))

    return {
        'soc_profile': soc,
        'final_soc': soc[-1],
        'total_energy': energy_used,
        'battery_capacity': battery_capacity,
        'min_soc': min(soc),
        'avg_efficiency': np.mean(soc)
    }
```

#### 2.6 可行性检查
```python
def _check_feasibility(self, kinematics: List,
                       power_profile: List,
                       battery: Dict) -> Dict:
    """
    检查路径的可行性

    返回: 可行性标志和警告信息
    """
    checks = {
        'feasible': True,
        'warnings': [],
        'critical': []
    }

    # 1. 姿态角约束
    max_phi = max(abs(k['attitude'][0]) for k in kinematics)
    max_theta = max(abs(k['attitude'][1]) for k in kinematics)

    if max_phi > np.radians(45):
        checks['critical'].append(f'最大滚转角 {np.degrees(max_phi):.1f}° 超限')
        checks['feasible'] = False

    if max_theta > np.radians(30):
        checks['warnings'].append(f'最大俯仰角 {np.degrees(max_theta):.1f}° 较大')

    # 2. 功率约束
    max_power = max(p['total'] for p in power_profile)
    if max_power > motor_max_power * 16:  # 16个旋翼
        checks['critical'].append(f'峰值功率 {max_power:.0f}W 超限')
        checks['feasible'] = False

    # 3. 电池约束
    if battery['final_soc'] < 0.1:  # SOC < 10%
        checks['critical'].append(f'电池SOC不足: {battery["final_soc"]*100:.1f}%')
        checks['feasible'] = False
    elif battery['final_soc'] < 0.2:  # SOC < 20%
        checks['warnings'].append(f'电池SOC较低: {battery["final_soc"]*100:.1f}%')

    # 4. 转向率约束
    max_turn_rate = max(abs(k['angular_rate'][2]) for k in kinematics)
    if max_turn_rate > np.radians(30):  # >30°/s
        checks['warnings'].append(f'最大转向率 {np.degrees(max_turn_rate):.1f}°/s')

    return checks
```

---

## 输出报告格式

### 路径对比报告
```json
{
  "comparison": {
    "best_energy": {
      "path_index": 2,
      "total_energy": "45.3 kWh",
      "flight_time": "5m 32s",
      "final_soc": "62.3%"
    },
    "best_time": {
      "path_index": 0,
      "total_energy": "48.7 kWh",
      "flight_time": "5m 15s",
      "final_soc": "59.4%"
    },
    "best_safety": {
      "path_index": 3,
      "total_energy": "47.2 kWh",
      "flight_time": "5m 45s",
      "final_soc": "60.7%",
      "min_safety_distance": "45.2 m"
    }
  },
  "detailed_analysis": [
    {
      "path_id": 0,
      "geometry": {
        "total_distance": "2817 m",
        "max_height": "240 m",
        "turns": 8
      },
      "energy": {
        "total": "48.7 kWh",
        "hover": "35.2 kWh",
        "climb": "8.5 kWh",
        "drag": "5.0 kWh",
        "peak_power": "125 kW",
        "avg_power": "88 kW"
      },
      "battery": {
        "initial_soc": "100%",
        "final_soc": "59.4%",
        "min_soc": "59.4%",
        "capacity_used": "40.6%"
      },
      "motor_efficiency": {
        "avg": "85.3%",
        "min": "82.1%",
        "max": "88.7%"
      },
      "kinematics": {
        "max_pitch": "12.3°",
        "max_roll": "8.5°",
        "max_turn_rate": "25.3°/s"
      },
      "feasibility": {
        "status": "feasible",
        "warnings": ["最大俯仰角12.3°接近限值"],
        "critical": []
      }
    },
    // ... 其他路径
  ],
  "recommendation": {
    "selected_path": 2,
    "reason": "在满足所有约束条件下能耗最低",
    "parameters": {
      "cruise_height": "200 m",
      "safety_buffer": "50 m",
      "turn_radius": "150 m"
    }
  }
}
```

---

## 实现优先级

### Phase 1: 核心功能 (1-2天)
1. 扩展V5.1支持多路径生成
2. 实现轨迹插值模块
3. 基础运动学仿真

### Phase 2: 物理评估 (2-3天)
4. 气动力计算
5. 功率消耗模型
6. 电池SOC预测

### Phase 3: 分析与可视化 (1-2天)
7. 可行性检查
8. 路径对比报告
9. 简单可视化(能耗剖面)

### Phase 4: 优化 (可选)
10. 参数敏感度分析
11. 路径平滑建议
12. 交互式可视化

---

## 性能预期

| 阶段 | 计算量 | 耗时 | 输出 |
|-----|-------|-----|-----|
| 几何搜索(5条路径) | 5 × 25节点 × 50ops | ~0.1秒 | 候选路径 |
| 轨迹插值(5条×500点) | 2500点 × 插值 | ~0.5秒 | 密集轨迹 |
| 运动学仿真 | 2500点 × 30ops | ~0.3秒 | 姿态数据 |
| 气动力计算 | 2500点 × 50ops | ~0.5秒 | 升阻力数据 |
| 功率计算 | 2500点 × 40ops | ~0.4秒 | 功率剖面 |
| **总计** | | **~1.8秒** | 完整报告 |

**对比V5**: 从**无法运行**到**2秒输出详细报告**
