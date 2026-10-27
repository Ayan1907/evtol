# -*- coding: utf-8 -*-
"""
配置集中管理 —— 避免参数散落在各个脚本里。

调参时只改这里，不要在 env / train / render 里硬编码。
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """环境配置"""

    # ---------- 场景 ----------
    map_file: str = "data/maps/map_dense.json"   # 相对于 ~/evtol 根目录
    stride: float = 25.0                  # 每步位移（m）—— 决定 episode 长度
    max_steps: int = 200                  # 最大步数（截断）

    # ---------- 到达判定 ----------
    goal_radius_h: float = 50.0           # 水平到达半径（m）
    goal_radius_v: float = 50.0           # 高度到达容差（m）

    # ---------- 碰撞检测 ----------
    check_step: float = 5.0               # 线段采样间隔（m），越小越保守越慢

    # ---------- 观测 ----------
    obs_patch: int = 8                    # 局部最高建筑高度图的分辨率（K×K）
    obs_radius: float = 200.0             # 局部高度图覆盖半径（m）

    # ---------- 国标飞行剖面约束（GB 38031-2022 / FlightProfile）----------
    #   原项目 evtol_v7.evtol_path_planning_v7.FlightProfile 已实现完整剖面：
    #     爬升段（距起点 <500 m）  min_h 自起点高度线性升到 120 m
    #     巡航段                   允许带 120–180 m
    #     进近段（距终点 <500 m）  min_h 自 120 m 线性降到终点高度
    #   本环境直接复用该类，不再自造高度规则。
    use_flight_profile: bool = True       # 是否启用高度剖面约束（关闭则退回"自由飞行"）
    height_hard_limit: float = 10.0       # 超出允许带多少米即判违规终止
                                          #   设 float("inf") → 永不终止，只做软惩罚
    height_penalty: float = 2.0           # 软惩罚系数：每违规 1 米扣多少奖励

    # ---------- 物理接口 ----------
    cruise_speed: float = 50.0            # 用于由 stride 推算 dt（m/s）
    use_energy: bool = True               # 是否启用单步能耗计算
    use_wind: bool = True                 # 是否启用风场

    # ---------- 风场参数（对应 evtol_path_planning_v4.WindField）----------
    wind_speed: float = 5.0               # 基础风速（m/s）
    wind_direction: float = 315.0         # 基础风向（度，0=东，90=南，180=西，270=北）
    turbulence: float = 0.1               # 湍流强度

    # ---------- 复现 ----------
    seed: int = 42

    @property
    def dt(self) -> float:
        """每步代表的物理时间（s）"""
        return self.stride / max(self.cruise_speed, 1e-6)


@dataclass
class TrainConfig:
    """训练配置（第 2 周用）"""

    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    seed: int = 0

    log_dir: str = "runs/"
    save_path: str = "ppo_evtol.zip"

    # 评估
    eval_episodes: int = 50
