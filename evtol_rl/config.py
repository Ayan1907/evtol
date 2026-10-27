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
    start_altitude: float = None          # 课程学习起点高度（None=地图默认 80m）
    shield: bool = True                   # 动作屏蔽（离散 shield）：碰撞→原地阻挡+小罚，训练期零死亡
    blocked_penalty: float = -0.5         # 被阻挡一步的惩罚（介于步惩罚与终端之间）

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

    # ---------- 奖励函数（v1 设计 2026-10-27，依据 .research/rl-reward-design-20261027）----------
    #   聚合 = 三层混合：合规硬终止（字典序 L1）+ 安全裕度缓冲走廊（L2）
    #          + 进度(严格PBRS)与能耗(固定E_ref归一)线性加权（主干）
    r_arrive: float = 100.0               # 到达终端奖励（文献典型 +100）
    r_collide: float = -100.0             # 撞楼/违规终端惩罚（TEeVTOL 同构 ±100）
    r_step: float = -0.01                 # 每步时间惩罚（防磨蹭；time/length 合并项的惩罚面）
    k_progress: float = 10.0              # PBRS 势幅度：Φ(s) = −K·d₃D/d₀（d₀=初始距离）\n                                          #   与 γ=0.999 配对：前进 +0.088/步 ≫ 悬停 −0.001/步\n                                          #   （Run1 K=100 γ=0.99 → 悬停蹭分；Run1b K=1 → 信号≈0 等效稀疏）
    pbrs_gamma: float = 0.999             # ⚠️ 必须与 TrainConfig.gamma 一致；0.999 匹配 116–200 步视界，\n                                          #   且 (1−γ) 残差缩小 10 倍（Run1b 教训：γ=0.99 时残差吞噬学习信号）
    w_energy: float = 0.0                 # 能耗项权重（0=关；文献锚点 0.2，engrXiv 2026）
    w_margin: float = 0.0                 # 安全裕度项权重（0=关；建议 0.05）
    margin_band: float = 20.0             # 走廊缓冲宽度（m）：距允许带边界 <20m 开始线性罚
    e_ref_kwh: float = 0.0                # 能耗归一参考 kWh（0 = 自动实测巡航单步能耗）

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
    gamma: float = 0.999                  # 与 EnvConfig.pbrs_gamma 必须一致（视界匹配 200 步 episode）
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    seed: int = 0

    log_dir: str = "runs/"
    save_path: str = "ppo_evtol.zip"

    # 评估
    eval_episodes: int = 50
