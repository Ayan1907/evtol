# AGENTS.md — AutoClaw / 任何 AI Agent 接手本仓库必读

> 本文件是跨机器交接的单一事实来源（2026-10-28 归档）。任何 agent 在这台机器上
> 开始工作前，先读完本文件，再读 `evtol_rl/README.md`（第五至十一节）。
> **不要凭记忆或常识重做已验证的结论——所有"为什么"都在下面。**

---

## 1. 项目一句话与当前状态

在获奖 eVTOL 城市航迹规划项目（低碳 A\* + NSGA-II 离线规划，北京市一等奖）之上，
派生强化学习在线重规划能力。

**当前已达成的验收口径（2026-10-28）**：

| 口径 | 到达率 | 说明 |
|---|---|---|
| 固定场景 + 国标硬终止 10m | 95% | 100 集，457k 样本 DAgger（`ppo_dagger6.zip`） |
| **分布意义**（起终点 ±400m 扰动 + 随机风）+ 硬终止 | **91%** | 全新种子 100 集验收（`ppo_dagger_rand6.zip`），违规 8%、超时 1%，能耗 2.335 kWh |

面试话术只能写分布意义的 91%——固定场景 95% 只是采样鲁棒性（SB3 load 复位 torch
RNG，同模型评估逐字节一致，"成功率"=固定场景采样鲁棒性）。

## 2. 环境事实与硬约束

- **两台机器**：Windows 笔记本（开发，`wsl.exe` 被安全策略拦截，访问 WSL 走 UNC
  `\\wsl.localhost\Ubuntu\...`）；Linux 原生主机（运行 ROS 2/PX4/SITL 的目标机）。
  代码以本仓库为唯一同步源，**不要用网盘拷贝**。
- **下载一律清华源**；清华被掐时降级阿里云/腾讯镜像 + PowerShell 通道直下 wheel +
  sha256 校验（pip 自身 TLS 可能被无差别重置，此路已验证）。已装：torch 2.9.1+cpu /
  SB3 2.9.2 / gymnasium 1.4.0 / numpy / scipy 1.18.1。
- **训练/评估命令一律加 `-u`**（python 管道缓冲会假死）；后台任务 cwd ≠ 仓库目录，
  **相对路径的输入输出文件全部用绝对路径**（npz 曾因此载入旧副本白跑 4 轮）。
- `PPO.load` 用绝对路径；`make_env(cfg)` 返回工厂函数要再调用一次。
- Windows 下 `NpzFile` 用完必须 `.close()`，否则写回同名文件 PermissionError。

## 3. 协作守则（对齐用户既定偏好，务必遵守）

> **完整版协作规范见 skill 仓库 `SKILL4S-Personal/AGENT-COLLABORATION.md`**（与本仓库同级 clone；语言/进度条/先问后动/
> 文件操作/配色/流程图/交付流程/技术约定全量版）。本节仅为其摘要，两者冲突时以
> AGENT-COLLABORATION.md 为准。

1. 不写"把对话整理一遍"的中间产物 md；要点直接写在回复里，落盘只落用户要交付的正文文件。
2. 用户焦虑/深夜时先接住情绪再谈事，不排任务不催进度；他没问"接下来怎么办"就别主动给计划。
3. **不整体覆盖用户/其他 agent 手工编辑过的文件**——先读实际版本做增量修改；别人删掉的=没用的。
4. 引用纪律：一处引用=一个具体技术陈述，逐字核对原文，不凑数。
5. 直接结论、明确推荐（推荐/不推荐）、理由按权重排序；反感冗长叙述与反复微调。
6. 调试纪律：先写测试脚本验证再改主代码；改参数用表格对比；**每轮只动一个变量**。
7. 用户习惯深夜工作：长任务收尾检查结果、总结下一步、给简洁状态摘要。
8. 一切 Python（numpy/scipy/PyTorch），不用 MATLAB；批量文件处理写脚本；下载走清华源。
9. 敏感事项：面试/文书场合不提"因力学难而换方向"这类表述；用户举例 ≠ 他本人的选择，
   提取启发原则即可，不要拿例子当靶子反驳。
10. 固定 seed 下实验完全可复现：**"再跑一轮"必须换 `--seed`**，否则是重放不是训练。

## 4. RL 技术结论（已验证，勿推翻重来）

### 奖励函数（v1，已实现于 `evtol_rl/evtol_env.py::_compute_reward`）
- **严格 PBRS**：F_t = γΦ(s′)−Φ(s)，Φ(s)=−K·d₃D/d₀（d₀=初始距离）；**失败终止不置零**
  （终态置零约定会在撞楼步产生 +K·d/d₀ 正残差，实测抵消 −100 惩罚）。
- 能耗项 E_ref 静态归一（巡航单步实测能耗）；**不要开 VecNormalize(norm_reward=True)**。
- 安全裕度缓冲走廊（距允许带边 <20m 线性罚）+ 高度违规软罚 −2/m。
- 三层聚合：合规=硬终止（字典序 L1）+ 裕度 shaping（L2）+ 进度/能耗线性加权（主干）。

### 超参配对（拆开即失败，实测多轮）
- **γ=0.999 与 K=10 配对**。γ=0.99 时残差 (1−γ)·K·d_norm 吞掉信号（等效稀疏）；
  K=100 时残差反超前进收益 → 悬停蹭分 0% 通关。
- 改奖励前先跑 `check_reward_scale.py`：前进 +0.088 > 0 > 悬停/后退，爬升段早飞 −3.9。

### 探索问题的正解 = DAgger，不是调参
- PPO 直微调（含 lr=5e-5）：value 网络随机初始化与 BC 策略失配，30 万步横评全 0%。**勿再试**。
- 正解：`dagger_train.py`（启发式专家全程标注 + BC 重训 + stochastic 评估择优）。
- 专家三要素（缺一即残废）：**band-aware**（z<lo+5 爬升/z>hi−5 下降，策略会忠实继承专家的
  合规缺陷）；**巡航只用 155m**（130m 撞进 150m 楼群死循环，42% 超时元凶）；**壁障感知**
  （首选动作被阻挡时在未阻挡动作中选余弦最大者——随机场景下盲推被 shield 卡死仅 23/30）。
- 示范/采集必须配软罚（`height_hard_limit=inf`）；硬终止 + ε 扰动 = 196/200 示范暴毙。

### 评估协议
- `eval_final.py --model X --episodes 100 --randomize --seed0 <新值>`：硬口径 stochastic
  100 集；**每次验收换新 seed0**；确定性评估有两步极限环（A→B→A，100/100 卡死 407.2m），
  勿用，治本要给 obs 加 blocked 标志（未做，Phase 3 候选）。
- 评估口径必须与验收口径一致：软口径数字会高估一倍（53% 软 = 25% 真 hard）。

## 5. 待办与路线

1. **RL 已达标收官**；可选增强（会破坏模型兼容，动前评估）：obs 加 blocked、W_MARGIN=0.05。
2. **控制层 Phase 2（下一个大任务）**：分层架构 RL(2Hz 路标点, ONNX) → MPC 跟踪层
   （CasADi+OSQP QP，质点模型，20Hz，速度/加速度/高度带约束）→ PX4 内环（不动）。
   方案 C（NMPC 旁路 PX4 位置控制）已论证不做。三周计划与 topics 表见对话纪要，
   Week1 = 裸链路（无 MPC）先通：Ubuntu 24.04 + Jazzy + gz + PX4 v1.15/1.16 +
   MicroXRCEAgent，`/fmu/in/trajectory_setpoint` 流式跟随路标点。
3. 诚实边界：gz 用 x500 四旋翼验证控制栈，README 明写"四旋翼平台验证分层架构，
   eVTOL 动力学为三自由度质点近似"——不吹 eVTOL 仿真。

## 6. 命令速查

```bash
cd ~/evtol/evtol_rl
python evtol_env.py                       # 环境自检
python check_reward_scale.py              # 奖励梯度方向自检（改奖励后必跑）
python pretrain_bc.py --episodes 200      # 重新生成 BC（一般不需要，数据集已存）
python dagger_train.py --randomize --jitter 400 \
  --dataset-in '<绝对路径>/dagger_dataset.npz' \
  --dataset-out '<绝对路径>/dagger_dataset.npz' \
  --rounds 4 --eps-per-round 120 --bc-epochs 15 --seed <新值> \
  --save-path ppo_dagger_vN.zip
python eval_final.py --model ppo_dagger_rand6.zip --episodes 100 \
  --randomize --seed0 <新值>              # 验收（100 集硬口径 stochastic）
python train_ppo.py ...                   # PPO 直调已弃用，仅存档
```

## 7. 参考档案

- 奖励设计文献依据（TEeVTOL 奖励逐字拆解、PBRS、多目标聚合、safe RL、归一化六路调研）：
  Windows 侧工作区 `.research/rl-reward-design-20261027/`（未入仓库，需要时拷入）。
- 原项目获奖口径：低碳 A\* 路径 +1.09% 换安全风险 −86.6%、合规 0%→90%+（注意：不是
  "能耗降低 30.6%"，那是误传）；NSGA-II 30.9s / 17 个 Pareto 解 / 硬口径 HV 3.04e5；
  MOEA/D 对比与其 Tchebycheff 未归一化缺陷（`evtol_v7/test_optimizer_compare.py`）。
- 诚实边界：RL 是**新增工作**，奖项属于原方案（A\*/NSGA-II）；对外表述
  "在获奖项目基础上派生"，绝不把奖项贴到 RL 头上。
