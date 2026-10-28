# -*- coding: utf-8 -*-
"""最终验收评估：硬终止口径（GB 38031-2022 高度带，超 10m 即违规终止）。

评估一律 deterministic=False —— shield 下确定性策略有死循环伪影
（阻挡→obs 不变→同动作循环，见 README 第九节）。

用法：
    python eval_final.py --model ppo_dagger.zip --episodes 100
    python eval_final.py --model ppo_dagger.zip --episodes 20 --seed0 9000   # 复现交接数字
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import EnvConfig  # noqa: E402
from pretrain_bc import make_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_dagger.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=20000, help="评估种子起点（各集 seed=seed0+ep）")
    ap.add_argument("--hard-limit", type=float, default=10.0, help="高度硬终止阈值 m（inf=关闭硬终止）")
    ap.add_argument("--soft", action="store_true",
                    help="软罚口径：hard_limit=inf，逐集记录最大越带深度，"
                         "另报『软到达且全程 ≤硬限』的硬口径折算到达率")
    ap.add_argument("--deterministic", action="store_true",
                    help="确定性评估（部署口径）。配合 stuck 检测：连续 30 步被 shield 阻挡"
                         "判 stuck（确定性策略对相同 obs 永远输出同一动作，无检测会磨到超时）")
    ap.add_argument("--stuck-steps", type=int, default=30)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model_path = args.model if os.path.isabs(args.model) else os.path.join(_HERE, args.model)
    model = PPO.load(model_path)

    cfg = EnvConfig()
    if args.soft:
        cfg.height_hard_limit = float("inf")
    else:
        cfg.height_hard_limit = args.hard_limit  # 硬口径：违规 > 阈值即终止
    env = make_env(cfg)

    res = {}
    energies, steps = [], []
    soft_max_viol = []          # 软口径下每集最大越带深度（硬口径折算用）
    outcomes = []               # 每集结局 reason（硬口径折算用）
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed0 + ep)
        e = 0.0
        max_viol = 0.0
        stuck_run = 0
        while True:
            a, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(int(a))
            e += info.get("energy_j", 0.0)
            max_viol = max(max_viol, float(info.get("height_violation_m", 0.0)))
            # shield 阻挡 = 本步原地不动；确定性策略会无限重复同一动作
            stuck_run = stuck_run + 1 if info.get("collided") else 0
            if stuck_run >= args.stuck_steps:
                k, term = "stuck", True
            if term or trunc:
                k = info.get("reason", "?")
                res[k] = res.get(k, 0) + 1
                energies.append(e / 3.6e6)
                steps.append(info["step"])
                soft_max_viol.append(max_viol)
                outcomes.append(k)
                print(f"  ep{ep:3d}: {k:16s} {info['step']:3d}步 "
                      f"剩余{info['distance']:7.1f}m 能耗{e / 3.6e6:.3f}kWh "
                      f"最大越带{max_viol:5.1f}m", flush=True)
                break

    n = args.episodes
    print("=" * 56)
    tag = f"软罚口径（hard_limit=inf, stochastic, {n} 集, seed0={args.seed0}）" if args.soft \
        else f"硬口径评估（hard_limit={args.hard_limit}m, stochastic, {n} 集, seed0={args.seed0}）"
    print(tag)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s}: {v:3d} ({v / n:.0%})")
    print(f"  到达率   : {res.get('arrived', 0) / n:.1%}")
    if args.soft:
        clean = sum(1 for i in range(n)
                    if outcomes[i] == "arrived" and soft_max_viol[i] <= args.hard_limit)
        print(f"  硬口径折算（软到达且全程越带≤{args.hard_limit}m）: "
              f"{clean} ({clean / n:.1%})")
        print(f"  越带>硬限的集数（软口径下靠软罚救回的）: "
              f"{sum(1 for v in soft_max_viol if v > args.hard_limit)}")
    if energies:
        print(f"  平均能耗 : {np.mean(energies):.3f} kWh")
        print(f"  平均步数 : {np.mean(steps):.1f}")


if __name__ == "__main__":
    main()
