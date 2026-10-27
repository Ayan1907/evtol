# -*- coding: utf-8 -*-
"""
随机策略演示 —— 第 1 周的验收物
=========================================================================
跑一个随机策略 episode，导出：
    random_policy.gif   轨迹动画（俯视图 + 高度曲线）
    random_policy.png   末帧静态图

用途：证明"环境能跑通"（Gym 接口正确、碰撞检测生效、能耗接口接通）。
⚠️ 随机策略当然到不了终点。这张 gif 的价值是展示环境本身，不是性能。

用法：
    cd ~/evtol/evtol_rl
    python render_random.py
    python render_random.py --map map_dense.json --stride 25 --seed 1
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from evtol_env import EVTOLEnv  # noqa: E402


def setup_cjk_font(root):
    """注册原项目自带的思源黑体，避免中文变方块。"""
    from matplotlib import font_manager

    cand = os.path.join(root, "assets/fonts", "SourceHanSansSC-Regular.otf")
    if os.path.exists(cand):
        font_manager.fontManager.addfont(cand)
        plt.rcParams["font.sans-serif"] = ["Source Han Sans SC", "DejaVu Sans"]
        return True
    return False


def rollout(env, seed=0, policy="random"):
    """跑一个 episode，返回轨迹与统计。"""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)

    traj = [env.pos.copy()]
    rewards = []
    energy_j = 0.0
    reason = ""
    n_violation = 0
    # 逐步记录"允许高度带"，右侧子图要画动态剖面而非固定 120–180
    alt_min = [info.get("height_min", float("nan"))]
    alt_max = [info.get("height_max", float("nan"))]

    while True:
        if policy == "random":
            action = int(rng.integers(env.N_ACTIONS))
        elif policy == "greedy":  # 朝目标直飞的朴素基线（用于对照）
            d = env.goal - env.pos
            d = d / (np.linalg.norm(d) + 1e-9)
            action = int(np.argmax(env._DIRS_UNIT @ d))
        else:
            raise ValueError(policy)

        obs, r, term, trunc, info = env.step(action)
        traj.append(env.pos.copy())
        rewards.append(r)
        energy_j += info.get("energy_j", 0.0)
        if info.get("height_violation_m", 0.0) > 0.0:
            n_violation += 1
        alt_min.append(info.get("height_min", float("nan")))
        alt_max.append(info.get("height_max", float("nan")))
        if term or trunc:
            reason = info["reason"]
            break

    return {
        "policy": policy,
        "traj": np.array(traj),
        "rewards": rewards,
        "return": float(np.sum(rewards)),
        "energy_kwh": energy_j / 3.6e6,
        "reason": reason,
        "final_distance": float(info["distance"]),
        "steps": len(rewards),
        "alt_min": np.array(alt_min),
        "alt_max": np.array(alt_max),
        "n_violation": n_violation,
    }


def draw(env, result, out_png, out_gif, fps=12):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax_map, ax_alt = axes

    # ---------------- 左：俯视图 ----------------
    polys = [
        [(b.x, b.y), (b.x_max, b.y), (b.x_max, b.y_max), (b.x, b.y_max)]
        for b in env.city.buildings
    ]
    if polys:
        pc = PolyCollection(
            polys, facecolors="#8d99ae", edgecolors="#5a6273", linewidths=0.4, alpha=0.85
        )
        ax_map.add_collection(pc)

    ax_map.plot(*env.start[:2], "o", ms=12, color="#2a9d8f", label="起点")
    ax_map.plot(*env.goal[:2], "*", ms=20, color="#e76f51", label="终点")
    ax_map.add_patch(plt.Circle(env.goal[:2], env.goal_radius_h,
                                color="#e76f51", alpha=0.2))
    ax_map.set_xlim(0, env.size_x)
    ax_map.set_ylim(0, env.size_y)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("俯视图 · 建筑与航迹")
    ax_map.grid(alpha=0.25)
    ax_map.legend(loc="upper left", fontsize=9)

    line, = ax_map.plot([], [], "-", lw=2.0, color="#264653", label="航迹")
    dot, = ax_map.plot([], [], "o", ms=6, color="#f4a261")
    ax_map.legend(loc="upper left", fontsize=9)

    # ---------------- 右：高度剖面 ----------------
    xs = np.arange(len(result["traj"]))
    lo, hi = result["alt_min"], result["alt_max"]
    if np.isfinite(lo).any():  # 已接入 FlightProfile → 画随航段变化的允许带
        ax_alt.fill_between(xs, lo, hi, color="#a8dadc", alpha=0.45,
                            label="国标允许高度带（爬升/巡航/进近）")
        ax_alt.plot(xs, lo, lw=1.0, color="#457b9d", alpha=0.9)
        ax_alt.plot(xs, hi, lw=1.0, color="#457b9d", alpha=0.9)
    else:
        ax_alt.axhspan(120, 180, color="#a8dadc", alpha=0.45,
                       label="巡航带 120–180 m（剖面未启用）")
    ax_alt.axhline(env.goal[2], lw=1.0, ls="--", color="#e76f51", alpha=0.8,
                   label=f"终点高度 {env.goal[2]:.0f} m")
    ax_alt.set_xlim(0, max(result["steps"], 1))
    ax_alt.set_ylim(0, env.size_z * 1.05)
    ax_alt.set_xlabel("步数")
    ax_alt.set_ylabel("高度 (m)")
    ax_alt.set_title("高度剖面 · 与国标允许带对照")
    ax_alt.grid(alpha=0.25)
    ax_alt.legend(loc="upper left", fontsize=8)

    alt_line, = ax_alt.plot([], [], "-", lw=2.0, color="#e76f51")
    alt_dot, = ax_alt.plot([], [], "o", ms=6, color="#f4a261")

    txt = fig.suptitle("", fontsize=11)

    traj = result["traj"]
    n = len(traj)

    def update(frame):
        i = min(frame + 1, n)
        line.set_data(traj[:i, 0], traj[:i, 1])
        dot.set_data([traj[i - 1, 0]], [traj[i - 1, 1]])
        alt_line.set_data(np.arange(i), traj[:i, 2])
        alt_dot.set_data([i - 1], [traj[i - 1, 2]])
        band_lo, band_hi = result["alt_min"][i - 1], result["alt_max"][i - 1]
        z = traj[i - 1, 2]
        band = (f"高度 {z:.0f} m（允许 {band_lo:.0f}–{band_hi:.0f} m）"
                if np.isfinite(band_lo) else f"高度 {z:.0f} m")
        txt.set_text(
            f"{result['policy']} 策略 · 第 {i - 1}/{result['steps']} 步 · {band} · "
            f"剩余 {np.linalg.norm(env.goal - traj[i - 1]):.0f} m · "
            f"累计能耗 {result['energy_kwh']:.3f} kWh"
        )
        return line, dot, alt_line, alt_dot, txt

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    update(n - 1)
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/maps/map_standard.json")
    ap.add_argument("--stride", type=float, default=25.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="random", choices=["random", "greedy"])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--no-profile", action="store_true",
                    help="关闭国标高度剖面约束（对照用：可看出无约束时存在取巧解）")
    args = ap.parse_args()

    env = EVTOLEnv(
        map_file=args.map,
        stride=args.stride,
        max_steps=args.max_steps,
        seed=args.seed,
        use_flight_profile=not args.no_profile,
        verbose=True,
    )

    root = os.path.dirname(_HERE)
    has_cjk = setup_cjk_font(root)
    if not has_cjk:
        print("⚠️ 未找到 fonts/SourceHanSansSC-Regular.otf，中文可能显示为方块")

    result = rollout(env, seed=args.seed, policy=args.policy)

    prefix = args.out_prefix or f"{args.policy}_policy"
    out_gif = os.path.join(_HERE, f"{prefix}.gif")
    out_png = os.path.join(_HERE, f"{prefix}.png")
    draw(env, result, out_png, out_gif, fps=args.fps)

    reason_cn = {
        "arrived": "到达目标",
        "collision": "撞建筑 / 越界",
        "height_violation": "偏离国标高度剖面",
        "timeout": "超时",
    }.get(result["reason"], result["reason"])

    print("\n" + "=" * 58)
    print(f"策略: {args.policy}   地图: {args.map}")
    print(f"步数: {result['steps']}   终止原因: {result['reason']}（{reason_cn}）")
    print(f"累计奖励: {result['return']:.2f}")
    print(f"剩余距离: {result['final_distance']:.1f} m")
    print(f"累计能耗: {result['energy_kwh']:.4f} kWh")
    print(f"高度违规步数: {result['n_violation']} / {result['steps']}"
          f"   （剖面来源: {env.flight_profile_source}）")
    print(f"轨迹点数: {len(result['traj'])}")
    print(f"输出: {out_gif}")
    print(f"      {out_png}")
    print("=" * 58)


if __name__ == "__main__":
    main()
