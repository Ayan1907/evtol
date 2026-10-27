#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强可视化分析模块 - 任务二
============================

生成以下可视化图表:
1. 进化过程监控图(3合1)
2. 最终帕累托前沿分析图(3合1)
3. 解集分布热力图
4. 目标间权衡关系图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_evolution_monitoring(result, output_path: str):
    """
    绘制进化过程监控图(3合1)
    
    左上: 超体积(HV)随代数变化曲线
    右上: 帕累托解数量随代数变化柱状图
    下方: 间距(Spacing)随代数变化曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NSGA-II 进化过程监控', fontsize=14, fontweight='bold')
    
    hv_history = result.convergence_history['hypervolume']
    spacing_history = result.convergence_history['spacing']
    pareto_size_history = result.convergence_history['pareto_size']
    generations = range(1, len(hv_history) + 1)
    
    # 图1: 超体积变化曲线 (左上)
    ax = axes[0, 0]
    ax.plot(generations, hv_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(generations, hv_history, alpha=0.3)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Hypervolume (HV)', fontsize=11)
    ax.set_title('超体积随代数变化曲线', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 标注最大值
    max_hv_idx = np.argmax(hv_history)
    max_hv = hv_history[max_hv_idx]
    ax.annotate(f'Max: {max_hv:.2f}\n(Gen {max_hv_idx+1})', 
                xy=(max_hv_idx+1, max_hv), 
                xytext=(max_hv_idx+5, max_hv*0.9),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 图2: 帕累托解数量柱状图 (右上)
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pareto_size_history)))
    ax.bar(generations, pareto_size_history, color=colors, edgecolor='navy', alpha=0.8)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Number of Pareto Solutions', fontsize=11)
    ax.set_title('帕累托解数量随代数变化', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注最终数量
    final_size = pareto_size_history[-1]
    ax.axhline(y=final_size, color='red', linestyle='--', alpha=0.7, label=f'Final: {final_size}')
    ax.legend()
    
    # 图3: 间距变化曲线 (下方左)
    ax = axes[1, 0]
    ax.plot(generations, spacing_history, 'g-', linewidth=2, marker='s', markersize=4)
    ax.fill_between(generations, spacing_history, alpha=0.3, color='green')
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Spacing Metric', fontsize=11)
    ax.set_title('间距(Spacing)随代数变化曲线', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 图4: 综合评估指标 (下方右)
    ax = axes[1, 1]
    
    # 归一化各指标
    hv_norm = np.array(hv_history) / max(hv_history) if max(hv_history) > 0 else np.zeros_like(hv_history)
    spacing_inv = 1 / (1 + np.array(spacing_history))  # 越小越好
    size_norm = np.array(pareto_size_history) / max(pareto_size_history) if max(pareto_size_history) > 0 else np.zeros_like(pareto_size_history)
    
    ax.plot(generations, hv_norm, 'b-', linewidth=2, label='HV (normalized)', alpha=0.8)
    ax.plot(generations, spacing_inv, 'g-', linewidth=2, label='1/(1+Spacing)', alpha=0.8)
    ax.plot(generations, size_norm, 'r-', linewidth=2, label='Pareto Size (normalized)', alpha=0.8)
    
    # 综合得分
    composite = (hv_norm + spacing_inv + size_norm) / 3
    ax.plot(generations, composite, 'k--', linewidth=2.5, label='Composite Score', alpha=0.9)
    
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Normalized Score', fontsize=11)
    ax.set_title('综合性能指标演变', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"进化过程监控图已保存: {output_path}")
    
    return output_path


def plot_pareto_front_analysis(result, output_path: str):
    """
    绘制最终帕累托前沿分析图(3合1)
    
    左: 能耗 vs. 飞行时间 二维散点图(用颜色表示安全评分)
    中: 航程 vs. 安全评分 二维散点图(用大小表示能耗)
    右: 三个主要目标的3D散点图(能耗、时间、安全评分)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('帕累托前沿多维度分析', fontsize=14, fontweight='bold')
    
    individuals = result.pareto_front.individuals
    if not individuals:
        print("警告: 没有帕累托解可供可视化")
        return None
    
    # 【修复】提取有效目标值
    valid_individuals = []
    valid_objectives = []
    for ind in individuals:
        if ind.objectives is not None and len(ind.objectives) > 0:
            if not np.any(np.isnan(ind.objectives)) and not np.any(np.isinf(ind.objectives)):
                valid_individuals.append(ind)
                valid_objectives.append(ind.objectives)
    
    if not valid_individuals:
        print("警告: 没有有效的帕累托解可供可视化")
        return None
    
    objectives = np.array(valid_objectives)
    
    # 目标索引: 0=能耗, 1=时间, 2=噪声, 3=安全, 4=长度
    energy = objectives[:, 0]
    time = objectives[:, 1]
    noise = objectives[:, 2]
    safety = objectives[:, 3]
    length = objectives[:, 4]
    
    # 归一化安全评分用于颜色映射
    safety_norm = (safety - safety.min()) / (safety.max() - safety.min() + 1e-10)
    
    # 图1: 能耗 vs 时间, 颜色=安全评分 (左)
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(energy, time, c=safety, cmap='RdYlGn_r', 
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Energy (kWh)', fontsize=11)
    ax1.set_ylabel('Flight Time (s)', fontsize=11)
    ax1.set_title('能耗 vs 时间\n(颜色=安全评分)', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Safety Score')
    ax1.grid(True, alpha=0.3)
    
    # 标注极端解
    min_energy_idx = np.argmin(energy)
    min_time_idx = np.argmin(time)
    ax1.annotate('Min Energy', xy=(energy[min_energy_idx], time[min_energy_idx]),
                 fontsize=8, ha='left', color='blue')
    ax1.annotate('Min Time', xy=(energy[min_time_idx], time[min_time_idx]),
                 fontsize=8, ha='right', color='red')
    
    # 图2: 航程 vs 安全, 大小=能耗 (中)
    ax2 = fig.add_subplot(132)
    # 能耗归一化用于大小
    energy_size = 50 + 200 * (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
    scatter2 = ax2.scatter(length, safety, s=energy_size, c='steelblue', 
                          alpha=0.7, edgecolors='darkblue', linewidth=0.5)
    ax2.set_xlabel('Path Length (m)', fontsize=11)
    ax2.set_ylabel('Safety Score', fontsize=11)
    ax2.set_title('航程 vs 安全评分\n(大小=能耗)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加大小图例
    for e_val, label in [(energy.min(), 'Min'), (energy.mean(), 'Mean'), (energy.max(), 'Max')]:
        idx = np.argmin(np.abs(energy - e_val))
        ax2.scatter([], [], s=50 + 200 * (e_val - energy.min()) / (energy.max() - energy.min() + 1e-10),
                   c='steelblue', alpha=0.7, label=f'{label}: {e_val:.3f}kWh')
    ax2.legend(title='能耗', loc='upper right', fontsize=8)
    
    # 图3: 3D散点图 (右)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 归一化各目标用于更好的可视化
    energy_n = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
    time_n = (time - time.min()) / (time.max() - time.min() + 1e-10)
    safety_n = (safety - safety.min()) / (safety.max() - safety.min() + 1e-10)
    
    # 颜色映射
    colors_3d = [plt.cm.plasma(v) for v in safety_norm]
    
    scatter3 = ax3.scatter(energy_n, time_n, safety_n, c=safety, cmap='coolwarm',
                          s=120, alpha=0.9, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Energy (normalized)', fontsize=9)
    ax3.set_ylabel('Time (normalized)', fontsize=9)
    ax3.set_zlabel('Safety (normalized)', fontsize=9)
    ax3.set_title('三维帕累托前沿\n(能耗-时间-安全)', fontsize=12)
    
    # 设置轴范围
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.6, pad=0.1)
    cbar3.set_label('Safety Score')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"帕累托前沿分析图已保存: {output_path}")
    
    return output_path


def plot_solution_distribution_heatmap(result, output_path: str, 
                                      bounds: tuple = ((0, 2500), (0, 2500), (30, 300))):
    """
    绘制解集分布热力图
    
    计算所有帕累托解在决策变量空间(航路点坐标)的分布
    用热图展示哪些航路点位置被频繁选择
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('帕累托解集分布热力图', fontsize=14, fontweight='bold')
    
    individuals = result.pareto_front.individuals
    if not individuals:
        print("警告: 没有帕累托解可供可视化")
        return None
    
    # 提取所有航路点的坐标
    x_coords = [[] for _ in range(10)]  # 假设10个关键点
    y_coords = [[] for _ in range(10)]
    z_coords = [[] for _ in range(10)]
    
    for ind in individuals:
        genes = ind.genes
        for i, gene in enumerate(genes):
            if i < 10:
                if hasattr(gene, '__getitem__'):
                    x_coords[i].append(gene[0])
                    y_coords[i].append(gene[1])
                    z_coords[i].append(gene[2])
                elif isinstance(gene, (list, tuple)):
                    x_coords[i].extend([gene[0]])
                    y_coords[i].extend([gene[1]])
                    z_coords[i].extend([gene[2]])
    
    # X坐标分布热力图
    ax = axes[0, 0]
    if x_coords and any(x_coords):
        x_data = [x for xs in x_coords if xs for x in xs]
        if x_data:
            x_hist, x_bins = np.histogram(x_data, bins=20)
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            ax.bar(x_centers, x_hist, width=x_bins[1]-x_bins[0], alpha=0.7, color='steelblue')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('X坐标分布')
    ax.grid(True, alpha=0.3)
    
    # Y坐标分布热力图
    ax = axes[0, 1]
    if y_coords and any(y_coords):
        y_data = [y for ys in y_coords if ys for y in ys]
        if y_data:
            y_hist, y_bins = np.histogram(y_data, bins=20)
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            ax.bar(y_centers, y_hist, width=y_bins[1]-y_bins[0], alpha=0.7, color='forestgreen')
    ax.set_xlabel('Y Coordinate (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Y坐标分布')
    ax.grid(True, alpha=0.3)
    
    # Z坐标分布热力图
    ax = axes[0, 2]
    if z_coords and any(z_coords):
        z_data = [z for zs in z_coords if zs for z in zs]
        if z_data:
            z_hist, z_bins = np.histogram(z_data, bins=15)
            z_centers = (z_bins[:-1] + z_bins[1:]) / 2
            ax.bar(z_centers, z_hist, width=z_bins[1]-z_bins[0], alpha=0.7, color='coral')
    ax.set_xlabel('Z Coordinate (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Z坐标分布(高度)')
    ax.grid(True, alpha=0.3)
    
    # 2D热力图: X-Y 位置
    ax = axes[1, 0]
    all_x = [x for xs in x_coords if xs for x in xs]
    all_y = [y for ys in y_coords if ys for y in ys]
    if all_x and all_y and len(all_x) == len(all_y):
        heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=15)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title('X-Y 位置热力图')
        plt.colorbar(im, ax=ax, label='Frequency')
    
    # 2D热力图: X-Z 位置
    ax = axes[1, 1]
    all_x_z = [x for xs in x_coords if xs for x in xs]
    all_z = [z for zs in z_coords if zs for z in zs]
    if all_x_z and all_z and len(all_x_z) == len(all_z):
        heatmap, xedges, zedges = np.histogram2d(all_x_z, all_z, bins=15)
        extent = [xedges[0], xedges[-1], zedges[0], zedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='YlGnBu', aspect='auto')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Z Coordinate (m)')
        ax.set_title('X-Z 位置热力图(垂直剖面)')
        plt.colorbar(im, ax=ax, label='Frequency')
    
    # 各关键点位置统计
    ax = axes[1, 2]
    positions = []
    for i in range(10):
        xs = x_coords[i] if i < len(x_coords) else []
        ys = y_coords[i] if i < len(y_coords) else []
        zs = z_coords[i] if i < len(z_coords) else []
        if xs and ys and zs:
            positions.append((np.mean(xs), np.mean(ys), np.std(xs) if len(xs) > 1 else 0))
    
    if positions:
        xs, ys, stds = zip(*positions)
        scatter = ax.scatter(xs, ys, c=stds, cmap='hot', s=200, alpha=0.8, edgecolors='black')
        ax.set_xlabel('Mean X (m)')
        ax.set_ylabel('Mean Y (m)')
        ax.set_title('各关键点中心位置\n(颜色=标准差)')
        plt.colorbar(scatter, ax=ax, label='Std Dev')
        ax.grid(True, alpha=0.3)
        
        # 连接各点显示路径趋势
        ax.plot(xs, ys, 'k--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"解集分布热力图已保存: {output_path}")
    
    return output_path


def plot_tradeoff_curves(result, output_path: str):
    """
    绘制目标间权衡关系图
    
    选择2-3个最重要的目标
    绘制它们之间的帕累托前沿曲线
    标注几个有代表性的解
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('目标间权衡关系分析', fontsize=14, fontweight='bold')
    
    individuals = result.pareto_front.individuals
    if not individuals:
        print("警告: 没有帕累托解可供可视化")
        return None
    
    # 【修复】提取有效目标值
    valid_individuals = []
    valid_objectives = []
    for ind in individuals:
        if ind.objectives is not None and len(ind.objectives) > 0:
            if not np.any(np.isnan(ind.objectives)) and not np.any(np.isinf(ind.objectives)):
                valid_individuals.append(ind)
                valid_objectives.append(ind.objectives)
    
    if not valid_individuals:
        print("警告: 没有有效的帕累托解可供可视化")
        return None
    
    objectives = np.array(valid_objectives)
    
    # 目标索引: 0=能耗, 1=时间, 2=噪声, 3=安全, 4=长度
    energy = objectives[:, 0]
    time = objectives[:, 1]
    noise = objectives[:, 2]
    safety = objectives[:, 3]
    length = objectives[:, 4]
    
    # 代表性解的标签
    labels = {
        'min_energy': ('最小能耗解', 'green'),
        'min_time': ('最快速解', 'blue'),
        'min_noise': ('最低噪声解', 'purple'),
        'min_safety': ('最安全解', 'orange'),
    }
    
    # 找到代表性解
    min_energy_idx = np.argmin(energy)
    min_time_idx = np.argmin(time)
    min_noise_idx = np.argmin(noise)
    min_safety_idx = np.argmin(safety)
    
    representative = [
        (min_energy_idx, 'min_energy'),
        (min_time_idx, 'min_time'),
        (min_noise_idx, 'min_noise'),
        (min_safety_idx, 'min_safety'),
    ]
    
    def annotate_solution(ax, idx, label_key):
        """标注代表性解"""
        label, color = labels[label_key]
        x, y = energy[idx], time[idx]
        ax.scatter([x], [y], c=color, s=200, marker='*', edgecolors='black', zorder=5)
        ax.annotate(label, xy=(x, y), xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 图1: 能耗 vs 时间
    ax = axes[0]
    scatter = ax.scatter(energy, time, c=noise, cmap='viridis', s=100, alpha=0.8, 
                        edgecolors='black', linewidth=0.5)
    
    # 绘制帕累托前沿线
    sorted_idx = np.argsort(energy)
    ax.plot(energy[sorted_idx], time[sorted_idx], 'r--', alpha=0.7, linewidth=2, label='Pareto Front')
    
    # 标注代表性解
    for idx, label_key in representative:
        annotate_solution(ax, idx, label_key)
    
    ax.set_xlabel('Energy (kWh)', fontsize=11)
    ax.set_ylabel('Flight Time (s)', fontsize=11)
    ax.set_title('能耗 vs 时间 权衡曲线', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Noise Level')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 图2: 能耗 vs 噪声
    ax = axes[1]
    scatter = ax.scatter(energy, noise, c=safety, cmap='RdYlGn', s=100, alpha=0.8,
                        edgecolors='black', linewidth=0.5)
    
    # 绘制帕累托前沿线
    sorted_idx = np.argsort(energy)
    ax.plot(energy[sorted_idx], noise[sorted_idx], 'r--', alpha=0.7, linewidth=2, label='Pareto Front')
    
    # 标注代表性解
    for idx, label_key in representative:
        annotate_solution(ax, idx, label_key)
    
    ax.set_xlabel('Energy (kWh)', fontsize=11)
    ax.set_ylabel('Noise Level', fontsize=11)
    ax.set_title('能耗 vs 噪声 权衡曲线', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Safety Score')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 图3: 时间 vs 安全
    ax = axes[2]
    scatter = ax.scatter(time, safety, c=energy, cmap='plasma', s=100, alpha=0.8,
                        edgecolors='black', linewidth=0.5)
    
    # 绘制帕累托前沿线
    sorted_idx = np.argsort(time)
    ax.plot(time[sorted_idx], safety[sorted_idx], 'r--', alpha=0.7, linewidth=2, label='Pareto Front')
    
    # 标注代表性解
    for idx, label_key in representative:
        annotate_solution(ax, idx, label_key)
    
    ax.set_xlabel('Flight Time (s)', fontsize=11)
    ax.set_ylabel('Safety Score', fontsize=11)
    ax.set_title('时间 vs 安全 权衡曲线', fontsize=12)
    plt.colorbar(scatter, ax=ax, label='Energy (kWh)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"目标间权衡关系图已保存: {output_path}")
    
    return output_path


def generate_comprehensive_report(result, output_dir: str, bounds: tuple = None):
    """
    生成完整的可视化报告
    
    参数:
        result: OptimizationResult对象
        output_dir: 输出目录
        bounds: 边界范围
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("开始生成增强可视化分析报告")
    print("=" * 70)
    
    outputs = []
    
    # 1. 进化过程监控图
    print("\n[1/4] 生成进化过程监控图...")
    path1 = os.path.join(output_dir, 'evolution_monitoring.png')
    plot_evolution_monitoring(result, path1)
    outputs.append(('进化过程监控', path1))
    
    # 2. 帕累托前沿分析图
    print("\n[2/4] 生成帕累托前沿分析图...")
    path2 = os.path.join(output_dir, 'pareto_front_analysis.png')
    plot_pareto_front_analysis(result, path2)
    outputs.append(('帕累托前沿分析', path2))
    
    # 3. 解集分布热力图
    print("\n[3/4] 生成解集分布热力图...")
    path3 = os.path.join(output_dir, 'solution_distribution_heatmap.png')
    plot_solution_distribution_heatmap(result, path3, bounds)
    outputs.append(('解集分布热力图', path3))
    
    # 4. 目标间权衡关系图
    print("\n[4/4] 生成目标间权衡关系图...")
    path4 = os.path.join(output_dir, 'tradeoff_curves.png')
    plot_tradeoff_curves(result, path4)
    outputs.append(('目标间权衡关系', path4))
    
    # 生成汇总报告
    print("\n" + "=" * 70)
    print("可视化报告汇总")
    print("=" * 70)
    
    print(f"\n帕累托前沿质量指标:")
    print(f"  - 帕累托解数量: {len(result.pareto_front.individuals)}")
    print(f"  - 超体积(HV): {result.pareto_front.hypervolume:.4f}")
    print(f"  - 间距(Spacing): {result.pareto_front.spacing:.4f}")
    print(f"  - 迭代次数: {result.n_iterations}")
    
    print(f"\n生成的可视化图表:")
    for name, path in outputs:
        print(f"  - {name}: {path}")
    
    return outputs


if __name__ == '__main__':
    print("增强可视化分析模块")
    print("请从 test_visualization.py 导入并使用")
