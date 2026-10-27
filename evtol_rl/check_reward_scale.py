# -*- coding: utf-8 -*-
"""奖励梯度方向自检：爬升/朝目标/后退/悬停(来回) 单步奖励对比（K=1 新尺度）。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evtol_env import EVTOLEnv

env = EVTOLEnv(map_file="data/maps/map_dense.json", k_progress=10.0)
up = int(np.where((env._DIRS == np.array([0, 0, 1])).all(axis=1))[0][0])
down = int(np.where((env._DIRS == np.array([0, 0, -1])).all(axis=1))[0][0])
to_goal = env.goal - env.pos
to_goal = np.array([to_goal[0], to_goal[1], 0.0])
fwd = int(np.argmax(env._DIRS_UNIT @ (to_goal / np.linalg.norm(to_goal))))
back = int(np.argmax(env._DIRS_UNIT @ (-to_goal / np.linalg.norm(to_goal))))

def one(a, warm=0):
    env.reset(seed=0)
    for _ in range(warm):
        env.step(up)
    return env.step(a)[1]

# 悬停 = 来回一对动作的平均单步（巡航段）
env.reset(seed=0)
env.step(up); env.step(up)          # 爬升到 130m（合法航线）
env.step(fwd); r2 = env.step(back)[1]

print(f"爬升步     R = {one(up):+.4f}   （min_h 未抬升，合规）")
print(f"爬升段早飞 R = {one(fwd):+.4f}   （违规 2m，应显著为负：合规罚生效）")
print(f"巡航段前进 R = {one(fwd, warm=2):+.4f}   <- 应为正且最大")
print(f"巡航段后退 R = {one(back, warm=2):+.4f}   <- 应为负")
print(f"巡航段悬停 R = {r2:+.4f}            <- 应 <= 巡航段前进")