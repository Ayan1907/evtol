#!/usr/bin/env bash
# 一键把 ~/evtol 变成 Git 仓库（在 WSL 内运行）：bash ~/evtol/git_setup.sh
set -e
cd "$(dirname "$0")"

if [ ! -d .git ]; then
  git init -b main
  echo "[ok] git 仓库已初始化"
else
  echo "[skip] 已是 git 仓库"
fi

# 首次使用需要配置身份
if [ -z "$(git config user.name)" ]; then
  echo "[提示] 未配置 git 身份，下面用默认值（可改）："
  git config user.name  "xujiuyan"
  git config user.email "xujiuyan@bit.edu.cn"
fi

git add -A
git commit -m "init: evtol 原项目 + evtol_rl 骨架（map_dense 已定，基线已实测）" || echo "[skip] 没有新改动可提交"
echo
echo "=== 下一步（手动，一次性）：到 GitHub/Gitee 建一个【私有】空仓库，然后： ==="
echo "git remote add origin git@github.com:<你的用户名>/evtol.git   # 或 Gitee 地址"
echo "git push -u origin main"
echo
echo "之后每完成一个阶段 commit 一次；主机 Linux 端 git clone 同一仓库即可续工作。"