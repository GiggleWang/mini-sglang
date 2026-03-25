#!/usr/bin/env python3
"""
画每个 timeline CSV 的引擎状态图：
  X 轴 = 距离首次记录的时间差 (ms)
  Y 轴 = 当前状态 (prefill / decode / idle)
  颜色 = 橙色(prefill) / 蓝色(decode) / 灰色(idle)

用法：
    python plot_state.py <folder>
    python plot_state.py <folder> --out state.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


STATE_Y  = {"prefill": 2, "decode": 1, "idle": 0}
STATE_COLOR = {"prefill": "#e07b54", "decode": "#4e8fcc", "idle": "#cccccc"}
LABELS   = {2: "prefill", 1: "decode", 0: "idle"}


def load_runs(path: Path):
    """
    以类型切换为分割点：连续同类型的 iter（不管 gap 多大）合并成一个 run。
    run 之间的空档为 idle。
    返回 [(t_start, t_end, state), ...] 列表，state 为 "prefill"/"decode"/"idle"。
    """
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t_start_ms"]), float(r["t_end_ms"]), r["type"]))

    if not rows:
        return []

    t0 = rows[0][0]

    # 按类型切换分组：run_start=第一个iter的t_start，run_end=最后一个iter的t_end
    runs_raw = []
    run_start, run_end, run_type = rows[0]
    for ts, te, typ in rows[1:]:
        if typ == run_type:
            run_end = te
        else:
            runs_raw.append((run_start, run_end, run_type))
            run_start, run_end, run_type = ts, te, typ
    runs_raw.append((run_start, run_end, run_type))

    # 在相邻 run 之间插入 idle 段
    result = []
    for i, (ts, te, typ) in enumerate(runs_raw):
        if i > 0:
            prev_end = runs_raw[i - 1][1]
            if ts - prev_end > 0.01:
                result.append((prev_end - t0, ts - t0, "idle"))
        result.append((ts - t0, te - t0, typ))

    return result


def plot(folder: Path, out: Path):
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"没有找到 CSV 文件：{folder}")
        return

    n = len(csv_files)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.2 * n))
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, csv_files):
        runs = load_runs(path)
        if not runs:
            continue

        total = runs[-1][1]

        for ts, te, state in runs:
            y = STATE_Y[state]
            ax.fill_between([ts, te], [y - 0.4, y - 0.4], [y + 0.4, y + 0.4],
                            color=STATE_COLOR[state], linewidth=0)

        ax.set_xlim(0, total)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["idle", "decode", "prefill"], fontsize=9)
        ax.set_xlabel("Time since start (ms)", fontsize=9)
        ax.set_title(path.stem, fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    # 图例
    legend_patches = [
        mpatches.Patch(color=STATE_COLOR[s], label=s)
        for s in ("prefill", "decode", "idle")
    ]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=9)

    fig.suptitle("Engine State Timeline", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (args.folder / "state_timeline.png")
    plot(args.folder, out)


if __name__ == "__main__":
    main()
