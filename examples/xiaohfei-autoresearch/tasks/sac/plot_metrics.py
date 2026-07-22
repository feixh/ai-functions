#!/usr/bin/env python3
"""Visualize metrics.jsonl from SAC training runs.

Usage:
    python plot_metrics.py <metrics.jsonl> [--out output.png]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- palette (validated reference palette) ---
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE      = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED   = "#898781"
GRID_COLOR   = "#e1e0d9"
AXIS_COLOR   = "#c3c2b7"

# Groups of metrics to display together (same y-scale ≈ same unit)
METRIC_PANELS = [
    {"title": "Cumulative Reward",  "keys": ["cumulative_reward"]},
    {"title": "Entropy",            "keys": ["entropy"]},
    {"title": "Q-Function Losses",  "keys": ["loss_q1", "loss_q2"]},
    {"title": "Value Loss",         "keys": ["loss_v"]},
    {"title": "Policy Loss",        "keys": ["loss_pi"]},
]


def load_runs(path: Path) -> list[list[dict]]:
    """Split the file into runs wherever global_step resets."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    runs: list[list[dict]] = []
    current: list[dict] = []
    for rec in records:
        if current and rec["global_step"] <= current[-1]["global_step"]:
            runs.append(current)
            current = []
        current.append(rec)
    if current:
        runs.append(current)
    return runs


def fmt_step(x, _pos):
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return str(int(x))


def plot(runs: list[list[dict]], out: Path | None):
    n_panels = len(METRIC_PANELS)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                             facecolor=SURFACE)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_flat:
        ax.set_facecolor(SURFACE)

    run_labels = [f"Run {i+1}" for i in range(len(runs))]

    for panel_idx, panel in enumerate(METRIC_PANELS):
        ax = axes_flat[panel_idx]
        keys = panel["keys"]

        # assign colors: if multiple runs, first key per-run; if multiple keys, per-key
        if len(runs) > 1 and len(keys) == 1:
            for run_i, run in enumerate(runs):
                xs = [r["global_step"] for r in run]
                ys = [r[keys[0]] for r in run]
                ax.plot(xs, ys, linewidth=1.2, color=SERIES_COLORS[run_i % len(SERIES_COLORS)],
                        solid_capstyle="round", solid_joinstyle="round", label=run_labels[run_i])
            if len(runs) > 1:
                ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)
        else:
            # all runs concatenated, color by key
            for key_i, key in enumerate(keys):
                for run_i, run in enumerate(runs):
                    xs = [r["global_step"] for r in run]
                    ys = [r[key] for r in run]
                    # only label the first run per key to avoid legend duplicates
                    label = key if (len(keys) > 1 and run_i == 0) else None
                    ax.plot(xs, ys, linewidth=1.2,
                            color=SERIES_COLORS[key_i % len(SERIES_COLORS)],
                            solid_capstyle="round", solid_joinstyle="round",
                            label=label)
            if len(keys) > 1:
                ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)

        ax.set_title(panel["title"], fontsize=11, fontweight="bold",
                     color=TEXT_PRIMARY, pad=8, loc="left")
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_step))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"))
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
            spine.set_linewidth(1)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1, linestyle="solid")
        ax.set_axisbelow(True)
        ax.set_xlabel("Step", fontsize=8, color=TEXT_MUTED)

    # hide unused panels
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("SAC Training Metrics", fontsize=14, fontweight="bold",
                 color=TEXT_PRIMARY, y=1.01)
    plt.tight_layout(pad=1.5)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        print(f"Saved → {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot SAC metrics.jsonl")
    parser.add_argument("path", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save to PNG instead of showing interactively")
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"File not found: {args.path}")

    runs = load_runs(args.path)
    print(f"Loaded {sum(len(r) for r in runs)} records across {len(runs)} run(s).")
    plot(runs, args.out)


if __name__ == "__main__":
    main()
