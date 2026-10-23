"""Draw Fig. 4 and save its numbers, from data/source_data_fig4 or a rerun (--evaluations)."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from src.evaluation.fig4 import DATA, STEP_US, load_groups, metric_mean, cooling_curve
from scripts.mpl_aps_style import TEXT_WIDTH_IN, TICK_SIZE, add_panel_label, apply_paper_style, check_lettering

FAMILIES = [("mlp", "Small MLP", "#0072B2"), ("mlp_large", "Large MLP", "#332288"), ("affine", "Affine actor", "#56B4E9"),
            ("differentiator", "Differentiator", "#009E73"), ("kalman", "Kalman filter", "#CC79A7"),
            ("particle", "Particle filter", "#D55E00")]
NO_FEEDBACK = ("none", "No feedback", "#7f7f7f")
FIGURE_POINTS = [("nominal", "25 MHz, 440 µK", "medium", "low"),
                 ("hot_delta12p5", "12.5 MHz, 800 µK", "high", "high")]
CURVE_SCENARIO = "hot_delta12p5"
METRICS = {"energy": "Energy at 546 µs,\nsurviving atoms (µK)", "retention": "Survival (%)"}
BREAKS = {"energy": (80, 360, [100, 150, 200, 250, 300, 350]), "retention": (80, 100, [80, 85, 90, 95, 100])}
STUB = 0.16


def broken_axis(ax, metric):
    low, high, ticks = BREAKS[metric]
    stub = STUB * (high - low)
    forward = lambda y: np.where(y < low, np.asarray(y) * stub / low, stub + np.asarray(y) - low)
    inverse = lambda v: np.where(v < stub, np.asarray(v) * low / stub, np.asarray(v) - stub + low)
    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_ylim(0, high)
    ax.set_yticks([0] + ticks)
    # white band across the bars and a double slash on the spine mark the break
    band = (0.3 * low, 0.7 * low)
    ax.axhspan(*band, color="white", zorder=3, lw=0)
    for y in band:
        ax.plot([0], [y], transform=ax.get_yaxis_transform(), marker=[(-1, -0.5), (1, 0.5)], ms=6,
                mew=0.8, color="k", clip_on=False, zorder=4)


def bar_panels(axes, data):
    """(a) energy and (b) survival at each operating point."""
    width, gap = 0.15, 0.3
    centers = np.arange(len(FIGURE_POINTS)) * (len(FAMILIES) * width + gap)
    for ax, metric in zip(axes, METRICS):
        for i, (family, label, colour) in enumerate(FAMILIES):
            for center, (scenario, *_rest) in zip(centers, FIGURE_POINTS):
                x = center + (i - (len(FAMILIES) - 1) / 2) * width
                mean, err = metric_mean(data[scenario][family], metric)
                legend_label = label if (metric == "energy" and center == centers[0]) else None
                ax.bar(x, mean, width * 0.92, yerr=err, color=colour, label=legend_label, zorder=2,
                       error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6, "zorder": 2})
        broken_axis(ax, metric)
        ax.axvline((centers[0] + centers[1]) / 2, color="0.6", lw=0.6, ls=":")
        ax.set_ylabel(METRICS[metric])
        ax.set_xticks(centers, [""] * len(centers))
        ax.tick_params(axis="x", length=0)
    labels = [f"{point}\nLorentzian NL: {readout}\nTrap NL: {motion}"
              for _, point, readout, motion in FIGURE_POINTS]
    axes[-1].set_xticks(centers, labels)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, legend_labels, ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   frameon=False, handlelength=1.0, columnspacing=1.2)


def curve_panel(ax, data):
    """(c) mean axial energy of surviving atoms versus time at the most nonlinear point."""
    groups = data[CURVE_SCENARIO]
    for family, label, colour in FAMILIES + [NO_FEEDBACK]:
        t, mean, sem = cooling_curve(groups[family])
        style = {"ls": "--"} if family == "none" else {}
        ax.plot(t, mean, color=colour, lw=1.0, label=label, **style)
        ax.fill_between(t, mean - sem, mean + sem, color=colour, alpha=0.2, lw=0)
    point = next(p for p in FIGURE_POINTS if p[0] == CURVE_SCENARIO)
    ax.set_title(f"{point[1]}: Lorentzian NL {point[2]}, trap NL {point[3]}", fontsize=TICK_SIZE)
    ax.axvline(20 * STEP_US / 1e3, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("Time after feedback starts (ms)")
    ax.set_ylabel("Axial energy (µK)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    # colours follow the legend of (a); only the no-feedback curve is labelled here
    t_none, mean_none, _ = cooling_curve(groups["none"])
    k = np.searchsorted(t_none, 1.6)
    ax.text(t_none[k], mean_none[k] + 18, "No feedback", color=NO_FEEDBACK[2], fontsize=TICK_SIZE,
            ha="center", va="bottom")


def operations_panel(ax):
    """(d) estimated operations per control update."""
    operations = json.loads((DATA / "operations.json").read_text())
    rows = sorted((operations["controllers"][key]["ops"], label, colour) for key, label, colour in FAMILIES)
    y = np.arange(len(rows))
    ax.barh(y, [r[0] for r in rows], color=[r[2] for r in rows], height=0.6)
    for yi, (ops, _, _) in zip(y, rows):
        exponent = int(np.floor(np.log10(ops)))
        text = f"{ops:,}" if ops < 1e5 else f"{ops / 10**exponent:.0f}×10$^{{{exponent}}}$"
        ax.text(ops * 1.6, yi, text, va="center", fontsize=TICK_SIZE)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e9)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Estimated operations per control update")


def render(data, output):
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 6.1), layout="constrained")
    grid = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.8, 1.55], width_ratios=[1.25, 1.0])
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :]), fig.add_subplot(grid[2, 0]),
            fig.add_subplot(grid[2, 1])]
    bar_panels(axes[:2], data)
    curve_panel(axes[2], data)
    operations_panel(axes[3])
    for ax, label in zip(axes, "abcd"):
        add_panel_label(ax, f"({label})")
    fig.canvas.draw()
    check_lettering(fig)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, help="Directory of a complete rerun (scripts/evaluate_fig4.py)")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/Fig4", help="Output stem")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = load_groups(args.evaluations)
    apply_paper_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    render(data, args.output)
    metrics = {s: {f: {m: dict(zip(("mean", "bootstrap_se"), metric_mean(g, m)))
                     for m in METRICS} for f, g in data[s].items()} for s, *_ in FIGURE_POINTS}
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"Saved {args.output}.pdf, .png and .json")
