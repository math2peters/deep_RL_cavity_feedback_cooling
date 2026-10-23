import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import yaml
from mpl_aps_style import TEXT_WIDTH_IN, add_panel_label, apply_paper_style, check_lettering


def load_config():
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    with open(script_dir / "params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return package_root, params["fig3"], params["output_dir"]


def get_dataset_name(csv_file, param_name):
    filename = csv_file.stem.lower()
    if filename.startswith("experimental_run_") or "scan_data" in filename:
        return "MLP (Expt.)"
    if filename.endswith("_mlp_sim_noisy"):
        return "Simulation (noisy)"
    if filename.endswith("_mlp_sim"):
        return "MLP (Sim.)"
    return None


def load_datasets(folder, param_name, simulation=None):
    """Bundled curves; the simulated ones come from `simulation` (scripts/sweep_fig3.py output) when given."""
    files = sorted(folder.glob("*.csv"))
    if simulation is not None:
        files = [f for f in files if f.stem.startswith("experimental_run_")]
        files += sorted(Path(simulation).glob(f"{param_name}_sweep_results_mlp_sim*.csv"))
    datasets = []
    for csv_file in files:
        dataset_name = get_dataset_name(csv_file, param_name)
        if dataset_name is None:
            continue
        df = pd.read_csv(csv_file)
        datasets.append((dataset_name, df))
    return datasets


def plot_panel(ax, datasets, param_name, training_point):
    if param_name == "detuning":
        x_label = "Detuning Δ/2π (MHz)"
        x_column_exp = "detuning"
        x_column_sim = "detuning"
    else:
        x_label = "Empty-cavity counts per bin"
        x_column_exp = "Calibrated_Power"
        x_column_sim = "photon_number"

    exp_color = "#D55E00"
    sim_color = "#0072B2"
    light_blue = "#E6F3FF"

    ideal_sim = None
    noisy_sim = None
    exp_data = None
    for name, df in datasets:
        if name == "MLP (Expt.)":
            exp_data = df.copy()
        elif name == "Simulation (noisy)":
            noisy_sim = df.copy()
        elif name == "MLP (Sim.)":
            ideal_sim = df.copy()

    if ideal_sim is not None and noisy_sim is not None:
        ideal_sorted = ideal_sim.sort_values(x_column_sim)
        noisy_sorted = noisy_sim.sort_values(x_column_sim)
        common_values = sorted(set(ideal_sorted[x_column_sim]).intersection(set(noisy_sorted[x_column_sim])))
        if common_values:
            ideal_fractions = []
            noisy_fractions = []
            for value in common_values:
                ideal_fractions.append(ideal_sorted.loc[ideal_sorted[x_column_sim] == value, "fraction_completed"].iloc[0])
                noisy_fractions.append(noisy_sorted.loc[noisy_sorted[x_column_sim] == value, "fraction_completed"].iloc[0])
            ax.fill_between(common_values, ideal_fractions, noisy_fractions, color=light_blue, alpha=1.0, edgecolor="none", zorder=0)

    if ideal_sim is not None:
        ax.plot(
            ideal_sim[x_column_sim],
            ideal_sim["fraction_completed"],
            label="Simulation: MLP",
            color=sim_color,
            markeredgecolor=sim_color,
            markerfacecolor=sim_color,
            marker="^",
            linestyle="-",
            markersize=3.5,
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )

    if exp_data is not None:
        if param_name == "photon_number":
            x_col = x_column_exp if x_column_exp in exp_data.columns else "photon_number"
            frac_col = "capture_fraction" if "capture_fraction" in exp_data.columns else "fraction_completed"
            frac_err_col = "capture_fraction_error" if "capture_fraction_error" in exp_data.columns else "se_fraction_completed"
        else:
            x_col = x_column_exp
            frac_col = "fraction_completed"
            frac_err_col = "se_fraction_completed"
        ax.errorbar(
            exp_data[x_col],
            exp_data[frac_col],
            yerr=exp_data[frac_err_col],
            label="Experiment: MLP (Expt.)",
            color=exp_color,
            markeredgecolor=exp_color,
            markerfacecolor=exp_color,
            marker="o",
            linestyle="-",
            markersize=3.5,
            capsize=1.5,
            linewidth=1.2,
            alpha=1.0,
            zorder=3,
        )

    ax.axvline(x=training_point, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Survival probability")
    ax.set_ylim(bottom=0, top=1.1)


POLICIES = [("MLP (Sim.)", "#0072B2", "^"), ("MLP (Expt.)", "#D55E00", "o")]
COHORTS = ["pretraining", "late_train", "fixed_expt"]


def count_sensitivity(data, draws=10000, seed=20261004):
    """RMS sensitivity of each actor's output to the count k bins in the past.

    Both frozen actors receive the same 37456 measured hardware histories (steps 22-99 of
    the pretraining, late-training and fixed-policy episodes); the derivative includes the
    count's moving-average descendants and holds the command history fixed. Intervals are
    pointwise 95% percentiles of a paired, cohort-stratified bootstrap over episodes.
    """
    sums, counts, cohort = data["episode_sum_squared_kernels"], data["episode_states"], data["episode_cohort"]
    rng = np.random.default_rng(seed)
    boot_sum, boot_n = np.zeros((draws, 14)), np.zeros(draws)
    for name in COHORTS:
        ids = np.flatnonzero(cohort == name)
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)
        boot_sum += weights @ sums[ids].reshape(len(ids), 14)
        boot_n += weights @ counts[ids]
    rms = np.sqrt(sums.sum(axis=0)/counts.sum())
    ci = np.quantile(np.sqrt(boot_sum.reshape(draws, 2, 7)/boot_n[:, None, None]), [.025, .975], axis=0)
    return rms, ci


def action_panel(fig, ax, depth):
    hexbins = ax.hexbin(depth[:, 0], depth[:, 1], gridsize=43, extent=(0, 1, 0, 1), mincnt=1, bins="log",
                        cmap="Blues", linewidths=0, rasterized=False)
    ax.plot([0, 1], [0, 1], color=".4", lw=.8, ls="--", zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=r"MLP (Sim.) depth ($U_t/U_\mathrm{max}$)",
           ylabel=r"MLP (Expt.) depth ($U_t/U_\mathrm{max}$)")
    ax.xaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.grid(False)
    ax.set_anchor("W")
    # attach the colour bar to the square axes so the column keeps its left edge
    bar = fig.colorbar(hexbins, cax=ax.inset_axes([1.04, .06, .045, .88]))
    bar.set_label("Histories per hexagon", fontsize=9, labelpad=4)
    bar.set_ticks(10.**np.arange(int(np.floor(np.log10(hexbins.get_array().max())))+1))


def depth_summary(data, draws=10000, seed=20261004):
    """History-weighted means with cohort-stratified episode-bootstrap uncertainty."""
    depth = data["depth"].astype(float)
    counts, cohort = data["episode_states"], data["episode_cohort"]
    ends = np.cumsum(counts)
    sums = np.array([depth[a:b].sum(axis=0) for a, b in zip(np.r_[0, ends[:-1]], ends)])
    rng = np.random.default_rng(seed)
    totals, sizes = np.zeros((draws, 2)), np.zeros(draws)
    for name in COHORTS:
        ids = np.flatnonzero(cohort == name)
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)
        totals += weights @ sums[ids]
        sizes += weights @ counts[ids]
    means = totals / sizes[:, None]
    return {"histories": len(depth), "episodes": len(counts), "policy_order": [p[0] for p in POLICIES],
            "mean_depth": depth.mean(axis=0).tolist(), "bootstrap_se": means.std(axis=0).tolist(),
            "ci95": np.quantile(means, [.025, .975], axis=0).tolist(),
            "fraction_experiment_deeper": float(np.mean(depth[:, 1] > depth[:, 0]))}


def sensitivity_panel(ax, rms, ci):
    x = np.arange(7)
    for i, (label, color, marker) in enumerate(POLICIES):
        ax.plot(x, rms[i], color=color, lw=1.2, zorder=3)
        ax.errorbar(x, rms[i], yerr=[rms[i]-ci[0, i], ci[1, i]-rms[i]], fmt=marker, color=color, markersize=4,
                    capsize=2, elinewidth=.8, markeredgewidth=.65,
                    markerfacecolor="white" if marker == "^" else color, label=label, zorder=4)
    ax.set(xlim=(-.25, 6.25), ylim=(0, .5), xticks=range(7), xlabel="Count lag (bins; 0 = newest)",
           ylabel="Count sensitivity (RMS)")
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False)


def main():
    parser = argparse.ArgumentParser(description="Fig. 3 from the bundled data.")
    parser.add_argument("--simulation", type=Path, help="Directory of new simulation CSVs from scripts/sweep_fig3.py")
    args = parser.parse_args()
    package_root, cfg, output_dir_name = load_config()
    output_dir = package_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    detuning_datasets = load_datasets(package_root / cfg["detuning_dir"], "detuning", args.simulation)
    power_datasets = load_datasets(package_root / cfg["power_dir"], "photon_number", args.simulation)

    with np.load(package_root / cfg["policy_comparison"], allow_pickle=False) as data:
        depth = data["depth"]
        rms, ci = count_sensitivity(data)
        summary = depth_summary(data)
    (output_dir / "Fig3.json").write_text(json.dumps(summary, indent=2) + "\n")

    apply_paper_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .18})

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.6), layout="constrained",
                             gridspec_kw={"height_ratios": [.68, .84]})
    fig.get_layout_engine().set(w_pad=.045, h_pad=.035, wspace=.06, hspace=.035)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    plot_panel(ax_a, detuning_datasets, "detuning", cfg["detuning_training_point"])
    plot_panel(ax_b, power_datasets, "photon_number", cfg["photon_training_point"])
    ax_a.legend(loc="lower center", frameon=False, fontsize=9)
    # (c, d): both frozen actors on the same measured hardware histories
    action_panel(fig, ax_c, depth)
    sensitivity_panel(ax_d, rms, ci)
    for ax, label in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, label, dx=-32, dy=5)
    check_lettering(fig)
    output_pdf = output_dir / cfg["output_pdf"]
    output_png = output_dir / cfg["output_pdf"].replace(".pdf", ".png")
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    plt.close(fig)


if __name__ == "__main__":
    main()
