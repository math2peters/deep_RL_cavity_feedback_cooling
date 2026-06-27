from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


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


def load_datasets(folder, param_name):
    datasets = []
    for csv_file in sorted(folder.glob("*.csv")):
        dataset_name = get_dataset_name(csv_file, param_name)
        if dataset_name is None:
            continue
        df = pd.read_csv(csv_file)
        datasets.append((dataset_name, df))
    return datasets


def plot_panel(ax, datasets, param_name, training_point):
    if param_name == "detuning":
        x_label = r"Probe Detuning $\Delta/2\pi$ (MHz)"
        x_column_exp = "detuning"
        x_column_sim = "detuning"
    else:
        x_label = "Photon Counts"
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
            label="MLP (Sim.)",
            color=sim_color,
            markeredgecolor=sim_color,
            markerfacecolor=sim_color,
            marker="^",
            linestyle="-",
            markersize=6,
            linewidth=2,
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
            label="MLP (Expt.)",
            color=exp_color,
            markeredgecolor=exp_color,
            markerfacecolor=exp_color,
            marker="o",
            linestyle="-",
            markersize=6,
            capsize=5,
            linewidth=2,
            alpha=1.0,
            zorder=3,
        )

    ax.axvline(x=training_point, color="black", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Survival probability", fontsize=14)
    ax.set_ylim(bottom=0, top=1.1)
    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1.5, labelsize=12)


def main():
    package_root, cfg, output_dir_name = load_config()
    output_dir = package_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    detuning_datasets = load_datasets(package_root / cfg["detuning_dir"], "detuning")
    power_datasets = load_datasets(package_root / cfg["power_dir"], "photon_number")

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    fig = plt.figure(figsize=(12, 5))
    ax_a = fig.add_subplot(1, 2, 1)
    ax_b = fig.add_subplot(1, 2, 2)
    plot_panel(ax_a, detuning_datasets, "detuning", cfg["detuning_training_point"])
    plot_panel(ax_b, power_datasets, "photon_number", cfg["photon_training_point"])
    ax_a.text(-0.15, 1.05, "(a)", transform=ax_a.transAxes, fontsize=18, fontweight="bold", va="top", ha="right")
    ax_b.text(-0.15, 1.05, "(b)", transform=ax_b.transAxes, fontsize=18, fontweight="bold", va="top", ha="right")
    ax_a.legend(fontsize=12)
    plt.tight_layout()
    output_pdf = output_dir / cfg["output_pdf"]
    output_png = output_dir / cfg["output_pdf"].replace(".pdf", ".png")
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
