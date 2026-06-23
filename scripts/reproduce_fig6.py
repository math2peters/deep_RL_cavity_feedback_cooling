from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d


def load_config():
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    with open(script_dir / "params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return package_root, params["fig6"], params["output_dir"]


def main():
    package_root, cfg, output_dir_name = load_config()
    output_dir = package_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_df = pd.read_csv(package_root / cfg["energy_csv"])
    energy_curve = energy_df[energy_df["E_y"] == 0].copy() # assume y-energy = 0; weak dependence on y-energy
    energy_interp = interp1d(
        energy_curve["Transmission"].values,
        energy_curve["E_z_uK"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    initial_transmission = cfg["initial_spcm_counts"] / cfg["initial_empty_cavity"]
    rel_spcm_err = cfg["initial_spcm_err"] / cfg["initial_spcm_counts"]
    rel_empty_err = cfg["initial_empty_err"] / cfg["initial_empty_cavity"]
    initial_transmission_err = initial_transmission * np.sqrt(rel_spcm_err**2 + rel_empty_err**2)
    energy_value = float(energy_interp(initial_transmission))
    energy_high = float(energy_interp(min(initial_transmission + initial_transmission_err, 1.0)))
    energy_low = float(energy_interp(max(initial_transmission - initial_transmission_err, 0.0)))
    energy_err = (energy_high - energy_low) / 2

    temperature_df = pd.read_csv(package_root / cfg["temperature_csv"])
    temperature_curve = temperature_df[temperature_df["T_y_uK"] == 0.0].copy()
    temperature_interp = interp1d(
        temperature_curve["Transmission"].values,
        temperature_curve["T_z_uK"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )
    min_transmission = float(temperature_curve["Transmission"].min())
    min_temperature = float(temperature_curve.loc[temperature_curve["Transmission"].idxmin(), "T_z_uK"])

    def temperature_from_transmission(value):
        if value <= min_transmission:
            return min_temperature
        return float(temperature_interp(value))

    final_transmission = cfg["final_transmission"]
    final_transmission_err = cfg["final_transmission_err"]
    temperature_value = temperature_from_transmission(final_transmission)
    temperature_high = temperature_from_transmission(min(final_transmission + final_transmission_err, 1.0))
    temperature_low = temperature_from_transmission(max(final_transmission - final_transmission_err, 0.0))
    temperature_err_lower = temperature_value - temperature_low
    temperature_err_upper = temperature_high - temperature_value

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    sim_color = "#0072B2"
    exp_color = "#D55E00"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    axes[0].text(-0.1, 1.1, "(a)", transform=axes[0].transAxes, fontsize=20, fontweight="bold", va="top", ha="right")
    axes[1].text(-0.1, 1.1, "(b)", transform=axes[1].transAxes, fontsize=20, fontweight="bold", va="top", ha="right")

    axes[0].plot(
        energy_curve["E_z_uK"].values,
        energy_curve["Transmission"].values,
        color=sim_color,
        alpha=0.8,
        label="Simulated Transmission",
        zorder=2,
    )
    axes[0].errorbar(
        energy_value,
        initial_transmission,
        xerr=energy_err,
        yerr=initial_transmission_err,
        fmt="o",
        markerfacecolor=exp_color,
        markeredgecolor=exp_color,
        ecolor=exp_color,
        elinewidth=2,
        capsize=5,
        zorder=3,
        label=f"Initial Energy: {energy_value:.0f}({energy_err:.0f}) µK",
    )
    axes[0].set_xlabel("Longitudinal Energy (µK)")
    axes[0].set_ylabel("Transmission Fraction")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].tick_params(axis="both", which="major", direction="in", length=3, width=1.5)

    axes[1].plot(
        temperature_curve["T_z_uK"].values,
        temperature_curve["Transmission"].values,
        color=sim_color,
        alpha=0.8,
        label="Simulated Transmission",
        zorder=2,
    )
    axes[1].errorbar(
        temperature_value,
        final_transmission,
        xerr=np.array([[temperature_err_lower], [temperature_err_upper]]),
        yerr=final_transmission_err,
        fmt="o",
        markerfacecolor=exp_color,
        markeredgecolor=exp_color,
        ecolor=exp_color,
        elinewidth=2,
        capsize=5,
        zorder=3,
        label=f"Final Temperature: {temperature_value:.0f}({temperature_err_upper/2+temperature_err_lower/2:.0f}) µK",
    )
    axes[1].set_xlabel("Longitudinal Temperature (µK)")
    axes[1].set_ylabel("Transmission Fraction")
    axes[1].set_ylim(0, 0.35)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].tick_params(axis="both", which="major", direction="in", length=3, width=1.5)

    plt.tight_layout()
    output_pdf = output_dir / cfg["output_pdf"]
    output_png = output_dir / cfg["output_pdf"].replace(".pdf", ".png")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
