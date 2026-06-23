from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(0)

# Physical constants
PI = np.pi
h = 6.62607015e-34
kB = 1.380649e-23
gamma = 2 * PI * 5.23e6
m = 2.21e-25

# Trap parameters
w0_y = 1.5e-6
w0_z = 13e-6
cavity_waist = 7.1e-6

# Cavity parameters
kappa = 2 * PI * 40e3
Delta_c = 0.0

U0 = h * 36e6
eta = 20.2
Delta_a = 2 * PI * 25e6
empty_cavity_counts = 32.185689
detection_efficiency = 0.125
observed_transmission = 0.208
atom_present_counts = observed_transmission * empty_cavity_counts
photon_number = atom_present_counts * (1 / (kappa * 20e-6 * detection_efficiency))
g_max = np.sqrt(eta * kappa * gamma / 4)

omega_y = np.sqrt(4 * U0 / (m * w0_y**2))
omega_z = np.sqrt(2 * U0 / (m * w0_z**2))


def transmission_with_broadening_2d(y_pos, z_pos):
    g = g_max * np.exp(-(y_pos**2 + z_pos**2) / cavity_waist**2)
    omega = 2 * g * np.sqrt(photon_number)
    denom = Delta_a**2 + (gamma / 2) ** 2 + 0.5 * omega**2

    local_u0 = (g**2 * Delta_a) / denom
    local_gamma = (g**2 * (gamma / 2)) / denom

    kappa_tot = kappa + 2 * local_gamma
    Delta_eff = Delta_c - local_u0

    return (kappa / 2) ** 2 / ((kappa_tot / 2) ** 2 + Delta_eff**2)


def sample_harmonic_boltzmann(T_y: float, T_z: float, n_samples: int = 100000):
    sigma_y = np.sqrt(kB * T_y / (m * omega_y**2))
    sigma_z = np.sqrt(kB * T_z / (m * omega_z**2))

    y_samples = np.random.normal(0, sigma_y, n_samples)
    z_samples = np.random.normal(0, sigma_z, n_samples)
    return y_samples, z_samples


def calculate_average_transmission(T_y: float, T_z: float, n_samples: int = 100000) -> float:
    y_samples, z_samples = sample_harmonic_boltzmann(T_y, T_z, n_samples)
    transmissions = np.array(
        [transmission_with_broadening_2d(y_pos, z_pos) for y_pos, z_pos in zip(y_samples, z_samples)]
    )
    return float(np.mean(transmissions))


def build_dataframe() -> pd.DataFrame:
    T_empty = (kappa / 2) ** 2 / ((kappa / 2) ** 2 + Delta_c**2)
    T_z_vals_uK = np.linspace(0, 200, 50)
    T_y_vals_uK = np.linspace(0, 500, 5)
    T_z_vals = T_z_vals_uK * 1e-6
    T_y_vals = T_y_vals_uK * 1e-6

    rows = []
    total = len(T_y_vals) * len(T_z_vals)
    count = 0

    for T_y_uK, T_y in zip(T_y_vals_uK, T_y_vals):
        for T_z_uK, T_z in zip(T_z_vals_uK, T_z_vals):
            count += 1
            progress = (count / total) * 100
            print(
                f"Progress: {count}/{total} ({progress:.1f}%) - "
                f"T_y={T_y_uK:.1f}uK, T_z={T_z_uK:.1f}uK",
                end="\r",
            )
            rows.append(
                {
                    "T_y": T_y,
                    "T_z": T_z,
                    "T_y_uK": T_y_uK,
                    "T_z_uK": T_z_uK,
                    "Transmission": calculate_average_transmission(T_y, T_z) / T_empty,
                }
            )

    print("\nCalculation complete.")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("thermal_transmission_calibration.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to write the generated CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataframe()
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
