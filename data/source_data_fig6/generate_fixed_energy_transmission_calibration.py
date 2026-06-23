from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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
detection_efficiency = 0.125

U0 = h * 36e6
eta = 20.2
Delta_a = 2 * PI * 25e6
empty_cavity_counts = 32
photon_number = (1 / (kappa * 20e-6 * detection_efficiency)) * 19
g_max = np.sqrt(eta * kappa * gamma / 4)


def force_2d(y_pos: float, z_pos: float) -> tuple[float, float]:
    """Calculate forces in y and z for the 2D trap."""
    denom = 1 + (z_pos / w0_z) ** 2

    f_y = -4 * U0 * y_pos / (w0_y**2) * (1 / denom) * np.exp(
        -2 * ((y_pos / w0_y) ** 2) * 1 / denom
    )

    if abs(y_pos) < 1e-10:
        f_z = -2 * U0 * z_pos / (w0_z**2 * denom**2)
    else:
        exp_term = np.exp(-2 * ((y_pos / w0_y) ** 2) * 1 / denom)
        f_z = -2 * U0 * z_pos / (w0_z**2 * denom**2) * exp_term
        f_z += -4 * U0 * (z_pos / w0_z**2) * ((y_pos / w0_y) ** 2) / (denom**3) * exp_term

    return f_y, f_z


def transmission_with_broadening_2d(y_pos: float, z_pos: float) -> float:
    """Return the one-sided normalized transmission at position (y, z)."""
    g = g_max * np.exp(-(y_pos**2 + z_pos**2) / cavity_waist**2)
    omega = 2 * g * np.sqrt(photon_number)
    denom = Delta_a**2 + (gamma / 2) ** 2 + 0.5 * omega**2

    local_u0 = (g**2 * Delta_a) / denom
    local_gamma = (g**2 * (gamma / 2)) / denom

    kappa_tot = kappa + 2 * local_gamma
    Delta_eff = Delta_c - local_u0

    return (kappa / 2) ** 2 / ((kappa_tot / 2) ** 2 + Delta_eff**2)


def classical_turning_point_y(E_y: float) -> float:
    if E_y >= U0:
        return 5 * cavity_waist

    arg = 1 - E_y / U0
    if arg <= 0:
        return 5 * cavity_waist

    return w0_y * np.sqrt(-0.5 * np.log(arg))


def classical_turning_point_z(E_z: float) -> float:
    if E_z >= U0:
        return 5 * cavity_waist
    return w0_z * np.sqrt(U0 / (U0 - E_z) - 1)


def atom_eom_2d(_t: float, state: np.ndarray) -> list[float]:
    y_pos, z_pos, v_y, v_z = state
    f_y, f_z = force_2d(y_pos, z_pos)
    return [v_y, v_z, f_y / m, f_z / m]


def simulate_transmission_2d(E_y: float, E_z: float, sim_periods: int = 50) -> float:
    if E_y < 1e-30 and E_z < 1e-30:
        return transmission_with_broadening_2d(0.0, 0.0)

    if E_y < 1e-30:
        z_max = classical_turning_point_z(E_z)
        y0 = [0.0, z_max, 0.0, 0.0]
    elif E_z < 1e-30:
        y_max = classical_turning_point_y(E_y)
        y0 = [y_max, 0.0, 0.0, 0.0]
    else:
        y_max = classical_turning_point_y(E_y)
        y0 = [y_max, 0.0, 0.0, np.sqrt(2 * E_z / m)]

    k_eff_y = 2 * U0 / w0_y**2
    k_eff_z = 2 * U0 / w0_z**2

    omega_y = np.sqrt(k_eff_y / m)
    omega_z = np.sqrt(k_eff_z / m)

    period_y = 2 * PI / omega_y
    period_z = 2 * PI / omega_z

    anharmonicity_y = np.clip(E_y / U0, 0, 0.9)
    anharmonicity_z = np.clip(E_z / U0, 0, 0.9)
    period_y *= 1 + 2 * anharmonicity_y
    period_z *= 1 + 2 * anharmonicity_z

    period = max(period_y, period_z)
    t_span = [0, sim_periods * period]
    t_eval = np.linspace(t_span[0], t_span[1], int(sim_periods * 1000))

    sol = solve_ivp(atom_eom_2d, t_span, y0, t_eval=t_eval, rtol=1e-7, atol=1e-9)

    y_vals = sol.y[0]
    z_vals = sol.y[1]

    max_distance = 20 * max(w0_y, w0_z)
    if np.max(np.abs(y_vals)) > max_distance or np.max(np.abs(z_vals)) > max_distance:
        mask = (np.abs(y_vals) < max_distance) & (np.abs(z_vals) < max_distance)
        y_vals = y_vals[mask]
        z_vals = z_vals[mask]
        if len(y_vals) < 10:
            return 1.0

    T_vals = np.array(
        [transmission_with_broadening_2d(y_pos, z_pos) for y_pos, z_pos in zip(y_vals, z_vals)]
    )
    return float(np.mean(T_vals))


def build_dataframe() -> pd.DataFrame:
    n_points = 51
    n_y_points = 5
    E_z_vals = np.linspace(0, 0.9 * U0, n_points)
    E_y_vals = np.linspace(0, 0.5 * U0, n_y_points)
    T_empty = (kappa / 2) ** 2 / ((kappa / 2) ** 2 + Delta_c**2)

    rows = []
    total_calcs = len(E_y_vals) * len(E_z_vals)
    calc_count = 0

    for E_y in E_y_vals:
        for E_z in E_z_vals:
            calc_count += 1
            progress_pct = (calc_count / total_calcs) * 100
            print(
                f"Progress: {calc_count}/{total_calcs} ({progress_pct:.1f}%) - "
                f"E_y={E_y / kB * 1e6:.1f}uK, E_z={E_z / kB * 1e6:.1f}uK",
                end="\r",
            )
            transmission = simulate_transmission_2d(E_y, E_z) / T_empty
            rows.append(
                {
                    "E_y": E_y,
                    "E_z": E_z,
                    "E_y_uK": E_y / kB * 1e6,
                    "E_z_uK": E_z / kB * 1e6,
                    "Transmission": transmission,
                }
            )

    print("\nCalculation complete.")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("fixed_energy_transmission_calibration.csv")
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
