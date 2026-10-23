"""Simulation curves of Fig. 3(a,b): the five nominal-point MLPs of Fig. 4 across the experimental sweeps.

  python scripts/sweep_fig3.py --sweep detuning [--noisy]

Sweeps the probe detuning (-105 to 105 MHz in 10-MHz steps) or the empty-cavity counts per bin
(1 to 81 in steps of 4), with every other parameter at its nominal value; the count normalization
follows the photon number. Nothing is retrained. At every value the same 1000 consecutive episodes
from --seed-start are split equally among the five training seeds. With --noisy, every episode also
redraws the experimental imperfections of noisy_measurements (probe power, trap depth and the
transverse position of the trap in the cavity mode). Writes the CSV that scripts/reproduce_fig3.py --simulation reads.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.fig4 import STEP20

SWEEPS = {"detuning": [float(d) for d in range(-105, 106, 10)],
          "photon_number": [float(n) for n in range(1, 82, 4)]}
EPISODES, CHUNK = 1000, 25


def condition_kwargs(sweep, value):
    if sweep == "detuning":
        return {"probe_detuning_input": 2*np.pi*value*1e6}
    return {"photon_number_input": value}


def run_chunk(task):
    policy, sweep, value, seeds, actors, noisy = task
    import torch
    torch.set_num_threads(1)
    from scripts.evaluate_fig4 import run_episode
    from src.evaluation.benchmark_controllers import BenchmarkEnv
    from src.evaluation.fig4 import load_controller
    controller = load_controller(policy, actors)
    env = None if noisy else BenchmarkEnv(**condition_kwargs(sweep, value))
    out = []
    try:
        for seed in seeds:
            if noisy:   # imperfections are drawn when the environment is built
                if env is not None:
                    env.close()
                env = BenchmarkEnv(study_seed=int(seed), noisy_measurements=True, **condition_kwargs(sweep, value))
            _, retained, energies = run_episode(env, controller, seed)
            out.append((retained, float(energies[STEP20])))
    finally:
        if env is not None:
            env.close()
    return value, out


def main():
    from src.evaluation.fig4 import selected_policies
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=SWEEPS, required=True)
    parser.add_argument("--noisy", action="store_true", help="Redraw the experimental imperfections every episode")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--actors", type=Path, help="Newly exported actor weights; default: frozen paper actors")
    parser.add_argument("--seed-start", type=int, default=5_000_001)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/fig3_sweeps")
    args = parser.parse_args()
    policies = selected_policies(scenario="nominal", families=["mlp"])
    share = EPISODES // len(policies)
    tasks = [(policy, args.sweep, value, list(range(start, min(start+CHUNK, first+share))), args.actors, args.noisy)
             for i, policy in enumerate(policies) for first in [args.seed_start + i*share]
             for value in SWEEPS[args.sweep] for start in range(first, first+share, CHUNK)]
    results = {value: [] for value in SWEEPS[args.sweep]}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for value, episodes in pool.map(run_chunk, tasks):
            results[value].extend(episodes)
    lines = [f"{args.sweep},fraction_completed,se_fraction_completed,energy_546us_uK"]
    for value, episodes in results.items():
        retained = np.array([r for r, _ in episodes], float)
        energies = np.array([e for r, e in episodes if r])
        lines.append(f"{value},{retained.mean()},{retained.std(ddof=1)/np.sqrt(len(retained))},"
                     f"{energies.mean() if len(energies) else ''}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/f"{args.sweep}_sweep_results_mlp_sim{'_noisy' if args.noisy else ''}.csv"
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
