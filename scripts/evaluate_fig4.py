"""Rerun the frozen Fig. 4 controllers on the shared test episodes.

Writes one record per controller to --output-dir; pass that directory to
scripts/reproduce_fig4.py --evaluations to redraw the figure from the rerun.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.fig4 import FULL_STEPS, LEARNED, SCENARIOS, compact_record, load_controller, selected_policies


def run_episode(env, controller, seed):
    """Total reward, survival of the full sequence, and the axial energy (uK) after every step.

    The energy uses the maximum-depth axial potential, so it does not change with the trap depth.
    """
    from src.rl_env.environment import m, kB
    observation, _ = env.reset(seed=int(seed))
    if hasattr(controller, "reset"):
        controller.reset()
    reward_sum, energies = 0., np.full(FULL_STEPS, np.nan)
    for step in range(FULL_STEPS):
        action, _ = controller.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        z = (env.atom.get_position()[2]-env.top_trap_offset_z)/env.top_trap_waist_z
        energies[step] = (.5*m*env.atom.get_velocity()[2]**2 + env.top_trap_U0_max*(z**2/(1+z**2)))/kB*1e6
        if terminated or truncated:
            break
    retained = bool(info["trapped_end"]) and env.time_elapsed >= env.t_max
    return reward_sum, retained, energies


def evaluate_policy(task):
    policy, seeds, directory, actors = task
    import torch
    torch.set_num_threads(1)
    from src.evaluation.benchmark_controllers import BenchmarkEnv
    started = time.perf_counter()
    controller = load_controller(policy, actors)
    env = BenchmarkEnv(**SCENARIOS[policy["scenario"]])
    try:
        episodes, diagnostics = [], []
        for seed in seeds:
            episodes.append(run_episode(env, controller, seed))
            diagnostics.append((env.covariance_counts.copy(),
                                getattr(controller, "covariance_counts", np.zeros(2, dtype=np.int64)).copy()))
    finally:
        env.close()
    record = compact_record(seeds, np.array([e[1] for e in episodes]), np.array([e[2] for e in episodes]))
    record["simulator_covariance_counts"] = np.array([d[0] for d in diagnostics])
    record["controller_covariance_counts"] = np.array([d[1] for d in diagnostics])
    np.savez_compressed(directory/f"{policy['id']}.npz", noise_model=np.array(env.noise_model), **record)
    return (f"{policy['id']}: {len(seeds)} episodes in {time.perf_counter()-started:.1f} s; "
            f"simulator covariance [evaluations, bounds] = {record['simulator_covariance_counts'].sum(axis=0).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--controller", choices=[*LEARNED, "differentiator", "kalman", "particle", "none"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=3000000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tuning", type=Path, help="Selections from a new tuning run; default: frozen paper parameters")
    parser.add_argument("--actors", type=Path, help="Newly exported actor weights; default: frozen paper actors")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/fig4_evaluations")
    args = parser.parse_args()
    policies = selected_policies(args.tuning, args.scenario, [args.controller] if args.controller else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start+args.episodes)
    tasks = [(policy, seeds, args.output_dir, args.actors) for policy in policies]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for result in pool.map(evaluate_policy, tasks):
            print(result, flush=True)


if __name__ == "__main__":
    main()
