"""Check the compiled SDE backend against the original sdeint implementation.

Both backends receive identical initial conditions and random draws. Discrete
observations, rewards, counts and termination decisions must match exactly;
continuous trajectories and terminal metrics must agree within STATE_RTOL.
This checks numerical equivalence on the tested episodes, not identical SAC
training runs: the original environment also draws unseeded SDE noise.

Usage: python scripts/verify_sde_backend.py [--episodes 5]
"""

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl_env.differentiator import DifferentiatorController  # noqa: E402
from src.rl_env.environment import CavityCoolingEnv  # noqa: E402

STATE_RTOL = 1e-9
EXACT_KEYS = ("initial_obs", "obs", "reward", "counts", "trapped", "terminated", "truncated")
CONTINUOUS_KEYS = ("state", "trajectory", "fields", "photon_sums", "final_temperature")


@contextmanager
def seeded_sde_noise(noise_seed):
    """Replace only unseeded default_rng calls with a reproducible sequence."""
    real_default_rng = np.random.default_rng
    calls = 0

    def seeded_default_rng(seed=None):
        nonlocal calls
        if seed is not None:
            return real_default_rng(seed)
        calls += 1
        return real_default_rng([noise_seed, calls])

    np.random.default_rng = seeded_default_rng
    try:
        yield
    finally:
        np.random.default_rng = real_default_rng


def run_episodes(backend, episode_seeds, noise_seed, *, env_kwargs=None, controller=None):
    """Run controlled episodes, including reset observations and terminal probe-off metrics."""
    episode_seeds = list(episode_seeds)
    if not episode_seeds:
        raise ValueError("At least one episode seed is required")
    episodes = []
    n_steps = 0
    start = time.perf_counter()
    with seeded_sde_noise(noise_seed):
        # Noisy constructor parameters are sampled before the constructor seeds NumPy.
        np.random.seed(11)
        env = CavityCoolingEnv(seed=11, sde_backend=backend, **(env_kwargs or {}))
        controller = controller if controller is not None else DifferentiatorController()
        try:
            for episode_seed in episode_seeds:
                np.random.seed(episode_seed)
                observation, _ = env.reset(seed=episode_seed)
                record = {key: [] for key in EXACT_KEYS + ("state",)}
                record["initial_obs"] = observation.copy()
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _ = controller.predict(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    n_steps += 1
                    record["obs"].append(observation.copy())
                    record["reward"].append(reward)
                    record["state"].append(np.concatenate(
                        [env.atom.get_position(), env.atom.get_velocity(), [env.a_r, env.a_i]]))
                    record["trapped"].append(env.atom.trapped)
                    record["terminated"].append(terminated)
                    record["truncated"].append(truncated)
                record["counts"] = env.total_count_list.copy()
                record["photon_sums"] = env.photon_counts_array.copy()
                record["trajectory"] = np.concatenate(
                    [env.atom.get_position_history(), env.atom.get_velocity_history()]).T
                record["fields"] = np.column_stack([env.a_r_history_list, env.a_i_history_list])
                record["final_temperature"] = info["final_temperature"]
                record["terminal_info"] = info
                episodes.append(record)
        finally:
            env.close()
    return episodes, (time.perf_counter() - start) / n_steps


def relative_error(reference, candidate):
    """Largest error scaled by each reference component's peak magnitude."""
    ref, new = np.asarray(reference), np.asarray(candidate)
    if ref.shape != new.shape or not (np.isfinite(ref).all() and np.isfinite(new).all()):
        return float("inf")
    if ref.size == 0:
        return 0.0
    scale = np.maximum(np.max(np.abs(ref), axis=0), 1e-300) if ref.ndim else max(abs(ref), 1e-300)
    return float(np.max(np.abs(ref - new) / scale))


def same_info(reference, candidate):
    """Compare all terminal metrics, preserving exact categorical values and None."""
    if reference.keys() != candidate.keys():
        return False
    for key, ref in reference.items():
        new = candidate[key]
        if isinstance(ref, dict):
            if not isinstance(new, dict) or not same_info(ref, new):
                return False
        elif ref is None:
            if new is not None:
                return False
        elif isinstance(ref, (bool, int, np.bool_, np.integer)):
            if ref != new:
                return False
        elif new is None or relative_error(ref, new) > STATE_RTOL:
            return False
    return True


def compare(reference, candidate):
    """Return (all_ok, worst continuous error); incomplete or empty runs fail."""
    if not reference or len(reference) != len(candidate):
        print(f"FAIL: episode counts {len(reference)} / {len(candidate)} (must be equal and nonzero)")
        return False, float("inf")
    all_ok = True
    worst_error = 0.0
    for i, (ref, new) in enumerate(zip(reference, candidate)):
        exact = bool(ref["reward"]) and all(np.array_equal(ref[key], new[key]) for key in EXACT_KEYS)
        errors = {key: relative_error(ref[key], new[key]) for key in CONTINUOUS_KEYS}
        continuous_error = max(errors.values())
        worst_error = max(worst_error, continuous_error)
        metrics_match = same_info(ref["terminal_info"], new["terminal_info"])
        ok = exact and continuous_error <= STATE_RTOL and metrics_match
        all_ok &= ok
        failed = [key for key, error in errors.items() if error > STATE_RTOL]
        print(f"episode {i}: {len(ref['reward'])} steps, discrete outputs "
              f"{'identical' if exact else 'DIFFER'}, max continuous error {continuous_error:.1e}, "
              f"terminal metrics {'match' if metrics_match else 'DIFFER'}, "
              f"final T {ref['final_temperature']*1e6:.4f} / {new['final_temperature']*1e6:.4f} uK"
              + (f", failed: {', '.join(failed)}" if failed else ""))
    return all_ok, worst_error


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--noise-seed", type=int, default=7)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    episode_seeds = [101 * (i + 1) for i in range(args.episodes)]
    run_episodes("numba", episode_seeds[:1], args.noise_seed)  # compile before timing
    reference, t_reference = run_episodes("sdeint", episode_seeds, args.noise_seed)
    candidate, t_candidate = run_episodes("numba", episode_seeds, args.noise_seed)
    all_ok, worst_error = compare(reference, candidate)
    print(f"\nsdeint {t_reference*1e3:.2f} ms/step, numba {t_candidate*1e3:.2f} ms/step "
          f"({t_reference/t_candidate:.0f}x)")
    print(f"max continuous error {worst_error:.1e} (tolerance {STATE_RTOL:.0e})")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
