"""Benchmark the pieces of a simulation training run and project a full run's wall time.

Measures (1) one environment step with each SDE backend, (2) one SAC gradient
step with the paper's network sizes at several torch thread counts, and (3)
projects the wall time of train_network_simulation.py (1.02M steps, 8 envs,
256 gradient steps per 256 vector steps).

Usage:
    python scripts/benchmark_training_speed.py [--skip-sdeint]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from src.rl_env.differentiator import DifferentiatorController  # noqa: E402
from src.rl_env.environment import CavityCoolingEnv  # noqa: E402

TOTAL_TIMESTEPS = 1_020_000
NUM_ENVS = 8
LEARNING_STARTS = 10_000
STEPS_PER_UPDATE = 256  # vector-env steps between training phases
GRADIENT_STEPS = 256


def time_env_step(backend, n_steps):
    env = CavityCoolingEnv(seed=1, sde_backend=backend)
    controller = DifferentiatorController()
    observation, _ = env.reset(seed=3)
    for _ in range(3):  # compile / warm caches
        observation, *_ = env.step(np.array([0.5]))
    start = time.perf_counter()
    for _ in range(n_steps):
        action, _ = controller.predict(observation)
        observation, _, done, _, _ = env.step(action)
        if done:
            observation, _ = env.reset()
    return (time.perf_counter() - start) / n_steps


def time_vec_env_step(n_steps=300):
    env = SubprocVecEnv([lambda i=i: CavityCoolingEnv(seed=7 + i) for i in range(NUM_ENVS)])
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):  # compile in the worker processes
        env.step(rng.uniform(-1, 1, (NUM_ENVS, 1)).astype(np.float32))
    start = time.perf_counter()
    for _ in range(n_steps):
        env.step(rng.uniform(-1, 1, (NUM_ENVS, 1)).astype(np.float32))
    elapsed = (time.perf_counter() - start) / n_steps
    env.close()
    return elapsed


def time_gradient_step(threads, n_steps=300):
    torch.set_num_threads(threads)
    env = DummyVecEnv([lambda: CavityCoolingEnv(seed=7)])
    model = SAC(policy="MlpPolicy", env=env, gamma=0.98, tau=0.01, learning_rate=1e-4,
                buffer_size=60_000, batch_size=256, train_freq=(256, "step"), gradient_steps=256,
                target_update_interval=4, ent_coef="auto_0.1", learning_starts=LEARNING_STARTS,
                policy_kwargs={"net_arch": dict(pi=[16, 16], qf=[512, 256, 128]),
                               "activation_fn": nn.LeakyReLU, "optimizer_class": AdamW,
                               "log_std_init": 0.0},
                seed=27, verbose=0, device="cpu")
    model.set_logger(configure(None, []))
    buffer = model.replay_buffer
    n_fill = 50_000
    rng = np.random.default_rng(0)
    for name in ("observations", "next_observations", "actions", "rewards"):
        array = getattr(buffer, name)
        array[:n_fill] = rng.uniform(-1, 1, array[:n_fill].shape)
    buffer.pos = n_fill
    model.train(gradient_steps=20, batch_size=256)  # warm up
    start = time.perf_counter()
    model.train(gradient_steps=n_steps, batch_size=256)
    env.close()
    return (time.perf_counter() - start) / n_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-sdeint", action="store_true", help="skip the slow original integrator")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    print("Environment step (single process, differentiator policy)")
    t_numba = time_env_step("numba", 1000)
    print(f"  numba : {t_numba*1e3:6.2f} ms")
    if not args.skip_sdeint:
        t_sdeint = time_env_step("sdeint", 100)
        print(f"  sdeint: {t_sdeint*1e3:6.2f} ms  ({t_sdeint/t_numba:.0f}x slower)")

    t_vec = time_vec_env_step()
    print(f"  {NUM_ENVS} envs in subprocesses: {t_vec*1e3:6.2f} ms per vector step")

    print("SAC gradient step (batch 256, actor 16x16, critics 512-256-128)")
    t_grad = {}
    for threads in args.threads:
        t_grad[threads] = time_gradient_step(threads)
        print(f"  torch threads {threads:2d}: {t_grad[threads]*1e3:6.2f} ms")
    best_threads = min(t_grad, key=t_grad.get)

    n_gradient_steps = (TOTAL_TIMESTEPS - LEARNING_STARTS) / (STEPS_PER_UPDATE * NUM_ENVS) * GRADIENT_STEPS
    print(f"Projected 1.02M-step run: {n_gradient_steps:,.0f} gradient steps = "
          f"{n_gradient_steps*t_grad[best_threads]/60:.1f} min with {best_threads} thread(s), plus "
          f"> {TOTAL_TIMESTEPS/NUM_ENVS*t_vec/60:.1f} min of environment stepping "
          f"(policy inference and replay-buffer writes come on top)")


if __name__ == "__main__":
    main()
