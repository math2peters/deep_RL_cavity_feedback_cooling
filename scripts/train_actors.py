"""Train one learned controller of Fig. 4 with SAC.

  python scripts/train_actors.py --actor mlp --scenario nominal --seed 11

Actors: "mlp" (two hidden layers of 16, the size deployed on the FPGA), "mlp_large" (two hidden
layers of 64, with four times more gradient updates per transition) and "affine" (no hidden layers).
Fig. 4 uses seeds 11, 22, 33, 44 and 55 of each actor at each operating point. A training seed fixes
the network initialization, the exploration noise and the training episodes. SAC resets
environment i with seed + i before learning, overriding its constructor seed. After every 10^5
environment steps the deterministic policy is scored on the same 100 validation episodes; the best
checkpoint is saved as best_model.zip. Checkpoints of an earlier run in the same directory are
removed at the start; config.json records the steps and the checkpoint hashes only once training
completes, and scripts/export_actors.py packs only checkpoints that match a completed run.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
from torch import nn
from torch.optim import AdamW

from src.evaluation.fig4 import LEARNED, SCENARIOS, sha256
from src.rl_env.noise_covariance import MODEL_VERSION

VALIDATION_SEED = 100_001
CHECKPOINTS = ("best_model.zip", "final.zip")
ACTORS = {"mlp": ([16, 16], 256), "mlp_large": ([64, 64], 1024), "affine": ([], 256)}   # hidden layers, updates


def run_directory(actor, scenario, seed):
    return ROOT/"outputs/training"/scenario/f"{actor}_{seed}"


def make_env(seed, scenario):
    from src.evaluation.benchmark_controllers import BenchmarkEnv
    return Monitor(BenchmarkEnv(study_seed=seed, **SCENARIOS[scenario]))


class FixedValidation(EvalCallback):
    """Score every checkpoint on the same validation episodes."""
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(VALIDATION_SEED)
        return super()._on_step()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actor", choices=LEARNED, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_020_000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    hidden, gradient_steps = ACTORS[args.actor]
    out = run_directory(args.actor, args.scenario, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out/"config.json").write_text(json.dumps({**vars(args), "noise_model": MODEL_VERSION}, indent=2) + "\n")
    for name in CHECKPOINTS + ("evaluations.npz",):   # left by an earlier or interrupted run
        (out/name).unlink(missing_ok=True)
    train_env = SubprocVecEnv([partial(make_env, 1000*args.seed + i, args.scenario) for i in range(8)],
                              start_method="spawn")
    validation_env = DummyVecEnv([partial(make_env, VALIDATION_SEED, args.scenario)])
    try:
        model = SAC("MlpPolicy", train_env, gamma=0.98, tau=0.01, learning_rate=1e-4,
                    buffer_size=500_000, batch_size=256, train_freq=(256, "step"),
                    gradient_steps=gradient_steps, target_update_interval=4, ent_coef="auto_0.1",
                    learning_starts=10_000, seed=args.seed, device="cpu", verbose=0,
                    policy_kwargs={"net_arch": {"pi": hidden, "qf": [512, 256, 128]},
                                   "activation_fn": nn.LeakyReLU, "optimizer_class": AdamW,
                                   "log_std_init": 0.0})
        callback = FixedValidation(validation_env, n_eval_episodes=100, eval_freq=12_500, deterministic=True,
                                   best_model_save_path=str(out), log_path=str(out), verbose=1)
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(out/"final")
        if not (out/"best_model.zip").exists():
            raise RuntimeError("Training ended before the first validation; no best checkpoint was saved")
        (out/"config.json").write_text(json.dumps({**vars(args), "noise_model": MODEL_VERSION, "total_timesteps": model.num_timesteps,
                                                  "gradient_updates": model._n_updates,
                                                  "checkpoints_sha256": {name: sha256(out/name) for name in CHECKPOINTS}},
                                                 indent=2) + "\n")
    finally:
        train_env.close()
        validation_env.close()


if __name__ == "__main__":
    main()
