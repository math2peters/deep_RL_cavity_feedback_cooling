from stable_baselines3 import SAC
import numpy as np
import os
import sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rl_env.train_utils import LinearSACPolicy
from src.rl_env.environment import CavityCoolingEnv
from src.rl_env.differentiator import DifferentiatorController

def build_differentiator_model():
    """Build the differentiator baseline wrapper used for path compatibility."""
    env = CavityCoolingEnv(architecture='MLP', full_observations=False, seed=42, frame_stack_number=7)
    model = SAC(
        policy=LinearSACPolicy,
        env=env,
        learning_rate=1e-4,
        buffer_size=1,
        batch_size=1,
        policy_kwargs={
            "net_arch": dict(pi=[], qf=[16, 16]),
            "log_std_init": -10.0,
        },
        verbose=0
    )
    model.differentiator = DifferentiatorController()

    def custom_predict(observation, deterministic=True, **kwargs):
        return model.differentiator.predict(observation, deterministic)

    model.predict = custom_predict
    return model, env


def main():
    parser = argparse.ArgumentParser(
        description="Create the differentiator baseline wrapper retained for model-path compatibility."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "differentiator",
        help="Output path for the differentiator compatibility wrapper (without or with .zip extension).",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path.suffix == ".zip":
        output_path = output_path.with_suffix("")

    os.makedirs(output_path.parent, exist_ok=True)
    model, env = build_differentiator_model()
    model.save(output_path)
    env.close()

    print("Created differentiator compatibility wrapper")
    print(f"Wrapper saved to: {output_path}.zip")


if __name__ == "__main__":
    main()
