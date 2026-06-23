from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import time
import torch.nn as nn
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from torch.optim import AdamW
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_OUTPUT_DIR = REPO_ROOT / "outputs" / "training"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import functions and classes from train_utils.py
from src.rl_env.train_utils import (
    LinearSACPolicy,
    InterruptCallback,
    PlottingCallback,
    CustomLoggingCallback,
    make_env,
    exponential_schedule
)

architecture = 'MLP'
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train RL network simulation')
    parser.add_argument('--run-name', type=str, default='default', help='Name for this training run')
    args = parser.parse_args()
    
    start_time = time.time()
    num_envs = 8
    gym_env = False
    full_observations = False
    frame_stacks = 7
    rcs = [1, 1, 1]
    use_linear_policy = False  # Set to True to use linear policy instead of MLP
    
    train_vec_env = SubprocVecEnv([make_env(i, architecture=architecture, reward_scale=1, reward_component_scale=rcs, 
                                            seed=i+7, gym_env=gym_env, full_observations=full_observations, 
                                            frame_stacks=frame_stacks) for i in range(num_envs)])
    
    eval_vec_env = SubprocVecEnv([make_env(0, architecture=architecture, render_mode='human', 
                                           seed=42, gym_env=gym_env, full_observations=full_observations, 
                                           frame_stacks=frame_stacks)])



    n_total_runs = int(1020e3)

    # Configure policy based on the selected type (linear or MLP)
    if use_linear_policy:
        # Use our custom linear policy
        policy_type = LinearSACPolicy
        policy_kwargs = {
            "net_arch": dict(
                pi=[],  # Need to include this key, even if empty
                qf=[512, 256,128]  # Critic network architecture
            ),
            "activation_fn": nn.LeakyReLU,
            "optimizer_class": AdamW,
            "log_std_init": 0.0,
        }
        model_save_path = str(TRAINING_OUTPUT_DIR / "linear_policy" / args.run_name)
    else:
        # Use the default MLP policy
        policy_type = "MlpPolicy"
        policy_kwargs = {
            "net_arch": dict(
                pi=[16, 16],  # Actor network
                qf=[512, 256, 128]  # Critic network
            ),
            "activation_fn": nn.LeakyReLU,
            "optimizer_class": AdamW,
            "log_std_init": 0.0,
        }
        model_save_path = str(TRAINING_OUTPUT_DIR / "mlp" / args.run_name)
    

    
    model = SAC(
            policy=policy_type,
            env=train_vec_env,
            gamma=0.98,
            tau=0.01,
            learning_rate=1e-4,
            buffer_size=500_000,
            batch_size=256,
            train_freq=(256, "step"),
            gradient_steps=256,
            target_update_interval=4,
            ent_coef="auto_0.1",
            learning_starts=10_000,
            policy_kwargs=policy_kwargs,
            seed=27,
            verbose=0
        )
    
    

    # Configure the logger to write to a CSV file
    log_folder = f"{model_save_path}/logs"
    new_logger = configure(folder=log_folder, format_strings=["stdout", "csv"])
    model.set_logger(new_logger)

    eval_callback2 = EvalCallback(eval_env=eval_vec_env, n_eval_episodes=100, eval_freq=12_500, deterministic=True, render=False, verbose=1, best_model_save_path=f"{model_save_path}/best_evaluated_model_det")

    interrupt_callback = InterruptCallback()

    cb = [eval_callback2, interrupt_callback, PlottingCallback(eval_vec_env), CustomLoggingCallback()]

    model.learn(total_timesteps=int(n_total_runs), callback=cb, progress_bar=True)

    model.save(f"{model_save_path}/trained_model", exclude=['weights_only'])
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds or {int((end_time - start_time) / 60)} minutes.")
    print(f"Policy type used: {'Linear' if use_linear_policy else 'MLP'}")
    print(f"Run name: {args.run_name}")
