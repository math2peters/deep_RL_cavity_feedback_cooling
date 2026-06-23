#!/usr/bin/env python3
"""
Collect Atom Trajectories

This script runs multiple episodes of atom simulations, collecting velocity and position 
histories of the atom. It uses multiprocessing to speed up data collection and only keeps
episodes where the atom remained trapped throughout the simulation.
"""

import os

import numpy as np
import sys
import argparse
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
TRAJECTORY_OUTPUT_DIR = REPO_ROOT / "outputs" / "trajectory_cache"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set up multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
    print("Using 'spawn' method for multiprocessing")
except RuntimeError:
    print("Could not set multiprocessing start method to 'spawn'")

# Import the environment and the SAC model
from src.rl_env.differentiator import DifferentiatorController, is_differentiator_model
from src.rl_env.environment import CavityCoolingEnv
from stable_baselines3 import SAC


def load_policy_model(model_path):
    if is_differentiator_model(model_path):
        return DifferentiatorController()
    return SAC.load(model_path)

def run_episode(model_path, seed, episode_id, deterministic=True, diffusion_on=True):
    """
    Run a single episode and collect trajectory data
    
    This function ensures proper randomization by:
    1. Using a unique seed for each episode (base seed + episode_id)
    2. Setting this seed for both numpy and torch random number generators
    3. Using a dedicated RandomState to make randomization consistent
    4. Passing the same seed to both environment creation and reset
    5. Ensuring thermal distributions are used for atom positions and velocities
    
    Args:
        model_path: Path to the saved model
        seed: Base seed for randomization
        episode_id: Identifier for this specific episode
        deterministic: Whether to use deterministic actions
        diffusion_on: Whether to enable diffusion in the simulation
        
    Returns:
        Dictionary containing trajectory data if atom remained trapped,
        None otherwise
    """
    # Set unique seed for this episode
    episode_seed = seed + episode_id
    
    # Set seeds for all random number generators
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    
    # Save original random functions for proper cleanup later
    original_random_normal = np.random.normal
    original_random_rand = np.random.rand
    
    # Force deterministic randomness using a separate RandomState
    det_rng = np.random.RandomState(episode_seed)
    
    def deterministic_normal(*args, **kwargs):
        return det_rng.normal(*args, **kwargs)
    
    def deterministic_rand(*args, **kwargs):
        return det_rng.rand(*args, **kwargs)
    
    # Apply the patches to ensure deterministic environment behavior
    np.random.normal = deterministic_normal
    np.random.rand = deterministic_rand
    
    # Create environment for this episode
    try:
        env = CavityCoolingEnv(
            architecture='MLP',
            full_observations=False,
            seed=episode_seed,
            truncate_if_untrapped=True,  
            frame_stack_number=7,
            diffusion_on=diffusion_on,
            frame_wait_mode=True
        )
        
        # Load the model
        model = load_policy_model(model_path)
        
        # Reset the environment - pass the same seed to ensure consistency
        obs, _ = env.reset(seed=episode_seed)
        
        # Run the episode
        done = False
        truncated = False
        
        # Initialize a counter to track steps
        step_count = 0

        while not done:

            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Update count values for next iteration
            prev_count = obs[-6]
            cur_count = obs[-2]  # Get current count from observation
            
            # Increment step counter
            step_count += 1
        
        # Get position and velocity histories for all episodes, not just trapped ones
        position_history = env.atom.get_position_history()
        velocity_history = env.atom.get_velocity_history()
        
        # Get transmission data (photon counts at each step)
        transmission_history = env.total_count_list.copy()
        
        # If frame_wait_mode is True, filter out initial frames where step_number < frame_stack_number
        if env.frame_wait_mode:
            frame_stack_number = env.frame_stack_number - 1
            # Calculate how many simulation steps correspond to one environment step
            steps_per_env_step = int(env.steps_per_us * env.t_step * 1e6)
            # Calculate the starting index to skip the warm-up phase
            start_idx = frame_stack_number * steps_per_env_step
            
            # Filter out the initial frames
            position_history = [pos[start_idx:] for pos in position_history]
            velocity_history = [vel[start_idx:] for vel in velocity_history]
            
            # Filter transmission history to match (transmission is per environment step, not simulation substep)
            transmission_history = transmission_history[frame_stack_number:]
            
            # Recalculate times array to match filtered data
            times = np.linspace(0, env.time_elapsed - frame_stack_number * env.t_step, len(position_history[0]))
        else:
            times = np.linspace(0, env.time_elapsed, len(position_history[0]))
        
        # Create results dictionary
        results = {
            'episode_id': episode_id,
            'top_trap_U0_max': env.top_trap_U0_max,
            'top_trap_U0': env.top_trap_U0,
            'top_trap_waist_z': env.top_trap_waist_z,
            'top_trap_offset_z': env.top_trap_offset_z,
            'times': times,
            'position_history': position_history,
            'velocity_history': velocity_history,
            'transmission_history': transmission_history,  # Add transmission data
            'energy_list': env.energy_list if hasattr(env, 'energy_list') else [],
            'reward': env.episode_reward if hasattr(env, 'episode_reward') else 0,
            'trapped': not truncated  # Add a flag to indicate if the atom remained trapped
        }
        
        env.close()
        # Restore original random functions
        np.random.normal = original_random_normal
        np.random.rand = original_random_rand
        return results
    except Exception as e:
        # Restore original random functions in case of error
        np.random.normal = original_random_normal
        np.random.rand = original_random_rand
        raise

def get_model_name(model_path):
    """Extract a simplified model name from the model path"""
    # Remove any .zip extension if present
    base_name = os.path.basename(model_path.replace('.zip', ''))
    # Get the directory name
    dir_name = os.path.basename(os.path.dirname(model_path))
    
    # Just use the last directory and base name
    return f"{dir_name}_{base_name}"

def collect_trajectories(model_path, num_episodes=10_000, num_workers=None, deterministic=True, diffusion_on=True, save_path=None):
    """
    Collect trajectory data from multiple episodes using multiprocessing
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to run
        num_workers: Number of parallel workers (default: CPU count)
        deterministic: Whether to use deterministic actions
        diffusion_on: Whether to enable diffusion in the simulation
        save_path: Path to save the collected data
    Returns:
        List of trajectory data dictionaries
    """
    start_time = time.time()
    
    # Set number of workers
    if num_workers is None:
        num_workers = 8#mp.cpu_count()
    
    print(f"Starting evaluation with {num_workers} workers for {num_episodes} episodes")
    print(f"Diffusion: {'ON' if diffusion_on else 'OFF'}")
    # Base seed for reproducibility
    base_seed = 42
    
    # Prepare inputs for parallel execution
    inputs = [(model_path, base_seed, i, deterministic, diffusion_on) for i in range(num_episodes)]
    
    successful_episodes = []
    
    if num_workers == 1:
        for i, episode_args in enumerate(inputs):
            try:
                result = run_episode(*episode_args)
                if result is not None:
                    successful_episodes.append(result)
                    trapped_count = sum(1 for ep in successful_episodes if ep.get('trapped', False))
                    print(f"Completed episode {i+1}/{num_episodes}, trapped rate: {trapped_count/(i+1):.4f}")
                else:
                    print(f"Episode {i+1} failed to run")
            except Exception as e:
                print(f"Error in episode {i+1}: {e}")
    else:
        # Execute episodes in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_episode, *args) for args in inputs]

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result is not None:
                        successful_episodes.append(result)
                        trapped_count = sum(1 for ep in successful_episodes if ep.get('trapped', False))
                        print(f"Completed episode {i+1}/{num_episodes}, trapped rate: {trapped_count/(i+1):.4f}")
                    else:
                        print(f"Episode {i+1} failed to run")
                except Exception as e:
                    print(f"Error in episode {i+1}: {e}")
    
    # Count how many episodes have trapped atoms
    trapped_count = sum(1 for ep in successful_episodes if ep.get('trapped', False))
    print(f"Collected {len(successful_episodes)} total episodes out of {num_episodes} attempted")
    if successful_episodes:
        print(f"Trapped rate: {trapped_count/len(successful_episodes):.4f} ({trapped_count} trapped episodes)")
    else:
        print("No successful episodes collected")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Save the collected data if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(successful_episodes, f)
        print(f"Saved trajectory data to {save_path}")
    
    return successful_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect atom trajectory data from multiple episodes")
    parser.add_argument("--model", type=str,
                        default=str(MODELS_DIR / "differentiator.zip"),
                        help=("Path to a trained MLP checkpoint, or to differentiator.zip "
                              "to select the source-code differentiator controller"))
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Number of episodes to run")
    parser.add_argument("--workers", type=int, default=8, 
                        help="Number of worker processes ")
    parser.add_argument("--deterministic", action="store_true", default=True, 
                        help="Use deterministic actions")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false",
                        help="Use non-deterministic actions")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save the collected data (default: auto-generated based on model name)")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Run a simple test without multiprocessing")
    
    # Fixed boolean flags using store_true/store_false actions
    parser.add_argument("--diffusion-on", action="store_true", dest="diffusion_on", default=True,
                        help="Enable diffusion in the simulation")
    parser.add_argument("--no-diffusion", action="store_false", dest="diffusion_on",
                        help="Disable diffusion in the simulation")

    
    args = parser.parse_args()
    
    # Print configuration for clarity
    print(f"Using {args.workers} workers.")
    print(f"Running with {'deterministic' if args.deterministic else 'non-deterministic'} actions.")
    print(f"Diffusion: {'ON' if args.diffusion_on else 'OFF'}")
    print(f"Model: {args.model}")
    print(f"Save path: {args.save_path}")
    
    # Generate default save path if not provided
    if args.save_path is None:
        model_name = get_model_name(args.model)
        diffusion_suffix = "" if args.diffusion_on else "_no_diffusion"
        args.save_path = str(TRAJECTORY_OUTPUT_DIR / f"atom_trajectories_{model_name}{diffusion_suffix}.pkl")
    
    # Update function calls to use the correctly named arguments
    if args.test:
        # Run simple test without multiprocessing
        print("Running simple test without multiprocessing")
        successful_episodes = []
        trapped_count = 0
        
        for i in range(args.episodes):
            result = run_episode(args.model, 42, i, args.deterministic, args.diffusion_on)
            if result is not None:
                successful_episodes.append(result)
                if result.get('trapped', False):
                    trapped_count += 1
                print(f"Episode {i+1} completed, trapped: {result.get('trapped', False)}")
            else:
                print(f"Episode {i+1} failed to run")
        
        print(f"Collected {len(successful_episodes)} total episodes")
        if successful_episodes:
            print(f"Trapped rate: {trapped_count/len(successful_episodes):.4f} ({trapped_count} trapped episodes)")
        else:
            print("No successful episodes collected")
        
        # Save the collected data
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'wb') as f:
            pickle.dump(successful_episodes, f)
        print(f"Saved trajectory data to {args.save_path}")
    else:
        trajectories = collect_trajectories(
            model_path=args.model,
            num_episodes=args.episodes, 
            num_workers=args.workers,
            deterministic=args.deterministic,
            diffusion_on=args.diffusion_on,
            save_path=args.save_path
        )
