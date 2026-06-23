"""
Multiprocessing Evaluation Script for Reinforcement Learning Feedback Cooling

This script provides a high-performance evaluation tool for analyzing saved reinforcement
learning models in the cavity cooling environment. It leverages parallel processing to
dramatically speed up evaluation and includes specialized features for visualization
of cooling dynamics.

Key Features:
- Parallel episode evaluation using multiprocessing
- Rendering of user-defined random episodes
- Collection and analysis of photon count traces
- Visualization of cooling dynamics with both corrected and uncorrected traces
- Estimation of cooling timescale
- FFT analysis of photon count signals
- Option to test with randomly initialized models (defaults to using the MLP (Sim.) policy)
- Control over deterministic vs. stochastic action selection

Usage Examples:
  # Basic evaluation with default settings (100 episodes, MLP (Sim.) policy)
  python -m src.evaluation.evaluate_multiprocess
  
  # Evaluate a specific model with 50 episodes and no rendering
  python -m src.evaluation.evaluate_multiprocess --model mlp_experimental --episodes 50 --render 0
  
  # Use 4 worker processes and disable trace truncation
  python -m src.evaluation.evaluate_multiprocess --workers 4 --no-truncate
  
  # Test with a random model using deterministic actions
  python -m src.evaluation.evaluate_multiprocess --use-random-model --deterministic true
  
  # Test with a random model using stochastic actions
  python -m src.evaluation.evaluate_multiprocess --use-random-model --deterministic false

Command Line Arguments:
  --model           : Model to evaluate ('mlp_sim', 'mlp_experimental', 'differentiator')
  --episodes        : Number of episodes to evaluate (default: 100)
  --render          : Number of random episodes to render (default: 0, disabled)
  --workers         : Number of worker processes (default: 10)
  --no-truncate     : Disable truncation of traces at threshold crossings
  --use-random-model: Use a randomly initialized model instead of a saved policy (default: False)
  --deterministic   : Use deterministic actions (default: True for saved policies, False for random models)
  --log-std-init    : Initial log standard deviation for random model (default: 0.0)
  --probe-detuning  : Probe detuning in MHz (default: use environment default)
  --photon-number   : Photon number (default: use environment default)
  --top-trap-u0-max : Top trap U0 max in MHz, converted to Hz internally (default: use environment default)
  --noisy-measurements: Enable noisy measurements to simulate experimental conditions
  
Output:
- Summary statistics of episode rewards, lengths, and temperatures
- Histogram of initial vs. final temperatures
- Mean photon count traces (loss-corrected and uncorrected)
- FFT analysis of photon count signals
- Estimated cooling timescale
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from collections import defaultdict
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter1d
import random
import argparse
import torch
import warnings
from pathlib import Path

# Filter out the specific warning about lr_schedule deserialization
warnings.filterwarnings("ignore", message="Could not deserialize object lr_schedule")
# Filter out the PyTorch serialization warning about torch.load with weights_only=False
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set up multiprocessing start method
# This helps ensure proper process isolation, especially with PyTorch models
# Not doing this can lead to deadlocks or corrupted states
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    print("Could not set multiprocessing start method to 'spawn'")

from src.rl_env.differentiator import DifferentiatorController, is_differentiator_model
from src.rl_env.environment import CavityCoolingEnv

kB = 1.380649e-23
m = 2.21e-25


def load_policy_model(model_path, custom_objects=None):
    if is_differentiator_model(model_path):
        return DifferentiatorController()
    return SAC.load(model_path, device='cpu', custom_objects=custom_objects)

# Function to pad or truncate arrays to a consistent length
def pad_or_truncate(arr, target_length):
    """Pads with zeros or truncates the array to match target_length."""
    current_length = len(arr)
    if current_length > target_length:
        return arr[:target_length]
    else:
        return np.pad(arr, (0, target_length - current_length), mode='constant')

def create_random_model(seed=None, log_std_init=0.0, probe_detuning_input=None, photon_number_input=None, top_trap_U0_max_input=None, temperature_input=None, noisy_measurements=False):
    """
    Create a randomly initialized model with the specified seed and log_std_init
    
    Args:
        seed: Random seed for reproducibility (None for complete randomness)
        log_std_init: Initial value for the log standard deviation in the policy
        probe_detuning_input: Probe detuning parameter for the environment (None to use environment default)
        photon_number_input: Photon number parameter for the environment (None to use environment default)
        top_trap_U0_max_input: Top trap U0 max parameter for the environment (None to use environment default)
        temperature_input: Temperature parameter for the environment in K (None to use environment default)
        noisy_measurements: Whether to add noise to measurements to simulate experimental conditions
        
    Returns:
        A randomly initialized SAC model
    """
    # Set seeds for reproducibility only if seed is not None
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # For true randomness
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Create a fresh environment for the model
    env_params = {
        'architecture': 'MLP',
        'full_observations': False,
        'seed': seed,  # Can be None for randomness
        'truncate_if_untrapped': True,
        'frame_stack_number': 7,
        'noisy_measurements': noisy_measurements
    }
    
    # Only add probe_detuning_input if it's explicitly specified
    if probe_detuning_input is not None:
        env_params['probe_detuning_input'] = probe_detuning_input
    
    # Only add photon_number_input if it's explicitly specified
    if photon_number_input is not None:
        env_params['photon_number_input'] = photon_number_input
    
    # Only add top_trap_U0_max_input if it's explicitly specified
    if top_trap_U0_max_input is not None:
        env_params['top_trap_U0_max_input'] = top_trap_U0_max_input
        
    # Only add temperature_input if it's explicitly specified
    if temperature_input is not None:
        env_params['temperature_input'] = temperature_input
    env = CavityCoolingEnv(**env_params)
    
    # Policy architecture - match what's used in the trained model
    policy_kwargs = {
        "net_arch": [16, 16],  # Same architecture as the trained model
        "log_std_init": log_std_init  # Set initial log standard deviation
    }
    
    # Create fresh SAC model with random weights
    random_model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        learning_starts=10000,
        seed=seed,  # Can be None for complete randomness
        verbose=0
    )
    
    print(f"Created random model with policy architecture: {policy_kwargs}")
    
    # Clean up environment
    env.close()
    
    return random_model

def evaluate_episode(model_path, seed, episode_id, collect_traces=False, truncate=True, deterministic=True, log_std_init=0.0, probe_detuning_input=None, photon_number_input=None, top_trap_U0_max_input=None, temperature_input=None, custom_objects=None, noisy_measurements=False):
    """
    Evaluate a single episode
    
    Args:
        model_path: Path to the saved model (ignored if use_random_model is True)
        seed: Base seed for randomization (None for complete randomness)
        episode_id: Identifier for this specific episode
        collect_traces: Whether to collect photon count traces
        truncate: Whether to truncate traces when photon counts exceed threshold
        deterministic: Whether to use deterministic actions
        log_std_init: Initial log standard deviation for random model
        probe_detuning_input: Probe detuning parameter for the environment (None to use environment default)
        photon_number_input: Photon number parameter for the environment (None to use environment default)
        top_trap_U0_max_input: Top trap U0 max parameter for the environment (None to use environment default)
        temperature_input: Temperature parameter for the environment in K (None to use environment default)
        custom_objects: Dictionary of custom objects needed for model loading (e.g., custom policies)
        noisy_measurements: Whether to add noise to measurements to simulate experimental conditions
    """
    # For complete randomness, do not set seeds
    episode_seed = seed
    
    # Create environment with same config as single-process script
    env_params = {
        'architecture': 'MLP',
        'full_observations': False,
        'seed': episode_seed,  # Can be None for randomness
        'truncate_if_untrapped': True,  # Match single-process script exactly
        'frame_stack_number': 7,
        'noisy_measurements': noisy_measurements
    }
    
    # Only add probe_detuning_input if it's explicitly specified
    if probe_detuning_input is not None:
        env_params['probe_detuning_input'] = probe_detuning_input
    
    # Only add photon_number_input if it's explicitly specified
    if photon_number_input is not None:
        env_params['photon_number_input'] = photon_number_input
    
    # Only add top_trap_U0_max_input if it's explicitly specified
    if top_trap_U0_max_input is not None:
        env_params['top_trap_U0_max_input'] = top_trap_U0_max_input
        
    # Only add temperature_input if it's explicitly specified
    if temperature_input is not None:
        env_params['temperature_input'] = temperature_input
        
    env = CavityCoolingEnv(**env_params)
    
    # Don't override random functions for true randomness
    original_random_normal = np.random.normal
    original_random_rand = np.random.rand
    
    # Only use deterministic functions if seed is not None
    det_rng = None
    if episode_seed is not None:
        det_rng = np.random.RandomState(episode_seed)
    
        def deterministic_normal(*args, **kwargs):
            return det_rng.normal(*args, **kwargs)
        
        def deterministic_rand(*args, **kwargs):
            return det_rng.rand(*args, **kwargs)
        
        # Apply the patches only if seeded
        np.random.normal = deterministic_normal
        np.random.rand = deterministic_rand
    
    try:
        # Force CPU usage to avoid CUDA issues across processes
        model = load_policy_model(model_path, custom_objects=custom_objects)
    
        if deterministic and hasattr(model, "policy") and hasattr(model.policy, "actor"):
            original_forward = model.policy.actor.forward
            
            def strictly_deterministic_forward(obs, deterministic=True):
                return original_forward(obs, deterministic=True)
            
            model.policy.actor.forward = strictly_deterministic_forward
    except Exception as e:
        print(f"Worker {episode_id} failed to setup model: {e}")
        # Restore the original random functions before raising the exception
        np.random.normal = original_random_normal
        np.random.rand = original_random_rand
        raise
    
    # Run episode
    obs, _ = env.reset()
    
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    # For trace collection
    raw_counts = []  # Store raw counts
    time_points = []
    action_history = []  # Track actions for debugging
    # For energy calculation
    z_energy_values = []  # Store z-energy values
    
    # Track if feedback has started (after frame_stack_number steps in frame_wait_mode)
    feedback_started = not env.frame_wait_mode  # True immediately if not in frame_wait_mode
    feedback_start_time = 0  # Will be set when feedback starts
    
    # Get initial values for energy calculation
    if not env.frame_wait_mode:
        initial_atom_velocity = env.atom.get_velocity()[2]
        initial_atom_position = env.atom.get_position()[2]

    # Episode loop
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_history.append(action.copy())
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Check if feedback has started (after frame_stack_number steps in frame_wait_mode)
        if env.frame_wait_mode and steps == env.frame_stack_number:
            feedback_started = True
            feedback_start_time = env.time_elapsed
        
        # Collect photon counts and energy values if requested
        if collect_traces:
            # Extract counts directly from environment
            raw_count = env.total_counts/env.expected_max_photon_number
            
            # Only collect data after feedback has started if in frame_wait_mode
            if feedback_started:
                # Adjust timestamps to be relative to feedback start
                current_time = env.step_number * env.t_step
                adjusted_time = current_time - feedback_start_time
                
                raw_counts.append(raw_count)
                time_points.append(adjusted_time)
                
                # Calculate and store z-energy
                velocity_z = env.atom.get_velocity()[2]
                position_z = env.atom.get_position()[2]
                # Calculate kinetic energy (1/2 m v_z^2)
                ke_z = 0.5 * m * velocity_z**2
                # Calculate potential energy at the top trap
                pe_z = -env.top_trap_U0_max / (1 + ((position_z - env.top_trap_offset_z) / env.top_trap_waist_z) ** 2) + env.top_trap_U0_max
                # Total z-energy
                total_z_energy = ke_z + pe_z
                # Convert to µK
                z_energy_values.append(total_z_energy / kB * 1e6)
    
    # Get final temperature and check if episode was successful
    episode_finished = False
    final_temperature = None
    trapped = False
    
    if env.frame_wait_mode:
        # Get history arrays
        position_history = env.atom.get_position_history()
        velocity_history = env.atom.get_velocity_history()
        
        # Calculate the index for the moment just before feedback starts
        # This gets the last timestep of the frame_stack_number-1 step
        feedback_pre_start_idx = (env.frame_stack_number) * int(env.steps_per_us * env.t_step * 1e6)
        
        # Make sure index is within bounds and positive
        if feedback_pre_start_idx >= 0 and feedback_pre_start_idx < position_history.shape[1]:
            initial_atom_position = position_history[2, feedback_pre_start_idx]
            initial_atom_velocity = velocity_history[2, feedback_pre_start_idx]
        else:
            # Fallback if index out of bounds
            initial_atom_position = env.atom.get_position()[2]
            initial_atom_velocity = env.atom.get_velocity()[2]
        
    V_t = -env.top_trap_U0_max / (1 + ((initial_atom_position - env.top_trap_offset_z) / env.top_trap_waist_z) ** 2) + env.top_trap_U0_max
    energy = V_t + 1/2 * m * initial_atom_velocity**2
    initial_temperature = (energy) / kB * 1e6
    
    if done and not truncated:  # Match exactly how single-process script handles this
        episode_finished = True
        # Get final velocity and position from environment history

        # Calculate final potential energy at the trap
        if env.atom.trapped:
            energy = env.energy_list[-1] * env.top_trap_U0_max
            # Convert to temperature in μK
            final_temperature = (energy) / kB * 1e6
        else:
            final_temperature = None
        
        # An atom is trapped if the episode finished naturally (done=True and truncated=False)
        trapped = True
    
    # Restore the original random functions
    if episode_seed is not None:
        np.random.normal = original_random_normal
        np.random.rand = original_random_rand
    
    # Clean up
    env.close()
    
    # Return episode results
    result = {
        'episode_id': episode_id,
        'total_reward': total_reward,
        'steps': steps,
        'finished': episode_finished,  # Only True if done and not truncated
        'initial_temperature': initial_temperature,
        'final_temperature': final_temperature, #if trapped else None,  # Only include final temp if trapped
        'trapped': trapped,
        'frame_wait_mode': env.frame_wait_mode,
        'frame_stack_number': env.frame_stack_number,
        'feedback_start_time': feedback_start_time
    }
    
    # Add trace data if collected
    if collect_traces:
        result['raw_counts'] = raw_counts
        result['time_points'] = time_points
        result['z_energy_values'] = z_energy_values
    
    return result

def render_episode(model_path, seed=None, render_mode='human', deterministic=True, log_std_init=0.0, probe_detuning_input=None, photon_number_input=None, top_trap_U0_max_input=None, temperature_input=None, custom_objects=None, noisy_measurements=False):
    """Render an episode with the trained or random model"""
    # Only set seeds if a seed is provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # For true randomness
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # Create environment for rendering using the same optional-parameter behavior
    # as evaluate_episode(): omit None values so environment defaults remain intact.
    env_params = {
        'architecture': 'MLP',
        'full_observations': False,
        'render_mode': render_mode,
        'seed': seed,  # Can be None for randomness
        'frame_stack_number': 7,
        'noisy_measurements': noisy_measurements
    }

    if probe_detuning_input is not None:
        env_params['probe_detuning_input'] = probe_detuning_input

    if photon_number_input is not None:
        env_params['photon_number_input'] = photon_number_input

    if top_trap_U0_max_input is not None:
        env_params['top_trap_U0_max_input'] = top_trap_U0_max_input

    if temperature_input is not None:
        env_params['temperature_input'] = temperature_input

    env = CavityCoolingEnv(**env_params)
    
    print(f"Rendering with seed {seed if seed is not None else 'None (random)'}")
    
    original_random_normal = np.random.normal
    original_random_rand = np.random.rand
    
    if seed is not None:
        det_rng = np.random.RandomState(seed)
        
        def deterministic_normal(*args, **kwargs):
            return det_rng.normal(*args, **kwargs)
        
        def deterministic_rand(*args, **kwargs):
            return det_rng.rand(*args, **kwargs)
        
        np.random.normal = deterministic_normal
        np.random.rand = deterministic_rand
    

    # Load model with CPU device for consistency
    render_model = load_policy_model(model_path, custom_objects=custom_objects)
    print(f"Rendering with model from {model_path}")
    
    # If using deterministic actions, patch the forward method for SB3 policies only.
    if deterministic and hasattr(render_model, "policy") and hasattr(render_model.policy, "actor"):
        original_forward = render_model.policy.actor.forward

        def strictly_deterministic_forward(obs, deterministic=True):
            return original_forward(obs, deterministic=True)

        render_model.policy.actor.forward = strictly_deterministic_forward
        print("Using strictly deterministic actions for rendering")
    elif deterministic:
        print("Using deterministic actions for rendering (non-SB3 controller)")
    else:
        print(f"Using stochastic actions for rendering (deterministic={deterministic})")
    
    render_obs, _ = env.reset()
    render_done = False
    render_truncate = False
    
    while not (render_done or render_truncate):
        render_action, _ = render_model.predict(render_obs, deterministic=deterministic)

        render_obs, _, render_done, render_truncate, _ = env.step(render_action)
    
    env.render(resolution=1)
    time.sleep(5)  # Show rendering for a few seconds
    
    # Restore original random functions if they were patched
    if seed is not None:
        np.random.normal = original_random_normal
        np.random.rand = original_random_rand
    
    env.close()

def evaluate_model_parallel(model_path, num_episodes=100, num_workers=None, render_episodes=None, 
                           trace_analysis=False, truncate_traces=True,
                           deterministic=True, log_std_init=0.0, probe_detuning_input=None,
                           photon_number_input=None, top_trap_U0_max_input=None, temperature_input=None,
                           do_plots=True, custom_objects=None, noisy_measurements=False):
    """
    Evaluate model performance using parallel episode evaluation
    
    Args:
        model_path: Path to the saved model (ignored if use_random_model is True)
        num_episodes: Total number of episodes to evaluate
        num_workers: Number of worker processes to use (defaults to CPU count)
        render_episodes: List of episode indices to render (None means don't render)
        trace_analysis: Whether to collect and analyze photon count traces
        truncate_traces: Whether to truncate traces when photon counts exceed threshold
        deterministic: Whether to use deterministic actions
        log_std_init: Initial log standard deviation for random model
        probe_detuning_input: Probe detuning parameter for the environment (None to use environment default)
        photon_number_input: Photon number parameter for the environment (None to use environment default)
        top_trap_U0_max_input: Top trap U0 max parameter for the environment (None to use environment default)
        temperature_input: Temperature parameter for the environment in K (None to use environment default)
        do_plots: Whether to generate plots
        custom_objects: Dictionary of custom objects needed for model loading (e.g., custom policies)
        noisy_measurements: Whether to add noise to measurements to simulate experimental conditions
        
    Returns:
        Dictionary containing evaluation results, including average metrics, cooling timescale,
        and energy trace data if trace_analysis is enabled
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Use at most 8 workers
    

    try:
        test_model = load_policy_model(model_path, custom_objects=custom_objects)
        print(f"Test loading model: Success - {type(test_model)}")
        del test_model  # Clean up test model
    except Exception as e:
        print(f"Error loading model in main process: {e}")
        raise

    model_type = "pre-trained"
    action_type = "deterministic" if deterministic else "stochastic"
    print(f"Starting parallel evaluation of {model_type} model with {action_type} actions")
    print(f"Using {num_workers} workers for {num_episodes} episodes")
    print(f"Running with completely random (unseeded) execution")
    
    if probe_detuning_input is not None:
        print(f"Probe detuning: {probe_detuning_input/(2*np.pi*1e6):.1f} MHz")
    else:
        print("Using environment default probe detuning")
    
    if photon_number_input is not None:
        print(f"Photon number: {photon_number_input}")
    else:
        print("Using environment default photon number")
    
    if top_trap_U0_max_input is not None:
        print(f"Top trap U0 max: {top_trap_U0_max_input}")
    else:
        print("Using environment default top trap U0 max")

    
    results = []
    start_time = time.time()

    def record_progress(completed_count):
        if completed_count % 50 == 0 or completed_count == num_episodes:
            episodes_finished = sum(r['finished'] for r in results)
            avg_reward = np.mean([r['total_reward'] for r in results])

            initial_temps = [r['initial_temperature'] for r in results]
            final_temps = [
                r['final_temperature'] for r in results
                if r['trapped'] and r['final_temperature'] is not None
            ]

            print(f"Progress: {completed_count}/{num_episodes} episodes")
            print(f"Current fraction of episodes that finished: {episodes_finished/completed_count:.4f}")
            print(f"Current average reward: {avg_reward:.4f}")
            if final_temps:
                print(f"Current average temperature final: {np.mean(final_temps):.4f} µK")
            print(f"Current average temperature initial: {(np.mean(initial_temps) if initial_temps else 0):.4f} µK")

    def run_episode_inline(episode_id):
        return evaluate_episode(
            model_path,
            None,
            episode_id,
            collect_traces=trace_analysis,
            truncate=truncate_traces,
            deterministic=deterministic,
            log_std_init=log_std_init,
            probe_detuning_input=probe_detuning_input,
            photon_number_input=photon_number_input,
            top_trap_U0_max_input=top_trap_U0_max_input,
            temperature_input=temperature_input,
            custom_objects=custom_objects,
            noisy_measurements=noisy_measurements
        )

    if num_workers == 1:
        for episode_id in range(num_episodes):
            try:
                result = run_episode_inline(episode_id)
                results.append(result)
                record_progress(episode_id + 1)
            except Exception as e:
                print(f"Error in episode evaluation: {e}")
                import traceback
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(
                evaluate_episode,
                model_path,
                None,
                i,
                collect_traces=trace_analysis,
                truncate=truncate_traces,
                deterministic=deterministic,
                log_std_init=log_std_init,
                probe_detuning_input=probe_detuning_input,
                photon_number_input=photon_number_input,
                top_trap_U0_max_input=top_trap_U0_max_input,
                temperature_input=temperature_input,
                custom_objects=custom_objects,
                noisy_measurements=noisy_measurements
            ) for i in range(num_episodes)]

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    record_progress(i + 1)
                except Exception as e:
                    print(f"Error in episode evaluation: {e}")
                    import traceback
                    traceback.print_exc()

    # Extract final statistics
    episode_rewards = [r['total_reward'] for r in results]
    episode_lengths = [r['steps'] for r in results]
    episodes_finished = sum(r['finished'] for r in results)
    
    atom_temperature_initial = [r['initial_temperature'] for r in results]
    # Only include temperatures from atoms that were successfully trapped
    atom_temperature_final = [r['final_temperature'] for r in results if r['trapped'] and r['final_temperature'] is not None]
    
    # Analyze energy traces if requested
    cooling_timescale = None
    energy_trace_data = None
    energy_result = None
    if trace_analysis and any('z_energy_values' in r for r in results):
        print("\nAnalyzing energy traces...")
        energy_result = analyze_energy_traces(results, truncate_traces, do_plots)
        if energy_result:
            if 'cooling_timescale' in energy_result:
                cooling_timescale = energy_result['cooling_timescale']
                print(f"Cooling timescale (tau): {cooling_timescale:.2f} µs")
            # Extract energy trace data
            if 'fit_times' in energy_result and 'fit_energies' in energy_result:
                energy_trace_data = {
                    'fit_times': energy_result['fit_times'],
                    'fit_energies': energy_result['fit_energies']
                }
                if 'fit_std_errs' in energy_result:
                    energy_trace_data['fit_std_errs'] = energy_result['fit_std_errs']
                if 'fit_episode_counts' in energy_result:
                    energy_trace_data['fit_episode_counts'] = energy_result['fit_episode_counts']
                print(f"Energy trace data collected with {len(energy_result['fit_times'])} time points")
    
    # Render selected episodes if requested
    if render_episodes is not None and len(render_episodes) > 0:
        print(f"\nRendering {len(render_episodes)} episodes...")
        for episode_idx in render_episodes:
            if 0 <= episode_idx < num_episodes:
                print(f"Rendering episode {episode_idx+1}")
                render_episode(
                    model_path, 
                    seed=None,  # Always use None for true randomness
                    deterministic=deterministic,
                    log_std_init=log_std_init,
                    probe_detuning_input=probe_detuning_input,
                    photon_number_input=photon_number_input,
                    top_trap_U0_max_input=top_trap_U0_max_input,
                    temperature_input=temperature_input,
                    custom_objects=custom_objects,
                    noisy_measurements=noisy_measurements
                )
            else:
                print(f"Episode index {episode_idx} out of range, skipping")
    
    # Final statistics
    trapped_atoms = sum(1 for r in results if r['trapped'])
    print("\nEvaluation completed.")
    print(f"Average reward: {np.mean(episode_rewards):.4f}, Average episode length: {np.mean(episode_lengths):.4f}")
    print(f"Episodes finished: {episodes_finished}/{num_episodes} ({episodes_finished/num_episodes:.2%})")
    print(f"Atoms trapped: {trapped_atoms}/{num_episodes} ({trapped_atoms/num_episodes:.2%})")
    
    if atom_temperature_final:
        print(f"Average final temperature: {np.mean(atom_temperature_final):.4f} µK (for {len(atom_temperature_final)} successfully completed episodes)")
    else:
        print("No episodes completed successfully")
    print(f"Average initial temperature: {(np.mean(atom_temperature_initial) if atom_temperature_initial else 0):.4f} µK")
    
    if cooling_timescale:
        print(f"Cooling timescale (tau): {cooling_timescale:.4f} µs")
    
    # Create visualizations if plots are enabled
    if do_plots:
        plt.close('all')
        
        # 1. Temperature histogram
        plt.figure(figsize=(10, 6))
        
        # Only proceed with histogram if we have data
        if atom_temperature_initial and atom_temperature_final:
            # Calculate logarithmic bins
            min_temp = min(min(atom_temperature_initial), min(atom_temperature_final))
            max_temp = max(max(atom_temperature_initial), max(atom_temperature_final))
            log_bins = np.logspace(np.log10(min_temp), np.log10(max_temp), 20)
            
            # Plot both histograms on the same axes
            plt.hist(atom_temperature_initial, bins=log_bins, alpha=0.6, color='red', label='Initial')
            plt.hist(atom_temperature_final, bins=log_bins, alpha=0.6, color='blue', label='Final')
            
            # Add mean lines
            plt.axvline(np.mean(atom_temperature_initial), color='red', linestyle='--',
                        label=f'Initial Mean: {np.mean(atom_temperature_initial):.1f} µK')
            plt.axvline(np.mean(atom_temperature_final), color='blue', linestyle='--',
                        label=f'Final Mean: {np.mean(atom_temperature_final):.1f} µK')
            
            plt.xscale('log')
            plt.xlabel('Temperature (µK)', fontsize=22)
            plt.ylabel('Count', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            model_type_str = "Trained"
            action_type_str = "Deterministic" if deterministic else "Stochastic"
            plt.title(f'Temperature Distribution ({model_type_str} Model, {action_type_str} Actions)', fontsize=22)
            plt.legend(fontsize=16)
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        # 2. Episode Steps Histogram
        plt.figure(figsize=(12, 6))
        
        # Create two separate histograms: one for successful episodes and one for truncated episodes
        successful_steps = [r['steps'] for r in results if r['trapped']]
        truncated_steps = [r['steps'] for r in results if not r['trapped']]
        
        # Determine appropriate bins
        max_steps = max(episode_lengths) if episode_lengths else 0
        min_steps = min(episode_lengths) if episode_lengths else 0
        bin_width = max(1, (max_steps - min_steps) // 25)  # Aim for about 25 bins
        bins = range(min_steps, max_steps + bin_width, bin_width)
        
        # Plot the histograms
        if truncated_steps:
            plt.hist(truncated_steps, bins=bins, alpha=0.7, color='orange', 
                     label=f'Truncated Episodes (n={len(truncated_steps)})')
        if successful_steps:
            plt.hist(successful_steps, bins=bins, alpha=0.7, color='green', 
                     label=f'Successfully Completed Episodes (n={len(successful_steps)})')
        
        # Add mean lines for each category
        if truncated_steps:
            plt.axvline(np.mean(truncated_steps), color='orange', linestyle='--',
                       label=f'Truncated Mean: {np.mean(truncated_steps):.1f} steps')
        if successful_steps:
            plt.axvline(np.mean(successful_steps), color='green', linestyle='--',
                       label=f'Successful Mean: {np.mean(successful_steps):.1f} steps')
        
        # Overall mean
        plt.axvline(np.mean(episode_lengths), color='black', linestyle='-',
                   label=f'Overall Mean: {np.mean(episode_lengths):.1f} steps')
        
        plt.xlabel('Episode Steps', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        model_type_str = "Trained"
        action_type_str = "Deterministic" if deterministic else "Stochastic"
        plt.title(f'Episode Length Distribution ({model_type_str} Model, {action_type_str} Actions)', fontsize=18)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Calculate speed improvement
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Evaluation completed in {total_time:.2f} seconds")
    print(f"Average time per episode: {total_time/num_episodes:.2f} seconds")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episodes_finished': episodes_finished,
        'atom_temperature_final': atom_temperature_final,
        'atom_temperature_initial': atom_temperature_initial,
        'total_time': total_time,
        'avg_trapped_steps': np.mean(episode_lengths),
        'avg_reward': np.mean(episode_rewards),
        'fraction_trapped': trapped_atoms / num_episodes,
        'probe_detuning': probe_detuning_input,
        'photon_number': photon_number_input,
        'top_trap_U0_max': top_trap_U0_max_input,
        'temperature': temperature_input,
        'cooling_timescale': cooling_timescale,
        'cooling_timescale_err': energy_result.get('cooling_timescale_err', None) if energy_result else None,
        'energy_trace_data': energy_trace_data
    }

def exponential_with_offset(t, E0, tau, offset):
    """
    Exponential decay function with offset
    E(t) = (E0 - offset) * exp(-t/tau) + offset
    
    Args:
        t: Time in seconds
        E0: Initial energy in µK
        tau: Time constant in seconds
        offset: Asymptotic energy value in µK
        
    Returns:
        Energy at time t in µK
    """
    return (E0 - offset) * np.exp(-t/tau) + offset

def analyze_energy_traces(results, truncate=True, do_plots=True):
    """
    Analyze and fit energy traces to extract cooling timescale
    
    Args:
        results: List of episode results containing time_points and z_energy_values
        truncate: Whether to truncate traces at threshold crossings
        do_plots: Whether to generate plots
        
    Returns:
        Dictionary with analysis results including cooling_timescale and energy trace data
    """
    # Filter results to only those with trace data (regardless of final trapping status)
    trace_results = [r for r in results if 'time_points' in r and 'z_energy_values' in r and len(r['time_points']) > 0]
    
    if not trace_results:
        print("No valid energy trace data available for analysis")
        return None
    
    print(f"Analyzing energy traces from {len(trace_results)} episodes with trace data")
    
    # Assuming all episodes have the same time steps (since they all start feedback at the same point)
    # First find the episode with the most time points to use as reference
    max_length_idx = np.argmax([len(r['time_points']) for r in trace_results])
    reference_times = np.array(trace_results[max_length_idx]['time_points'])
    
    # Skip the first time point where feedback just starts (potential zero values)
    if len(reference_times) > 1:
        # This ensures we start at the first real feedback action, not the initial step
        start_idx = 0
        reference_times = reference_times[start_idx:]
    else:
        start_idx = 0
    
    # Create arrays to hold summed energy values and counts for each time point
    total_time_points = len(reference_times)
    summed_energy = np.zeros(total_time_points)
    summed_energy_squared = np.zeros(total_time_points)  # For standard error calculation
    energy_counts = np.zeros(total_time_points)
    
    # Sum up energy values across all episodes for each corresponding time point
    for result in trace_results:
        if not result['time_points'] or not result['z_energy_values']:
            continue
            
        # Get time points and energy values for this episode
        episode_times = np.array(result['time_points'])
        episode_energies = np.array(result['z_energy_values'])
        
        # Skip the first point if needed (same as before)
        if len(episode_times) > 1:
            episode_times = episode_times[start_idx:]
            episode_energies = episode_energies[start_idx:]
        
        # Skip if no valid points after adjustment
        if len(episode_times) == 0:
            continue
            
        # For each time point in the reference, find if this episode has data at that time
        for ref_idx, ref_time in enumerate(reference_times):
            # Find the closest time point in this episode (should be exact match in most cases)
            time_diffs = np.abs(episode_times - ref_time)
            
            # Only include if we have a very close time match (within 1% of time step)
            time_step = reference_times[1] - reference_times[0] if len(reference_times) > 1 else 1e-6
            min_diff_idx = np.argmin(time_diffs)
            
            if time_diffs[min_diff_idx] <= 0.01 * time_step:  # Close enough time match
                energy_value = episode_energies[min_diff_idx]
                
                # Only include non-zero energy values
                if energy_value > 0:
                    summed_energy[ref_idx] += energy_value
                    summed_energy_squared[ref_idx] += energy_value ** 2
                    energy_counts[ref_idx] += 1
    
    # Calculate mean energy and standard error at each time point, avoiding division by zero
    mean_energy = np.zeros_like(summed_energy)
    std_err_energy = np.zeros_like(summed_energy)
    for i in range(total_time_points):
        if energy_counts[i] > 0:
            mean_energy[i] = summed_energy[i] / energy_counts[i]
            if energy_counts[i] > 1:  # Need at least 2 data points for std error
                variance = (summed_energy_squared[i] / energy_counts[i]) - (mean_energy[i] ** 2)
                variance = max(0, variance)  # Ensure variance is not negative due to numerical errors
                std_err_energy[i] = np.sqrt(variance / energy_counts[i])  # Standard error of the mean
            else:
                std_err_energy[i] = 0  # Cannot calculate standard error with only 1 data point
    
    # Filter out time points with no valid energy values (count = 0)
    valid_indices = energy_counts > 0
    fit_times = reference_times[valid_indices]
    fit_energies = mean_energy[valid_indices]
    fit_std_errs = std_err_energy[valid_indices]
    fit_episode_counts = energy_counts[valid_indices]  # Number of episodes at each time point
    
    # Ensure we have enough points for fitting
    if len(fit_times) < 3:
        print("Not enough valid data points for fitting")
        # Return the energy data even if we couldn't fit
        return {
            'fit_times': fit_times, 
            'fit_energies': fit_energies,
            'fit_std_errs': fit_std_errs,
            'fit_episode_counts': fit_episode_counts
        }
    
    # Skip initial points if they're zero
    first_nonzero_idx = 0
    while first_nonzero_idx < len(fit_energies) and fit_energies[first_nonzero_idx] <= 0:
        first_nonzero_idx += 1
    
    # Check if we have any non-zero points left
    if first_nonzero_idx >= len(fit_energies):
        print("No non-zero energy values found for fitting")
        return {
            'fit_times': fit_times, 
            'fit_energies': fit_energies,
            'fit_std_errs': fit_std_errs,
            'fit_episode_counts': fit_episode_counts
        }
    
    # Adjust time to start at zero for the first valid energy point
    orig_fit_times = fit_times.copy()  # Save original times before adjusting
    orig_fit_energies = fit_energies.copy()  # Save original energies
    orig_fit_std_errs = fit_std_errs.copy()  # Save original standard errors
    orig_fit_episode_counts = fit_episode_counts.copy()  # Save original episode counts
    
    fit_times = fit_times[first_nonzero_idx:] - fit_times[first_nonzero_idx]
    fit_energies = fit_energies[first_nonzero_idx:]
    fit_std_errs_for_fitting = fit_std_errs[first_nonzero_idx:]
    
    # Initial guesses for fitting parameters
    initial_energy = fit_energies[0]
    final_energy = fit_energies[-1]
    # Initial guess for tau (cooling timescale) - assume it's about 1/8 of the total time
    tau_guess = (fit_times[-1] - fit_times[0]) / 8
    
    # Perform the fit
    try:
        from scipy.optimize import curve_fit
        
        # Initial parameter guesses [E0, tau, offset]
        p0 = [initial_energy, tau_guess, final_energy]
        # Basic bounds: all parameters positive
        bounds = ([0, -2000e-6, 0], [5e3, 2000e-6, 5e3])
        
        popt, pcov = curve_fit(
            exponential_with_offset, 
            fit_times, 
            fit_energies, 
            p0=p0,
            bounds=bounds
        )
        
        # Extract fitted parameters
        E0, tau, offset = popt
        
        # Calculate errors
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate fit quality metrics
        y_fit = exponential_with_offset(fit_times, *popt)
        residuals = fit_energies - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((fit_energies - np.mean(fit_energies))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Convert tau from seconds to microseconds for display
        tau_us = tau * 1e6
        tau_err_us = perr[1] * 1e6
        
        if do_plots:
            # Plot the mean energy trace and fit
            plt.figure(figsize=(12, 8))
            
            # Convert time from seconds to microseconds for plotting
            fit_times_us = fit_times * 1e6
            
            # Plot original data
            plt.scatter(fit_times_us, fit_energies, label='Mean Energy', color='blue', alpha=0.7, s=15)
            
            # Plot fitted curve
            t_fine = np.linspace(min(fit_times), max(fit_times), 1000)
            t_fine_us = t_fine * 1e6  # Convert to microseconds for plotting
            e_fit = exponential_with_offset(t_fine, *popt)
            plt.plot(t_fine_us, e_fit, 'r-', linewidth=2, 
                    label=f'Fit: E(t) = {E0:.1f}·exp(-t/{tau_us:.1f}µs) + {offset:.1f} (R² = {r_squared:.4f})')
            
            plt.xlabel('Time (µs)', fontsize=14)
            plt.ylabel('Energy (µK)', fontsize=14)
            plt.title('Energy vs. Time with Exponential Fit', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # Print fit results
        print(f"Exponential fit results:")
        print(f"E0 = {E0:.4f} ± {perr[0]:.4f} µK")
        print(f"tau = {tau_us:.4f} ± {tau_err_us:.4f} µs")
        print(f"offset = {offset:.4f} ± {perr[2]:.4f} µK")
        print(f"R-squared = {r_squared:.6f}")
        
        return {
            'cooling_timescale': tau_us,  # in microseconds
            'cooling_timescale_err': tau_err_us,  # error in microseconds
            'E0': E0,  # in µK
            'offset': offset,  # in µK
            'r_squared': r_squared,
            'fit_times': orig_fit_times,  # Original time points in seconds
            'fit_energies': orig_fit_energies,  # Original energy values in µK
            'fit_std_errs': orig_fit_std_errs,
            'fit_episode_counts': orig_fit_episode_counts
        }
        
    except Exception as e:
        print(f"Energy trace fitting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'fit_times': orig_fit_times,
            'fit_energies': orig_fit_energies,
            'fit_std_errs': orig_fit_std_errs,
            'fit_episode_counts': orig_fit_episode_counts,
            'fit_error': str(e)
        }

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Evaluate RL models for cavity cooling with improved visualization')
    parser.add_argument('--model', type=str, default='mlp_sim',
                        choices=['mlp_sim', 'mlp_experimental', 'differentiator'],
                        help='Model to evaluate (default: mlp_sim)')
    parser.add_argument('--episodes', type=int, default=400, help='Number of episodes to evaluate (default: 400)')
    parser.add_argument('--render', type=int, default=0, help='Number of random episodes to render (default: 0, disabled)')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker processes (default: 10)')
    parser.add_argument('--no-truncate', action='store_true', help='Disable truncation of traces at threshold crossings')
    parser.add_argument('--deterministic', type=lambda x: str(x).lower() == 'true', default=True, 
                        help='Use deterministic actions (default: True)')
    parser.add_argument('--probe-detuning', type=float, default=None,
                        help='Probe detuning in MHz (default: use environment default)')
    parser.add_argument('--photon-number', type=float, default=None,
                        help='Photon number (default: use environment default)')
    parser.add_argument('--top-trap-u0-max', type=float, default=None,
                        help='Top trap U0 max in MHz, converted to Hz internally (default: use environment default)')
    parser.add_argument('--noisy-measurements', action='store_true', default=False,
                        help='Enable noisy measurements to simulate experimental conditions')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature in µK (default: use environment default)')
    parser.add_argument('--plots', action='store_true', default=False,
                        help='Show interactive diagnostic plots (default: disabled)')
    
    args = parser.parse_args()
    

    # Available models to evaluate - same paths as single-process script
    models = {
        'mlp_sim': str(MODELS_DIR / 'mlp_sim.zip'),
        'mlp_experimental': str(MODELS_DIR / 'mlp_experimental.zip'),
        'differentiator': str(MODELS_DIR / 'differentiator.zip'),
    }
    
    # Select which model to evaluate
    model_key = args.model
    model_path = models[model_key]
    
    # Set up number of episodes and renders
    num_episodes = args.episodes
    num_renders = args.render
    render_episodes = random.sample(range(num_episodes), min(num_renders, num_episodes)) if num_renders > 0 else None
    
    # Convert probe detuning from MHz to rad/s if specified
    probe_detuning_input = None
    if args.probe_detuning is not None:
        probe_detuning_input = 2 * np.pi * args.probe_detuning * 1e6
        print(f"Using specified probe detuning: {args.probe_detuning} MHz")
    else:
        print("Using environment default probe detuning")
    
    # Set photon number if specified
    photon_number_input = args.photon_number
    if photon_number_input is not None:
        print(f"Using specified photon number: {photon_number_input}")
    else:
        print("Using environment default photon number")
    
    # Set top trap U0 max if specified
    top_trap_U0_max_input = args.top_trap_u0_max
    if top_trap_U0_max_input is not None:
        # Convert from MHz to Hz
        top_trap_U0_max_input = top_trap_U0_max_input * 1e6
        print(f"Using specified top trap U0 max: {args.top_trap_u0_max} MHz ({top_trap_U0_max_input:.1e} Hz)")
    else:
        print("Using environment default top trap U0 max")
    
    # Set temperature_input if specified via args.temperature
    temperature_input = None
    if hasattr(args, 'temperature') and args.temperature is not None:
        # Convert from µK to K
        temperature_input = args.temperature * 1e-6
        print(f"Using specified temperature: {args.temperature} µK ({temperature_input:.2e} K)")
    else:
        print("Using environment default temperature")
    
    # Print evaluation settings

    print(f"Evaluating paper model: {model_key} from path: {model_path}")
    
    print(f"Action selection: {'Deterministic' if args.deterministic else 'Stochastic'}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Noisy measurements: {args.noisy_measurements}")
    if render_episodes:
        print(f"Will render {len(render_episodes)} episodes: {render_episodes}")
    
    # Run evaluation with parallelization
    results = evaluate_model_parallel(
        model_path=model_path,
        num_episodes=num_episodes,
        num_workers=args.workers,
        render_episodes=render_episodes,
        trace_analysis=True,
        truncate_traces=not args.no_truncate,
        deterministic=args.deterministic,
        probe_detuning_input=probe_detuning_input,
        photon_number_input=photon_number_input,
        top_trap_U0_max_input=top_trap_U0_max_input,
        temperature_input=temperature_input,
        noisy_measurements=args.noisy_measurements,
        do_plots=args.plots
    )
