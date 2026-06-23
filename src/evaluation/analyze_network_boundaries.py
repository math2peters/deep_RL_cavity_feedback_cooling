import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import time
import warnings
from pathlib import Path

# Filter out the specific warning about lr_schedule deserialization
warnings.filterwarnings("ignore", message="Could not deserialize object lr_schedule")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
FIG4_DATA_DIR = REPO_ROOT / "data" / "source_data_fig4"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the evaluate_model_parallel function from evaluate_multiprocess
from src.evaluation.evaluate_multiprocess import evaluate_model_parallel


def resolve_model_path(model_type, model_name):
    """Resolve a paper model checkpoint from the local models directory."""
    model_slug = make_model_slug(model_type, model_name)
    model_name_path = Path(model_name)
    if model_name_path.is_absolute() or model_name_path.exists():
        return str(model_name_path)

    candidates = [
        MODELS_DIR / f"{model_slug}.zip",
        MODELS_DIR / model_name,
        MODELS_DIR / f"{model_name}.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def make_model_slug(model_type, model_name):
    if model_type == 'baseline':
        return model_name
    return f"{model_type}_{model_name}"

def sweep_parameter(model_path, param_name, param_range, num_episodes=100, num_workers=10, model_type='mlp', noisy_measurements=False):
    """
    Sweep through different parameter values and collect performance metrics
    
    Args:
        model_path: Path to the RL model to evaluate
        param_name: Name of the parameter to sweep ('detuning', 'photon_number', or 'temperature')
        param_range: List of parameter values to sweep through
        num_episodes: Number of episodes to evaluate for each parameter value
        num_workers: Number of parallel workers to use
        model_type: Type of paper model family ('mlp' or 'baseline')
        noisy_measurements: Whether to add noise to measurements to simulate experimental conditions
        
    Returns:
        Dictionary containing results for each parameter value
    """
    results = {}
    
    # Print sweep parameters
    print(f"Starting {param_name} sweep from {min(param_range)} to {max(param_range)}")
    print(f"Evaluating {num_episodes} episodes per value using {num_workers} workers")
    print(f"Using model from {model_path}, model type: {model_type}")
    print(f"Noisy measurements: {noisy_measurements}")
    
    start_time = time.time()
    
    # Setup parameter inputs for evaluation
    for i, param_value in enumerate(param_range):
        print(f"\n{'='*80}")
        print(f"Evaluating {param_name} = {param_value} ({i+1}/{len(param_range)})")
        print(f"{'='*80}")
        
        # Setup parameter inputs
        probe_detuning_input = None
        photon_number_input = None
        temperature_input = None
        
        if param_name == 'detuning':
            # Convert detuning from MHz to rad/s
            probe_detuning_input = 2 * np.pi * param_value * 1e6
        elif param_name == 'photon_number':
            photon_number_input = param_value
        elif param_name == 'temperature':
            # Store temperature in K (convert from µK)
            temperature_input = param_value * 1e-6
        
        # Run evaluation for this parameter value without generating plots
        result = evaluate_model_parallel(
            model_path=model_path,
            num_episodes=num_episodes,
            num_workers=num_workers,
            render_episodes=None,
            trace_analysis=True,  # Enable trace analysis to get cooling timescale
            truncate_traces=True,
            deterministic=True,
            probe_detuning_input=probe_detuning_input,
            photon_number_input=photon_number_input,
            temperature_input=temperature_input,
            do_plots=False,  # Disable plots for the sweep
            noisy_measurements=noisy_measurements  # Pass noisy_measurements parameter
        )
        
        # Calculate average final temperature (only for trapped atoms)
        avg_final_temp = np.mean(result['atom_temperature_final']) if result['atom_temperature_final'] else None
        
        # Get cooling timescale from the results
        cooling_timescale = result.get('cooling_timescale', None)
        # Value is already in microseconds, no need to convert
        if cooling_timescale is None:
            print(f"Warning: No cooling timescale data available for {param_name}={param_value}")
            print("This can happen when there are insufficient valid energy traces for fitting.")
            print("Ensure that trapped atoms are properly cooling and energy values are non-zero.")
        
        # Get cooling timescale error if available
        cooling_timescale_err = result.get('cooling_timescale_err', None)
        
        # Calculate standard errors for each metric
        std_trapped_steps = np.std(result['episode_lengths'])
        se_trapped_steps = std_trapped_steps / np.sqrt(num_episodes)
        
        std_reward = np.std(result['episode_rewards'])
        se_reward = std_reward / np.sqrt(num_episodes)
        
        # Calculate standard error for fraction trapped
        # Using binomial standard error for proportion: sqrt(p*(1-p)/n)
        fraction_trapped = result['fraction_trapped']
        se_fraction_trapped = np.sqrt(fraction_trapped * (1 - fraction_trapped) / num_episodes)
        
        # Calculate standard error for final temperature (if available)
        if result['atom_temperature_final']:
            std_final_temp = np.std(result['atom_temperature_final'])
            se_final_temp = std_final_temp / np.sqrt(len(result['atom_temperature_final']))
        else:
            se_final_temp = None
        
        # Store the results for this parameter value
        results[param_value] = {
            'avg_trapped_steps': result['avg_trapped_steps'],
            'std_trapped_steps': std_trapped_steps,
            'se_trapped_steps': se_trapped_steps,
            'avg_reward': result['avg_reward'],
            'std_reward': std_reward,
            'se_reward': se_reward,
            'fraction_trapped': result['fraction_trapped'],  # Now represents fraction of episodes successfully completed
            'se_fraction_trapped': se_fraction_trapped,
            'avg_final_temp': avg_final_temp,
            'se_final_temp': se_final_temp,
            'cooling_timescale': cooling_timescale,  # Already in µs
            'cooling_timescale_err': cooling_timescale_err,  # Error in the cooling timescale
            'param_name': param_name,
            'param_value': param_value
        }
        
        # Store energy trace data if available
        if 'energy_trace_data' in result and result['energy_trace_data'] is not None:
            # Store the times in microseconds and energies in µK
            fit_times = result['energy_trace_data']['fit_times']
            fit_energies = result['energy_trace_data']['fit_energies']
            fit_std_errs = result['energy_trace_data'].get('fit_std_errs', None)
            fit_episode_counts = result['energy_trace_data'].get('fit_episode_counts', None)
            
            # Convert time from seconds to microseconds for storage
            fit_times_us = fit_times * 1e6
            
            results[param_value]['energy_trace_times_us'] = fit_times_us
            results[param_value]['energy_trace_energies_uk'] = fit_energies
            if fit_std_errs is not None:
                results[param_value]['energy_trace_std_errs_uk'] = fit_std_errs
            if fit_episode_counts is not None:
                results[param_value]['energy_trace_episode_counts'] = fit_episode_counts
            print(f"Stored energy trace data with {len(fit_times)} time points")
        else:
            print("No energy trace data available to store")
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Completed {i+1}/{len(param_range)} values in {elapsed:.1f} seconds")
        estimated_total = elapsed / (i+1) * len(param_range)
        print(f"Estimated total time: {estimated_total:.1f} seconds, remaining: {estimated_total - elapsed:.1f} seconds")
        
        # Print cooling timescale if available
        if cooling_timescale is not None:
            if cooling_timescale_err is not None:
                print(f"Cooling timescale (τ): {cooling_timescale:.2f} ± {cooling_timescale_err:.2f} µs")
            else:
                print(f"Cooling timescale (τ): {cooling_timescale:.2f} µs")
            print(f"This represents the exponential decay time constant of energy after fitting.")
    
    return results

def plot_sweep_results(results, output_dir=None, model_type='mlp', model_filename='sim', noisy_measurements=False):
    """
    Plot the results of the parameter sweep
    
    Args:
        results: Dictionary containing results for each parameter value
        output_dir: Directory to save plots and CSV files
        model_type: Type of paper model family ('mlp' or 'baseline')
        model_filename: The filename of the model
        noisy_measurements: Whether noisy measurements were used
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = FIG4_DATA_DIR / make_model_slug(model_type, model_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the parameter name from the first result
    param_values = sorted(results.keys())
    param_name = results[param_values[0]]['param_name']
    
    # Format the parameter name for axis labels
    if param_name == 'detuning':
        param_label = 'Probe Detuning (MHz)'
    elif param_name == 'photon_number':
        param_label = 'Photon Number'
    elif param_name == 'temperature':
        param_label = 'Initial Temperature (µK)'
    else:
        param_label = param_name.replace('_', ' ').title()
    
    # Extract metrics
    avg_trapped_steps = [results[p]['avg_trapped_steps'] for p in param_values]
    se_trapped_steps = [results[p]['se_trapped_steps'] for p in param_values]
    
    avg_rewards = [results[p]['avg_reward'] for p in param_values]
    se_rewards = [results[p]['se_reward'] for p in param_values]
    
    fraction_trapped = [results[p]['fraction_trapped'] for p in param_values]
    se_fraction_trapped = [results[p]['se_fraction_trapped'] for p in param_values]
    
    avg_final_temps = [results[p]['avg_final_temp'] for p in param_values]
    se_final_temps = [results[p]['se_final_temp'] for p in param_values]
    
    cooling_timescales = [results[p]['cooling_timescale'] for p in param_values]
    cooling_timescale_errors = [results[p].get('cooling_timescale_err', None) for p in param_values]
    
    # Create figure with 5 subplots, each with its own x-axis
    fig, axes = plt.subplots(5, 1, figsize=(12, 25), sharex=False)
    
    # Function to set up finer grid for each subplot
    def set_finer_grid(ax):
        # Major grid
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
        # Minor grid
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        # Set background grid color
        ax.set_axisbelow(True)
    
    # Plot 1: Average trapped steps with error bars
    axes[0].errorbar(param_values, avg_trapped_steps, yerr=se_trapped_steps, 
                    fmt='o-', color='blue', linewidth=2, markersize=8, 
                    capsize=5, ecolor='blue', elinewidth=1, alpha=0.8)
    axes[0].set_xlabel(param_label, fontsize=14)
    axes[0].set_ylabel('Average Episode Length (steps)', fontsize=14)
    axes[0].set_title(f'Effect of {param_label} on Average Episode Length', fontsize=16)
    axes[0].set_ylim(bottom=0)  # Make y-axis start at 0
    set_finer_grid(axes[0])
    
    # Plot 2: Average reward with error bars
    axes[1].errorbar(param_values, avg_rewards, yerr=se_rewards, 
                    fmt='o-', color='green', linewidth=2, markersize=8, 
                    capsize=5, ecolor='green', elinewidth=1, alpha=0.8)
    axes[1].set_xlabel(param_label, fontsize=14)
    axes[1].set_ylabel('Average Reward', fontsize=14)
    axes[1].set_title(f'Effect of {param_label} on Average Reward', fontsize=16)
    axes[1].set_ylim(bottom=0)  # Make y-axis start at 0
    set_finer_grid(axes[1])
    
    # Plot 3: Fraction of atoms trapped at the end with error bars
    axes[2].errorbar(param_values, fraction_trapped, yerr=se_fraction_trapped, 
                    fmt='o-', color='red', linewidth=2, markersize=8, 
                    capsize=5, ecolor='red', elinewidth=1, alpha=0.8)
    axes[2].set_xlabel(param_label, fontsize=14)
    axes[2].set_ylabel('Fraction Successfully Completed', fontsize=14)
    axes[2].set_title(f'Effect of {param_label} on Fraction of Successfully Completed Episodes', fontsize=16)
    axes[2].set_ylim(bottom=0)  # Make y-axis start at 0
    set_finer_grid(axes[2])
    
    # Plot 4: Average final temperature (for trapped atoms) with error bars
    valid_temps = [(p, temp, err) for p, temp, err in zip(param_values, avg_final_temps, se_final_temps) if temp is not None]
    if valid_temps:
        valid_params, valid_temperatures, valid_temp_errors = zip(*valid_temps)
        axes[3].errorbar(valid_params, valid_temperatures, yerr=valid_temp_errors, 
                        fmt='o-', color='purple', linewidth=2, markersize=8, 
                        capsize=5, ecolor='purple', elinewidth=1, alpha=0.8)
    axes[3].set_xlabel(param_label, fontsize=14)
    axes[3].set_ylabel('Average Final Temperature (μK)', fontsize=14)
    axes[3].set_title(f'Effect of {param_label} on Average Final Temperature', fontsize=16)
    axes[3].set_ylim(bottom=0)  # Make y-axis start at 0
    set_finer_grid(axes[3])
    
    # Plot 5: Cooling timescale (tau) with error bars
    valid_taus = [(p, tau, err) for p, tau, err in zip(param_values, cooling_timescales, cooling_timescale_errors) if tau is not None]
    if valid_taus:
        valid_params, valid_tau_values, valid_tau_errors = zip(*valid_taus)
        axes[4].errorbar(valid_params, valid_tau_values, yerr=valid_tau_errors,
                       fmt='o-', color='orange', linewidth=2, markersize=8, 
                       capsize=5, ecolor='orange', elinewidth=1, alpha=0.8)
    axes[4].set_xlabel(param_label, fontsize=14)
    axes[4].set_ylabel('Cooling Timescale τ (μs)', fontsize=14)
    axes[4].set_title(f'Effect of {param_label} on Cooling Timescale', fontsize=16)
    axes[4].set_ylim(bottom=0)  # Make y-axis start at 0
    set_finer_grid(axes[4])
    
    # Common settings for all plots
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
    
    plt.tight_layout()
    
    # Format model type for filename
    model_type_str = "MLP" if model_type == 'mlp' else "Baseline"
    
    # Add noise indicator to filename
    noise_suffix = "_noisy" if noisy_measurements else ""
    
    # Save the figure with model type and parameter name in filename
    plt.savefig(os.path.join(output_dir, f'{param_name}_sweep_results_{model_type}_{model_filename}{noise_suffix}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{param_name}_sweep_results_{model_type}_{model_filename}{noise_suffix}.pdf'))
    
    # Also save the raw data as CSV
    import csv
    with open(os.path.join(output_dir, f'{param_name}_sweep_results_{model_type}_{model_filename}{noise_suffix}.csv'), 'w', newline='') as csvfile:
        fieldnames = [
            f'{param_name}', 
            'avg_trapped_steps', 'se_trapped_steps',
            'avg_reward', 'se_reward',
            'fraction_completed', 'se_fraction_completed',
            'avg_final_temp', 'se_final_temp',
            'cooling_timescale', 'cooling_timescale_err'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for p in param_values:
            writer.writerow({
                f'{param_name}': p,
                'avg_trapped_steps': results[p]['avg_trapped_steps'],
                'se_trapped_steps': results[p]['se_trapped_steps'],
                'avg_reward': results[p]['avg_reward'],
                'se_reward': results[p]['se_reward'],
                'fraction_completed': results[p]['fraction_trapped'],  # Keep using fraction_trapped internally for backward compatibility
                'se_fraction_completed': results[p]['se_fraction_trapped'],
                'avg_final_temp': results[p]['avg_final_temp'],
                'se_final_temp': results[p]['se_final_temp'],
                'cooling_timescale': results[p]['cooling_timescale'],
                'cooling_timescale_err': results[p].get('cooling_timescale_err', None)
            })
    
    # Create a directory for energy trace data
    energy_traces_dir = os.path.join(output_dir, "energy_traces")
    os.makedirs(energy_traces_dir, exist_ok=True)
    
    # Save energy trace data if available
    for p in param_values:
        if 'energy_trace_times_us' in results[p] and 'energy_trace_energies_uk' in results[p]:
            times = results[p]['energy_trace_times_us']
            energies = results[p]['energy_trace_energies_uk']
            std_errs = results[p].get('energy_trace_std_errs_uk', None)
            episode_counts = results[p].get('energy_trace_episode_counts', None)
            
            # Adjust time to start from 0
            if len(times) > 0:
                times_adjusted = times - times[0]
            else:
                times_adjusted = times
            
            # Save as NumPy file for easier loading in future analysis
            energy_filename = f'{param_name}_{p}_energy_trace_{model_type}_{model_filename}{noise_suffix}.npz'
            npz_data = {
                'times_us': times_adjusted,
                'energies_uk': energies,
                'param_name': param_name,
                'param_value': p,
                'model_type': model_type,
                'model_filename': model_filename,
                'cooling_timescale': results[p]['cooling_timescale'],
                'cooling_timescale_err': results[p].get('cooling_timescale_err', None)
            }
            if std_errs is not None:
                npz_data['std_errs_uk'] = std_errs
            if episode_counts is not None:
                npz_data['episode_counts'] = episode_counts
            np.savez(os.path.join(energy_traces_dir, energy_filename), **npz_data)
            
            # Also save as CSV for easier manual inspection
            energy_csv_filename = f'{param_name}_{p}_energy_trace_{model_type}_{model_filename}{noise_suffix}.csv'
            with open(os.path.join(energy_traces_dir, energy_csv_filename), 'w', newline='') as csvfile:
                # Determine fieldnames based on available data
                fieldnames = ['time_us', 'energy_uk']
                if std_errs is not None:
                    fieldnames.append('energy_uk_std_err')
                if episode_counts is not None:
                    fieldnames.append('episode_count')
                    
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i, (t, e) in enumerate(zip(times_adjusted, energies)):
                    row_data = {
                        'time_us': t,
                        'energy_uk': e
                    }
                    if std_errs is not None and i < len(std_errs):
                        row_data['energy_uk_std_err'] = std_errs[i]
                    if episode_counts is not None and i < len(episode_counts):
                        row_data['episode_count'] = int(episode_counts[i])  # Convert to int for cleaner CSV
                    writer.writerow(row_data)
    
    # Save a readme file explaining the energy trace data format
    with open(os.path.join(energy_traces_dir, 'README.txt'), 'w') as f:
        f.write(f"Energy trace data for {param_name} sweep with {model_type}_{model_filename}\n")
        f.write("=================================================================\n\n")
        f.write("Each parameter value has its own .npz and .csv file containing:\n")
        f.write("- times_us: Time points in microseconds (starting from 0)\n")
        f.write("- energies_uk: Energy values in microkelvin (mean across episodes)\n")
        f.write("- std_errs_uk: Standard error of the mean energy in microkelvin\n")
        f.write("- episode_counts: Number of episodes contributing to each time point\n\n")
        f.write("CSV files include headers: time_us, energy_uk, energy_uk_std_err, episode_count\n\n")
        f.write("These files can be used for custom fitting if the automatic fit failed.\n")
        f.write("Example usage for loading data from a .npz file:\n\n")
        f.write("import numpy as np\n")
        f.write("data = np.load('filename.npz')\n")
        f.write("times = data['times_us']\n")
        f.write("energies = data['energies_uk']\n")
        f.write("std_errs = data['std_errs_uk']  # if available\n")
        f.write("episode_counts = data['episode_counts']  # if available\n\n")
        f.write("For fitting, you can use scipy.optimize.curve_fit with a custom function:\n\n")
        f.write("from scipy.optimize import curve_fit\n")
        f.write("def exponential_with_offset(t, E0, tau, offset):\n")
        f.write("    return (E0 - offset) * np.exp(-t/tau) + offset\n\n")
        f.write("# Convert times from microseconds to seconds for fitting\n")
        f.write("times_s = times / 1e6\n")
        f.write("# Initial guesses for E0, tau (in seconds), and offset\n")
        f.write("p0 = [energies[0], (times_s[-1] - times_s[0])/8, energies[-1]]\n")
        f.write("popt, pcov = curve_fit(exponential_with_offset, times_s, energies, p0=p0)\n")
        f.write("# Convert tau back to microseconds\n")
        f.write("tau_us = popt[1] * 1e6\n")
    
    noise_text = "with noisy measurements" if noisy_measurements else "without noisy measurements"
    print(f"Plots and data saved to {output_dir}/ with parameter {param_name}, model type {model_type}_{model_filename} {noise_text}")
    print(f"Energy trace data saved to {energy_traces_dir}/ for each parameter value")
    return fig

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze RL model performance across different parameter values')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'baseline'],
                        help='Which policy family to use (default: mlp)')
    parser.add_argument('--model-name', type=str, default='sim',
                        choices=['sim', 'experimental', 'differentiator'],
                        help='Which paper model to use (default: sim)')
    parser.add_argument('--parameter', type=str, default='photon_number', 
                        choices=['detuning', 'photon_number', 'temperature'],
                        help='Which parameter to sweep (default: photon_number)')
    parser.add_argument('--min-value', type=float, default=None,
                        help='Minimum parameter value')
    parser.add_argument('--max-value', type=float, default=None,
                        help='Maximum parameter value')
    parser.add_argument('--step', type=float, default=None,
                        help='Step size for parameter sweep')
    parser.add_argument('--episodes', type=int, default=4000,
                        help='Number of episodes to evaluate per parameter value (default: 4000)')
    parser.add_argument('--workers', type=int, default=20,
                        help='Number of worker processes to use (default: 20)')
    parser.add_argument('--noisy-measurements', action='store_true', default=False,
                        help='Enable noisy measurements to simulate experimental conditions')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Directory to write sweep CSVs and plots. Defaults to data/source_data_fig4/<model_slug>/')
    
    args = parser.parse_args()
    
    # Set model path based on argument
    model_path = resolve_model_path(args.model, args.model_name)
    
    # Set default values based on parameter type
    if args.parameter == 'detuning':
        min_default = -105.0
        max_default = 105.0
        step_default = 10.0
        param_name = 'detuning'
    elif args.parameter == 'photon_number':
        min_default = 1
        max_default = 81
        step_default = 4
        param_name = 'photon_number'
    elif args.parameter == 'temperature':
        min_default = 0  # In µK (100 µK = 100e-6 K)
        max_default = 6000  # In µK (900 µK = 900e-6 K)
        step_default = 400  # In µK
        param_name = 'temperature'
    
    # Use provided values or defaults
    min_value = args.min_value if args.min_value is not None else min_default
    max_value = args.max_value if args.max_value is not None else max_default
    step_size = args.step if args.step is not None else step_default
    
    # Generate parameter range
    param_range = np.arange(min_value, max_value + step_size/2, step_size)
    
    print(f"Sweeping {param_name} from {min_value} to {max_value} in steps of {step_size}")
    print(f"Noisy measurements: {args.noisy_measurements}")
    
    # Run the sweep
    results = sweep_parameter(
        model_path=model_path,
        param_name=param_name,
        param_range=param_range,
        num_episodes=args.episodes,
        num_workers=args.workers,
        model_type=args.model,
        noisy_measurements=args.noisy_measurements
    )
    
    # Create the output directory with model and model_name as subfolder
    output_dir = args.output_dir if args.output_dir is not None else FIG4_DATA_DIR / make_model_slug(args.model, args.model_name)
    
    # Plot results
    plot_sweep_results(results, output_dir=output_dir, model_type=args.model, 
                       model_filename=args.model_name, noisy_measurements=args.noisy_measurements)
    
if __name__ == "__main__":
    main()
