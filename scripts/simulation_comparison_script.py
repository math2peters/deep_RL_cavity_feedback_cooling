#!/usr/bin/env python3
"""
Script to create comparison plots between different model types across parameter sweeps
Compares MLP experiment, MLP simulation, and differentiator models
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd
import argparse
from pathlib import Path
from scipy.optimize import curve_fit
import warnings


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIG4_SOURCE_DIR = REPO_ROOT / 'data' / 'source_data_fig4'
DEFAULT_COMPARISON_OUTPUT_DIR = REPO_ROOT / 'Figure 4' / 'Simulation_Comparison'
DEFAULT_FIT_VERIFICATION_FILE = REPO_ROOT / 'outputs' / 'cooling_timescale_fit_verification.pdf'


MODEL_FILE_ALIASES = {
    'differentiator': ('differentiator', 'baseline_differentiator'),
    'baseline_differentiator': ('baseline_differentiator', 'differentiator'),
}


def get_model_file_aliases(model_folder):
    aliases = list(MODEL_FILE_ALIASES.get(model_folder, (model_folder,)))
    if model_folder not in aliases:
        aliases.insert(0, model_folder)
    return aliases


def alias_rank(model_suffix, model_folder):
    aliases = get_model_file_aliases(model_folder)
    if model_suffix in aliases:
        return aliases.index(model_suffix)
    return len(aliases)


def find_sweep_result_file(folder_path, param_name, model_folder):
    aliases = get_model_file_aliases(model_folder)
    for model_suffix in aliases:
        candidate = os.path.join(folder_path, f"{param_name}_sweep_results_{model_suffix}.csv")
        if os.path.exists(candidate):
            return candidate

    try:
        matching_files = [
            f for f in os.listdir(folder_path)
            if f.startswith(f"{param_name}_sweep_results_")
            and f.endswith('.csv')
            and "_noisy" not in f
        ]
    except OSError:
        return None

    if not matching_files:
        return None

    def sort_key(filename):
        model_suffix = filename.rsplit(f"{param_name}_sweep_results_", 1)[1].rsplit('.csv', 1)[0]
        return (alias_rank(model_suffix, model_folder), filename)

    return os.path.join(folder_path, sorted(matching_files, key=sort_key)[0])


def parse_energy_trace_filename(filename, param_name):
    prefix = f"{param_name}_"
    token = "_energy_trace_"
    suffix = ".csv"
    if not (filename.startswith(prefix) and filename.endswith(suffix) and token in filename):
        return None

    try:
        value_text, model_suffix = filename[len(prefix):-len(suffix)].split(token, 1)
        return float(value_text), model_suffix
    except ValueError:
        return None


def find_energy_trace_files(energy_traces_path, param_name, model_folder):
    candidates_by_value = {}
    try:
        filenames = os.listdir(energy_traces_path)
    except OSError:
        return []

    for filename in filenames:
        parsed = parse_energy_trace_filename(filename, param_name)
        if parsed is None:
            continue
        param_value, model_suffix = parsed
        candidates_by_value.setdefault(param_value, []).append((model_suffix, filename))

    selected_files = []
    for param_value, candidates in sorted(candidates_by_value.items()):
        model_suffix, filename = sorted(
            candidates,
            key=lambda item: (alias_rank(item[0], model_folder), item[1])
        )[0]
        selected_files.append((param_value, filename))

    return selected_files


def find_energy_trace_file(energy_traces_path, param_name, param_value, model_folder):
    for trace_value, filename in find_energy_trace_files(energy_traces_path, param_name, model_folder):
        if np.isclose(trace_value, float(param_value)):
            return os.path.join(energy_traces_path, filename)
    return None


def load_sweep_results(base_directory, model_folders, param_name='photon_number'):
    """
    Load sweep results for multiple models from CSV files in specified folders
    
    Args:
        base_directory: Base directory containing model subfolders
        model_folders: List of model folders to load (format: "{model_type}_{model_name}")
        param_name: Name of the parameter that was swept
        
    Returns:
        Dictionary of DataFrames containing results for each model
    """
    results = {}
    
    for folder in model_folders:
        folder_path = os.path.join(base_directory, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            continue
            
        csv_path = find_sweep_result_file(folder_path, param_name, folder)
        if csv_path is None:
            print(f"Warning: Could not find data for {folder} in {folder_path}")
            continue
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                if 'fraction_completed' in df.columns and 'fraction_trapped' not in df.columns:
                    df['fraction_trapped'] = df['fraction_completed']
                    print(f"Note: Using 'fraction_completed' field for compatibility with old 'fraction_trapped' field")
                
                if 'se_trapped_steps' not in df.columns:
                    print(f"Note: No standard error columns found in {folder} data")
                    df['se_trapped_steps'] = 0
                    df['se_reward'] = 0
                    df['se_fraction_trapped'] = 0
                    df['se_final_temp'] = 0
                    df['se_mean_ke_z'] = 0
                    
                if 'se_fraction_completed' in df.columns and 'se_fraction_trapped' not in df.columns:
                    df['se_fraction_trapped'] = df['se_fraction_completed']
                    
                if 'avg_mean_ke_z' not in df.columns:
                    print(f"Note: No avg_mean_ke_z field found in {folder} data, adding NaN values")
                    df['avg_mean_ke_z'] = np.nan
                    df['se_mean_ke_z'] = 0
                
                if 'legacy_cooling_timescale' not in df.columns and 'cooling_timescale' in df.columns:
                    df = df.rename(columns={
                        'cooling_timescale': 'legacy_cooling_timescale',
                        'cooling_timescale_err': 'legacy_cooling_timescale_err',
                    })
                    print("Note: Renamed source cooling_timescale fields to legacy_cooling_timescale; final fits are recomputed from energy traces")
                
                if 'legacy_cooling_timescale_err' not in df.columns:
                    df['legacy_cooling_timescale_err'] = 0
                
                df['cooling_timescale'] = np.nan
                df['cooling_timescale_err'] = np.nan
                    
                results[folder] = df
                print(f"Loaded data for {folder}")
            except Exception as e:
                print(f"Error loading CSV file {csv_path}: {e}")
        else:
            print(f"Warning: Could not find data for {folder} at {csv_path}")
    
    return results


def get_model_styling(model_folder):
    """
    Get consistent styling for different model types
    
    Args:
        model_folder: Folder name of the model
        
    Returns:
        Dictionary with styling parameters
    """
    # Color scheme consistent with detunings_plot_script.py
    global sim_color, sim_rob_color, exp_color, diff_color
    exp_color = '#D55E00'  # Orange for experiment
    sim_color = '#0072B2'  # Blue for simulation
    sim_rob_color = '#6BA6CD'  # Lighter blue for robust simulation (more distinct from main blue)
    diff_color = '#009E73'  # Green for differentiator (colorblind-friendly)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    
    if model_folder == 'mlp_sim':
        return {
            'label': 'MLP (Sim.)',
            'color': sim_color,
            'edge_color': sim_color,
            'face_color': sim_color,
            'marker': '^',
            'linestyle': '-',
            'alpha': 0.8,
            'zorder': 2
        }
    if model_folder == 'mlp_experimental':
        return {
            'label': 'MLP (Expt.)',
            'color': exp_color,
            'edge_color': exp_color,
            'face_color': exp_color,
            'marker': markers[0],
            'linestyle': '--',
            'alpha': 1.0,
            'zorder': 1
        }
    if model_folder == 'differentiator':
        return {
            'label': 'Differentiator',
            'color': diff_color,
            'edge_color': diff_color,
            'face_color': diff_color,
            'marker': 's',
            'linestyle': '-',
            'alpha': 0.7,
            'zorder': 3
        }
    
    # Default styling
    return {
        'label': model_folder,
        'color': '#666666',
        'edge_color': '#666666',
        'face_color': '#666666',
        'marker': markers[0],
        'linestyle': '-',
        'alpha': 0.8,
        'zorder': 1
    }


def create_comparison_plots(param_name='detuning', base_directory=None, output_directory=None, save_fit_verification=True):
    """
    Create comparison plots for the specified parameter
    
    Args:
        param_name: Parameter to sweep ('detuning' or 'photon_number')
        base_directory: Directory containing model sweep folders. Defaults to data/source_data_fig4.
        output_directory: Directory where comparison PDFs are saved.
    """
    base_directory = Path(base_directory) if base_directory is not None else DEFAULT_FIG4_SOURCE_DIR
    output_directory = Path(output_directory) if output_directory is not None else DEFAULT_COMPARISON_OUTPUT_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Load results for all models
    model_folders = ['mlp_sim', 'differentiator', 'mlp_experimental']
    results = load_sweep_results(str(base_directory), model_folders, param_name=param_name)
    
    # Load 20-step energy data from energy traces
    energy_results = load_20_step_energy_from_traces(str(base_directory), model_folders, param_name=param_name)
    
    # Merge energy data with main results
    results = merge_energy_data_with_sweep_results(results, energy_results)
    
    # Fit cooling timescales from energy traces
    cooling_results = fit_cooling_timescale_from_trace(str(base_directory), model_folders, param_name=param_name)
    
    # Merge fitted cooling data with main results
    results = merge_cooling_data_with_sweep_results(results, cooling_results)
    
    if save_fit_verification and cooling_results:
        save_fit_verification_pdf(
            str(base_directory),
            {param_name: cooling_results},
            DEFAULT_FIT_VERIFICATION_FILE,
        )
    
    if not results:
        print(f"No data found for parameter {param_name}")
        return cooling_results
    
    # Format the parameter name for axis labels
    if param_name == 'detuning':
        param_label = r'Probe Detuning $\Delta/2\pi$ (MHz)'
        x_min, x_max = -110, 110
        output_file = output_directory / 'sweep_detuning.pdf'
    elif param_name == 'photon_number':
        param_label = 'Photon Counts'
        x_min, x_max = -2, 90
        output_file = output_directory / 'sweep_power.pdf'
    else:
        param_label = param_name.replace('_', ' ').title()
        x_min, x_max = None, None
        output_file = output_directory / f'sweep_{param_name}.pdf'

    # Styling parameters
    fonttype = 'Times New Roman'
    linewidth = 2
    markersize = 6
    ticksize = 14
    fontsize = 16
    legendsize = 12

    # Metrics to plot with their corresponding standard error columns
    metrics = [
        ('fraction_trapped', 'se_fraction_trapped', 'Survival probability'),
        ('cooling_timescale', 'cooling_timescale_err', 'Cooling timescale (μs)'),
        ('20_step_energy', '20_step_energy_err', 'Energy (μK)')
    ]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)
    plt.rcParams['font.family'] = fonttype
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    plt.rcParams['legend.fontsize'] = legendsize

    # Plot each metric
    for i, (metric, se_metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        
        # Keep track of actual plotted x values for this metric to set axis limits
        plotted_param_values = []
        
        # Plot data for each model with error bars
        for model_folder, df in results.items():
            styling = get_model_styling(model_folder)
            
            # Handle special cases for energy, temperature and cooling timescale (may have NaN values or outliers)
            if metric in ['20_step_energy', 'cooling_timescale']:
                # Filter out NaN values
                valid_data = df[~df[metric].isna()].copy()
                if not valid_data.empty:
                    # Filter outliers for specific metrics
                    if metric == '20_step_energy':
                        # Additional filter: only plot points with sufficient episodes at 20-step timepoint
                        if '20_step_episode_count' in valid_data.columns:
                            initial_count = len(valid_data)
                            valid_data = valid_data[valid_data['20_step_episode_count'] >= 200]
                            filtered_count = initial_count - len(valid_data)
                            if filtered_count > 0:
                                print(f"Filtered out {filtered_count} points for {styling['label']} with <200 episodes at 20-step timepoint")
                        
                        # Remove data with very large error bars or extreme energies
                        if se_metric in valid_data.columns:
                            valid_data = valid_data[valid_data[se_metric] < 50]
                        valid_data = valid_data[valid_data[metric] < 600]
                    elif metric == 'cooling_timescale':
                        # Remove data with very large error bars
                        if se_metric in valid_data.columns:
                            if param_name == 'detuning':
                                valid_data = valid_data[valid_data[se_metric] < 100]
                            else:
                                valid_data = valid_data[valid_data[se_metric] < 200]
                        valid_data = valid_data[valid_data[metric] > 1]
                    
                    if not valid_data.empty:
                        # Track parameter values for this metric's axis limits
                        plotted_param_values.extend(valid_data[param_name].values)
                        
                        # Plot without error bars
                        ax.plot(
                            valid_data[param_name], 
                            valid_data[metric],
                            marker=styling['marker'],
                            markeredgecolor=styling['edge_color'],
                            markerfacecolor=styling['face_color'],
                            color=styling['color'],
                            linestyle=styling['linestyle'],
                            linewidth=linewidth,
                            markersize=markersize,
                            label=styling['label'],
                            alpha=styling['alpha'],
                            zorder=styling['zorder']
                        )
            else:
                # Standard handling for fraction_trapped and other metrics
                # Track parameter values for this metric's axis limits
                plotted_param_values.extend(df[param_name].values)
                
                # Plot without error bars
                ax.plot(
                    df[param_name], 
                    df[metric],
                    marker=styling['marker'],
                    markeredgecolor=styling['edge_color'],
                    markerfacecolor=styling['face_color'],
                    color=styling['color'],
                    linestyle=styling['linestyle'],
                    linewidth=linewidth,
                    markersize=markersize,
                    label=styling['label'],
                    alpha=styling['alpha'],
                    zorder=styling['zorder']
                )
        
        # Set titles and labels
        ax.set_xlabel(param_label, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(bottom=0)  # Make y-axis start at 0
        
        # Set x-axis limits based on actual plotted data for this metric
        if plotted_param_values:
            data_x_min = min(plotted_param_values)
            data_x_max = max(plotted_param_values)
            # Add small margin (5% of range)
            x_range = data_x_max - data_x_min
            margin = max(x_range * 0.05, 1.0)  # At least 1 unit margin
            plot_x_min = data_x_min - margin
            plot_x_max = data_x_max + margin
            ax.set_xlim(plot_x_min, plot_x_max)
        else:
            # Use configured limits when no data was plotted.
            ax.set_xlim(x_min, x_max)
        
        # Add training point vertical lines and annotations
        if param_name == 'detuning':
            ax.axvline(x=25, color=sim_color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        elif param_name == 'photon_number':
            ax.axvline(x=33, color=sim_color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        
        ax.legend(fontsize=12, loc='best')
        
        # Style tick parameters
        ax.tick_params(axis='both', which='major', direction='in', length=3, width=1.5, labelsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot as {output_file}")
    return cooling_results


def load_20_step_energy_from_traces(base_directory, model_folders, param_name='photon_number'):
    """
    Load 20-step energy data from energy trace CSV files for multiple models
    
    Args:
        base_directory: Base directory containing model subfolders
        model_folders: List of model folders to load
        param_name: Name of the parameter that was swept
        
    Returns:
        Dictionary of DataFrames with 20-step energy data for each model
    """
    energy_results = {}
    
    for folder in model_folders:
        folder_path = os.path.join(base_directory, folder)
        energy_traces_path = os.path.join(folder_path, 'energy_traces')
        
        if not os.path.exists(energy_traces_path):
            print(f"Warning: Energy traces folder not found for {folder}")
            continue
        
        energy_data = []
        
        try:
            # Get list of raw energy trace CSV files for this parameter.
            energy_files = find_energy_trace_files(energy_traces_path, param_name, folder)
            
            for param_value, energy_file in energy_files:
                # Load the raw energy trace CSV.
                energy_trace_path = os.path.join(energy_traces_path, energy_file)
                try:
                    energy_df = pd.read_csv(energy_trace_path)
                    
                    # Find the 20-step energy (518.7 μs) - this should be row 20 (index 19)
                    target_time = 518.7
                    closest_idx = (energy_df['time_us'] - target_time).abs().idxmin()
                    
                    if abs(energy_df.loc[closest_idx, 'time_us'] - target_time) > 1.0:  # Tolerance of 1 μs
                        print(f"Warning: No data point close to {target_time} μs in {energy_file}")
                        continue
                    
                    energy_20_step = energy_df.loc[closest_idx, 'energy_uk']
                    energy_20_step_err = energy_df.loc[closest_idx, 'energy_uk_std_err']
                    episode_count_20_step = energy_df.loc[closest_idx, 'episode_count']
                    
                    # Filter: only include points with at least 200 episodes at the 20-step timepoint
                    if episode_count_20_step < 200:
                        print(f"Skipping {param_name}={param_value} in {folder}: only {episode_count_20_step} episodes at 20-step timepoint")
                        continue
                    
                    energy_data.append({
                        param_name: param_value,
                        '20_step_energy': energy_20_step,
                        '20_step_energy_err': energy_20_step_err,
                        '20_step_episode_count': episode_count_20_step
                    })
                    
                except Exception as e:
                    print(f"Error loading energy trace {energy_file}: {e}")
                    continue
            
            if energy_data:
                energy_df = pd.DataFrame(energy_data)
                energy_df = energy_df.sort_values(param_name)  # Sort by parameter value
                energy_results[folder] = energy_df
                print(f"Loaded 20-step energy data for {folder}: {len(energy_data)} points")
            else:
                print(f"No valid 20-step energy data found for {folder}")
                
        except OSError as e:
            print(f"Error accessing energy traces for {folder}: {e}")
            continue
    
    return energy_results


def merge_energy_data_with_sweep_results(sweep_results, energy_results):
    """
    Merge 20-step energy data with the main sweep results
    
    Args:
        sweep_results: Dictionary of main sweep DataFrames
        energy_results: Dictionary of 20-step energy DataFrames
        
    Returns:
        Updated sweep_results with 20-step energy data merged in
    """
    for folder in sweep_results.keys():
        if folder in energy_results:
            # Merge on the parameter column (detuning or photon_number)
            param_cols = [col for col in sweep_results[folder].columns if col in ['detuning', 'photon_number']]
            if param_cols:
                param_col = param_cols[0]
                merged_df = pd.merge(
                    sweep_results[folder], 
                    energy_results[folder], 
                    on=param_col, 
                    how='left'
                )
                sweep_results[folder] = merged_df
                print(f"Merged 20-step energy data for {folder}")
            else:
                print(f"Warning: No parameter column found for merging in {folder}")
        else:
            # Add empty columns for consistency
            sweep_results[folder]['20_step_energy'] = np.nan
            sweep_results[folder]['20_step_energy_err'] = np.nan
            print(f"No 20-step energy data available for {folder}, added NaN columns")
    
    return sweep_results


def exponential_decay(t, A, tau, B):
    """
    Exponential decay function: A * exp(-t/tau) + B
    
    Args:
        t: Time array
        A: Amplitude
        tau: Time constant (what we want to extract)
        B: Offset
    """
    return A * np.exp(-t / tau) + B


def is_monotonically_decreasing(energy_trace, start_idx=0):
    """
    Check if energy trace is monotonically decreasing (cooling)
    
    Args:
        energy_trace: Array of energy values
        start_idx: Index to start checking from (default 0)
        
    Returns:
        Boolean indicating if trace is monotonically decreasing
    """
    # Use a smoothed version to handle noise
    # Check if the overall trend is decreasing by comparing segments
    trace = energy_trace[start_idx:]
    if len(trace) < 5:
        return False
    
    # Check if the end is significantly lower than the beginning
    start_avg = np.mean(trace[:5])
    end_avg = np.mean(trace[-5:])
    
    # Also check that most consecutive points show decreasing trend
    decreasing_count = 0
    total_comparisons = 0
    
    # Use a sliding window approach to be more robust to noise
    window_size = 3
    for i in range(len(trace) - window_size):
        current_avg = np.mean(trace[i:i+window_size])
        next_avg = np.mean(trace[i+1:i+1+window_size])
        if next_avg < current_avg:
            decreasing_count += 1
        total_comparisons += 1
    
    # Require both overall decrease and majority of local decreases
    overall_decrease = end_avg < start_avg * 0.95  # At least 5% decrease
    local_decrease_fraction = decreasing_count / total_comparisons if total_comparisons > 0 else 0
    
    return overall_decrease and local_decrease_fraction > 0.6


def calculate_r_squared(observed, fitted):
    residuals = observed - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def fit_cooling_timescale_from_trace(base_directory, model_folders, param_name='photon_number'):
    """
    Fit cooling timescales from energy trace data using exponential decay fitting
    
    Args:
        base_directory: Base directory containing model subfolders
        model_folders: List of model folders to load
        param_name: Name of the parameter that was swept
        
    Returns:
        Dictionary of DataFrames with fitted cooling timescale data for each model
    """
    cooling_results = {}
    
    for folder in model_folders:
        folder_path = os.path.join(base_directory, folder)
        energy_traces_path = os.path.join(folder_path, 'energy_traces')
        
        if not os.path.exists(energy_traces_path):
            print(f"Warning: Energy traces folder not found for {folder}")
            continue
        
        cooling_data = []
        
        try:
            # Get list of raw energy trace CSV files for this parameter.
            energy_files = find_energy_trace_files(energy_traces_path, param_name, folder)
            
            for param_value, energy_file in energy_files:
                # Load the raw energy trace CSV.
                energy_trace_path = os.path.join(energy_traces_path, energy_file)
                try:
                    energy_df = pd.read_csv(energy_trace_path)
                    
                    # Prepare data for fitting (convert time to seconds)
                    time_us = energy_df['time_us'].values
                    time_s = time_us / 1e6  # Convert μs to seconds
                    energy_uk = energy_df['energy_uk'].values
                    
                    # Fit the complete trace segment supported by enough
                    # surviving episodes. This avoids choosing a subjective
                    # reheating cutoff while still excluding late averages that
                    # are dominated by too few trajectories.
                    valid_mask = energy_df['episode_count'] >= 200
                    if valid_mask.sum() < 10:  # Need at least 10 points for good fit
                        print(f"Skipping {param_name}={param_value} in {folder}: insufficient valid data points for fitting")
                        continue
                    
                    fit_time = time_s[valid_mask]
                    fit_energy = energy_uk[valid_mask]
                    fit_sigma = energy_df.loc[valid_mask, 'energy_uk_std_err'].to_numpy()
                    finite_sigma = np.isfinite(fit_sigma) & (fit_sigma > 0)
                    if not np.all(finite_sigma):
                        fallback_sigma = np.nanmedian(fit_sigma[finite_sigma]) if np.any(finite_sigma) else 1.0
                        fit_sigma = np.where(finite_sigma, fit_sigma, fallback_sigma)
                    
                    # Initial guess for fitting parameters. The offset is
                    # constrained only by non-negative energy and the observed
                    # scale, so the full survivor-supported trace determines
                    # the long-time asymptote.
                    baseline_point_count = min(10, len(fit_energy))
                    late_base = float(np.mean(fit_energy[-baseline_point_count:]))
                    B_guess = max(min(late_base, np.max(fit_energy)), 0)
                    A_guess = max(fit_energy[0] - B_guess, 1e-9)  # Amplitude
                    tau_guess = max((fit_time[-1] - fit_time[0]) / 3, 1e-6)
                    B_upper = max(np.max(fit_energy), 1e-9)
                    
                    # Bounds for fitting (tau must be positive, reasonable ranges)
                    bounds = ([0, 1e-6, 0], [np.inf, 1.0, B_upper])  # tau between 1μs and 1s
                    
                    try:
                        # Fit the exponential decay
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            popt, pcov = curve_fit(
                                exponential_decay, 
                                fit_time, 
                                fit_energy,
                                sigma=fit_sigma,
                                absolute_sigma=True,
                                p0=[A_guess, tau_guess, B_guess],
                                bounds=bounds,
                                maxfev=5000
                            )
                        
                        fitted_A, fitted_tau, fitted_B = popt
                        fitted_energy = exponential_decay(fit_time, *popt)
                        r_squared = calculate_r_squared(fit_energy, fitted_energy)
                        
                        # Calculate fitting errors
                        param_errors = np.sqrt(np.diag(pcov))
                        tau_error = param_errors[1]
                        
                        # Quality check: reasonable timescale and good selected-window fit
                        if fitted_tau > 0 and fitted_tau < 1.0 and tau_error < fitted_tau and r_squared > 0.9:
                            fitted_tau_us = fitted_tau * 1e6
                            tau_error_us = tau_error * 1e6
                            
                            cooling_data.append({
                                param_name: param_value,
                                'fitted_cooling_timescale': fitted_tau_us,
                                'fitted_cooling_timescale_err': tau_error_us,
                                'fit_A': fitted_A,
                                'fit_B': fitted_B,
                                'fit_r_squared': r_squared,
                                'fit_window_end_us': fit_time[-1] * 1e6,
                                'fit_window_points': len(fit_time),
                                'fit_quality': 'good'
                            })
                            print(f"Fitted τ = {fitted_tau_us:.1f} ± {tau_error_us:.1f} μs for {param_name}={param_value} in {folder}")
                        else:
                            print(f"Poor fit quality for {param_name}={param_value} in {folder}: τ={fitted_tau*1e6:.1f}μs, error={tau_error*1e6:.1f}μs, R²={r_squared:.3f}")
                            
                    except Exception as fit_error:
                        print(f"Fitting failed for {param_name}={param_value} in {folder}: {fit_error}")
                        continue
                        
                except Exception as e:
                    print(f"Error loading/processing energy trace {energy_file}: {e}")
                    continue
            
            if cooling_data:
                cooling_df = pd.DataFrame(cooling_data)
                cooling_df = cooling_df.sort_values(param_name)  # Sort by parameter value
                cooling_results[folder] = cooling_df
                print(f"Successfully fitted cooling timescales for {folder}: {len(cooling_data)} points")
            else:
                print(f"No valid cooling timescale fits found for {folder}")
                
        except OSError as e:
            print(f"Error accessing energy traces for {folder}: {e}")
            continue
    
    return cooling_results


def merge_cooling_data_with_sweep_results(sweep_results, cooling_results):
    """
    Merge fitted cooling timescale data with the main sweep results
    
    Args:
        sweep_results: Dictionary of main sweep DataFrames
        cooling_results: Dictionary of fitted cooling timescale DataFrames
        
    Returns:
        Updated sweep_results with fitted cooling timescale data merged in
    """
    for folder in sweep_results.keys():
        if folder in cooling_results:
            # Merge on the parameter column (detuning or photon_number)
            param_cols = [col for col in sweep_results[folder].columns if col in ['detuning', 'photon_number']]
            if param_cols:
                param_col = param_cols[0]
                # Add fitted data, replacing existing cooling_timescale columns
                merged_df = pd.merge(
                    sweep_results[folder], 
                    cooling_results[folder], 
                    on=param_col, 
                    how='left'
                )
                
                if 'fitted_cooling_timescale' in merged_df.columns:
                    merged_df['cooling_timescale'] = merged_df['fitted_cooling_timescale']
                    merged_df['cooling_timescale_err'] = merged_df['fitted_cooling_timescale_err']
                
                sweep_results[folder] = merged_df
                print(f"Merged fitted cooling timescale data for {folder}")
            else:
                print(f"Warning: No parameter column found for merging cooling data in {folder}")
        else:
            # Add empty columns for consistency
            sweep_results[folder]['fitted_cooling_timescale'] = np.nan
            sweep_results[folder]['fitted_cooling_timescale_err'] = np.nan
            print(f"No fitted cooling timescale data available for {folder}, added NaN columns")
    
    return sweep_results


def save_fit_verification_pdf(base_directory, cooling_results_by_param, output_file=DEFAULT_FIT_VERIFICATION_FILE):
    """Save one dense PDF page containing all cooling traces and accepted fits."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    param_order = [
        param for param in ['detuning', 'photon_number']
        if param in cooling_results_by_param
    ]
    model_order = ['mlp_sim', 'differentiator', 'mlp_experimental']
    row_specs = []
    trace_files_by_row = {}
    for param_name in param_order:
        for model in model_order:
            energy_traces_path = os.path.join(base_directory, model, 'energy_traces')
            trace_files = find_energy_trace_files(energy_traces_path, param_name, model)
            if trace_files:
                row_specs.append((param_name, model))
                trace_files_by_row[(param_name, model)] = trace_files

    if not row_specs:
        print("No cooling traces available for verification plot")
        return

    max_cols = max(
        len(trace_files_by_row[(param_name, model)])
        for param_name, model in row_specs
    )
    fig_width = max(12, 2.15 * max_cols)
    fig_height = max(8, 1.85 * len(row_specs))
    fig, axes = plt.subplots(
        len(row_specs),
        max_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
    )

    for row_idx, (param_name, model) in enumerate(row_specs):
        cooling_df = cooling_results_by_param.get(param_name, {}).get(model, pd.DataFrame())
        if not cooling_df.empty:
            cooling_df = cooling_df.set_index(param_name)
        energy_traces_path = os.path.join(base_directory, model, 'energy_traces')
        styling = get_model_styling(model)
        trace_files = trace_files_by_row[(param_name, model)]

        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(trace_files):
                ax.axis('off')
                continue

            param_value, trace_filename = trace_files[col_idx]
            energy_trace_path = os.path.join(energy_traces_path, trace_filename)
            if energy_trace_path is None:
                ax.axis('off')
                continue

            energy_df = pd.read_csv(energy_trace_path)
            time_us = energy_df['time_us'].to_numpy()
            time_ms = time_us / 1000
            energy_uk = energy_df['energy_uk'].to_numpy()
            valid_mask = energy_df['episode_count'].to_numpy() >= 200

            ax.plot(
                time_ms[~valid_mask],
                energy_uk[~valid_mask],
                'x',
                color='0.55',
                markersize=2.0,
                alpha=0.8,
            )
            ax.plot(time_ms[valid_mask], energy_uk[valid_mask], 'o', color=styling['color'], markersize=2.0)

            fit_row = None
            if not cooling_df.empty and param_value in cooling_df.index:
                fit_row = cooling_df.loc[param_value]

            fit_time_us = time_us[valid_mask]
            if fit_row is not None and len(fit_time_us) > 1:
                fit_time_s = np.linspace(fit_time_us[0] / 1e6, fit_time_us[-1] / 1e6, 120)
                fitted_curve = exponential_decay(
                    fit_time_s,
                    fit_row['fit_A'],
                    fit_row['fitted_cooling_timescale'] / 1e6,
                    fit_row['fit_B'],
                )
                ax.plot(fit_time_s * 1000, fitted_curve, '-', color='red', linewidth=1.1)
                fit_label = (
                    f"tau={fit_row['fitted_cooling_timescale']:.0f}"
                    f"+/-{fit_row['fitted_cooling_timescale_err']:.0f} us"
                )
            else:
                fit_label = "fit rejected"

            value_label = r'$\Delta$' if param_name == 'detuning' else 'N'
            ax.set_title(
                f"{value_label}={param_value:g}\n"
                f"{fit_label}",
                fontsize=6,
            )
            ax.tick_params(axis='both', labelsize=5, length=2)
            ax.grid(True, alpha=0.25)

        axes[row_idx, 0].set_ylabel(f"{styling['label']}\n{param_name}\nEnergy (uK)", fontsize=7)

    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Time (ms)", fontsize=7)

    fig.suptitle(
        "Cooling-timescale fit verification: color = fitted data (episode_count >= 200), gray x = excluded data, red = accepted weighted exponential fit",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.8, w_pad=0.4)

    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Fit verification saved to: {output_file}")


def main():
    """Main function to create both detuning and power sweep plots"""
    parser = argparse.ArgumentParser(description='Create simulation comparison plots')
    parser.add_argument('--param', choices=['detuning', 'photon_number', 'both'], 
                       default='both', help='Parameter to sweep')
    parser.add_argument('--base-directory', default=str(DEFAULT_FIG4_SOURCE_DIR),
                       help='Directory containing mlp_sim, differentiator, and mlp_experimental folders')
    parser.add_argument('--output-directory', default=str(DEFAULT_COMPARISON_OUTPUT_DIR),
                       help='Directory where sweep comparison plots are saved')
    
    args = parser.parse_args()
    
    if args.param == 'both':
        print("Creating detuning sweep plots...")
        detuning_cooling = create_comparison_plots(
            'detuning',
            args.base_directory,
            args.output_directory,
            save_fit_verification=False,
        )
        print("\nCreating photon number sweep plots...")
        photon_cooling = create_comparison_plots(
            'photon_number',
            args.base_directory,
            args.output_directory,
            save_fit_verification=False,
        )
        save_fit_verification_pdf(
            args.base_directory,
            {'detuning': detuning_cooling, 'photon_number': photon_cooling},
            DEFAULT_FIT_VERIFICATION_FILE,
        )
    else:
        print(f"Creating {args.param} sweep plots...")
        create_comparison_plots(args.param, args.base_directory, args.output_directory)


if __name__ == "__main__":
    main() 
