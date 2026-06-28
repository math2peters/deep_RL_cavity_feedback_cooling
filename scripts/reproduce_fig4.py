#!/usr/bin/env python3
"""
Combined Figure Four Script

Creates a 3x6 grid where each plot spans 2 columns:
Row 1: (a) Survival, (b) Cooling timescale, (c) Energy - Detuning sweeps
Row 2: (d) Survival, (e) Cooling timescale, (f) Energy - Photon counts sweeps
Row 3: (g) MLP histogram, (h) Differentiator histogram, (i) Force vs velocity binned
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
import os
import sys
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import pickle
from collections import defaultdict
import warnings
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = Path(script_dir).parent
with open(os.path.join(script_dir, 'params.yaml'), 'r', encoding='utf-8') as handle:
    PARAMS = yaml.safe_load(handle)
FIG4_CFG = PARAMS['fig4']
OUTPUT_DIR = package_root / PARAMS['output_dir']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, script_dir)

# Import functions from simulation_comparison_script.py
from simulation_comparison_script import (
    load_sweep_results, 
    load_20_step_energy_from_traces,
    merge_energy_data_with_sweep_results,
    fit_cooling_timescale_from_trace,
    merge_cooling_data_with_sweep_results,
    get_model_styling,
    save_fit_verification_pdf,
)

# Try to import cmocean for additional colormaps
try:
    import cmocean.cm as cmo
    CMOCEAN_AVAILABLE = True
except ImportError:
    CMOCEAN_AVAILABLE = False
    print("Warning: cmocean not available. Using fallback colormap.")

# Physical constants
h = 6.62607015e-34  # Planck constant in Js
kB = 1.380649e-23   # Boltzmann constant in J/K
m = 2.21e-25        # Cesium mass in kg

# Shared force-axis limits for Fig. 4 panels g-i (friction force / mass in m/s^2)
FORCE_YLIM = (-450, 450)


def load_trajectory_data(data_path):
    """Load cached turning-point trajectories from a pickle file."""
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)


def analyze_trajectories(trajectories, turning_point_type='position'):
    """Analyze preprocessed trajectories containing turning-point data."""
    results = {
        'episode_ids': [],
        'turning_points': defaultdict(list),
        'force_velocity_pairs': [],
    }

    for traj in trajectories:
        episode_id = traj['episode_id']
        results['episode_ids'].append(episode_id)

        turning_point_data = [
            tp for tp in traj['turning_points']
            if tp.get('type', 'velocity') == turning_point_type
        ]
        if not turning_point_data:
            continue

        for i, tp_info in enumerate(turning_point_data):
            results['turning_points'][i].append({
                'episode_id': episode_id,
                'time': tp_info['time'],
                'position_z': tp_info['position_z'],
                'velocity_z': tp_info.get('velocity_z', 0.0),
                'energy': tp_info['energy'],
                'max_velocity': tp_info['max_velocity'],
            })

        for i in range(len(turning_point_data) - 2):
            tp1 = turning_point_data[i]
            tp3 = turning_point_data[i + 2]

            dv = tp3['max_velocity'] - tp1['max_velocity']
            dt = tp3['time'] - tp1['time']
            if dt == 0:
                continue

            results['force_velocity_pairs'].append({
                'episode_id': episode_id,
                'avg_velocity': (tp3['max_velocity'] + tp1['max_velocity']) / 2,
                'dvdt': dv / dt,
                'tp1_idx': i,
                'tp2_idx': i + 2,
                'step_size': 2,
                'time_midpoint': (tp1['time'] + tp3['time']) / 2,
            })

    return results


def get_force_velocity_binned_data(results):
    """Calculate binned force-velocity data for one processed dataset."""
    velocities = [pair['avg_velocity'] for pair in results['force_velocity_pairs']]
    forces = [pair['dvdt'] for pair in results['force_velocity_pairs']]
    if not velocities or not forces or len(velocities) < 2:
        return None

    valid_indices = np.isfinite(velocities) & np.isfinite(forces)
    velocities = np.array(velocities)[valid_indices]
    forces = np.array(forces)[valid_indices]
    if len(velocities) < 10:
        return None

    try:
        bins = np.linspace(np.min(velocities), np.max(velocities), 50)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        bin_indices = np.digitize(velocities, bins)
        bin_means = []
        for i in range(1, len(bins)):
            bin_forces = forces[bin_indices == i]
            if len(bin_forces) > 0:
                bin_means.append(np.mean(bin_forces))
            else:
                bin_means.append(np.nan)

        valid_indices = ~np.isnan(bin_means)
        bin_centers = bin_centers[valid_indices]
        bin_means = np.array(bin_means)[valid_indices]
        if len(bin_centers) == 0:
            return None

        return {'bin_centers': bin_centers, 'bin_means': bin_means}
    except Exception as exc:
        print(f"Error in get_force_velocity_binned_data: {exc}")
        return None


def calculate_histogram_max(results):
    """Calculate a shared vmax for force-velocity histograms."""
    velocities = [pair['avg_velocity'] for pair in results['force_velocity_pairs']]
    forces = [pair['dvdt'] for pair in results['force_velocity_pairs']]
    if not velocities or not forces:
        return 0

    valid_indices = np.isfinite(velocities) & np.isfinite(forces)
    velocities = np.array(velocities)[valid_indices]
    forces = np.array(forces)[valid_indices]
    if len(velocities) == 0:
        return 0

    v_min, v_max = np.min(velocities), np.max(velocities)
    if v_min >= v_max:
        v_max = v_min + 1e-9 if v_min == 0 else v_min + abs(v_min * 1e-9) + 1e-9
    f_min, f_max = FORCE_YLIM

    hist, _, _ = np.histogram2d(
        velocities,
        forces,
        bins=50,
        range=[[v_min, v_max], [f_min, f_max]],
    )
    return np.max(hist)


def plot_force_velocity_histogram_on_ax(ax, results, title_suffix, global_vmax=None, colormap='inferno', bg_color='black'):
    """Plot a 2D force-velocity histogram on the provided axes."""
    velocities = [pair['avg_velocity'] for pair in results['force_velocity_pairs']]
    forces = [pair['dvdt'] for pair in results['force_velocity_pairs']]

    if not velocities or not forces:
        ax.text(0.5, 0.5, "No data for histogram", ha='center', va='center', transform=ax.transAxes, color='white')
        ax.set_title(f'{title_suffix}', fontsize=20)
        ax.set_facecolor(bg_color)
        return

    valid_indices = np.isfinite(velocities) & np.isfinite(forces)
    velocities = np.array(velocities)[valid_indices]
    forces = np.array(forces)[valid_indices]
    if len(velocities) == 0:
        ax.text(0.5, 0.5, "No valid data for histogram", ha='center', va='center', transform=ax.transAxes, color='white')
        ax.set_title(f'{title_suffix}', fontsize=20)
        ax.set_facecolor(bg_color)
        return

    v_min, v_max = np.min(velocities), np.max(velocities)
    if v_min >= v_max:
        v_max = v_min + 1e-9 if v_min == 0 else v_min + abs(v_min * 1e-9) + 1e-9
    f_min, f_max = FORCE_YLIM

    ax.set_facecolor(bg_color)
    vmax_to_use = global_vmax if global_vmax is not None else None
    hist = ax.hist2d(
        velocities,
        forces,
        bins=50,
        cmap=colormap,
        range=[[v_min, v_max], [f_min, f_max]],
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax_to_use),
        edgecolors='none',
        rasterized=True,
    )

    cbar = plt.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count')
    cbar.ax.tick_params(labelsize=8)

    ax.axhline(y=0, color='#FF4040', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=0, color='#FF4040', linestyle='--', alpha=0.7, linewidth=1)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('dv/dt (m/s²)')
    ax.set_ylim(FORCE_YLIM)
    ax.set_title(f'{title_suffix}', fontsize=20)


def load_and_process_sweep_data(script_dir, model_folders, param_name, return_cooling=False):
    """Load and process sweep data for a given parameter type"""
    print(f"Loading {param_name} sweep data...")
    results = load_sweep_results(script_dir, model_folders, param_name=param_name)
    
    # Load 20-step energy data from energy traces
    print(f"Loading 20-step energy data for {param_name}...")
    energy_results = load_20_step_energy_from_traces(script_dir, model_folders, param_name=param_name)
    
    # Merge energy data with main results
    results = merge_energy_data_with_sweep_results(results, energy_results)
    
    # Fit cooling timescales from energy traces
    print(f"Fitting cooling timescales for {param_name}...")
    cooling_results = fit_cooling_timescale_from_trace(script_dir, model_folders, param_name=param_name)
    
    # Merge fitted cooling data with main results
    results = merge_cooling_data_with_sweep_results(results, cooling_results)
    
    if return_cooling:
        return results, cooling_results
    return results


def plot_sweep_metric_on_ax(ax, results, metric_info, param_name, x_min, x_max, param_label):
    """Plot a single metric for parameter sweep on given axis"""
    metric, se_metric, ylabel = metric_info
    
    # Keep track of actual plotted x values for this metric to set axis limits
    plotted_param_values = []
    
    # Plot data for each model
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
                        linewidth=2,
                        markersize=6,
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
                linewidth=2,
                markersize=6,
                label=styling['label'],
                alpha=styling['alpha'],
                zorder=styling['zorder']
            )
    
    # Set labels and limits
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
        # Add vertical line at 25 MHz for detuning plots
        sim_color = '#0072B2'
        ax.axvline(x=25, color=sim_color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    elif param_name == 'photon_number':
        # Add vertical line at 33 photons for photon number plots
        sim_color = '#0072B2'
        ax.axvline(x=33, color=sim_color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(axis='both', which='major', direction='in', length=3, width=1.5, labelsize=12)


def load_and_process_trajectory_data(data_cache_dir):
    """Load and process trajectory data for force-velocity analysis"""
    data_files_info = {
        "MLP (Sim.)": os.path.join(data_cache_dir, "mlp_sim_40k_turning_points.pkl"),
        "Differentiator": os.path.join(data_cache_dir, "differentiator_40k_turning_points.pkl")
    }
    
    processed_data = {}
    for label, path in data_files_info.items():
        if not os.path.exists(path):
            print(f"Data file not found: {path}")
            continue
        print(f"Loading data for {label} from {os.path.basename(path)}...")
        trajectories = load_trajectory_data(path)
        print(f"Analyzing trajectories for {label}...")
        # Analyze position turning points for the force-velocity summaries.
        processed_data[label] = analyze_trajectories(trajectories, turning_point_type='position')
    
    return processed_data


def plot_force_velocity_binned_on_ax(ax, processed_data):
    """Plot combined force vs velocity binned data on given axis"""
    colors = {
        "MLP (Sim.)": "#0072B2",  
        "Differentiator": "#009E73"
    }
    linestyles = {
        "MLP (Sim.)": "-",
        "Differentiator": "-"
    }
    
    for label in ["MLP (Sim.)", "Differentiator"]:
        if label not in processed_data: 
            continue
        results = processed_data[label]
        binned_data = get_force_velocity_binned_data(results)
        if binned_data:
            ax.plot(binned_data['bin_centers'], binned_data['bin_means'], 
                      label=label, color=colors[label], linestyle=linestyles[label], linewidth=1.5)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Velocity (m/s)', fontsize=14)
    ax.set_ylabel('Friction Force / Mass (m/s²)', fontsize=14)
    ax.set_ylim(FORCE_YLIM)
    ax.legend(loc='lower left', fontsize=10)
    ax.tick_params(axis='both', which='major', direction='in', length=3, width=1.5, labelsize=12)


def main():
    """Main function to create the combined Figure 4"""
    
    # Set matplotlib styling consistent with style_example.py
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.titlesize': 14,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    data_root = package_root / FIG4_CFG['data_root']
    data_cache_dir = data_root / 'data_cache'
    model_folders = FIG4_CFG['model_folders']
    
    # Load detuning and photon number data
    print("="*60)
    print("LOADING SIMULATION COMPARISON DATA")
    print("="*60)
    
    detuning_results, detuning_cooling = load_and_process_sweep_data(
        str(data_root), model_folders, 'detuning', return_cooling=True
    )
    photon_results, photon_cooling = load_and_process_sweep_data(
        str(data_root), model_folders, 'photon_number', return_cooling=True
    )
    save_fit_verification_pdf(
        str(data_root),
        {'detuning': detuning_cooling, 'photon_number': photon_cooling},
        OUTPUT_DIR / 'cooling_timescale_fit_verification.pdf',
    )
    
    # Load trajectory data for force-velocity analysis
    print("\n" + "="*60)
    print("LOADING TRAJECTORY DATA")
    print("="*60)
    
    trajectory_data = load_and_process_trajectory_data(str(data_cache_dir))
    
    # Create figure with 3 rows x 6 columns, each plot spans 2 columns
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 6, figure=fig)
    
    # Define metrics to plot
    metrics = [
        ('fraction_trapped', 'se_fraction_trapped', 'Survival probability'),
        ('cooling_timescale', 'cooling_timescale_err', 'Cooling timescale (μs)'),
        ('20_step_energy', '20_step_energy_err', 'Energy (μK)')
    ]
    
    # Row 1: Detuning plots (a, b, c)
    print("\n" + "="*40)
    print("GENERATING DETUNING PLOTS")
    print("="*40)
    
    detuning_axes = []
    for i, metric_info in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i*2:(i+1)*2])  # Each plot spans 2 columns
        detuning_axes.append(ax)
        
        if detuning_results:
            plot_sweep_metric_on_ax(
                ax, detuning_results, metric_info, 'detuning', 
                -110, 110, r'Probe Detuning $\Delta/2\pi$ (MHz)'
            )
        else:
            ax.text(0.5, 0.5, 'Detuning data not available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Row 2: Photon counts plots (d, e, f)
    print("\n" + "="*40)
    print("GENERATING PHOTON NUMBER PLOTS")
    print("="*40)
    
    photon_axes = []
    for i, metric_info in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i*2:(i+1)*2])  # Each plot spans 2 columns
        photon_axes.append(ax)
        
        if photon_results:
            plot_sweep_metric_on_ax(
                ax, photon_results, metric_info, 'photon_number',
                -2, 90, 'Photon Counts'
            )
        else:
            ax.text(0.5, 0.5, 'Photon counts data not available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Row 3: Force-velocity plots (g, h, i)
    print("\n" + "="*40)
    print("GENERATING FORCE-VELOCITY PLOTS")
    print("="*40)
    
    # Calculate global maximum for consistent histogram normalization
    global_max = 0
    for dataset_name in ["MLP (Sim.)", "Differentiator"]:
        if dataset_name in trajectory_data:
            dataset_max = calculate_histogram_max(trajectory_data[dataset_name])
            global_max = max(global_max, dataset_max)
    
    # Use global max, but ensure it's at least 1 for LogNorm
    global_max = max(global_max, 1)
    
    # Histogram colormap
    if CMOCEAN_AVAILABLE:
        histogram_colormap = cmo.thermal
        histogram_background_color = histogram_colormap(0.0)
    else:
        histogram_colormap = 'inferno'
        histogram_background_color = 'black'
    
    # (g) MLP (Sim.) histogram
    ax_g = fig.add_subplot(gs[2, 0:2])
    if "MLP (Sim.)" in trajectory_data:
        plot_force_velocity_histogram_on_ax(
            ax_g, trajectory_data["MLP (Sim.)"], "MLP (Sim.)", 
            global_vmax=global_max, colormap=histogram_colormap, 
            bg_color=histogram_background_color
        )
        # Override font settings
        ax_g.set_xlabel(ax_g.get_xlabel(), fontsize=14)
        ax_g.set_ylabel("Friction Force / Mass (m/s²)", fontsize=14)
        ax_g.set_title(ax_g.get_title(), fontsize=14)
        ax_g.tick_params(axis='both', which='major', labelsize=12)
    else:
        ax_g.text(0.5, 0.5, "MLP (Sim.) data not available", 
                 ha='center', va='center', transform=ax_g.transAxes)
        ax_g.set_title('MLP (Sim.)', fontsize=14)
    
    # (h) Differentiator histogram
    ax_h = fig.add_subplot(gs[2, 2:4])
    if "Differentiator" in trajectory_data:
        plot_force_velocity_histogram_on_ax(
            ax_h, trajectory_data["Differentiator"], "Differentiator", 
            global_vmax=global_max, colormap=histogram_colormap, 
            bg_color=histogram_background_color
        )
        # Override font settings
        ax_h.set_xlabel(ax_h.get_xlabel(), fontsize=14)
        ax_h.set_ylabel("Friction Force / Mass (m/s²)", fontsize=14)
        ax_h.set_title(ax_h.get_title(), fontsize=14)
        ax_h.tick_params(axis='both', which='major', labelsize=12)
    else:
        ax_h.text(0.5, 0.5, "Differentiator data not available", 
                 ha='center', va='center', transform=ax_h.transAxes)
        ax_h.set_title('Differentiator', fontsize=14)
    
    # (i) Force vs velocity binned
    ax_i = fig.add_subplot(gs[2, 4:6])
    if trajectory_data:
        plot_force_velocity_binned_on_ax(ax_i, trajectory_data)
        ax_i.set_title('Force vs. Velocity (Binned)', fontsize=14)
    else:
        ax_i.text(0.5, 0.5, "Trajectory data not available", 
                 ha='center', va='center', transform=ax_i.transAxes)
        ax_i.set_title('Force vs. Velocity (Binned)', fontsize=14)
    
    # Add subplot labels (a) through (i)
    all_axes = detuning_axes + photon_axes + [ax_g, ax_h, ax_i]
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    
    for ax, label in zip(all_axes, labels):
        ax.text(-0.12, 1.12, label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='right')
    
    # Save the figure
    output_path = OUTPUT_DIR / FIG4_CFG['output_pdf']
    png_path = OUTPUT_DIR / FIG4_CFG['output_pdf'].replace('.pdf', '.png')
    plt.tight_layout(pad=1., h_pad=1., w_pad=1.)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"\nFigure saved to {output_path}")
    print(f"Figure saved to {png_path}")
    
    plt.close()


if __name__ == "__main__":
    main() 
