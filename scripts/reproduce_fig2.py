import os
from pathlib import Path
import numpy as np
import yaml
def sem(values, axis=0, ddof=1):
    """Compute the standard error of the mean without SciPy."""
    values = np.asarray(values, dtype=float)
    count = np.sum(~np.isnan(values), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nanstd(values, axis=axis, ddof=ddof) / np.sqrt(count)

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec 
from scipy.optimize import curve_fit 

import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
with open(SCRIPT_DIR / "params.yaml", "r", encoding="utf-8") as handle:
    PARAMS = yaml.safe_load(handle)
FIG2_CFG = PARAMS["fig2"]
OUTPUT_DIR = PACKAGE_ROOT / PARAMS["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.85
FILTER_STEP = 110
TARGET_STEP = 100
WINDOW_SIZE = 5
SMOOTH_WIDTH = 3
GAUSSIAN_TRUNCATE = 4.0

def load_transmission_to_temperature_mapping(csv_file):
    """Load transmission and photon count to temperature mapping from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        unique_N = sorted(df['N'].unique())
        interp_data = {}
        
        for n in unique_N:
            n_data = df[df['N'] == n]
            transmissions = n_data['Transmission'].values
            z_temps = n_data['E_z_uK'].values
            sort_idx = np.argsort(transmissions)
            interp_data[n] = (transmissions[sort_idx], z_temps[sort_idx])

        print(f"Loaded 2D transmission-to-temperature mapping from {os.path.basename(csv_file)}")
        print(f"Photon counts (N) range: {min(unique_N)} to {max(unique_N)}")

        def temp_interpolator(N, transmission):
            if isinstance(N, (list, np.ndarray)) and isinstance(transmission, (list, np.ndarray)):
                results = np.zeros_like(transmission, dtype=float)
                # Ensure N and transmission have compatible shapes for element-wise processing
                if isinstance(N, (int, float)):
                    N_array = np.full_like(transmission, N, dtype=float)
                else:
                    N_array = np.array(N, dtype=float)

                for i in range(len(transmission)):
                    results[i] = temp_interpolator(N_array[i], transmission[i]) 
                return results

            # Single value interpolation logic
            nearest_n_values = sorted(unique_N, key=lambda x: abs(x - N))[:2]
            
            if len(nearest_n_values) == 1 or nearest_n_values[0] == nearest_n_values[1]:
                n = nearest_n_values[0]
                transmissions, z_temps = interp_data[n]
                interp = interp1d(transmissions, z_temps, bounds_error=False, fill_value="extrapolate")
                return max(0.0, float(interp(transmission)))
            else:
                n1, n2 = sorted(nearest_n_values[:2]) # Ensure n1 <= n2
                transmissions1, z_temps1 = interp_data[n1]
                transmissions2, z_temps2 = interp_data[n2]
                
                interp1 = interp1d(transmissions1, z_temps1, bounds_error=False, fill_value="extrapolate")
                interp2 = interp1d(transmissions2, z_temps2, bounds_error=False, fill_value="extrapolate")
                
                temp1 = float(interp1(transmission))
                temp2 = float(interp2(transmission))
                
                if n1 == n2: # Avoid division by zero if n values are identical
                    return temp1
                else:
                    # Ensure weight is capped between 0 and 1, handle N outside unique_N range
                    weight = max(0, min(1, (N - n1) / (n2 - n1)))
                    return max(0.0, temp1 * (1 - weight) + temp2 * weight)
        
        return temp_interpolator
    except Exception as e:
        print(f"Error loading transmission mapping from {csv_file}: {e}")
        return None

# --- Data processing function ---

def process_folder_for_figure(folder_path, threshold, filter_step, target_step, window_size, smooth_width, temp_mapping_func):
    """Process curated trace tables, apply filtering, and calculate average traces."""
    shots_df = pd.read_csv(PACKAGE_ROOT / FIG2_CFG["shots_csv"])
    traces_df = pd.read_csv(PACKAGE_ROOT / FIG2_CFG["traces_csv"])
    condition_shots = shots_df[shots_df["condition"] == folder_path].copy()
    condition_traces = traces_df[traces_df["condition"] == folder_path].copy()
    trace_map = {
        file_id: group.sort_values("step")
        for file_id, group in condition_traces.groupby("file_id")
    }
    print(f"Processing {len(condition_shots)} traces in {folder_path}...")
    
    valid_counts_traces = []
    valid_reward_traces = []
    valid_ecm_list = []
    step_durations_ns = []
    skipped_files = 0
    filtered_out_files = 0
    
    filter_start = max(0, filter_step - window_size//2)
    filter_end = filter_step + window_size//2 + 1
    
    trace_len_target = target_step + 1 # We need steps 0 to 100 inclusive
    smooth_radius = int(GAUSSIAN_TRUNCATE * smooth_width + 0.5)
    required_len = max(filter_step + window_size // 2 + smooth_radius + 1, trace_len_target)

    for row in condition_shots.itertuples(index=False):
        replay_valid = bool(row.replay_buffer_valid)
        empty_cavity_mean = float(row.empty_cavity_mean) if not pd.isna(row.empty_cavity_mean) else None
        dt_ns = float(row.dt_ns) if not pd.isna(row.dt_ns) else None
        trace_df = trace_map.get(row.file_id)
        if not replay_valid or empty_cavity_mean is None or empty_cavity_mean <= 0 or trace_df is None:
            skipped_files += 1
            continue
        if dt_ns is None or dt_ns <= 0:
            skipped_files += 1
            continue

        counts = trace_df["plus_spcm_counts"].to_numpy(dtype=float)
        if len(counts) < required_len:
            skipped_files += 1
            continue
        step_durations_ns.append(dt_ns)

        smoothed_counts = gaussian_filter1d(counts.astype(float), smooth_width)
        filter_indices = range(filter_start, min(filter_end, len(smoothed_counts)))
        if not filter_indices:
            skipped_files += 1
            continue

        filter_avg_counts = np.mean(smoothed_counts[filter_indices])
        transmission_filter = filter_avg_counts / empty_cavity_mean
        if transmission_filter >= threshold:
            filtered_out_files += 1
            continue

        counts_trace = counts[:trace_len_target].astype(float)
        valid_counts_traces.append(counts_trace)
        valid_ecm_list.append(empty_cavity_mean)

        rewards = trace_df["reward"].to_numpy(dtype=float)
        if np.isnan(rewards).all():
            skipped_files += 1
            valid_counts_traces.pop()
            valid_ecm_list.pop()
            step_durations_ns.pop()
            continue
        if len(rewards) >= trace_len_target:
            rewards_trace = rewards[:trace_len_target]
        elif len(rewards) == trace_len_target - 1:
            rewards_trace = np.concatenate([rewards, [0.0]])
        else:
            rewards_trace = np.pad(rewards, (0, max(0, trace_len_target - len(rewards))), mode='constant')
        valid_reward_traces.append(np.nan_to_num(rewards_trace, nan=0.0))
            
    print(f"  Found {len(valid_counts_traces)} valid traces after filtering.")
    print(f"  Skipped {skipped_files} files (invalid/missing data).")
    print(f"  Filtered out {filtered_out_files} files (threshold condition).")

    if not valid_counts_traces:
        print("  No valid data found for this folder.")
        return None

    # --- Calculate average results ---
    
    min_len = min(len(tr) for tr in valid_counts_traces)
    if valid_reward_traces:
        min_len = min(min_len, min(len(tr) for tr in valid_reward_traces))
    print(f"  min_len = {min_len}")
    aligned_counts = np.array([tr[:min_len] for tr in valid_counts_traces])
    aligned_rewards = np.array([tr[:min_len] for tr in valid_reward_traces]) if valid_reward_traces else None
    
    avg_counts = np.mean(aligned_counts, axis=0)
    sem_counts = sem(aligned_counts, axis=0)
    avg_reward = np.mean(aligned_rewards, axis=0) if aligned_rewards is not None else None
    avg_cum_reward = np.cumsum(avg_reward) if avg_reward is not None else None
    avg_ecm = np.mean(valid_ecm_list) # Average ECM for passed traces
    print(f"  Avg Empty Cavity Mean (for passed traces): {avg_ecm:.3f}")

    avg_temp = None
    sem_temp = None # SEM for temperature is not calculated this way
    
    if temp_mapping_func and avg_ecm > 0:
        # Calculate avg transmission from avg counts and avg ECM
        avg_transmission = avg_counts / avg_ecm
        # Use the empty-cavity count setting for the calibration N axis.
        avg_temp = temp_mapping_func(avg_counts, avg_transmission)
        print("  Calculated average temperature trace from average transmission and empty-cavity calibration.")
    elif not temp_mapping_func:
        print("  Temperature mapping function not available.")
    else:
        print("  Average Empty Cavity Mean is zero or invalid, cannot calculate transmission/temperature.")

    # Calculate average time step and create time axis
    if not step_durations_ns:
        print("  Warning: No valid time step durations found. Cannot create time axis.")
        time_us = np.arange(min_len) # Fallback to step index if time fails
    else:
        avg_dt_ns = np.mean(step_durations_ns)
        avg_dt_us = avg_dt_ns / 1000.0
        time_us = np.arange(min_len) * avg_dt_us
        print(f"  Average step duration: {avg_dt_us:.3f} µs")

    return {
        "time_us": time_us,
        "avg_counts": avg_counts,
        "sem_counts": sem_counts,
        "avg_reward": avg_reward,
        "avg_cum_reward": avg_cum_reward,
        "avg_temp": avg_temp,
        "sem_temp": sem_temp, # Will be None
        "count": len(valid_counts_traces),
        "avg_ecm": avg_ecm # Add avg_ecm to the returned dictionary
    }

# --- Helper function to load single trace data --- 

def get_single_trace_data(file_path, num_steps):
    """Read counts, time axis, feedback action, empty-cavity mean, and rewards from one example trace CSV."""
    example_csv = Path(file_path)
    if not example_csv.exists():
        print(f"Warning: Example file not found: {example_csv}")
        return None, None, None, None, None # Added None for action, empty_cavity_mean, and rewards
    try:
        df = pd.read_csv(example_csv)
        if len(df) < num_steps:
            print(f"  Skipping {example_csv.name}: Not enough steps ({len(df)}/{num_steps}).")
            return None, None, None, None, None
        segment = df.iloc[:num_steps].copy()
        counts_segment = segment["plus_spcm_counts"].to_numpy(dtype=float)
        time_us = segment["time_us"].to_numpy(dtype=float)
        action_segment = segment["feedback_action"].to_numpy(dtype=float) if "feedback_action" in segment else None
        rewards_segment = segment["reward"].to_numpy(dtype=float) if "reward" in segment else None
        empty_cavity_mean = float(segment["empty_cavity_mean"].dropna().iloc[0])
        return counts_segment, time_us, action_segment, empty_cavity_mean, rewards_segment
    except Exception as e:
        print(f"  Error processing single file {example_csv.name}: {e}")
        return None, None, None, None, None

# --- Exponential Fit Function ---
def exp_decay_func(t, A, tau, B):
    """Exponential decay function A*exp(-t/tau) + B."""
    return A * np.exp(-t / tau) + B

# --- Plotting function ---

def plot_figure(data_with_feedback, data_without_feedback, example_files, output_filename="figure2_plot.png"):
    """Generate the three-panel plot for publication with exponential fits."""
    
    if not data_with_feedback or not data_without_feedback:
        print("Missing average data for one or both conditions. Cannot generate plot.")
        return

    # --- Load Example Trace Data ---
    example_data = {}
    num_steps_for_examples = len(data_with_feedback['time_us']) # Match length of average data
    for key, file_path in example_files.items():
        counts, time, action, empty_cavity_mean, rewards = get_single_trace_data(file_path, num_steps_for_examples) 
        if counts is not None and time is not None:
            example_data[key] = {
                'counts': counts,
                'time_us': time,
                'action': action,
                'empty_cavity_mean': empty_cavity_mean,
                'rewards': rewards,
            }
        else:
            print(f"Could not load example trace for '{key}': {file_path}")

    # --- Setup Plot (Apply Style from Example) ---
    plt.style.use('seaborn-v0_8-paper') 
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.figsize': (10, 8.5)
    })

    colors = {
        'avg_with': '#D55E00',
        'avg_without': '#0072B2',
        'fit_with': '#BC3E3B',
        'shot_noise': 'gray',
        'action': '#4D4D4D'
    }
    
    labels = {
        'avg_with': 'MLP (Expt.) Counts', 
        'avg_without': 'No Feedback Counts',
        'action': 'MLP (Expt.) Action (Trap Depth)'
    }

    fig = plt.figure()

    # Use GridSpec similar to example, 2 rows, 2 columns. Panel (a) spans top row.
    gs = gridspec.GridSpec(2, 2, figure=fig) 
    
    ax_a = fig.add_subplot(gs[0, :]) # Panel (a) spans the top row (index 0, all columns)
    ax_b = fig.add_subplot(gs[1, 0]) # Panel (b) is bottom-left (index 1, col 0)
    ax_c = fig.add_subplot(gs[1, 1]) # Panel (c) is bottom-right (index 1, col 1)

    subplot_labels = ['(a)', '(b)', '(c)'] # Labels for panels


    label_trans = mtransforms.ScaledTranslation(-40/72, 5/72, fig.dpi_scale_trans)
    ax_a.text(0.0, 1.0, subplot_labels[0], transform=ax_a.transAxes + label_trans,
              fontsize=18, fontweight='bold', va='bottom', ha='left')
    max_time_a = 0
    
    # Create secondary axis for action data
    ax_a_action = ax_a.twinx()
    ax_a_action.set_ylim(0, 1.05) # Action is (val+1)/2, so 0 to 1 range
    ax_a_action.set_ylabel("Action (Trap Depth)", fontsize=16, color=colors['action']) # Match axes label size (new)
    ax_a_action.tick_params(axis='y', labelsize=12, labelcolor=colors['action']) # Match tick label size (new)


    if 'with' in example_data and example_data['with'] is not None:
        time_ex_w = example_data['with']['time_us'] / 1000.0  # Convert to milliseconds
        counts_ex_w = example_data['with']['counts']
        action_ex_w = example_data['with'].get('action')
        rewards_ex_w = example_data['with'].get('rewards')
        empty_cavity_mean = example_data['with'].get('empty_cavity_mean')
        
        ax_a.plot(time_ex_w, counts_ex_w, color=colors['avg_with'], label=labels['avg_with'], linewidth=2.5)
        
        if action_ex_w is not None:
            action_transformed = (action_ex_w + 1) / 2.0
            ax_a_action.plot(time_ex_w, action_transformed, color=colors['action'], label=labels['action'], linewidth=2.5)

        y_min_a, y_max_a = 0, 65
        reward_sm = None
        if rewards_ex_w is not None:
            n = min(len(time_ex_w), len(rewards_ex_w))
            if n > 0:
                rewards_slice = rewards_ex_w[:n]
                reward_min = float(np.min(rewards_slice))
                reward_max = float(np.max(rewards_slice))
                if reward_max == reward_min:
                    reward_max = reward_min + 1.0
                norm = mcolors.PowerNorm(gamma=1.5, vmin=reward_min, vmax=reward_max)
                reward_band = np.vstack([rewards_slice, rewards_slice])
                reward_im = ax_a.imshow(
                    reward_band,
                    extent=[time_ex_w[0], time_ex_w[n - 1], y_min_a, y_max_a],
                    origin='lower',
                    aspect='auto',
                    cmap='Greens',
                    norm=norm,
                    alpha=0.75,
                    zorder=0,
                )
                reward_sm = plt.cm.ScalarMappable(norm=norm, cmap='Greens')
                reward_sm.set_array([])
                cax = inset_axes(
                    ax_a,
                    width="30%",
                    height="3%",
                    loc="lower right",
                    bbox_to_anchor=(0.0, 0.3, 0.9675, 1.0),
                    bbox_transform=ax_a.transAxes,
                    borderpad=0.8,
                )
                cb = fig.colorbar(reward_sm, cax=cax, orientation='horizontal')
                cb.set_label('Reward', fontsize=12, labelpad=4)
                cb.ax.xaxis.set_label_position('top')
                cb.ax.xaxis.set_ticks_position('bottom')
                cb.ax.tick_params(labelsize=10)
                # cax.set_facecolor('white')
                # cax.patch.set_alpha(1.0)


        # Add empty cavity mean line
        if empty_cavity_mean is not None and empty_cavity_mean > 0:
            ax_a.axhline(y=empty_cavity_mean, color='red', linestyle='--', 
                        linewidth=1.5, alpha=0.7, label='Empty cavity mean')
        
        if len(time_ex_w) > 0: max_time_a = max(max_time_a, time_ex_w[-1])

    ax_a.set_ylabel("Photon Counts", color=colors['avg_with']) # Keep axis label
    ax_a.tick_params(axis='y', labelcolor=colors['avg_with'])
    ax_a.grid(True) # Grid settings controlled by rcParams
    ax_a.set_ylim(bottom=0, top=65)
    ax_a.set_xlabel("Time (ms)")

    # Add secondary x-axis for oscillation periods (180 µs = 0.18 ms per period)
    def ms_to_periods(x):
        return x / 0.18  # Convert ms to oscillation periods
    
    def periods_to_ms(x):
        return x * 0.18  # Convert oscillation periods to ms
    
    ax_a_top = ax_a.secondary_xaxis('top', functions=(ms_to_periods, periods_to_ms))
    ax_a_top.set_xlabel('Oscillation Periods', fontsize=16)
    ax_a_top.tick_params(axis='x', labelsize=12)

    # Add legend directly to panel (a) - combine handles from both axes
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_a_action, labels_a_action = ax_a_action.get_legend_handles_labels()
    ax_a.legend(
        handles_a + handles_a_action ,
        labels_a + labels_a_action,
        loc=[0.63,0.475]
    )

    # --- Panel (b): Average Photon Counts ---
    # Manually position (b) label aligned with (a)
    ax_b.text(0.0, 1.0, subplot_labels[1], transform=ax_b.transAxes + label_trans,
              fontsize=18, fontweight='bold', va='bottom', ha='left')
    
    time_w_avg = data_with_feedback['time_us'] / 1000.0
    avg_c_w = data_with_feedback['avg_counts']
    ax_b.plot(time_w_avg, avg_c_w, color=colors['avg_with'], label=labels['avg_with'])

    time_wo_avg = data_without_feedback['time_us'] / 1000.0
    avg_c_wo = data_without_feedback['avg_counts']
    ax_b.plot(time_wo_avg, avg_c_wo, color=colors['avg_without'], label=labels['avg_without'])


    # Fit for Feedback Counts
    if len(time_w_avg) > 2 and len(avg_c_w) > 2:
        time_fit_c = time_w_avg[2:]
        counts_to_fit = avg_c_w[2:]
        try:
            p0_c = [counts_to_fit[0] - np.mean(counts_to_fit[-len(counts_to_fit)//5:]), time_fit_c[-1]/3, np.mean(counts_to_fit[-len(counts_to_fit)//5:])]
            params_c, cov_c = curve_fit(exp_decay_func, time_fit_c, counts_to_fit, p0=p0_c, maxfev=5000)
            perr_c = np.sqrt(np.diag(cov_c))

            print("\nPanel (b) Feedback Counts Fit Results:")
            print(f"  A = {params_c[0]:.2f} +/- {perr_c[0]:.2f}")
            print(f"  tau = {params_c[1]:.3f} +/- {perr_c[1]:.3f} ms")
            print(f"  B = {params_c[2]:.2f} +/- {perr_c[2]:.2f} counts")
            
            line_fit_c, = ax_b.plot(time_fit_c, exp_decay_func(time_fit_c, *params_c), color=colors['fit_with'], linestyle='-')
            ax_b.set_ylim(bottom=0)
            
        except RuntimeError:
            print("Panel (b) Feedback Counts: Could not fit exponential decay.")
            ax_b.plot(time_w_avg, avg_c_w, color=colors['avg_with'], label=f"{labels['avg_with']} (Fit Failed)", linestyle=':')

    ax_b.set_ylabel("Photon Counts")
    ax_b.set_xlabel("Time (ms)")  # Update x-axis label to milliseconds
    ax_b.grid(True)
    ax_b.set_ylim(bottom=0)
    # Set specific x-axis ticks for panel (b)
    ax_b.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax_b.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])

    # Add secondary x-axis for oscillation periods every 2.5 periods
    ax_b_top = ax_b.secondary_xaxis('top', functions=(ms_to_periods, periods_to_ms))
    ax_b_top.set_xlabel('Oscillation Periods', fontsize=16)
    ax_b_top.tick_params(axis='x', labelsize=12)
    # Set ticks every 2.5 oscillation periods
    period_values = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
    ax_b_top.set_xticks(period_values)
    ax_b_top.set_xticklabels([f'{p:.1f}' for p in period_values])

    # --- Calculate and Plot Shot Noise Boundaries for Panel (b) --- 
    avg_ecm_w = data_with_feedback.get('avg_ecm') # Get avg ECM for feedback
    shot_noise_data = None
    shot_noise_lines_b_handles = [] # Store handles for potential legend entry if needed
    if len(avg_c_w) >= 7:
        last_7_indices = slice(-7, None)
        avg_last_7_c = np.mean(avg_c_w[last_7_indices])
        if avg_last_7_c > 0:
            shot_noise_c = np.sqrt(avg_last_7_c)
            count_upper = avg_last_7_c + shot_noise_c
            count_lower = avg_last_7_c - shot_noise_c
            
            shot_noise_data = {
                'avg_c': avg_last_7_c,
                'noise_c': shot_noise_c,
                'count_upper': count_upper,
                'count_lower': count_lower
            }

            print(f"Panel (b) Feedback Avg Last 7 Counts: {avg_last_7_c:.2f}, Shot Noise: ±{shot_noise_c:.2f}")
            
    # Add legend to panel (b) AFTER all plotting elements are added
    ax_b.legend(loc='upper right') 

    # --- Panel (c): Average Z-Energy (Temperature) ---
    ax_c.text(0.0, 1.0, subplot_labels[2], transform=ax_c.transAxes + label_trans,
              fontsize=18, fontweight='bold', va='bottom', ha='left')
    
    temp_mapping_func_from_data = data_with_feedback.get('temp_mapping_func') 
    fit_label_c = None # To store the fit label with tau

    if data_with_feedback.get('avg_temp') is not None and data_without_feedback.get('avg_temp') is not None:
        avg_t_w = data_with_feedback['avg_temp']
        line_avg_w_c, = ax_c.plot(time_w_avg, avg_t_w, color=colors['avg_with'], label=labels['avg_with'])

        avg_t_wo = data_without_feedback['avg_temp']
        line_avg_wo_c, = ax_c.plot(time_wo_avg, avg_t_wo, color=colors['avg_without'], label=labels['avg_without'])

        # --- Calculate Shot Noise Boundaries for Panel (c) --- 
        shot_noise_temps = {} # Store calculated temps for full span plotting
        if shot_noise_data and temp_mapping_func_from_data and avg_ecm_w is not None and avg_ecm_w > 0:
            try:
                trans_upper = shot_noise_data['count_upper'] / avg_ecm_w
                trans_lower = shot_noise_data['count_lower'] / avg_ecm_w
                temp_upper = temp_mapping_func_from_data(shot_noise_data['count_upper'], trans_upper)
                temp_lower = temp_mapping_func_from_data(shot_noise_data['count_lower'], trans_lower)
                shot_noise_temps['upper'] = temp_upper
                shot_noise_temps['lower'] = temp_lower
                print(f"Panel (c) Temp bounds from shot noise: Upper={temp_upper:.2f} µK, Lower={temp_lower:.2f} µK")
            except Exception as e:
                print(f"Panel (c): Error calculating temperature shot noise bounds: {e}")
        
        # Fit for Feedback Temperature
        if len(time_w_avg) > 2 and len(avg_t_w) > 2:
            time_fit_t = time_w_avg[2:]
            temp_to_fit = avg_t_w[2:]
            if np.all(np.isfinite(temp_to_fit)):
                try:
                    p0_t = [temp_to_fit[0] - np.mean(temp_to_fit[-len(temp_to_fit)//5:]), 
                            time_fit_t[-1]/3, 
                            np.mean(temp_to_fit[-len(temp_to_fit)//5:])]
                    params_t, cov_t = curve_fit(exp_decay_func, time_fit_t, temp_to_fit, p0=p0_t, maxfev=5000)
                    perr_t = np.sqrt(np.diag(cov_t))
                    
                    line_fit_t, = ax_c.plot(time_fit_t, exp_decay_func(time_fit_t, *params_t), color=colors['fit_with'], linestyle='-')
                    ax_c.set_ylim(bottom=0)
                    
                    print("\nPanel (c) Feedback Temperature Fit Results:")
                    print(f"  A = {params_t[0]:.2f} +/- {perr_t[0]:.2f} µK")
                    print(f"  tau = {params_t[1]:.3f} +/- {perr_t[1]:.3f} ms")
                    print(f"  B = {params_t[2]:.2f} +/- {perr_t[2]:.2f} µK")

                except RuntimeError:
                    print("Panel (c) Feedback Temperature: Could not fit exponential decay.")
                    ax_c.plot(time_w_avg, avg_t_w, color=colors['avg_with'], label=f"{labels['avg_with']} (Fit Failed)", linestyle=':')
            else:
                print("Panel (c) Feedback Temperature: Contains NaN/inf, skipping fit.")

        ax_c.set_ylabel("Energy (µK)")
        ax_c.set_xlabel("Time (ms)")  # Update x-axis label to milliseconds
        ax_c.grid(True)
        ax_c.set_ylim(bottom=0)
        # Set specific x-axis ticks for panel (c)
        ax_c.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax_c.set_xticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
        
        # Add secondary x-axis for oscillation periods every 2.5 periods
        ax_c_top = ax_c.secondary_xaxis('top', functions=(ms_to_periods, periods_to_ms))
        ax_c_top.set_xlabel('Oscillation Periods', fontsize=16)
        ax_c_top.tick_params(axis='x', labelsize=12)
        # Set ticks every 2.5 oscillation periods
        period_values = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
        ax_c_top.set_xticks(period_values)
        ax_c_top.set_xticklabels([f'{p:.1f}' for p in period_values])
        
        ax_c.legend(loc='upper right')

    else:
        ax_c.text(0.5, 0.5, 'Avg Temp data not available', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax_c.transAxes)
        ax_c.set_xlabel("Time (ms)")  # Update x-axis label to milliseconds
        ax_c.set_ylabel("Z-Energy (µK)")
        ax_c.grid(True)
        
        # Add secondary x-axis for oscillation periods even when temp data not available (every 2.5 periods)
        ax_c_top = ax_c.secondary_xaxis('top', functions=(ms_to_periods, periods_to_ms))
        ax_c_top.set_xlabel('Oscillation Periods', fontsize=16)
        ax_c_top.tick_params(axis='x', labelsize=12)
        # Set ticks every 2.5 oscillation periods
        period_values = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
        ax_c_top.set_xticks(period_values)
        ax_c_top.set_xticklabels([f'{p:.1f}' for p in period_values])

    max_time_bc = 0
    if 'time_us' in data_with_feedback and len(data_with_feedback['time_us']) > 0 : 
        max_time_bc = max(max_time_bc, data_with_feedback['time_us'][-1] / 1000.0)  # Convert to milliseconds
    if 'time_us' in data_without_feedback and len(data_without_feedback['time_us']) > 0 : 
        max_time_bc = max(max_time_bc, data_without_feedback['time_us'][-1] / 1000.0)  # Convert to milliseconds
    
    final_max_time = 0
    if max_time_a > 0: final_max_time = max(final_max_time, max_time_a)
    if max_time_bc > 0: final_max_time = max(final_max_time, max_time_bc)

    if final_max_time > 0:
        # Set consistent x-limits for all panels
        ax_a.set_xlim(0, final_max_time)  # Use full time range for panel (a) too
        ax_b.set_xlim(0, final_max_time)
        ax_c.set_xlim(0, final_max_time)

    # Update legends after adding shot noise lines
    ax_b.legend(loc='upper right')
    ax_c.legend(loc='upper right')

    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.8)

    plt.savefig(output_filename, dpi=600) # Keep dpi=600
    pdf_filename = os.path.splitext(output_filename)[0] + '.pdf'
    plt.savefig(pdf_filename)
    print(f"Figure saved to {output_filename} and {pdf_filename}")
    plt.close()

# --- Main execution ---

def main():
    csv_file = str(PACKAGE_ROOT / FIG2_CFG["calibration_csv"])
    example_files = {
        "with": str(PACKAGE_ROOT / FIG2_CFG["example_with_csv"]),
    }
    print(f"Using Feedback Example: {os.path.basename(example_files['with'])}")

    output_plot_file = str(OUTPUT_DIR / FIG2_CFG["output_pdf"].replace('.pdf', '.png'))

    # Load the temperature mapping function
    temp_mapping_func = load_transmission_to_temperature_mapping(csv_file)
    # No need to stop if mapping fails, panel (c) will show message

    processed_data = {}
    
    # Process average data for each folder
    for name in ["with_feedback", "without_feedback"]:
        folder_key = name.replace("_", " ").title() 
        print(f"\n--- Processing Average Data: {folder_key} ---")
        results = process_folder_for_figure(
            name,
            THRESHOLD,
            FILTER_STEP,
            TARGET_STEP,
            WINDOW_SIZE,
            SMOOTH_WIDTH,
            temp_mapping_func
        )
        if results:
            processed_data[name] = results
        else:
            print(f"Could not process average data for {folder_key}")
            processed_data[name] = None
            
    # Generate the plot using processed average data and example file paths
    if processed_data.get("with_feedback") and processed_data.get("without_feedback"):
         # Pass temp mapping func to plotting function if loaded
        if temp_mapping_func:
             # Check if temp_mapping_func is needed in data_with_feedback dict anymore
             # It is needed for shot noise temp calculation within plot_figure
             processed_data["with_feedback"]["temp_mapping_func"] = temp_mapping_func 
             
        plot_figure(processed_data["with_feedback"], 
                    processed_data["without_feedback"], 
                    example_files,
                    output_plot_file)
    else:
         print("\nCould not generate plot due to missing processed average data for one or both conditions.")

if __name__ == "__main__":
    main() 




















