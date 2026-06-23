from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
def sem(values, axis=0, ddof=1):
    """Compute the standard error of the mean without SciPy."""
    values = np.asarray(values, dtype=float)
    count = np.sum(~np.isnan(values), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nanstd(values, axis=axis, ddof=ddof) / np.sqrt(count)

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
with open(SCRIPT_DIR / "params.yaml", "r", encoding="utf-8") as handle:
    PARAMS = yaml.safe_load(handle)
FIG1_CFG = PARAMS["fig1"]
OUTPUT_DIR = PACKAGE_ROOT / PARAMS["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.85
FILTER_STEP = 210
TARGET_STEP = 200
WINDOW_SIZE = 5
SMOOTH_WIDTH = 3
MAX_TAU = 30
GAUSSIAN_TRUNCATE = 4.0

def calculate_g2_correlation(intensity_trace, max_tau=None):
    """Calculate g2(tau) correlation function for a single intensity trace."""
    if max_tau is None:
        max_tau = len(intensity_trace) // 4  # Use 1/4 of trace length as default
    
    max_tau = min(max_tau, len(intensity_trace) - 1)
    
    # Calculate mean intensity
    mean_intensity = np.mean(intensity_trace)
    if mean_intensity <= 0:
        return None, None
    
    taus = np.arange(max_tau + 1)
    g2_values = np.zeros(len(taus))
    
    for i, tau in enumerate(taus):
        if tau == 0:
            # g2(0) - handle zero-delay case
            correlation = np.mean(intensity_trace**2)
        else:
            # Calculate correlation for tau > 0
            if len(intensity_trace) > tau:
                correlation = np.mean(intensity_trace[:-tau] * intensity_trace[tau:])
            else:
                correlation = 0
        
        g2_values[i] = correlation / (mean_intensity**2)
    
    return taus, g2_values

def gaussian(x, amplitude, center, width, offset):
    """Gaussian function for peak fitting."""
    return amplitude * np.exp(-((x - center) / width)**2) + offset

def fit_peak(frequencies_khz, power, center_guess_khz, fit_range_khz=2.0):
    """Fit a Gaussian to the peak around center_guess_khz."""
    # Define fitting region
    fit_mask = (frequencies_khz >= center_guess_khz - fit_range_khz) & \
               (frequencies_khz <= center_guess_khz + fit_range_khz)
    
    if np.sum(fit_mask) < 4:  # Need at least 4 points for 4-parameter fit
        raise ValueError("Not enough data points in fitting region")
    
    freq_fit = frequencies_khz[fit_mask]
    power_fit = power[fit_mask]
    
    # Initial parameter guesses
    amplitude_guess = np.max(power_fit) - np.min(power_fit)
    center_guess = freq_fit[np.argmax(power_fit)]
    width_guess = 0.5  # kHz
    offset_guess = np.min(power_fit)
    
    initial_guess = [amplitude_guess, center_guess, width_guess, offset_guess]
    
    try:
        popt, pcov = curve_fit(gaussian, freq_fit, power_fit, p0=initial_guess)
        return popt, pcov, fit_mask
    except Exception as e:
        print(f"Peak fitting failed: {e}")
        return None, None, fit_mask

# --- Data processing functions ---

def process_folder_for_fourier(folder_path, threshold, filter_step, target_step, window_size, smooth_width):
    """Process curated trace tables and calculate Fourier transforms."""
    shots_df = pd.read_csv(PACKAGE_ROOT / FIG1_CFG["shots_csv"])
    traces_df = pd.read_csv(PACKAGE_ROOT / FIG1_CFG["traces_csv"])
    trace_map = {
        file_id: group.sort_values("step")["plus_spcm_counts"].to_numpy(dtype=float)
        for file_id, group in traces_df.groupby("file_id")
    }
    print(f"Processing {len(shots_df)} traces in fig1_shots.csv...")
    
    valid_power_spectra = []
    step_durations_ns = []
    skipped_files = 0
    filtered_out_files = 0
    
    filter_start = max(0, filter_step - window_size//2)
    filter_end = filter_step + window_size//2 + 1
    
    trace_len_target = target_step + 1 # We need steps 0 to target_step inclusive
    smooth_radius = int(GAUSSIAN_TRUNCATE * smooth_width + 0.5)
    required_len = max(filter_step + window_size // 2 + smooth_radius + 1, trace_len_target)

    for row in shots_df.itertuples(index=False):
        replay_valid = bool(row.replay_buffer_valid)
        empty_cavity_mean = float(row.empty_cavity_mean) if not pd.isna(row.empty_cavity_mean) else None
        dt_ns = float(row.dt_ns) if not pd.isna(row.dt_ns) else None
        counts = trace_map.get(row.file_id)

        if not replay_valid or empty_cavity_mean is None or empty_cavity_mean <= 0 or counts is None:
            skipped_files += 1
            continue
        if dt_ns is None or dt_ns <= 0:
            skipped_files += 1
            continue
        step_durations_ns.append(dt_ns)
        if len(counts) < required_len:
            skipped_files += 1
            continue

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
        normalized_trace = counts_trace / empty_cavity_mean
        fft_result = np.fft.fft(normalized_trace)
        power_spectrum = np.abs(fft_result) ** 2
        valid_power_spectra.append(power_spectrum)
            
    print(f"  Found {len(valid_power_spectra)} valid traces after filtering.")
    print(f"  Skipped {skipped_files} files (invalid/missing data).")
    print(f"  Filtered out {filtered_out_files} files (threshold condition).")

    if not valid_power_spectra:
        print("  No valid data found for this folder.")
        return None

    # --- Calculate average power spectrum ---
    min_len = min(len(ps) for ps in valid_power_spectra)
    aligned_spectra = np.array([ps[:min_len] for ps in valid_power_spectra])
    
    avg_power_spectrum = np.mean(aligned_spectra, axis=0)
    sem_power_spectrum = sem(aligned_spectra, axis=0)

    # Calculate frequency axis
    if not step_durations_ns:
        print("  Warning: No valid time step durations found. Cannot create frequency axis.")
        frequencies = np.arange(min_len) # Fallback to index if time fails
    else:
        avg_dt_ns = np.mean(step_durations_ns)
        avg_dt_s = avg_dt_ns / 1e9  # Convert to seconds
        sample_rate = 1.0 / avg_dt_s
        frequencies = np.fft.fftfreq(min_len, avg_dt_s)
        print(f"  Average step duration: {avg_dt_ns:.3f} ns")
        print(f"  Sample rate: {sample_rate:.3f} Hz")

    return {
        "frequencies": frequencies,
        "avg_power_spectrum": avg_power_spectrum,
        "sem_power_spectrum": sem_power_spectrum,
        "count": len(valid_power_spectra)
    }

def process_folder_for_g2(folder_path, threshold, filter_step, target_step, window_size, smooth_width, max_tau=None):
    """Process curated trace tables and calculate g2(tau) correlations."""
    shots_df = pd.read_csv(PACKAGE_ROOT / FIG1_CFG["shots_csv"])
    traces_df = pd.read_csv(PACKAGE_ROOT / FIG1_CFG["traces_csv"])
    trace_map = {
        file_id: group.sort_values("step")["plus_spcm_counts"].to_numpy(dtype=float)
        for file_id, group in traces_df.groupby("file_id")
    }
    print(f"Processing {len(shots_df)} traces in fig1_shots.csv...")
    
    valid_g2_functions = []
    step_durations_ns = []
    skipped_files = 0
    filtered_out_files = 0
    
    filter_start = max(0, filter_step - window_size//2)
    filter_end = filter_step + window_size//2 + 1
    
    trace_len_target = target_step + 1 # We need steps 0 to target_step inclusive
    smooth_radius = int(GAUSSIAN_TRUNCATE * smooth_width + 0.5)
    required_len = max(filter_step + window_size // 2 + smooth_radius + 1, trace_len_target)

    for row in shots_df.itertuples(index=False):
        replay_valid = bool(row.replay_buffer_valid)
        empty_cavity_mean = float(row.empty_cavity_mean) if not pd.isna(row.empty_cavity_mean) else None
        dt_ns = float(row.dt_ns) if not pd.isna(row.dt_ns) else None
        counts = trace_map.get(row.file_id)

        if not replay_valid or empty_cavity_mean is None or empty_cavity_mean <= 0 or counts is None:
            skipped_files += 1
            continue
        if dt_ns is None or dt_ns <= 0:
            skipped_files += 1
            continue
        step_durations_ns.append(dt_ns)
        if len(counts) < required_len:
            skipped_files += 1
            continue

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
        normalized_trace = counts_trace / empty_cavity_mean
        taus, g2_values = calculate_g2_correlation(normalized_trace, max_tau)
        if taus is not None and g2_values is not None:
            valid_g2_functions.append(g2_values)
            
    print(f"  Found {len(valid_g2_functions)} valid traces after filtering.")
    print(f"  Skipped {skipped_files} files (invalid/missing data).")
    print(f"  Filtered out {filtered_out_files} files (threshold condition).")

    if not valid_g2_functions:
        print("  No valid data found for this folder.")
        return None

    # --- Calculate average g2 function ---
    min_len = min(len(g2) for g2 in valid_g2_functions)
    aligned_g2 = np.array([g2[:min_len] for g2 in valid_g2_functions])
    
    avg_g2 = np.mean(aligned_g2, axis=0)
    sem_g2 = sem(aligned_g2, axis=0)

    # Calculate tau axis in time units
    if not step_durations_ns:
        print("  Warning: No valid time step durations found. Using step indices for tau.")
        tau_values = np.arange(min_len)
        tau_units = "steps"
    else:
        avg_dt_ns = np.mean(step_durations_ns)
        avg_dt_us = avg_dt_ns / 1000.0  # Convert to microseconds
        tau_values = np.arange(min_len) * avg_dt_us
        tau_units = "µs"
        print(f"  Average step duration: {avg_dt_ns:.3f} ns")

    return {
        "tau_values": tau_values,
        "tau_units": tau_units,
        "avg_g2": avg_g2,
        "sem_g2": sem_g2,
        "count": len(valid_g2_functions)
    }

# --- Plotting functions ---

def create_fourier_plot(data, condition, ax):
    """Create Fourier transform plot on the given axis."""
    if condition == 'with_feedback':
        color = '#0072B2'  # Blue for experiment
        label = 'MLP (Expt.)'
        marker = 's-'
    else:
        color = '#0072B2'  # Blue for no feedback
        label = 'No Feedback'
        marker = 's-'
    
    # Get data for this condition
    freq = data['frequencies']
    power = data['avg_power_spectrum']
    sem = data['sem_power_spectrum']
    
    # Take positive frequencies only and exclude DC component (index 0)
    n_points = len(freq) // 2
    
    # Skip DC component (index 0) to avoid large y-axis scaling
    freq_pos = freq[1:n_points]
    power_pos = power[1:n_points]
    sem_pos = sem[1:n_points]
    
    # Convert frequencies to kHz
    freq_pos_khz = freq_pos / 1000.0
    
    # Plot with error bars, markers, and connecting lines
    ax.errorbar(freq_pos_khz, power_pos, yerr=sem_pos, 
                fmt=f'{marker}',
                markeredgecolor=color,
                markerfacecolor=color,
                color=color,
                ecolor=color,
                label=label,
                linewidth=1.5,
                markersize=4,
                capsize=2,
                capthick=0.5,
                alpha=0.8)
    
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD")
    ax.grid(True)
    ax.set_ylim(bottom=0, top=75)  # Ensure y-axis starts at 0
    
    # Set frequency limits
    if len(freq_pos_khz) > 0:
        ax.set_xlim(freq_pos_khz[0], freq_pos_khz[-1])
    
    # Find and annotate actual peaks in the data
    if len(freq_pos_khz) > 0 and len(power_pos) > 0:
        # Find peaks in the power spectrum
        peaks, _ = find_peaks(power_pos, height=np.max(power_pos) * 0.1, distance=10)
        
        # Look for peaks near expected harmonics
        peak_freqs = freq_pos_khz[peaks]
        peak_powers = power_pos[peaks]
        
        # Find peak closest to 5.5 kHz (1st harmonic)
        first_harmonic_candidates = peaks[np.abs(peak_freqs - 5.5) < 2.0]
        if len(first_harmonic_candidates) > 0:
            peak_idx = first_harmonic_candidates[np.argmax(peak_powers[np.abs(peak_freqs - 5.5) < 2.0])]
            peak_freq = freq_pos_khz[peak_idx]
            peak_power = power_pos[peak_idx]
            
            # Fit Gaussian around the peak

            fit_range = 1.0  # kHz
            fit_mask = np.abs(freq_pos_khz - peak_freq) < fit_range
            if np.sum(fit_mask) > 4:
                popt, _ = fit_peak(freq_pos_khz[fit_mask], power_pos[fit_mask], peak_freq, fit_range)
                fitted_freq = popt[1]
                fitted_power = popt[0] + popt[3]

        
        # Find peak closest to 11 kHz (2nd harmonic)
        second_harmonic_candidates = peaks[np.abs(peak_freqs - 11.0) < 2.0]
        if len(second_harmonic_candidates) > 0:
            peak_idx = second_harmonic_candidates[np.argmax(peak_powers[np.abs(peak_freqs - 11.0) < 2.0])]
            peak_freq = freq_pos_khz[peak_idx]
            peak_power = power_pos[peak_idx]
            
            # Fit Gaussian around the peak
            try:
                fit_range = 1.0  # kHz
                fit_mask = np.abs(freq_pos_khz - peak_freq) < fit_range
                if np.sum(fit_mask) > 4:
                    popt, _ = fit_peak(freq_pos_khz[fit_mask], power_pos[fit_mask], peak_freq, fit_range)
                    fitted_freq = popt[1]
                    fitted_power = popt[0] + popt[3]
                    
                    # Annotate with arrow
                    ax.annotate('2nd harmonic', xy=(fitted_freq+0.1, fitted_power+0.5), 
                               xytext=(fitted_freq +2, fitted_power + 20),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                               fontsize=10, color='black', ha='center')
                else:
                    # Fallback annotation
                    ax.annotate('2nd harmonic', xy=(peak_freq+0.1, peak_power), 
                               xytext=(peak_freq +2, peak_power + 20),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                               fontsize=10, color='black', ha='center')
            except:
                # Simple annotation if fitting fails
                ax.annotate('2nd harmonic', xy=(peak_freq+0.1, peak_power), 
                           xytext=(peak_freq +2, peak_power + 20),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                           fontsize=10, color='black', ha='center')

def create_g2_plot(data, condition, ax):
    """Create g2 correlation plot on the given axis."""
    if condition == 'with_feedback':
        color = '#D55E00'  # Orange for experiment
        label = 'MLP (Expt.)'
        marker = 's'  
    else:
        color = '#0072B2'  # Blue for no feedback
        label = 'No Feedback'
        marker = 's'
    
    # Get data for this condition
    tau = data['tau_values']
    g2 = data['avg_g2']
    sem = data['sem_g2']
    tau_units = data['tau_units']
    
    # Plot with error bars, markers, and connecting lines
    ax.errorbar(tau, g2, yerr=sem, 
                fmt=f'{marker}-',
                markeredgecolor=color,
                markerfacecolor=color,
                color=color,
                ecolor=color,
                label=label,
                linewidth=1.5,
                markersize=4,
                capsize=2,
                capthick=0.5,
                alpha=0.8)
    
    ax.set_xlabel(f"Delay time, τ ({tau_units})")
    ax.set_ylabel("g²(τ)")
    ax.grid(True)
    
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    if len(tau) > 1:
        ax.set_xlim(0, tau[-1])

def create_inset_plot(ax):
    """Create the inset plot (Figure1_insert) on the given axis."""
    inset_df = pd.read_csv(PACKAGE_ROOT / FIG1_CFG["inset_csv"])
    counts = inset_df["plus_spcm_counts"].to_numpy(dtype=float)
    times = inset_df["time_ms"].to_numpy(dtype=float)
    empty_cavity_mean = float(inset_df["empty_cavity_mean"].dropna().iloc[0])

    # Plot parameters
    linewidth = 2.0
    curve_color_counts = '#0072B2'

    # Plot the trace (first 140 points to match original)
    plot_points = min(140, len(counts))
    ax.plot(
        times[:plot_points], counts[:plot_points],
        color=curve_color_counts,
        linewidth=linewidth,
        zorder=0
    )
    
    print(f"Empty cavity mean: {empty_cavity_mean}")
    
    # Add horizontal line for empty cavity mean
    if empty_cavity_mean is not None:
        ax.axhline(y=empty_cavity_mean, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='Empty cavity mean')
    
    ax.set_xlim(0, 2.75)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Photon counts', color='black')
    ax.tick_params(axis='both', which='both',
                    direction='in', top=True, right=False)
    ax.set_yticks([tick for tick in ax.get_yticks() if tick != 0 and tick != 50])

def plot_figure1():
    """Generate Figure 1 with arranged subplots."""
    
    # --- Setup Plot ---
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
        'figure.figsize': (10, 5)
    })

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    print("Generating panel (a) - Inset plot...")
    create_inset_plot(ax_a)

    # Generate panel (b) - Fourier transform analysis
    print("Generating panel (b) - Fourier transform analysis...")
    fourier_data = {}
    results = process_folder_for_fourier(
        None,
        THRESHOLD,
        FILTER_STEP,
        TARGET_STEP,
        WINDOW_SIZE,
        SMOOTH_WIDTH
    )
    if results:
        fourier_data["without_feedback"] = results
    if fourier_data.get("without_feedback"):
        create_fourier_plot(fourier_data["without_feedback"], 'without_feedback', ax_b)
    else:
        ax_b.text(0.5, 0.5, 'Fourier data not available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax_b.transAxes)
        ax_b.set_xlabel("Frequency (kHz)")
        ax_b.set_ylabel("Power Spectral Density")
        ax_b.grid(True)

    # Generate panel (c) - g2 correlation analysis
    print("Generating panel (c) - g2 correlation analysis...")
    g2_data = {}
    results = process_folder_for_g2(
        None,
        THRESHOLD,
        FILTER_STEP,
        TARGET_STEP,
        WINDOW_SIZE,
        SMOOTH_WIDTH,
        MAX_TAU
    )
    if results:
        g2_data["without_feedback"] = results
    if g2_data.get("without_feedback"):
        create_g2_plot(g2_data["without_feedback"], 'without_feedback', ax_c)
    else:
        ax_c.text(0.5, 0.5, 'g2 data not available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax_c.transAxes)
        ax_c.set_xlabel("τ (µs)")
        ax_c.set_ylabel("g²(τ)")
        ax_c.grid(True)

    ax_a.text(-0.08, 1.04, '(a)', transform=ax_a.transAxes,
              fontsize=18, fontweight='bold', va='top', ha='right')
    ax_b.text(-0.18, 1.08, '(b)', transform=ax_b.transAxes,
              fontsize=18, fontweight='bold', va='top', ha='right')
    ax_c.text(-0.18, 1.08, '(c)', transform=ax_c.transAxes,
              fontsize=18, fontweight='bold', va='top', ha='right')

    plt.tight_layout(pad=2.0, h_pad=1.8, w_pad=2.5)
    output_filename = OUTPUT_DIR / FIG1_CFG["output_pdf"].replace(".pdf", ".png")
    pdf_filename = OUTPUT_DIR / FIG1_CFG["output_pdf"]
    plt.savefig(output_filename, dpi=800, bbox_inches='tight')
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Figure saved to {output_filename} and {pdf_filename}")
    plt.close()

if __name__ == "__main__":
    plot_figure1() 