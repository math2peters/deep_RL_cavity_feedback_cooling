from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, ConnectionPatch, FancyBboxPatch
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
from mpl_aps_style import LABEL_SIZE, PANEL_LABEL_SIZE, TEXT_WIDTH_IN, TICK_SIZE, add_panel_label, apply_paper_style

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
        tau_units = "μs"
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
                linewidth=0.8,
                elinewidth=0.5,
                markersize=2,
                capsize=1,
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
                               arrowprops=dict(arrowstyle='->', color='black', lw=0.6, shrinkA=1, shrinkB=1),
                               fontsize=TICK_SIZE, color='black', ha='center')
                else:
                    # Fallback annotation
                    ax.annotate('2nd harmonic', xy=(peak_freq+0.1, peak_power), 
                               xytext=(peak_freq +2, peak_power + 20),
                               arrowprops=dict(arrowstyle='->', color='black', lw=0.6, shrinkA=1, shrinkB=1),
                               fontsize=TICK_SIZE, color='black', ha='center')
            except:
                # Simple annotation if fitting fails
                ax.annotate('2nd harmonic', xy=(peak_freq+0.1, peak_power), 
                           xytext=(peak_freq +2, peak_power + 20),
                           arrowprops=dict(arrowstyle='->', color='black', lw=0.6, shrinkA=1, shrinkB=1),
                           fontsize=TICK_SIZE, color='black', ha='center')

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
                linewidth=0.8,
                elinewidth=0.5,
                markersize=2,
                capsize=1,
                capthick=0.5,
                alpha=0.8)
    
    ax.set_xlabel(rf"Delay time, $\tau$ ({tau_units})")
    ax.set_ylabel(r"$g^{(2)}(\tau)$")
    ax.grid(True)
    
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
    if len(tau) > 1:
        ax.set_xlim(0, tau[-1])

def create_inset_plot(ax):
    """Create the inset plot (Figure1_insert) on the given axis."""
    inset_df = pd.read_csv(PACKAGE_ROOT / FIG1_CFG["inset_csv"])
    counts = inset_df["plus_spcm_counts"].to_numpy(dtype=float)
    times = inset_df["time_ms"].to_numpy(dtype=float)
    empty_cavity_mean = float(inset_df["empty_cavity_mean"].dropna().iloc[0])

    # Plot parameters
    linewidth = 1.0
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
                  linewidth=0.8, alpha=0.7, label='Empty cavity mean')
    
    ax.set_xlim(0, 2.75)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Photon counts', color='black')
    ax.tick_params(axis='both', which='both',
                    direction='in', top=True, right=False)
    y_limits = ax.get_ylim()
    ax.set_yticks(np.arange(10, y_limits[1], 10))
    ax.set_ylim(y_limits)
    return times[:plot_points], counts[:plot_points]

# --- Panel (a): schematic artwork, level diagram, and atom-position insets ---

# The schematic is the original PowerPoint artwork with its level-diagram bubble removed;
# the bubble is redrawn below. Artwork coordinates are points of that export (y down).
SCHEMATIC_SIZE = (973.0, 398.0)
ATOM_EDGES = ((142.4, 226.4), (154.7, 226.4))  # where the bubble connectors meet the atom
CONNECTOR_STARTS = ((80.4, 163.6), (216.4, 163.6))
BUBBLE_X = (6.0, 219.0)  # clear of the mirror, which starts at x = 224
BUBBLE_BOTTOM = 163.6

PROBE_COLOR = '#ED7D31'
CAVITY_COLOR = '#2E75B6'
BUBBLE_COLOR = '#9DB4DD'
INSET_ARROW_COLOR = '#ED7D31'

# Points of the panel (b) trace highlighted by the atom-position insets: a transmission
# maximum (atom at the edge of the cavity mode) and a minimum (atom at the mode center).
PEAK_WINDOW_MS = (0.40, 0.48)
VALLEY_WINDOW_MS = (2.33, 2.38)


def draw_level_diagram(ax, width, height):
    """Two-level atom with probe, cavity, and atomic frequencies (not to scale)."""
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')

    y0, y1 = 0.08 * height, 0.70 * height  # |0>, |1>
    yp = y1 + 0.11 * height  # probe = cavity frequency; Delta is exaggerated
    x_lines = (0.20 * width, 0.98 * width)
    x_w0, x_wp, x_wc = 0.29 * width, 0.53 * width, 0.77 * width

    for y, ket in ((y0, r'$|0\rangle$'), (y1, r'$|1\rangle$')):
        ax.plot(x_lines, [y, y], color='black', lw=0.9, solid_capstyle='butt')
        ax.text(x_lines[0] - 1.5, y, ket, ha='right', va='center', fontsize=LABEL_SIZE)
    ax.plot([x_wp - 0.10 * width, x_lines[1]], [yp, yp], color='0.35', lw=0.6, ls=(0, (2.0, 1.5)))

    def up_arrow(x, y_start, y_end, color, lw):
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=5,
                                    shrinkA=0, shrinkB=0))

    up_arrow(x_w0, y0, y1, 'black', 0.7)
    up_arrow(x_wp, y0, yp, PROBE_COLOR, 1.1)
    up_arrow(x_wc, y0, yp, CAVITY_COLOR, 1.1)
    y_mid = 0.5 * (y0 + y1)
    for x, text, color in ((x_w0, r'$\omega_0$', 'black'),
                           (x_wp, r'$\omega_p$', PROBE_COLOR),
                           (x_wc, r'$\omega_c$', CAVITY_COLOR)):
        ax.text(x + 1.5, y_mid, text, ha='left', va='center', fontsize=LABEL_SIZE, color=color)

    # Delta: gap between |1> and the probe frequency, labelled right next to it
    x_delta = 0.93 * width
    ax.annotate('', xy=(x_delta, yp), xytext=(x_delta, y1),
                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=0.5, mutation_scale=3,
                                shrinkA=0, shrinkB=0))
    ax.text(x_delta + 1.0, yp + 2.0, r'$\Delta \equiv \omega_p - \omega_0$',
            ha='right', va='bottom', fontsize=TICK_SIZE)


def draw_atom_in_mode(ax, atom_z):
    """Atom (purple) in the tweezer (red) inside the cavity mode (yellow)."""
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect('equal')
    ax.axis('off')
    clip = Circle((0, 0), 1.0, transform=ax.transData)

    zz, xx = np.mgrid[-1:1:241j, -1:1:241j]
    beam_w = 0.14 * np.sqrt(1 + (zz / 0.45) ** 2)
    beam = (0.14 / beam_w) ** 0.5 * np.exp(-2 * xx ** 2 / beam_w ** 2)
    rgba = np.zeros(zz.shape + (4,))
    rgba[..., :3] = to_rgb('#F2545B')
    rgba[..., 3] = 0.9 * beam
    beam_im = ax.imshow(rgba, extent=(-1, 1, -1, 1), origin='lower', interpolation='bilinear', zorder=1)
    beam_im.set_clip_path(clip)

    x = np.linspace(-1, 1, 101)
    for sign in (1, -1):
        mode_edge, = ax.plot(x, sign * 0.36 * np.sqrt(1 + (x / 1.2) ** 2),
                             color='#FFC000', lw=0.8, zorder=2)
        mode_edge.set_clip_path(clip)

    ax.add_patch(Circle((0, atom_z), 0.13, facecolor='#8E92D0', edgecolor='#5A5E9E', lw=0.4, zorder=3))
    ax.add_patch(Circle((-0.04, atom_z + 0.04), 0.05, facecolor='white', edgecolor='none', alpha=0.6, zorder=3))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor='#2F5290', lw=0.7, zorder=4))


def place_axes(fig, left_in, bottom_in, width_in, height_in, **kwargs):
    fig_w, fig_h = fig.get_size_inches()
    return fig.add_axes([left_in / fig_w, bottom_in / fig_h, width_in / fig_w, height_in / fig_h], **kwargs)


def plot_figure1():
    """Generate Figure 1: schematic (a), example trace (b), PSD (c), and g2 (d)."""
    apply_paper_style()

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.50))
    fig_w, fig_h = fig.get_size_inches()  # some GUI backends round to whole pixels

    # (a) schematic, bottom-left, with free space above it for the enlarged level diagram
    schematic_w = 4.30
    scale = schematic_w / SCHEMATIC_SIZE[0]  # inches per artwork point
    top_art = SCHEMATIC_SIZE[1] - fig_h / scale  # artwork y at the top edge of the figure
    ax_s = place_axes(fig, 0, 0, schematic_w, fig_h)
    ax_s.imshow(plt.imread(PACKAGE_ROOT / FIG1_CFG["schematic_png"]),
                extent=(0, SCHEMATIC_SIZE[0], SCHEMATIC_SIZE[1], 0), interpolation='antialiased', zorder=0)
    ax_s.set_xlim(0, SCHEMATIC_SIZE[0])
    ax_s.set_ylim(SCHEMATIC_SIZE[1], top_art)
    ax_s.axis('off')

    pt = 1 / (72 * scale)  # one printed point in artwork units
    bubble_top = top_art + 14 * pt
    ax_s.add_patch(FancyBboxPatch(
        (BUBBLE_X[0], bubble_top), BUBBLE_X[1] - BUBBLE_X[0], BUBBLE_BOTTOM - bubble_top,
        boxstyle=f'round,pad=0,rounding_size={6 * pt}', facecolor='white', edgecolor=BUBBLE_COLOR,
        lw=0.9, ls=(0, (1, 1)), zorder=2))
    for start, end in zip(CONNECTOR_STARTS, ATOM_EDGES):
        ax_s.plot([start[0], end[0]], [start[1], end[1]], color=BUBBLE_COLOR, lw=0.7, ls=(0, (1, 1)), zorder=2)

    pad = 3 * pt
    lvl_w_in = (BUBBLE_X[1] - BUBBLE_X[0] - 2 * pad) * scale
    lvl_h_in = (BUBBLE_BOTTOM - bubble_top - 2 * pad) * scale
    ax_lvl = place_axes(fig, (BUBBLE_X[0] + pad) * scale, (SCHEMATIC_SIZE[1] - BUBBLE_BOTTOM + pad) * scale,
                        lvl_w_in, lvl_h_in)
    draw_level_diagram(ax_lvl, lvl_w_in * 72, lvl_h_in * 72)
    fig.text(0, 1, '(a)', ha='left', va='top', fontsize=PANEL_LABEL_SIZE, fontweight='bold')

    # (b) example trace and (c, d) averaged PSD and g2 in the right column
    right = schematic_w + 0.08
    ylabel_room = 0.34
    ax_b = place_axes(fig, right + ylabel_room, 1.58, fig_w - right - ylabel_room - 0.03, 0.75)
    width_cd = (fig_w - right - 2 * ylabel_room - 0.03 - 0.02) / 2
    ax_c = place_axes(fig, right + ylabel_room, 0.28, width_cd, 0.62)
    ax_d = place_axes(fig, right + 2 * ylabel_room + width_cd + 0.02, 0.28, width_cd, 0.62)

    print("Generating panel (b) - Example trace...")
    times, counts = create_inset_plot(ax_b)

    print("Generating panel (c) - Fourier transform analysis...")
    results = process_folder_for_fourier(
        None,
        THRESHOLD,
        FILTER_STEP,
        TARGET_STEP,
        WINDOW_SIZE,
        SMOOTH_WIDTH
    )
    if results:
        create_fourier_plot(results, 'without_feedback', ax_c)
    else:
        ax_c.text(0.5, 0.5, 'Fourier data not available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax_c.transAxes)
        ax_c.set_xlabel("Frequency (kHz)")
        ax_c.set_ylabel("Power Spectral Density")
        ax_c.grid(True)

    print("Generating panel (d) - g2 correlation analysis...")
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
        create_g2_plot(results, 'without_feedback', ax_d)
    else:
        ax_d.text(0.5, 0.5, 'g2 data not available',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax_d.transAxes)
        ax_d.set_xlabel(r"$\tau$ (μs)")
        ax_d.set_ylabel(r"$g^{(2)}(\tau)$")
        ax_d.grid(True)

    for ax, label in ((ax_b, '(b)'), (ax_c, '(c)'), (ax_d, '(d)')):
        add_panel_label(ax, label, dx=-ylabel_room * 72 + 1)

    # atom-position insets under (b), pointing at a transmission maximum and minimum
    inset_d = 0.42
    b_left, b_width = ax_b.get_position().x0 * fig_w, ax_b.get_position().width * fig_w
    for window, pick, atom_z, x_frac in ((PEAK_WINDOW_MS, np.argmax, 0.36, 0.06),
                                         (VALLEY_WINDOW_MS, np.argmin, 0.0, 0.90)):
        in_window = np.flatnonzero((times >= window[0]) & (times <= window[1]))
        target = in_window[pick(counts[in_window])]
        ax_in = place_axes(fig, b_left + x_frac * b_width - inset_d / 2, 1.0, inset_d, inset_d)
        draw_atom_in_mode(ax_in, atom_z)
        ax_b.plot(times[target], counts[target], 'o', color=INSET_ARROW_COLOR, ms=3, zorder=5, clip_on=False)
        fig.add_artist(ConnectionPatch(
            xyA=(times[target], counts[target]), coordsA=ax_b.transData,
            xyB=(0, 1.0), coordsB=ax_in.transData,
            arrowstyle='-|>', mutation_scale=6, color=INSET_ARROW_COLOR, lw=0.9, shrinkA=1.5, shrinkB=1.0,
            zorder=6))

    output_filename = OUTPUT_DIR / FIG1_CFG["output_pdf"].replace(".pdf", ".png")
    pdf_filename = OUTPUT_DIR / FIG1_CFG["output_pdf"]
    plt.savefig(output_filename, dpi=600)
    plt.savefig(pdf_filename, dpi=600)
    print(f"Figure saved to {output_filename} and {pdf_filename}")
    plt.close()

if __name__ == "__main__":
    plot_figure1() 