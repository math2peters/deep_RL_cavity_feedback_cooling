
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.interpolate import interp1d
import warnings

# Import functions from the main simulation script
from simulation_comparison_script import (
    load_sweep_results, 
    load_20_step_energy_from_traces,
    merge_energy_data_with_sweep_results,
    fit_cooling_timescale_from_trace,
    merge_cooling_data_with_sweep_results,
    get_model_styling
)


def find_peak_performance(detuning_values, performance_values, metric_type):
    """
    Find the peak performance point for a given metric
    
    Args:
        detuning_values: Array of detuning values
        performance_values: Array of performance values
        metric_type: 'maximize' for survival probability, 'minimize' for cooling/energy
        
    Returns:
        Tuple of (peak_detuning, peak_performance, peak_index)
    """
    # Remove NaN values
    valid_mask = ~np.isnan(performance_values)
    if not valid_mask.any():
        return np.nan, np.nan, -1
    
    valid_detuning = detuning_values[valid_mask]
    valid_performance = performance_values[valid_mask]
    
    if metric_type == 'maximize':
        peak_idx = np.argmax(valid_performance)
    else:  # minimize
        peak_idx = np.argmin(valid_performance)
    
    return valid_detuning[peak_idx], valid_performance[peak_idx], peak_idx


def find_performance_range(detuning_values, performance_values, peak_detuning, peak_performance, metric_type, performance_threshold=0.5):
    """
    Find the detuning range where performance reaches specified threshold of peak value
    
    Args:
        detuning_values: Array of detuning values
        performance_values: Array of performance values
        peak_detuning: Detuning at peak performance
        peak_performance: Peak performance value
        metric_type: 'maximize' or 'minimize'
        performance_threshold: Fraction of peak performance (e.g., 0.5 for 50%, 0.75 for 75%)
        
    Returns:
        Tuple of (left_detuning, right_detuning, range_width)
    """
    # Remove NaN values and sort by detuning
    valid_mask = ~np.isnan(performance_values)
    if not valid_mask.any() or valid_mask.sum() < 3:
        return np.nan, np.nan, np.nan
    
    valid_detuning = detuning_values[valid_mask]
    valid_performance = performance_values[valid_mask]
    
    # Sort by detuning
    sort_idx = np.argsort(valid_detuning)
    sorted_detuning = valid_detuning[sort_idx]
    sorted_performance = valid_performance[sort_idx]
    
    # Calculate target performance based on threshold
    if metric_type == 'maximize':
        # For survival probability: threshold fraction of peak value
        target_performance = peak_performance * performance_threshold
    else:
        # For cooling timescale/energy: threshold increase from peak (minimum)
        # If peak is 100 and threshold is 0.5, then target would be 150 (50% worse)
        degradation_factor = 1 + (1 - performance_threshold)
        target_performance = peak_performance * degradation_factor
    
    try:
        # Create interpolation function
        # Need to handle potential non-monotonic data
        interp_func = interp1d(sorted_detuning, sorted_performance, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Check performance at boundaries
        min_detuning = sorted_detuning.min()
        max_detuning = sorted_detuning.max()
        performance_at_min = interp_func(min_detuning)
        performance_at_max = interp_func(max_detuning)
        
        # Determine if boundaries are above threshold
        if metric_type == 'maximize':
            min_above_threshold = performance_at_min >= target_performance
            max_above_threshold = performance_at_max >= target_performance
        else:
            min_above_threshold = performance_at_min <= target_performance
            max_above_threshold = performance_at_max <= target_performance
        
        # Initialize range boundaries
        left_crossing = min_detuning if min_above_threshold else np.nan
        right_crossing = max_detuning if max_above_threshold else np.nan
        
        # Find crossings with target performance
        # We'll sample the interpolated function and find crossings
        detuning_fine = np.linspace(min_detuning, max_detuning, 1000)
        performance_fine = interp_func(detuning_fine)
        
        # Find crossings
        if metric_type == 'maximize':
            # Find where performance drops below target
            below_target = performance_fine < target_performance
        else:
            # Find where performance rises above target
            below_target = performance_fine > target_performance
        
        # Find transitions
        crossings = np.where(np.diff(below_target.astype(int)))[0]
        
        if len(crossings) > 0:
            # Get all crossing points
            crossing_detunings = detuning_fine[crossings]
            
            # Find leftmost and rightmost crossings relative to peak
            left_crossings = [d for d in crossing_detunings if d < peak_detuning]
            right_crossings = [d for d in crossing_detunings if d > peak_detuning]
            
            # Update boundaries if we found internal crossings
            if left_crossings and not min_above_threshold:
                left_crossing = max(left_crossings)  # Closest to peak on left
            if right_crossings and not max_above_threshold:
                right_crossing = min(right_crossings)  # Closest to peak on right
        
        # If we still don't have valid boundaries, try fallback approach
        if np.isnan(left_crossing) or np.isnan(right_crossing):
            peak_idx = np.argmin(np.abs(detuning_fine - peak_detuning))
            
            if np.isnan(left_crossing):
                # Left side
                left_side_performance = performance_fine[:peak_idx]
                left_side_detuning = detuning_fine[:peak_idx]
                if len(left_side_performance) > 0:
                    left_idx = np.argmin(np.abs(left_side_performance - target_performance))
                    left_crossing = left_side_detuning[left_idx]
                else:
                    left_crossing = min_detuning
            
            if np.isnan(right_crossing):
                # Right side
                right_side_performance = performance_fine[peak_idx:]
                right_side_detuning = detuning_fine[peak_idx:]
                if len(right_side_performance) > 0:
                    right_idx = np.argmin(np.abs(right_side_performance - target_performance))
                    right_crossing = right_side_detuning[right_idx]
                else:
                    right_crossing = max_detuning
        
        # Calculate range width
        if not (np.isnan(left_crossing) or np.isnan(right_crossing)):
            range_width = right_crossing - left_crossing
            return left_crossing, right_crossing, range_width
        else:
            return np.nan, np.nan, np.nan
            
    except Exception as e:
        print(f"Error in interpolation: {e}")
        return np.nan, np.nan, np.nan


def calculate_performance_metric(results, param_name='detuning', performance_threshold=0.5):
    """
    Calculate the performance metric for each model across all three metrics
    
    Args:
        results: Dictionary of DataFrames containing results for each model
        param_name: Parameter name ('detuning')
        performance_threshold: Fraction of peak performance for range calculation (e.g., 0.5 for 50%)
        
    Returns:
        Dictionary with performance metrics for each model
    """
    if param_name != 'detuning':
        print("Warning: Performance metric is designed for detuning sweeps only")
        return {}
    
    # Define the three metrics to analyze
    metrics = [
        ('fraction_trapped', 'maximize', 'Survival Probability'),
        ('cooling_timescale', 'minimize', 'Cooling Timescale'),
        ('20_step_energy', 'minimize', '20-Step Energy')
    ]
    
    # Store results for each model and metric
    model_metrics = {}
    global_ranges = {}  # Store global min/max detuning for each metric
    global_peaks = {}   # Store global best performance for each metric
    max_widths = {}     # Store largest width for each metric
    
    print("Analyzing peak performance and ranges...")
    
    # First pass: find global ranges, peaks, and maximum widths
    for metric_name, metric_type, metric_label in metrics:
        all_detunings = []
        all_peaks = []
        all_widths = []
        
        for model_folder, df in results.items():
            if metric_name not in df.columns:
                continue
                
            # Get valid data for this metric
            valid_mask = ~df[metric_name].isna()
            if not valid_mask.any():
                continue
                
            valid_df = df[valid_mask]
            detuning_vals = valid_df[param_name].values
            performance_vals = valid_df[metric_name].values
            
            all_detunings.extend(detuning_vals)
            
            # Find peak for this model
            peak_det, peak_perf, _ = find_peak_performance(detuning_vals, performance_vals, metric_type)
            if not np.isnan(peak_perf):
                all_peaks.append(peak_perf)
                
                # Find range width for this model
                left_det, right_det, range_width = find_performance_range(
                    detuning_vals, performance_vals, peak_det, peak_perf, metric_type, performance_threshold
                )
                if not np.isnan(range_width):
                    all_widths.append(range_width)
        
        if all_detunings and all_peaks:
            global_ranges[metric_name] = (min(all_detunings), max(all_detunings))
            if metric_type == 'maximize':
                global_peaks[metric_name] = max(all_peaks)
            else:
                global_peaks[metric_name] = min(all_peaks)
        
        if all_widths:
            max_widths[metric_name] = max(all_widths)
        else:
            max_widths[metric_name] = 1.0  # fallback
        
        print(f"{metric_label}: Global detuning range = {global_ranges.get(metric_name, (np.nan, np.nan))}")
        print(f"{metric_label}: Global best performance = {global_peaks.get(metric_name, np.nan)}")
        print(f"{metric_label}: Maximum width = {max_widths.get(metric_name, np.nan)}")
    
    # Second pass: calculate metrics for each model
    for model_folder, df in results.items():
        styling = get_model_styling(model_folder)
        model_label = styling['label']
        
        print(f"\nAnalyzing {model_label}...")
        
        model_metrics[model_folder] = {
            'label': model_label,
            'metric_scores': {},
            'details': {}
        }
        
        for metric_name, metric_type, metric_label in metrics:
            if metric_name not in df.columns:
                print(f"  {metric_label}: No data available")
                model_metrics[model_folder]['metric_scores'][metric_name] = np.nan
                continue
            
            # Get valid data for this metric
            valid_mask = ~df[metric_name].isna()
            if not valid_mask.any():
                print(f"  {metric_label}: No valid data")
                model_metrics[model_folder]['metric_scores'][metric_name] = np.nan
                continue
            
            valid_df = df[valid_mask]
            detuning_vals = valid_df[param_name].values
            performance_vals = valid_df[metric_name].values
            
            # Find peak performance
            peak_det, peak_perf, peak_idx = find_peak_performance(detuning_vals, performance_vals, metric_type)
            
            if np.isnan(peak_perf):
                print(f"  {metric_label}: Could not find peak")
                model_metrics[model_folder]['metric_scores'][metric_name] = np.nan
                continue
            
            # Find performance range
            left_det, right_det, range_width = find_performance_range(
                detuning_vals, performance_vals, peak_det, peak_perf, metric_type, performance_threshold
            )
            
            if np.isnan(range_width):
                print(f"  {metric_label}: Could not determine 50% range")
                model_metrics[model_folder]['metric_scores'][metric_name] = np.nan
                continue
            
            # Calculate metric components
            max_width = max_widths[metric_name]
            range_fraction = range_width / max_width if max_width > 0 else np.nan
            
            global_best = global_peaks[metric_name]
            if metric_type == 'maximize':
                performance_fraction = peak_perf / global_best if global_best > 0 else np.nan
            else:
                performance_fraction = global_best / peak_perf if peak_perf > 0 else np.nan
            
            # Final metric: geometric mean of range_fraction and performance_fraction
            metric_score = np.sqrt(range_fraction * performance_fraction) if not (np.isnan(range_fraction) or np.isnan(performance_fraction)) else np.nan
            
            model_metrics[model_folder]['metric_scores'][metric_name] = metric_score
            model_metrics[model_folder]['details'][metric_name] = {
                'peak_detuning': peak_det,
                'peak_performance': peak_perf,
                'left_50_detuning': left_det,
                'right_50_detuning': right_det,
                'range_width': range_width,
                'range_fraction': range_fraction,
                'performance_fraction': performance_fraction,
                'metric_score': metric_score
            }
            
            threshold_percent = int(performance_threshold * 100)
            print(f"  {metric_label}:")
            print(f"    Peak: {peak_perf:.3f} at {peak_det:.1f} MHz")
            print(f"    {threshold_percent}% range: {left_det:.1f} to {right_det:.1f} MHz (width: {range_width:.1f})")
            print(f"    Range fraction: {range_fraction:.3f}")
            print(f"    Performance fraction: {performance_fraction:.3f}")
            print(f"    Metric score: {metric_score:.4f}")
    
    # Third pass: calculate overall performance metric (average of all three)
    for model_folder in model_metrics.keys():
        scores = model_metrics[model_folder]['metric_scores']
        metric_values = [scores.get(metric_name, np.nan) for metric_name, _, _ in metrics]
        
        # Calculate average only if all metrics are available
        if all(not np.isnan(val) for val in metric_values):
            overall_metric = np.mean(metric_values)
        else:
            overall_metric = np.nan
        
        model_metrics[model_folder]['overall_metric'] = overall_metric
    
    return model_metrics


def print_performance_summary(model_metrics, performance_threshold=0.5):
    """
    Print a summary of performance metrics for all models
    
    Args:
        model_metrics: Dictionary with performance metrics for each model
        performance_threshold: Threshold used for range calculation
    """
    print("\n" + "="*80)
    print("PERFORMANCE METRIC SUMMARY")
    print("="*80)
    
    # Sort models by overall metric (highest first)
    sorted_models = sorted(
        [(folder, data) for folder, data in model_metrics.items()],
        key=lambda x: x[1]['overall_metric'] if not np.isnan(x[1]['overall_metric']) else -1,
        reverse=True
    )
    
    # Table 1: Width Metrics (Range Fractions)
    threshold_percent = int(performance_threshold * 100)
    print(f"\n1. WIDTH METRICS (Range at {threshold_percent}% of Peak / Maximum Width)")
    print("="*80)
    print(f"{'Model':<25} {'Survival':<12} {'Cooling':<12} {'Energy':<12} {'Overall':<12}")
    print("-" * 80)
    
    for model_folder, data in sorted_models:
        label = data['label']
        details = data.get('details', {})
        
        survival_width = details.get('fraction_trapped', {}).get('range_fraction', np.nan)
        cooling_width = details.get('cooling_timescale', {}).get('range_fraction', np.nan)
        energy_width = details.get('20_step_energy', {}).get('range_fraction', np.nan)
        
        # Calculate overall width metric (average of all three)
        width_values = [survival_width, cooling_width, energy_width]
        if all(not np.isnan(val) for val in width_values):
            overall_width = np.mean(width_values)
        else:
            overall_width = np.nan
        
        print(f"{label:<25} {survival_width:<12.4f} {cooling_width:<12.4f} {energy_width:<12.4f} {overall_width:<12.4f}")
    
    # Table 2: Peak Performance Metrics (Performance Fractions)
    print("\n2. PEAK PERFORMANCE METRICS (Model Peak / Global Best)")
    print("="*80)
    print(f"{'Model':<25} {'Survival':<12} {'Cooling':<12} {'Energy':<12} {'Overall':<12}")
    print("-" * 80)
    
    for model_folder, data in sorted_models:
        label = data['label']
        details = data.get('details', {})
        
        survival_perf = details.get('fraction_trapped', {}).get('performance_fraction', np.nan)
        cooling_perf = details.get('cooling_timescale', {}).get('performance_fraction', np.nan)
        energy_perf = details.get('20_step_energy', {}).get('performance_fraction', np.nan)
        
        # Calculate overall performance metric (average of all three)
        perf_values = [survival_perf, cooling_perf, energy_perf]
        if all(not np.isnan(val) for val in perf_values):
            overall_perf = np.mean(perf_values)
        else:
            overall_perf = np.nan
        
        print(f"{label:<25} {survival_perf:<12.4f} {cooling_perf:<12.4f} {energy_perf:<12.4f} {overall_perf:<12.4f}")
    
    # Table 3: Combined Metrics (Width × Peak Performance)
    print("\n3. COMBINED METRICS (Width × Peak Performance)")
    print("="*80)
    print(f"{'Model':<25} {'Survival':<12} {'Cooling':<12} {'Energy':<12} {'Overall':<12}")
    print("-" * 80)
    
    for model_folder, data in sorted_models:
        label = data['label']
        scores = data['metric_scores']
        overall = data['overall_metric']
        
        survival_score = scores.get('fraction_trapped', np.nan)
        cooling_score = scores.get('cooling_timescale', np.nan)
        energy_score = scores.get('20_step_energy', np.nan)
        
        print(f"{label:<25} {survival_score:<12.4f} {cooling_score:<12.4f} {energy_score:<12.4f} {overall:<12.4f}")
    
    print("\nInterpretation:")
    print("- Width metrics: How robust is the performance relative to the best model (larger = more robust)")
    print("- Peak performance metrics: How good is the best performance (larger = better peak)")
    print("- Combined metrics: Width × Peak Performance (larger = better overall)")
    print("- Overall metric: Average of all three individual metrics")


def main():
    """Main function to calculate and report DETUNING performance metrics"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate DETUNING performance metrics')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Performance degradation threshold (e.g., 0.5 for 50%%, 0.75 for 75%%) [default: 0.5]')
    args = parser.parse_args()
    
    performance_threshold = args.threshold
    threshold_percent = int(performance_threshold * 100)
    
    # Get the script's directory as base
    script_dir = Path(__file__).parent
    
    # Load results for all models (detuning sweeps only)
    model_folders = ['mlp_sim', 'differentiator', 'mlp_experimental']
    
    print("="*60)
    print("DETUNING SWEEP PERFORMANCE METRIC ANALYSIS")
    print(f"Performance Threshold: {threshold_percent}% of peak")
    print("="*60)
    print("Loading detuning sweep data...")
    results = load_sweep_results(script_dir, model_folders, param_name='detuning')
    
    # Load 20-step energy data from energy traces
    print("Loading 20-step energy data...")
    energy_results = load_20_step_energy_from_traces(script_dir, model_folders, param_name='detuning')
    
    # Merge energy data with main results
    results = merge_energy_data_with_sweep_results(results, energy_results)
    
    # Fit cooling timescales from energy traces
    print("Fitting cooling timescales...")
    cooling_results = fit_cooling_timescale_from_trace(script_dir, model_folders, param_name='detuning')
    
    # Merge fitted cooling data with main results
    results = merge_cooling_data_with_sweep_results(results, cooling_results)
    
    if not results:
        print("No data found for detuning sweeps")
        return
    
    # Calculate performance metrics
    model_metrics = calculate_performance_metric(results, param_name='detuning', performance_threshold=performance_threshold)
    
    # Print summary
    print_performance_summary(model_metrics, performance_threshold)
    
    # Save detailed results to CSV
    output_file = script_dir / 'detuning_performance_metrics.csv'
    save_detailed_results(model_metrics, output_file)
    print(f"\nDetailed DETUNING results saved to: {output_file}")


def save_detailed_results(model_metrics, output_file):
    """
    Save detailed performance metrics to CSV file
    
    Args:
        model_metrics: Dictionary with performance metrics for each model
        output_file: Path to output CSV file
    """
    rows = []
    
    for model_folder, data in model_metrics.items():
        base_row = {
            'model_folder': model_folder,
            'model_label': data['label'],
            'overall_metric': data['overall_metric']
        }
        
        # Add individual metric scores
        for metric_name, score in data['metric_scores'].items():
            base_row[f'{metric_name}_score'] = score
            
            # Add detailed info if available
            if metric_name in data.get('details', {}):
                details = data['details'][metric_name]
                for detail_key, detail_val in details.items():
                    base_row[f'{metric_name}_{detail_key}'] = detail_val
        
        rows.append(base_row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main() 