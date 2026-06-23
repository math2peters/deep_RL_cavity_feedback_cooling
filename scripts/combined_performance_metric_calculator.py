
import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse

# Import the main calculation functions from both scripts
from detuning_performance_metric_calculator import (
    calculate_performance_metric as calc_detuning_metrics,
    load_sweep_results, 
    load_20_step_energy_from_traces,
    merge_energy_data_with_sweep_results,
    fit_cooling_timescale_from_trace,
    merge_cooling_data_with_sweep_results,
    get_model_styling
)

from photon_number_performance_metric_calculator import (
    calculate_performance_metric as calc_photon_metrics
)


def load_and_process_data(script_dir, model_folders, param_name):
    """
    Load and process data for a given parameter type
    
    Args:
        script_dir: Script directory path
        model_folders: List of model folders
        param_name: Parameter name ('detuning' or 'photon_number')
        
    Returns:
        Processed results dictionary
    """
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
    
    return results


def combine_metrics(detuning_metrics, photon_metrics):
    """
    Combine detuning and photon number metrics by taking their product
    
    Args:
        detuning_metrics: Metrics from detuning analysis
        photon_metrics: Metrics from photon number analysis
        
    Returns:
        Combined metrics dictionary
    """
    combined_metrics = {}
    
    # Get all model folders present in both analyses
    common_models = set(detuning_metrics.keys()) & set(photon_metrics.keys())
    
    for model_folder in common_models:
        det_data = detuning_metrics[model_folder]
        phot_data = photon_metrics[model_folder]
        
        combined_metrics[model_folder] = {
            'label': det_data['label'],
            'metric_scores': {},
            'details': {},
            'width_metrics': {},
            'peak_metrics': {}
        }
        
        # Metrics to combine
        metrics = ['fraction_trapped', 'cooling_timescale', '20_step_energy']
        
        # Combine individual metric scores (combined metrics)
        for metric in metrics:
            det_score = det_data['metric_scores'].get(metric, np.nan)
            phot_score = phot_data['metric_scores'].get(metric, np.nan)
            
            if not np.isnan(det_score) and not np.isnan(phot_score):
                combined_score = np.sqrt(det_score * phot_score)
            elif not np.isnan(det_score):
                combined_score = det_score
            elif not np.isnan(phot_score):
                combined_score = phot_score
            else:
                combined_score = np.nan
                
            combined_metrics[model_folder]['metric_scores'][metric] = combined_score
        
        # Combine width metrics
        for metric in metrics:
            det_width = det_data['details'].get(metric, {}).get('range_fraction', np.nan)
            phot_width = phot_data['details'].get(metric, {}).get('range_fraction', np.nan)
            
            if not np.isnan(det_width) and not np.isnan(phot_width):
                combined_width = np.sqrt(det_width * phot_width)
            elif not np.isnan(det_width):
                combined_width = det_width
            elif not np.isnan(phot_width):
                combined_width = phot_width
            else:
                combined_width = np.nan
                
            combined_metrics[model_folder]['width_metrics'][metric] = combined_width
        
        # Combine peak performance metrics
        for metric in metrics:
            det_peak = det_data['details'].get(metric, {}).get('performance_fraction', np.nan)
            phot_peak = phot_data['details'].get(metric, {}).get('performance_fraction', np.nan)
            
            if not np.isnan(det_peak) and not np.isnan(phot_peak):
                combined_peak = np.sqrt(det_peak * phot_peak)
            elif not np.isnan(det_peak):
                combined_peak = det_peak
            elif not np.isnan(phot_peak):
                combined_peak = phot_peak
            else:
                combined_peak = np.nan
                
            combined_metrics[model_folder]['peak_metrics'][metric] = combined_peak
        
        # Calculate overall metrics for each table (average of three metrics)
        # Overall combined metric
        combined_scores = [combined_metrics[model_folder]['metric_scores'][m] for m in metrics]
        if all(not np.isnan(val) for val in combined_scores):
            combined_metrics[model_folder]['overall_metric'] = np.mean(combined_scores)
        else:
            combined_metrics[model_folder]['overall_metric'] = np.nan
        
        # Overall width metric
        width_scores = [combined_metrics[model_folder]['width_metrics'][m] for m in metrics]
        if all(not np.isnan(val) for val in width_scores):
            combined_metrics[model_folder]['overall_width'] = np.mean(width_scores)
        else:
            combined_metrics[model_folder]['overall_width'] = np.nan
        
        # Overall peak metric
        peak_scores = [combined_metrics[model_folder]['peak_metrics'][m] for m in metrics]
        if all(not np.isnan(val) for val in peak_scores):
            combined_metrics[model_folder]['overall_peak'] = np.mean(peak_scores)
        else:
            combined_metrics[model_folder]['overall_peak'] = np.nan
    
    return combined_metrics


def print_combined_summary(combined_metrics, performance_threshold):
    """
    Print a summary of combined performance metrics for all models
    
    Args:
        combined_metrics: Dictionary with combined performance metrics for each model
        performance_threshold: Performance degradation threshold
    """
    print("\n" + "="*80)
    print("COMBINED PERFORMANCE METRIC SUMMARY")
    print("(Geometric Mean of Detuning and Photon Counts Sweeps)")
    print(f"Performance Threshold: {performance_threshold*100}% of peak")
    print("="*80)
    
    # Sort models by overall combined metric (highest first)
    sorted_models = sorted(
        [(folder, data) for folder, data in combined_metrics.items()],
        key=lambda x: x[1]['overall_metric'] if not np.isnan(x[1]['overall_metric']) else -1,
        reverse=True
    )
    
    # Table 1: Combined Width Metrics
    print("\n1. COMBINED WIDTH METRICS (Geometric Mean of Detuning & Photon Counts)")
    print("="*80)
    print(f"{'Model':<25} {'Survival':<12} {'Cooling':<12} {'Energy':<12} {'Overall':<12}")
    print("-" * 80)
    
    for model_folder, data in sorted_models:
        label = data['label']
        
        survival_width = data['width_metrics'].get('fraction_trapped', np.nan)
        cooling_width = data['width_metrics'].get('cooling_timescale', np.nan)
        energy_width = data['width_metrics'].get('20_step_energy', np.nan)
        overall_width = data['overall_width']
        
        print(f"{label:<25} {survival_width:<12.4f} {cooling_width:<12.4f} {energy_width:<12.4f} {overall_width:<12.4f}")
    
    # Table 2: Combined Peak Performance Metrics
    print("\n2. COMBINED PEAK PERFORMANCE METRICS (Geometric Mean of Detuning & Photon Counts)")
    print("="*80)
    print(f"{'Model':<25} {'Survival':<12} {'Cooling':<12} {'Energy':<12} {'Overall':<12}")
    print("-" * 80)
    
    for model_folder, data in sorted_models:
        label = data['label']
        
        survival_perf = data['peak_metrics'].get('fraction_trapped', np.nan)
        cooling_perf = data['peak_metrics'].get('cooling_timescale', np.nan)
        energy_perf = data['peak_metrics'].get('20_step_energy', np.nan)
        overall_perf = data['overall_peak']
        
        print(f"{label:<25} {survival_perf:<12.4f} {cooling_perf:<12.4f} {energy_perf:<12.4f} {overall_perf:<12.4f}")
    
    # Table 3: Combined Final Metrics
    print("\n3. COMBINED FINAL METRICS (Geometric Mean of Detuning & Photon Counts)")
    print("="*80)
    print(f"{'Model':<25} {'Survival':<12} {'Cooling':<12} {'Energy':<12} {'Overall':<12}")
    print("-" * 80)
    
    for model_folder, data in sorted_models:
        label = data['label']
        
        survival_score = data['metric_scores'].get('fraction_trapped', np.nan)
        cooling_score = data['metric_scores'].get('cooling_timescale', np.nan)
        energy_score = data['metric_scores'].get('20_step_energy', np.nan)
        overall_score = data['overall_metric']
        
        print(f"{label:<25} {survival_score:<12.4f} {cooling_score:<12.4f} {energy_score:<12.4f} {overall_score:<12.4f}")
    
    print("\nInterpretation:")
    print("- All metrics are the geometric mean of detuning and photon number sweep results")
    print("- Width metrics: How robust is the performance relative to the best model")
    print("- Peak performance metrics: How good is the best performance relative to global best")
    print("- Final metrics: Width × Peak Performance, geometric mean across parameter sweeps")
    print("- Overall metrics: Average of survival, cooling, and energy for each table")


def save_combined_results(combined_metrics, output_file):
    """
    Save combined performance metrics to CSV file
    
    Args:
        combined_metrics: Dictionary with combined performance metrics for each model
        output_file: Path to output CSV file
    """
    rows = []
    
    for model_folder, data in combined_metrics.items():
        row = {
            'model_folder': model_folder,
            'model_label': data['label'],
            'overall_combined_metric': data['overall_metric'],
            'overall_width_metric': data['overall_width'],
            'overall_peak_metric': data['overall_peak']
        }
        
        # Add individual metric scores
        for metric_name in ['fraction_trapped', 'cooling_timescale', '20_step_energy']:
            row[f'{metric_name}_combined_score'] = data['metric_scores'].get(metric_name, np.nan)
            row[f'{metric_name}_width_score'] = data['width_metrics'].get(metric_name, np.nan)
            row[f'{metric_name}_peak_score'] = data['peak_metrics'].get(metric_name, np.nan)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)


def main():
    """Main function to calculate and report COMBINED performance metrics"""
    parser = argparse.ArgumentParser(description='Calculate COMBINED performance metrics')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Performance degradation threshold (e.g., 0.5 for 50%%, 0.75 for 75%%) [default: 0.5]')
    args = parser.parse_args()
    
    performance_threshold = args.threshold
    threshold_percent = int(performance_threshold * 100)
    
    # Get the script's directory as base
    script_dir = Path(__file__).parent
    
    # Load results for all models
    model_folders = ['mlp_sim', 'differentiator', 'mlp_experimental']
    
    print("="*60)
    print("COMBINED PERFORMANCE METRIC ANALYSIS")
    print("(Detuning + Photon Counts Sweeps)")
    print(f"Performance Threshold: {threshold_percent}% of peak")
    print("="*60)
    
    # Process detuning data
    print("\n" + "="*40)
    print("PROCESSING DETUNING DATA")
    print("="*40)
    detuning_results = load_and_process_data(script_dir, model_folders, 'detuning')
    
    if not detuning_results:
        print("No detuning data found")
        return
    
    # Calculate detuning metrics
    print("Calculating detuning performance metrics...")
    detuning_metrics = calc_detuning_metrics(detuning_results, param_name='detuning', performance_threshold=performance_threshold)
    
    # Process photon number data
    print("\n" + "="*40)
    print("PROCESSING PHOTON NUMBER DATA")
    print("="*40)
    photon_results = load_and_process_data(script_dir, model_folders, 'photon_number')
    
    if not photon_results:
        print("No photon number data found")
        return
    
    # Calculate photon number metrics
    print("Calculating photon number performance metrics...")
    photon_metrics = calc_photon_metrics(photon_results, param_name='photon_number', performance_threshold=performance_threshold)
    
    # Combine metrics
    print("\n" + "="*40)
    print("COMBINING METRICS")
    print("="*40)
    combined_metrics = combine_metrics(detuning_metrics, photon_metrics)
    
    # Print combined summary
    print_combined_summary(combined_metrics, performance_threshold)
    
    # Save detailed results to CSV
    output_file = script_dir / 'combined_performance_metrics.csv'
    save_combined_results(combined_metrics, output_file)
    print(f"\nDetailed COMBINED results saved to: {output_file}")


if __name__ == "__main__":
    main() 