#!/usr/bin/env python3
import subprocess
import sys
import argparse
import warnings
from pathlib import Path

# Filter out the specific warning about lr_schedule deserialization
warnings.filterwarnings("ignore", message="Could not deserialize object lr_schedule")

REPO_ROOT = Path(__file__).resolve().parents[2]

def run_sweep(model_type, model_name, parameter='detuning', noisy_measurements=False, output_dir=None):
    """Run parameter sweep for a specific model using defaults from the original script.
    
    Args:
        model_type: Type of policy model ('mlp' or 'baseline')
        model_name: Paper model name ('sim', 'experimental', or 'differentiator')
        parameter: Parameter to sweep ('detuning', 'photon_number', 'temperature')
        noisy_measurements: Whether to use noisy measurements
        output_dir: Optional directory for generated sweep outputs
    """
    print(f"\n{'='*80}")
    print(f"Running sweep for {model_type}_{model_name} with parameter: {parameter}")
    print(f"Noisy measurements: {noisy_measurements}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, str(REPO_ROOT / "src" / "evaluation" / "analyze_network_boundaries.py"),
        "--model", model_type,
        "--model-name", model_name,
        "--parameter", parameter
    ]
    
    if noisy_measurements:
        cmd.append("--noisy-measurements")
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parameter sweeps for different models')
    parser.add_argument('--parameters', nargs='+', default=['detuning', 'photon_number'],
                        choices=['detuning', 'photon_number', 'temperature'],
                        help='List of parameters to sweep (default: detuning photon_number)')
    parser.add_argument('--sweep-noisy', action='store_true',
                        help='Also run sweeps with noisy measurements')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Optional directory to write sweep outputs. Defaults to data/source_data_fig4/<model_slug>/')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['mlp', 'baseline'],
                        help='Type of policy model to run (if specified, only this model type will be run)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name of the model to run (if specified, only this model will be run)')
    args = parser.parse_args()
    
    # Default models to sweep
    default_models = [
        ('mlp', 'sim'),
        ('baseline', 'differentiator'),
        ('mlp', 'experimental'),
    ]
    
    # If model-type and model-name are specified, only run that model
    if args.model_type and args.model_name:
        models = [(args.model_type, args.model_name)]
        print(f"Running sweep for specific model: {args.model_type}/{args.model_name}")
    else:
        models = default_models
        print(f"Running sweep for all default models")
    
    # Run sweep for each model and parameter combination
    for model_type, model_name in models:
        for parameter in args.parameters:
            # Run with non-noisy measurements
            run_sweep(model_type, model_name, parameter, noisy_measurements=False, output_dir=args.output_dir)
            
            # If sweep_noisy is set, also run with noisy measurements
            if args.sweep_noisy:
                run_sweep(model_type, model_name, parameter, noisy_measurements=True, output_dir=args.output_dir)
        
    print("\nAll sweeps completed!")

if __name__ == "__main__":
    main() 