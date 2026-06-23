#!/usr/bin/env python3
"""
Trajectory Conversion Script

This script loads large trajectory pickle files containing full position and velocity
histories, extracts only the necessary turning point information, and saves it
to a new, smaller pickle file compatible with cooling_rate_calculation.py.

The script can extract two types of turning points:
1. Velocity turning points: where velocity_z crosses zero
2. Position turning points: where position_z crosses zero
"""

import numpy as np
from scipy.interpolate import interp1d
import pickle
import os
import sys
import argparse
from collections import defaultdict
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_OUTPUT_DIR = REPO_ROOT / "outputs" / "trajectory_cache"

# Physical constants (copied from cooling_rate_calculation.py)
h = 6.62607015e-34  # Planck constant in Js
kB = 1.380649e-23   # Boltzmann constant in J/K
m = 2.21e-25        # Cesium mass in kg

# --- Helper functions copied from cooling_rate_calculation.py ---

def load_trajectory_data(data_path):
    """Load trajectory data from pickle file"""
    print(f"Loading original data from {data_path}...", flush=True)
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories.", flush=True)
    return trajectories

def find_turning_points(times, velocity_z, max_points=10):
    """
    Find turning points where velocity_z crosses zero.
    (Identical to the function in cooling_rate_calculation.py)
    """
    sign_changes = np.where(np.diff(np.signbit(velocity_z)))[0]
    if len(sign_changes) > max_points:
        sign_changes = sign_changes[:max_points]
    
    turning_points = []
    for idx in sign_changes:
        t1, t2 = times[idx], times[idx+1]
        v1, v2 = velocity_z[idx], velocity_z[idx+1]
        if v1 == 0:
            turning_points.append((t1, idx))
            continue
        if v2 == 0:
            turning_points.append((t2, idx+1))
            continue
        t_interp = t1 - v1 * (t2 - t1) / (v2 - v1)
        turning_points.append((t_interp, idx))
    return turning_points

def find_position_turning_points(times, position_z, max_points=10):
    """
    Find turning points where position_z crosses zero.
    Similar to the velocity turning points function but for position.
    """
    sign_changes = np.where(np.diff(np.signbit(position_z)))[0]
    if len(sign_changes) > max_points:
        sign_changes = sign_changes[:max_points]
    
    turning_points = []
    for idx in sign_changes:
        t1, t2 = times[idx], times[idx+1]
        p1, p2 = position_z[idx], position_z[idx+1]
        if p1 == 0:
            turning_points.append((t1, idx))
            continue
        if p2 == 0:
            turning_points.append((t2, idx+1))
            continue
        t_interp = t1 - p1 * (t2 - t1) / (p2 - p1)
        turning_points.append((t_interp, idx))
    return turning_points

def calculate_trap_energy(position_z, top_trap_U0_max, top_trap_offset_z, top_trap_waist_z):
    """
    Calculate trap energy along the z-axis.
    (Identical to the function in cooling_rate_calculation.py)
    """
    return -top_trap_U0_max / (1 + ((position_z - top_trap_offset_z) / top_trap_waist_z) ** 2) + top_trap_U0_max

def calculate_max_velocity(trap_energy, position_z):
    """
    Calculate the expected maximum velocity at the center of the trap.
    (Identical to the function in cooling_rate_calculation.py)
    """
    speed = np.sqrt(2 * abs(trap_energy) / m)
    if position_z > 0:
        return -speed
    elif position_z < 0:
        return speed
    else:
        return 0.0

# --- Main Conversion Logic ---

def convert_trajectory(original_traj, max_turning_points):
    """
    Processes a single trajectory to extract turning point info for both
    velocity and position turning points.
    """
    try:
        times = original_traj['times']
        velocity_z = original_traj['velocity_history'][2]
        position_z = original_traj['position_history'][2]
        episode_id = original_traj['episode_id']
        
        # Get transmission data if available
        transmission_history = original_traj.get('transmission_history', [])
        
        # Extract trap parameters, using defaults if not present
        top_trap_U0_max = original_traj.get('top_trap_U0_max', h*36e6*13.5/10.5)
        top_trap_offset_z = original_traj.get('top_trap_offset_z', 0.0)
        top_trap_waist_z = original_traj.get('top_trap_waist_z', 13e-6)
        
        # Find velocity turning points
        vel_turning_points_raw = find_turning_points(times, velocity_z, max_turning_points)
        
        # Find position turning points
        pos_turning_points_raw = find_position_turning_points(times, position_z, max_turning_points)
        
        # Create interpolation functions for position and velocity
        # Check if times are unique and sorted, handle duplicates if necessary
        unique_times, indices = np.unique(times, return_index=True)
        if len(unique_times) < len(times):
            print(f"Warning: Duplicate time points found in episode {episode_id}. Using unique points for interpolation.", flush=True)
            times_interp = unique_times
            position_z_interp = position_z[indices]
            velocity_z_interp = velocity_z[indices]
        else:
            times_interp = times
            position_z_interp = position_z
            velocity_z_interp = velocity_z
            
        if len(times_interp) < 2:
             print(f"Warning: Not enough unique time points (<2) for interpolation in episode {episode_id}. Skipping.", flush=True)
             return None
             
        interp_pos_z = interp1d(times_interp, position_z_interp, kind='linear', fill_value="extrapolate")
        interp_vel_z = interp1d(times_interp, velocity_z_interp, kind='linear', fill_value="extrapolate")

        processed_turning_points = []
        
        # Process velocity turning points
        for turn_time, idx in vel_turning_points_raw:
            # Ensure turn_time is within interpolation bounds
            if turn_time < times_interp.min() or turn_time > times_interp.max():
                 print(f"Warning: Velocity turning point time {turn_time} outside interpolation range [{times_interp.min()}, {times_interp.max()}] for episode {episode_id}. Skipping point.", flush=True)
                 continue
                 
            turn_pos_z = float(interp_pos_z(turn_time))
            turn_vel_z = 0.0  # Velocity is zero at velocity turning points
            
            # For velocity turning points, energy is from the trap potential
            energy = calculate_trap_energy(turn_pos_z, top_trap_U0_max, top_trap_offset_z, top_trap_waist_z)
            max_velocity = calculate_max_velocity(energy, turn_pos_z)
            
            processed_turning_points.append({
                'type': 'velocity',
                'time': turn_time,
                'position_z': turn_pos_z,
                'velocity_z': turn_vel_z,
                'energy': energy,
                'max_velocity': max_velocity
            })
        
        # Process position turning points
        for turn_time, idx in pos_turning_points_raw:
            # Ensure turn_time is within interpolation bounds
            if turn_time < times_interp.min() or turn_time > times_interp.max():
                 print(f"Warning: Position turning point time {turn_time} outside interpolation range [{times_interp.min()}, {times_interp.max()}] for episode {episode_id}. Skipping point.", flush=True)
                 continue
            
            # Position is zero at position turning points
            turn_pos_z = 0.0
            
            # Calculate velocity at position turning point using interpolation
            turn_vel_z = float(interp_vel_z(turn_time))
            
            # For position turning points, calculate energy from kinetic energy
            # E = 1/2 * m * v^2
            energy = 0.5 * m * turn_vel_z**2
            
            # For position turning points, the max velocity is just the actual velocity
            # since we're already at the trap center (z=0)
            max_velocity = turn_vel_z
            
            processed_turning_points.append({
                'type': 'position',
                'time': turn_time,
                'position_z': turn_pos_z,
                'velocity_z': turn_vel_z,
                'energy': energy,
                'max_velocity': max_velocity
            })
        
        # Sort all turning points by time
        processed_turning_points.sort(key=lambda x: x['time'])
        
        # Only return if we have at least 2 turning points (needed for dv/dt)
        if len(processed_turning_points) < 2:
             print(f"Warning: Fewer than 2 valid turning points found for episode {episode_id}. Skipping trajectory.", flush=True)
             return None

        converted_data = {
            'episode_id': episode_id,
            'trap_params': {
                'top_trap_U0_max': top_trap_U0_max,
                'top_trap_offset_z': top_trap_offset_z,
                'top_trap_waist_z': top_trap_waist_z
            },
            'turning_points': processed_turning_points, # List of dicts with type field
            'transmission_history': transmission_history  # Add transmission data
        }
        return converted_data
        
    except Exception as e:
        print(f"Error processing episode {original_traj.get('episode_id', 'UNKNOWN')}: {e}", flush=True)
        return None


def process_file(input_file, max_points=20):
    """Process a single trajectory file and save the turning points"""
    if "turning_points" in input_file:
        print(f"Skipping {input_file} as it already contains turning points.", flush=True)
        return
        
    output_file = input_file.replace('.pkl', '_turning_points.pkl')
    
    # Load original data
    original_trajectories = load_trajectory_data(input_file)
    
    # Process trajectories
    converted_trajectories = []
    total = len(original_trajectories)
    print(f"Converting {total} trajectories from {input_file}...", flush=True)
    for i, traj in enumerate(original_trajectories):
        converted = convert_trajectory(traj, max_points)
        if converted:
            converted_trajectories.append(converted)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total} trajectories...", flush=True)

    print(f"Finished conversion. Retained {len(converted_trajectories)} valid trajectories.", flush=True)

    # Save converted data
    print(f"Saving converted data to {output_file}...", flush=True)
    with open(output_file, 'wb') as f:
        pickle.dump(converted_trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Conversion complete for {input_file}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert large trajectory pkl files to smaller turning-point-only pkl files.")
    parser.add_argument("--max-points", type=int, default=20, 
                        help="Maximum number of turning points to extract per trajectory")
    parser.add_argument("--data-dir", type=str, default=str(TRAJECTORY_OUTPUT_DIR),
                        help="Directory containing trajectory data files")
    
    args = parser.parse_args()
    
    # Find all .pkl files in the data directory
    data_files = glob.glob(os.path.join(args.data_dir, "*.pkl"))
    
    if not data_files:
        print(f"No .pkl files found in {args.data_dir}", flush=True)
        sys.exit(1)
    
    print(f"Found {len(data_files)} .pkl files in {args.data_dir}", flush=True)
    
    # Process each file
    for file_path in data_files:
        if "turning_points" not in file_path:
            process_file(file_path, args.max_points) 