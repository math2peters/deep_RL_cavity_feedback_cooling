Energy trace data for detuning sweep with mlp_sim
=================================================================

Each parameter value has its own .npz and .csv file containing:
- times_us: Time points in microseconds (starting from 0)
- energies_uk: Energy values in microkelvin (mean across episodes)
- std_errs_uk: Standard error of the mean energy in microkelvin
- episode_counts: Number of episodes contributing to each time point

CSV files include headers: time_us, energy_uk, energy_uk_std_err, episode_count

These files can be used for custom fitting if the automatic fit failed.
Example usage for loading data from a .npz file:

import numpy as np
data = np.load('filename.npz')
times = data['times_us']
energies = data['energies_uk']
std_errs = data['std_errs_uk']  # if available
episode_counts = data['episode_counts']  # if available

For fitting, you can use scipy.optimize.curve_fit with a custom function:

from scipy.optimize import curve_fit
def exponential_with_offset(t, E0, tau, offset):
    return (E0 - offset) * np.exp(-t/tau) + offset

# Convert times from microseconds to seconds for fitting
times_s = times / 1e6
# Initial guesses for E0, tau (in seconds), and offset
p0 = [energies[0], (times_s[-1] - times_s[0])/8, energies[-1]]
popt, pcov = curve_fit(exponential_with_offset, times_s, energies, p0=p0)
# Convert tau back to microseconds
tau_us = popt[1] * 1e6
