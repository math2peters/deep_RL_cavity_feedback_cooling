import numpy as np
import sdeint
import matplotlib.pyplot as plt
import gymnasium
import pandas as pd
from numba import jit
import time

# CONSTANTS
PI = np.pi
m = 2.21e-25 # cesium mass
h = 6.62607015e-34  # Planck constant in Js
kB = 1.380649e-23   # Boltzmann constant in J/K
hbar = h / (2*PI) # reduced Planck constant
c = 299792458 # speed of light
k_852 = 2*PI/(852e-9) # wavevector of 852 nm
gamma = 2*PI*5.23e6 # decay rate of cs atom
I_sat = 27.1 # saturation intensity in W/m^2

# Numba-optimized core physics functions

@jit(nopython=True)
def g_profile(yz, waist_yz, g_max):
    y, z = yz
    cavity_y, cavity_z = waist_yz
        
    return g_max*np.exp(-((y / cavity_y)**2 + (z / cavity_z)**2))

@jit(nopython=True)
def trap_potential(xyz, waist_xyz, offset_xyz, depth):
    """Calculate trap potential and its derivatives."""
    x, y, z = xyz
    offset_x, offset_y, offset_z = offset_xyz
    trap_x, trap_y, trap_z = waist_xyz
    
    denom = 1 + ((z-offset_z) / trap_z) ** 2
    V =  -depth*(1 / denom) * np.exp(-2 * (((x-offset_x)/trap_x)**2 + ((y-offset_y)/trap_y)**2)*1/denom)

    # Compute derivatives
    dVx =  -(4 * (x-offset_x)) / (trap_x**2 * denom) * V
    dVy =  -(4 * (y-offset_y)) / (trap_y**2 * denom) * V
    dVz =  -V * (z-offset_z) / trap_z**2 /denom *(- 4 * (((x-offset_x)/trap_x)**2 + ((y-offset_y)/trap_y)**2) / denom + 2)
    return V, dVx, dVy, dVz


@jit(nopython=True)
def generate_photons(P_c, cur_dt, photon_detection_efficiency):
    """Generate photon counts based on Poisson distribution."""
    poisson_mean = P_c * cur_dt * photon_detection_efficiency
    if poisson_mean > 0:
        return np.random.poisson(poisson_mean)
    return 0


@jit(nopython=True)
def calculate_energy(vxyz, U_total):
    """Calculate total energy."""
    return 0.5 * m * np.dot(vxyz, vxyz) - U_total

@jit(nopython=True)
def calculate_normalized_energy(vxyz, U_total):
    """Calculate normalized energy in MHz."""
    ke = 0.5 * m * np.dot(vxyz, vxyz) / (h*1e6)
    pe = U_total / (h*1e6)
    return ke + pe

@jit(nopython=True)
def calculate_diffusion_coefficient(scatter_rate, mean_photon_number, Gamma_a, f_2, grad_f):
    u_bar_2 = 0.4
    d1 = 1/2 * scatter_rate
    d2 = 2 * Gamma_a * max(mean_photon_number - 1/2, 0)  * (np.dot((hbar * grad_f), (hbar * grad_f)) + hbar **2 * k_852 **2 *u_bar_2 * f_2) / m ** 2
    d3 = Gamma_a * np.sqrt(mean_photon_number*f_2) * hbar * grad_f / m
    return d1, d2, d3


@jit(nopython=True)
def compute_diffusion(Y, d1, d2, d3):
    x, y, z, vx, vy, vz, a_r, a_i = Y
    
    G = np.zeros((len(Y), len(Y)))
    sqrt_d1 = np.sqrt(d1)
    sqrt_d2 = np.sqrt(d2)
    G[3, 3] = sqrt_d2
    G[4, 4] = sqrt_d2
    G[5, 5] = sqrt_d2

    G[6, 6] = sqrt_d1
    G[7, 7] = sqrt_d1

    cos_theta = a_r/(np.sqrt(a_r**2 + a_i**2)+1e-10)
    sin_theta = a_i/(np.sqrt(a_r**2 + a_i**2)+1e-10)
    
    # Calculate norm of d3 manually without using abs()
    d3_norm = np.sqrt(d3[0]**2 + d3[1]**2 + d3[2]**2)
    
    if d3_norm > 0:
        # Calculate D_vA components without using abs()
        D_vA_perp = np.sqrt(d3_norm) * d3 / d3_norm
        D_vA_para = np.sqrt(d3_norm) * d3 / d3_norm
    else:
        D_vA_perp = np.zeros(3)
        D_vA_para = np.zeros(3)

    G[3, 6] = D_vA_perp[0] * cos_theta - D_vA_para[0] * sin_theta
    G[3, 7] = D_vA_perp[0] * sin_theta + D_vA_para[0] * cos_theta

    G[4, 6] = D_vA_perp[1] * cos_theta - D_vA_para[1] * sin_theta
    G[4, 7] = D_vA_perp[1] * sin_theta + D_vA_para[1] * cos_theta

    G[5, 6] = D_vA_perp[2] * cos_theta - D_vA_para[2] * sin_theta
    G[5, 7] = D_vA_perp[2] * sin_theta + D_vA_para[2] * cos_theta
    
    G = G

    return G

def lorentzian(x, x_0, gamma):
    return gamma**2 / (gamma**2 + 4*(x-x_0)**2)

class Atom:
    """Atom state with thermal initial position and velocity."""

    def __init__(self, random_source, spread_xyz, spread_vxyz, offset_xyz, atom_capture=False, time_evolve_min=0, time_evolve_max=500e-6):
        """
        Initialize an atom with thermal position and velocity distributions.

        Args:
            random_source: Integer seed, NumPy-compatible RNG, or None. None uses
                a fresh OS-seeded RandomState.
            spread_xyz: Standard deviations [σx, σy, σz] for thermal position distribution
            spread_vxyz: Standard deviations [σvx, σvy, σvz] for thermal velocity distribution
            offset_xyz: Center position [x0, y0, z0] of the trap
            atom_capture: Whether to evolve the state backwards in time
            time_evolve_min: Minimum time for backwards evolution if atom_capture=True
            time_evolve_max: Maximum time for backwards evolution if atom_capture=True
        """

        if random_source is None:
            self.atom_idx = None
            rng = np.random.RandomState()
        elif isinstance(random_source, (int, np.integer)):
            self.atom_idx = int(random_source)
            rng = np.random.RandomState(self.atom_idx)
        else:
            self.atom_idx = None
            rng = random_source
        
        # Draw initial position and velocity from thermal Gaussian distributions
        initial_position = rng.normal(0, spread_xyz, 3) + offset_xyz
        initial_velocity = rng.normal(0, spread_vxyz, 3)
        
        # time evolve the state backwards in time by a uniform amount of time between 100 us and 1.1 ms
        if atom_capture:
            t_init = rng.uniform(time_evolve_min, time_evolve_max)
            initial_position = initial_position - initial_velocity * t_init
            

        self.state = [initial_position, initial_velocity]  # Initial state of the atom
        self.trapped = True
        self.position_history = np.empty((0, 3))  # To store the full history of x, y, z positions
        self.velocity_history = np.empty((0, 3))  # To store the full history of vx, vy, vz velocities
        
    def append_history(self, positions, velocities):
        """Appends the entire history of positions and velocities for one SDE time step."""
        self.position_history = np.vstack([self.position_history, positions])  # Appending all positions for the step
        self.velocity_history = np.vstack([self.velocity_history, velocities])  # Appending all velocities for the step
    
    def get_position_history(self):
        """Returns the position history."""
        return self.position_history.T  # Transpose to get separate x, y, z arrays
    
    def get_velocity_history(self):
        """Returns the velocity history."""
        return self.velocity_history.T  # Transpose to get separate vx, vy, vz arrays
    
    def set_position(self, position):
        """Updates the current position."""
        self.state[0] = position
    
    def set_velocity(self, velocity):
        """Updates the current velocity."""
        self.state[1] = velocity
    
    def get_position(self):
        """Gets the current position."""
        return self.state[0]
    
    def get_velocity(self):
        """Gets the current velocity."""
        return self.state[1]


class CavityCoolingEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, architecture='MLP', t_step=27.3e-6, t_max=2.73e-3, render_mode=None,
                 frame_stack_number=7, full_observations=False, atom_capture=False, truncate_if_untrapped=True,
                 reward_scale=1.0, reward_component_scale=[1, 1, 1], 
                 probe_detuning_input=2*np.pi*(25e6),
                 photon_number_input=33,
                 top_trap_U0_max_input=36e6,
                 temperature_input=440e-6, 
                 noisy_measurements=False,
                 diffusion_on=True,
                 frame_wait_mode=True,
                 seed=None, verbose=False):
        
        self.kappa = 2*PI*39e3 # linewidth of cavity
        if noisy_measurements:
            photon_number_input = photon_number_input + photon_number_input*0.25*np.random.randn()
            top_trap_U0_max_input = top_trap_U0_max_input + top_trap_U0_max_input*0.1*np.random.randn()
            self.temperature = temperature_input
            self.radial_temperature = 180e-6 # 3x the measured temperature, very sensitive to alignment
            self.eta_max = 21 
            self.photon_detection_efficiency = 0.125 * 0.5 #1
            self.detuning_noise =  self.kappa / 6
            self.top_trap_offset_y = 2.03e-6 * np.random.normal() # 15% change in cooperativity in rms
            self.top_trap_offset_z = 2.03e-6 * np.random.normal()# 15% change in cooperativity in rms
        else:
            self.temperature = temperature_input
            self.radial_temperature = 60e-6
            self.eta_max = 21 
            self.detuning_noise =  self.kappa / 12
            self.top_trap_offset_y = 0 
            self.top_trap_offset_z = 0
            self.photon_detection_efficiency = 0.125
        run_000_offset = 0
        g_labscript = 2*np.pi*1.05e6
        emf_factor = self.kappa / (self.kappa + g_labscript**2/probe_detuning_input**2 * gamma) # 1 
        if seed is not None:
            np.random.seed(seed)
            
        super(CavityCoolingEnv, self).__init__()
        self.function_entered_count = 0
        self.render_mode = render_mode
        self.architecture = architecture
        self.atom_capture = atom_capture    
        self.diffusion_on = diffusion_on
        # Definite time parameters of simulation
        self.t_max = t_max
        self.t_step = t_step
        self.time_elapsed = 0
        self.steps_per_us = 14
        self.dt = 1e-6 / self.steps_per_us

        # Full State observations:
        self.frame_stack_number = frame_stack_number
        self.frame_stack = []  # Data structure to store the last `frame_stack_number` observations
        self.full_observations = full_observations
        self.frame_wait_mode = frame_wait_mode

                        
        # Define action space as a Box with low and high limits
        self.action_labels = ['Top Trap Power']
        self.action_space = gymnasium.spaces.Box(low=np.array([-1.0]*len(self.action_labels)), high=np.array([1.0]*len(self.action_labels)), dtype=np.float32)

        self.observation_labels = ["Action {}".format(i) for i in range(len(self.action_labels))]
        self.observation_labels += ['Normalized Counts', 'Normalized Mean', 'Normalized Mean Difference']
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([-1] * len(self.observation_labels) * self.frame_stack_number),
            high=np.array([1] * len(self.observation_labels) * self.frame_stack_number),
            dtype=np.float32
            )      

        self.top_trap_U0_max = h * top_trap_U0_max_input # MHz top_trap depth
        self.top_trap_waist_x = 1.5e-6
        self.top_trap_waist_y = 1.5e-6
        self.top_trap_waist_z = 13e-6
        self.top_trap_offset_x = 0  # offset of top trap from origin
        self.top_trap_unit_vec = k_852*np.array([0, 0, 1])
        self.top_trap_frequency_x = np.sqrt(4 * self.top_trap_U0_max / (m * self.top_trap_waist_x**2))
        self.top_trap_frequency_y = np.sqrt(4 * self.top_trap_U0_max / (m * self.top_trap_waist_y**2))
        self.top_trap_frequency_z = np.sqrt(2 * self.top_trap_U0_max / (m * self.top_trap_waist_z**2))
        self.ramp_time = 500e-9
        self.delay_time = 7.3e-6

        self.cavity_waist = 7.1e-6
        self.cavity_rayleigh = PI * self.cavity_waist**2 / 852e-9
        self.g_max = np.sqrt(self.eta_max * self.kappa * gamma/4)
        self.cavity_k_vec = k_852*np.array([1, 0, 0])
        self.cavity_unit_vec = self.cavity_k_vec / np.linalg.norm(self.cavity_k_vec)
        
        detected_to_intra_cavity_photon = photon_number_input * (1 / (self.kappa * t_step * self.photon_detection_efficiency  ))
        self.cavity_probe_intensity_max = np.sqrt(self.kappa ** 2 / 4 * detected_to_intra_cavity_photon) 
        self.cavity_probe_intensity = self.cavity_probe_intensity_max * 1
        self.cavity_probe_atom_detuning = probe_detuning_input
        self.cavity_probe_cavity_detuning_max = 2*PI*50e3
        self.cavity_probe_cavity_detuning = 0
        self.cavity_detuning_drift = 0
        self.cavity_probe_k_vec = self.cavity_k_vec
        self.cavity_probe_unit_vec = self.cavity_unit_vec
        

        bin_count_size = int(self.t_step/(1e-6/self.steps_per_us))
        bin_row_total = int(self.t_max//self.t_step)+1
        self.photon_counts_array = np.zeros((bin_row_total+1))
        self.total_count_list = []
        if self.frame_wait_mode:
            self.photon_counts_array = np.zeros((bin_row_total+1+self.frame_stack_number))
        
        self.expected_max_photon_number = photon_number_input
        self.expected_min_photon_number = self.expected_max_photon_number * \
            lorentzian(0, (g_labscript)**2*(self.cavity_probe_atom_detuning-run_000_offset)/((self.cavity_probe_atom_detuning-run_000_offset)**2 + gamma**2/4), self.kappa)* \
                emf_factor
        
        self.velocity_normalization = [0.4, 0.4, 0.1]
        self.position_normalization = [self.top_trap_waist_x, self.top_trap_waist_y, self.top_trap_waist_z]
        self.atom_trapped_values = 4*np.array([self.top_trap_waist_x, self.top_trap_waist_y, self.top_trap_waist_z])
        self.truncate_if_untrapped = truncate_if_untrapped
        
        # reward stuff
        self.count_std_turn_on = int(1e-3/self.t_step * 0.25)
        self.running_count_beta_slow = 0.99
        self.running_count_beta_fast = 0.935
        self.reward_scale = reward_scale
        self.reward_component_scale = reward_component_scale
        self.reset()
        
        self.verbose = verbose
        if self.verbose:
            print("Noise parameters: ", noisy_measurements)
            print(f"Max eta: {self.eta_max}")
            print(f"Max g: {self.g_max/2/PI/1e6:.2f} MHz")
            print(f"Max trapping frequency along x axis: {self.top_trap_frequency_x/(2*PI*1e3)} kHz")
            print(f"Max trapping frequency along y axis: {self.top_trap_frequency_y/(2*PI*1e3)} kHz")
            print(f"Max trapping frequency along z axis: {self.top_trap_frequency_z/(2*PI*1e3)} kHz")
            print("Expected min fraction: ", self.expected_min_photon_number/self.expected_max_photon_number)
            print("Real min fraction: ",  
            lorentzian(0, (self.g_max)**2*(self.cavity_probe_atom_detuning)/((self.cavity_probe_atom_detuning)**2 + gamma**2/4), self.kappa))
            
        self.figure, self.axes = None, None  # Initialize figure and axes

    
    # https://arxiv.org/pdf/quant-ph/0010061
    def equations_of_motion(self, Y, t):
        # the SDE solver calls the function twice per iteration, so we need to check if the time has changed
        # cur_dt is only positive if it's called the first time
        cur_dt = t - self.last_time
        self.last_time = t
        
        # for record keeping purposes
        self.function_entered_count += 1
        
        x, y, z = Y[0:3]
        vx, vy, vz = Y[3:6]
        a_r, a_i = Y[6:8]
        
        action_t = min(max(t-self.delay_time, 0), self.ramp_time) 
        current_trap_U0 = (self.top_trap_U0 - self.last_top_trap_U0) * \
           action_t / self.ramp_time + self.last_top_trap_U0
        current_probe_intensity = self.cavity_probe_intensity 
        current_cavity_probe_detuning = self.cavity_probe_cavity_detuning 
        
        g = g_profile(np.array([y, z]), 
                         np.array([self.cavity_waist, self.cavity_waist]), 
                         self.g_max)

        f = g/self.g_max
        f_2 = f**2
        grad_f = f*np.array([0, 
                           -2*y/self.cavity_waist**2, 
                           -2*z/self.cavity_waist**2])
        mean_photon_number = (a_r**2 + a_i**2)
        omega = 2 * g * np.sqrt(mean_photon_number)
        denom = self.cavity_probe_atom_detuning**2 + (gamma/2)**2 + 0.5 * omega**2

        # dispersive shift and scattering‐induced loss
        U0_a = g**2 * self.cavity_probe_atom_detuning / denom
        Gamma_a = (g**2 * (gamma/2))  / denom
        scatter_rate = (self.kappa/2 + Gamma_a)
        self.cavity_potential = U0_a * hbar * mean_photon_number

        
        self.d1, self.d2, self.d3 = calculate_diffusion_coefficient(scatter_rate, mean_photon_number, Gamma_a, f_2, grad_f)
        
        da_r = - current_probe_intensity + (U0_a - current_cavity_probe_detuning) * a_i - \
            scatter_rate * a_r
            
        da_i = - (U0_a - current_cavity_probe_detuning) * a_r - scatter_rate * a_i
        
        if cur_dt > 0:
            
            if current_probe_intensity > 0:

                self.photon_counts_array[self.step_number] += mean_photon_number

        
        # Calculate trap forces using Numba-optimized function
        V_t, dVx_t, dVy_t, dVz_t = trap_potential(
            np.array([x, y, z]),
            np.array([self.top_trap_waist_x, self.top_trap_waist_y, self.top_trap_waist_z]),
            np.array([self.top_trap_offset_x, self.top_trap_offset_y, self.top_trap_offset_z]),
            current_trap_U0
        )
        
        F_trap_t = -np.array([dVx_t, dVy_t, dVz_t])
        F_cavity_t = - hbar * U0_a * max((mean_photon_number - 1/2), 0) * 2 * grad_f / (f + 1e-6)
        F_scatter = hbar * k_852 *  Gamma_a * mean_photon_number* np.array([1, 0, 0])

        a = (F_trap_t + F_cavity_t + F_scatter)/ m
        

        dYdt = np.zeros_like(Y)
        # Update the derivative of the state
        # velocities no change
        # The derivative of position is velocity
        dYdt[0:3] = [vx, vy, vz]  # dx/dt = velocity
        
        # The derivative of velocity is acceleration
        dYdt[3:6] = a  # dv/dt = acceleration
        
        dYdt[6:8] = [da_r, da_i]
        
        if abs(x) > self.atom_trapped_values[0] or abs(y) > self.atom_trapped_values[1] or abs(z) > self.atom_trapped_values[2]:
            self.atom.trapped = False
        else:
            self.atom.trapped = True

        return dYdt
    
    # Diffusion (stochastic part)
    def diffusion(self, Y, t):
        
        G = compute_diffusion(Y, self.d1, self.d2, self.d3)
        if not self.diffusion_on:
            G = np.zeros_like(G)
       

        return G 
    
    def _get_stacked_observation(self):
        # Return the flattened and concatenated observation directly
        return np.concatenate(self.frame_stack, axis=0)
    


    def step(self, action):
        action = np.clip(action, -1, 1) 
        # Execute one time step within the environment
        if self.step_number < self.frame_stack_number and self.frame_wait_mode:
            if self.step_number == 0:
                self.input_reward_scale = self.reward_scale
            action = [1]
            self.reward_scale = 0
            self.time_elapsed = 0 # don't start counting time until the first action
        elif self.step_number >= self.frame_stack_number and self.frame_wait_mode:
            self.reward_scale = self.input_reward_scale
        
        
        self.action_list.append(action)
        # Collect the states of the atom
        atom_position = np.array(self.atom.get_position())  # Ensure it's a numpy array
        atom_velocity = np.array(self.atom.get_velocity())  # Ensure it's a numpy array
        total_atom_state = np.concatenate((atom_position, atom_velocity))  # Correct concatenation of position and velocity
        total_atom_field_state = np.concatenate((total_atom_state, np.array([self.a_r, self.a_i])))
        self.last_top_trap_U0 = self.top_trap_U0
        self.last_cavity_probe_cavity_detuning = self.cavity_probe_cavity_detuning 

            

        self.top_trap_U0 = self.top_trap_U0_max * ((action[0] + 1) / 2) ** 2 # square to mimic AOD
            
        self.top_trap_U0 = np.clip(self.top_trap_U0, 0.001 * self.top_trap_U0_max, self.top_trap_U0_max)
            
        self.last_cavity_probe_intensity = self.cavity_probe_intensity
        self.cavity_probe_cavity_detuning = 0
        self.cavity_detuning_drift = np.random.normal()*self.detuning_noise*(1-0.5) + self.cavity_detuning_drift*0.5
        self.cavity_probe_cavity_detuning += self.cavity_detuning_drift
        total_simulation_steps = int(self.steps_per_us * self.t_step * 1e6)
        simulation_time = np.linspace(0, self.t_step, total_simulation_steps)
        
        self.time_elapsed += self.t_step
        self.iteration_count = 0
        self.last_time = 0
        self.count_column = 0  # For the photon counts array

        atom_solutions = sdeint.stratSRS2(self.equations_of_motion, self.diffusion, total_atom_field_state, simulation_time)
        
        # Append the entire position and velocity history for the step
        positions = atom_solutions[1:, 0:3]  # x, y, z positions
        velocities = atom_solutions[1:, 3:6]  # vx, vy, vz velocities
        fields = atom_solutions[1:, 6:8]
        
        self.a_r_history_list = np.concatenate((self.a_r_history_list, fields[1:, 0]))
        self.a_i_history_list = np.concatenate((self.a_i_history_list, fields[1:, 1]))
        self.a_r = fields[-1, 0]
        self.a_i = fields[-1, 1]

        self.atom.append_history(positions, velocities)
        self.atom.set_position(positions[-1])
        self.atom.set_velocity(velocities[-1])

        # This is correct in the mean, but not exactly correct for the variance. 
        # In the limit of larger photon number, it is exactly correct. 
        # To fully calculate this correctly, one needs to calculate the full solution to the equations 
        # in the EoM when the photon number outside the cavity is given by an operator b and 
        # rederive the stochastic EoM for the atom-cavity system 
        total_counts = np.random.poisson(self.photon_counts_array[self.step_number]\
            *self.kappa*self.t_step*self.photon_detection_efficiency / (total_simulation_steps))
        self.total_count_list.append(total_counts)
        
      
        self.time_fraction = self.time_elapsed / self.t_max
        

        if self.step_number > 2:
            self.running_count_mean_fast = self.running_count_beta_fast * self.running_count_mean_fast + \
                (1-self.running_count_beta_fast) * total_counts
            self.running_count_std_fast = self.running_count_beta_fast * self.running_count_std_fast + \
                (1-self.running_count_beta_fast) * abs(total_counts - self.running_count_mean_fast)**2
        else:
            self.running_count_mean_fast = total_counts
            self.running_count_std_fast = total_counts
            
        if self.step_number < int(1.0/(1-self.running_count_beta_fast)):
            self.running_count_mean_slow = self.running_count_mean_fast 
            self.running_count_std_slow = self.running_count_std_fast
        else:
            self.running_count_mean_slow = self.running_count_beta_slow * self.running_count_mean_slow + \
                (1-self.running_count_beta_slow) * total_counts
            self.running_count_std_slow = self.running_count_beta_slow * self.running_count_std_slow + \
                (1-self.running_count_beta_slow) * abs(total_counts - self.running_count_mean_fast)**2

        
        
        normalized_counts = (total_counts-self.expected_min_photon_number)/(self.expected_max_photon_number - self.expected_min_photon_number)
        normalized_mean = (self.running_count_mean_fast-self.expected_min_photon_number)/(self.expected_max_photon_number - self.expected_min_photon_number)
        normalized_difference = (self.running_count_mean_slow - self.running_count_mean_fast)/(self.expected_max_photon_number - self.expected_min_photon_number)
        normalized_std = self.running_count_std_fast/(self.expected_max_photon_number - self.expected_min_photon_number)
        normalized_std_difference = (self.running_count_std_slow - self.running_count_std_fast)/(self.expected_max_photon_number - self.expected_min_photon_number)
        
        self.total_counts = total_counts
        self.normalized_counts = normalized_counts

            
        
        current_observation = [
            *np.array(action),
            normalized_counts - 0.5,
            normalized_mean - 0.5,
            normalized_difference*2]

                               
        

        # Calculate reward and check if the episode is done
        done = False
        truncated = False
        reward = 0

        # Initialize individual reward components
        counts_std_reward = 0
        counts_velocity_reward = 0
        time_bonus_reward = 0 
        mean_change_reward = 0
        action_penalty_reward = 0
        mean_energy_reward = 0
                
        offset_norm = (self.top_trap_offset_z*1e6)
        offset_correction = (offset_norm*0.15+1)
        
        transmission_truncate_condition = False
        if self.step_number > 15 and normalized_mean > 0.8 and\
            (normalized_mean > self.observation_list[-10][1] + 0.5 or not self.diffusion_on) and\
               not self.atom.trapped:
            transmission_truncate_condition = True

        if transmission_truncate_condition and self.truncate_if_untrapped:
            done = True
            truncated = True
        
        if self.time_elapsed >= self.t_max:
            done = True


        
        if self.step_number > self.count_std_turn_on:# and self.step_number > self.counts_top_average*3:
            counts_std_reward += 0*offset_norm / (normalized_std**2/5 + 0.05) * (min(normalized_mean, 1)-1)**2 * self.reward_scale
        
        reward += counts_std_reward

        if self.step_number >= int(1.0/(1-self.running_count_beta_fast)):
            mean_energy_reward = 7*(max(min(normalized_mean, 1.0), 0)-1.0)**8* self.reward_component_scale[0] * self.reward_scale
        else:
            mean_energy_reward = 0

        reward += mean_energy_reward


        mean_change_reward = 7.5*normalized_difference * self.reward_component_scale[1]  * self.reward_scale
            
        reward += mean_change_reward
        
        action_penalty_reward = 0#-((action[0] ** 10)) * 1
        reward += action_penalty_reward
        
        
        if self.atom.trapped:
            self.total_steps_trapped += 1
            time_bonus_reward += 0.5 * self.reward_component_scale[2] * self.reward_scale # + 5 * np.exp(-self.step_number/15)
            reward += time_bonus_reward
            
        self.step_number += 1
        info = {}  # Additional info for debugging


        
        self.plotting_observation_list.append(current_observation)
    
        # Update the frame stack with the new observation
        if len(self.frame_stack) >= self.frame_stack_number:
            self.frame_stack.pop(0)  # Remove the oldest frame if stack size limit is reached
        self.frame_stack.append(current_observation)
        
        self.observation_list.append(current_observation)  # Add the single observation, not the stacked version
        self.running_count_mean_fast_list.append(self.running_count_mean_fast/self.expected_max_photon_number)
        self.running_count_mean_slow_list.append(self.running_count_mean_slow/self.expected_max_photon_number)
        self.running_count_std_fast_list.append(np.sqrt(self.running_count_std_fast/self.expected_max_photon_number**2))
        self.running_count_std_slow_list.append(np.sqrt(self.running_count_std_slow/self.expected_max_photon_number**2))
        final_atom_position = self.atom.get_position_history()[2,-1]
        final_atom_velocity = self.atom.get_velocity_history()[2,-1]
        V_t = -self.top_trap_U0 /(1 + ((final_atom_position-self.top_trap_offset_z) / self.top_trap_waist_z) ** 2)
        # V_t += self.cavity_potential
        V_t += self.top_trap_U0
        energy = (V_t + 1/2 * m * final_atom_velocity**2)/self.top_trap_U0_max
        
        self.energy_list.append(energy)
        
        self.reward_list.append(reward)
        
        self.reward_components['counts_std'].append(counts_std_reward)
        self.reward_components['counts_velocity'].append(counts_velocity_reward)
        self.reward_components['time_bonus'].append(time_bonus_reward)
        self.reward_components['action_penalty'].append(action_penalty_reward)
        self.reward_components['mean_change'].append(mean_change_reward)
        self.reward_components['mean_energy'].append(mean_energy_reward)
        

        
        

        if self.frame_wait_mode and self.frame_stack_number <= self.step_number <  self.frame_stack_number + 15  and self.atom.trapped:
            # Calculate and store mean KE_z for this step
            velocities_z = atom_solutions[1:, 5] # vz is the 6th element (index 5)
            ke_z_substeps = 0.5 * m * velocities_z**2
            mean_ke_z_joules = np.mean(ke_z_substeps)
            self.ke_z_list.append(mean_ke_z_joules)
        elif not self.frame_wait_mode and self.atom.trapped and self.step_number < 15:
            # Calculate and store mean KE_z for this step
            velocities_z = atom_solutions[1:, 5] # vz is the 6th element (index 5)
            ke_z_substeps = 0.5 * m * velocities_z**2
            mean_ke_z_joules = np.mean(ke_z_substeps)
            self.ke_z_list.append(mean_ke_z_joules)

        # Return the stacked observation
        stacked_current_observation = self._get_stacked_observation()
        if done:

            # Convert energy_list (normalized by U0_max) back to Kelvin for mean calculation
            energy_list_joules = np.array(self.energy_list) * self.top_trap_U0_max
            if len(energy_list_joules) > 0:
                mean_long_temp_joules = np.mean(energy_list_joules)
                mean_long_temperature = mean_long_temp_joules / kB
            else:
                mean_long_temperature = None


            # --- Perform final step with probe off ---
            # Store current state and probe intensity
            original_probe_intensity = self.cavity_probe_intensity
            Y_before_final_step = np.concatenate((self.atom.get_position(), self.atom.get_velocity(), np.array([self.a_r, self.a_i])))

        
            # Turn probe off and simulate one dt
            self.cavity_probe_intensity = 0
            # Ensure last_time is set correctly for EOM function if it relies on differences
            self.last_time = 0 # Reset last_time for the final mini-step calculation
            self.last_top_trap_U0 = self.top_trap_U0
            atom_solutions_final = sdeint.stratSRS2(self.equations_of_motion, self.diffusion, Y_before_final_step, simulation_time)
            final_state = atom_solutions_final[-1]

            # Extract final position and velocity
            final_pos = final_state[0:3]
            final_vel = final_state[3:6]

            # Calculate final trap potential (shifted so bottom is zero)
            V_t_final = -self.top_trap_U0 /(1 + ((final_pos[-1]-self.top_trap_offset_z) / self.top_trap_waist_z) ** 2)
            V_trap_final_shifted = V_t_final + self.top_trap_U0 # Shift relative to trap bottom

            # Calculate final kinetic energy
            KE_final = 0.5 * m * np.dot(final_vel[-1], final_vel[-1])

            # Calculate final total energy (trap + kinetic) and convert to Kelvin
            E_total_final_joules = KE_final + V_trap_final_shifted
            final_temperature_kelvin = E_total_final_joules / kB


            # Restore original probe intensity
            self.cavity_probe_intensity = original_probe_intensity
            # --- End final step ---
            if self.atom.trapped:
                self.energy_list.append(E_total_final_joules/self.top_trap_U0_max)
            else:
                self.energy_list.append(None)
            
            if self.ke_z_list and len(self.ke_z_list) > 0:
                m_ke_z = np.mean(self.ke_z_list)/kB
            else:
                m_ke_z = None


            info = {
                'episode': {
                    'r': np.sum(self.reward_list), # Use sum of rewards collected
                    'l': self.total_steps_trapped,
                    't': self.time_fraction,
                },
                'trapped_steps': self.total_steps_trapped,
                'trapped_end': self.atom.trapped,
                'mean_temperature': mean_long_temperature, # Average temp during RL steps
                'final_temperature': final_temperature_kelvin, # Temp after probe off step
                'mean_ke_z': m_ke_z # Mean KE_z during RL steps
            }


        self.done = done
        return stacked_current_observation, reward, done, truncated, info


    def seed(self, seed=None):

        np.random.seed(seed)
        return [seed]

    def reset(self, options=None, seed=None):
        """Reset the state of the environment to an initial state
        
        This creates a new atom with a random initial position and velocity drawn from 
        thermal Gaussian distributions derived from the trap frequencies and 
        temperature. Proper seeding ensures reproducibility while still having
        different initial conditions for each episode.
        
        Args:
            options: Optional reset options
            seed: Seed for random state initialization. Explicit integer seeds
                  produce reproducible atom initial conditions. When None, each
                  reset uses a fresh OS-seeded RandomState.
                  
        Returns:
            Tuple of (observation, info) where observation is the initial observation and
            info is an empty dictionary.
        """
        if seed is not None:
            self.seed(seed)
            # Explicit seeding of our RandomState for reproducibility
            rng = np.random.RandomState(seed)
        else:
            rng = None
            
        self.last_time = 0
        self.done = False
        
        action = [1]*self.action_space.shape[0]
        self.top_trap_U0 = self.top_trap_U0_max * (action[0] + 1) / 2
        spread_xyz = [np.sqrt(kB *self.radial_temperature / (self.top_trap_frequency_x**2 * m)), 
                      np.sqrt(kB *self.radial_temperature / (self.top_trap_frequency_y**2 * m)), 
                      np.sqrt(kB * self.temperature / (self.top_trap_frequency_z**2 * m))]
        spread_vxyz = [np.sqrt(kB * self.radial_temperature / m), 
                       np.sqrt(kB * self.radial_temperature / m), 
                       np.sqrt(kB * self.temperature / m)]
        
        # Fixed offset values without additional randomization
        offset_xyz = np.array([self.top_trap_offset_x, self.top_trap_offset_y, self.top_trap_offset_z])
        
        self.atom = Atom(rng, spread_xyz, spread_vxyz,
                         offset_xyz=offset_xyz, atom_capture=self.atom_capture)
        self.a_r = 0
        self.a_i = 0
        
        self.last_probe_intensity = self.cavity_probe_intensity
        self.last_top_trap_U0 = self.top_trap_U0
        self.cavity_probe_cavity_detuning = 0
        positions = self.atom.get_position()
        velocities = self.atom.get_velocity()
        V_t, dVx_t, dVy_t, dVz_t = trap_potential(positions, (self.top_trap_waist_x, self.top_trap_waist_y, self.top_trap_waist_z), (self.top_trap_offset_x, self.top_trap_offset_y, self.top_trap_offset_z), self.top_trap_U0)

        self.mean_E = calculate_normalized_energy(velocities, V_t)
        self.last_mean_E = self.mean_E

        self.step_number = 0
        self.total_steps_trapped = 0
        self.total_photons_scattered_before_trapped = 0
        self.first_time_trapped = False
        self.counts_measured = False
        self.time_fraction = 0
        self.time_elapsed = 0
        self.count_column = 0
        self.last_energy = 0
        self.action_list = []
        self.reward_list = []
        self.a_r_history_list = np.array([])
        self.a_i_history_list = np.array([])
        self.observation_list = []
        self.plotting_observation_list = []
        self.running_count_mean_slow_list = []  # List to store slow mean values
        self.running_count_mean_fast_list = []  # List to store fast mean values
        self.running_count_std_slow_list = []  # List to store slow std values
        self.running_count_std_fast_list = []  # List to store fast std values
        self.energy_list = []
        self.cumulative_reward = 0
        self.photon_counts_array.fill(0)  # Reset the photon counts
        self.total_count_list = []

        self.ke_z_list = [] # Initialize list for mean z-axis KE

        self.probe_power_mean_stack = []
        self.running_count_mean_fast = 0.0
        self.running_count_mean_slow = 0.0
        self.running_count_std_fast = 0.0
        self.running_count_std_slow = 0.0
        self.reward_components = {
        'counts_std': [],
        'time_bonus': [],
        'action_penalty': [],
        'counts_velocity': [],
        'mean_change': [],
        'mean_energy': []
            }
        
        current_observation = [
                    *np.array(action),
                    0.,
                    -0.5,
                    -1]
        

        # Initialize the frame stack with zero-filled observations
        self.frame_stack = [np.array(current_observation, dtype=np.float32) 
                            for _ in range(self.frame_stack_number)]
        
        self.plotting_observation_list.append(current_observation)
    
        # Update the frame stack with the new observation
        if len(self.frame_stack) >= self.frame_stack_number:
            self.frame_stack.pop(0)  # Remove the oldest frame if stack size limit is reached
        self.frame_stack.append(current_observation)
        
        self.observation_list.append(current_observation)  # Add the single observation, not the stacked version
        self.plotting_observation_list.append(current_observation)

        # The initial observation should be zero-filled as well
        stacked_observation = self._get_stacked_observation()
        
        info = {}
        self.next_step_counts= 0  # Initialize next step counts

        return stacked_observation, info  # Return the stacked observation
        

    def render(self, mode='human', interactive_plot=True, keep_open=False, resolution=4):
        if (mode == 'human' and interactive_plot and (self.step_number % resolution == 0 or self.done) and len(self.observation_list) > 0) or keep_open:
            if self.figure is None or self.axes is None:
                plt.ion()  # Turn on interactive mode for updates
                self.figure, self.axes = plt.subplots(3, 2, figsize=(16, 8))

            if keep_open:
                plt.ioff()

            # Clear the current axes and figure
            for ax in self.axes.flatten():
                ax.clear()

            # Subplot 1: X, Y, Z positions over time
            position_history = np.array(self.atom.get_position_history())  # Ensure it's a numpy array
            if position_history.shape[0] != 3:  # Check if we have x, y, z dimensions
                print(f"Warning: Unexpected shape of position_history: {position_history.shape}")
                return  # Exit early if the shape is not as expected
            times = np.linspace(0, self.time_elapsed, position_history.shape[1])
            self.axes[0, 0].plot(times, position_history[0], label='x(t)', color='r')
            self.axes[0, 0].plot(times, position_history[1], label='y(t)', color='g')
            self.axes[0, 0].plot(times, position_history[2], label='z(t)', color='b')
            self.axes[0, 0].set_xlabel('Time (s)')
            self.axes[0, 0].set_ylabel('Position (m)')
            self.axes[0, 0].set_title('Position (X, Y, Z) Over Time')
            self.axes[0, 0].legend(loc='upper left', fontsize='small')
            self.axes[0, 0].grid(True)

            # Subplot 2: Observations over time
            observation_times = np.linspace(0, self.time_elapsed, len(self.plotting_observation_list))
            observations = np.array(self.plotting_observation_list)
            observations = observations.reshape((max(len(self.plotting_observation_list), 1), len(self.observation_labels)))
            for i in range(observations.shape[1]):
                cur_label = self.observation_labels[i]
                self.axes[0, 1].plot(observation_times, observations[:, i], label=cur_label)
            self.axes[0, 1].set_xlabel('Time (s)')
            self.axes[0, 1].set_ylabel('Observation Values')
            self.axes[0, 1].set_title('Observations Over Time')
            self.axes[0, 1].legend(loc='upper left', fontsize='small')
            self.axes[0, 1].grid(True)

            # Subplot 3: Actions over time
            action_times = np.linspace(0, self.time_elapsed, len(self.action_list))
            for i, label in enumerate(self.action_labels):
                action_values = [action[i] for action in self.action_list]
                self.axes[1, 0].plot(action_times, action_values, label=label)
            self.axes[1, 0].set_xlabel('Time (s)')
            self.axes[1, 0].set_ylabel('Action Value')
            self.axes[1, 0].set_title('Actions Taken Over Time')
            self.axes[1, 0].legend(loc='upper left', fontsize='small')
            self.axes[1, 0].grid(True)

            # Subplot 4: Rewards over time with components
            reward_times = np.linspace(0, self.time_elapsed, len(self.reward_list))
            self.axes[1, 1].plot(reward_times, self.reward_list, label='Total Reward', color='k', linewidth=2)

            # Plot individual reward components
            for component, color in zip(['counts_std', 'time_bonus', 'action_penalty', 'counts_velocity', 'mean_change', 'mean_energy'], ['r', 'g', 'b', 'm', 'c', 'y']):
                component_values = self.reward_components[component]
                if len(component_values) < len(reward_times):
                    # Align the lengths by padding with zeros if necessary
                    component_values = np.pad(component_values, (len(reward_times) - len(component_values), 0), 'constant')
                self.axes[1, 1].plot(reward_times, component_values, label=component, color=color, linestyle='--')

            self.axes[1, 1].set_xlabel('Time (s)')
            self.axes[1, 1].set_ylabel('Reward Value')
            self.axes[1, 1].set_title('Rewards Over Time')
            self.axes[1, 1].legend(loc='upper left', fontsize='small')
            self.axes[1, 1].grid(True)

            # New Subplot: Cavity fields a_r and a_i over time
            field_times = np.linspace(0, self.time_elapsed, len(self.a_r_history_list))
            a_r_values = self.a_r_history_list
            a_i_values = self.a_i_history_list 
            self.axes[2, 0].plot(field_times, a_r_values, label='a_r(t)', color='c')
            self.axes[2, 0].plot(field_times, a_i_values, label='a_i(t)', color='m')
            self.axes[2, 0].set_xlabel('Time (s)')
            self.axes[2, 0].set_ylabel('Cavity Field')
            self.axes[2, 0].set_title('Cavity Fields a_r and a_i Over Time')

            # New Subplot: Running count means over time
            mean_times = np.linspace(0, self.time_elapsed, len(self.running_count_mean_slow_list))
            self.axes[2, 1].plot(mean_times, self.running_count_mean_slow_list, label='Running Count Mean Slow', color='b')
            self.axes[2, 1].plot(mean_times, self.running_count_mean_fast_list, label='Running Count Mean Fast', color='r')
            self.axes[2, 1].plot(mean_times, 
                                 np.array(self.running_count_mean_fast_list)-np.array(self.running_count_mean_slow_list), 
                                 label='Running Count Mean Difference', color='g')
            self.axes[2, 1].plot(mean_times, self.running_count_std_slow_list, label='Running Count Std Slow', color='y')
            self.axes[2, 1].plot(mean_times, self.running_count_std_fast_list, label='Running Count Std Fast', color='m')
            self.axes[2, 1].set_xlabel('Time (s)')
            self.axes[2, 1].set_ylabel('Running Count Mean')
            self.axes[2, 1].set_title('Running Count Means Over Time')
            self.axes[2, 1].legend(loc='upper right', fontsize='small')
            self.axes[2, 1].grid(True)

            plt.tight_layout()
            self.figure.canvas.draw()
            plt.pause(1e-4)  # Pause to allow for plot updates

            if keep_open:
                plt.show()


    def close(self):
        # Cleanup
        pass


if __name__ == '__main__':

    env = CavityCoolingEnv(architecture='MLP', seed=None, 
                           atom_capture=False, verbose=True, full_observations=False, 
                           diffusion_on=True,  t_max=20e-3, probe_detuning_input=2*np.pi*33e6, 
                           temperature_input=1e-3, frame_stack_number=7)

    # Example usage:
    obs = env.reset()
    done = False
    j = 0
    reward_sum = 0
    gain = 0.0
    current_counts = previous_count = 0
    while not done:
        action = [np.clip(0.9 - gain * (current_counts - previous_count), -1, 1)]
        obs, reward, done, trunc, info = env.step(action)
        
        previous_count = obs[-7]
        current_counts = obs[-3]
        

        j += 1
        env.render(resolution=25)
        reward_sum += reward
    
    print("Photon Count std:", np.std(env.total_count_list))
    print("Photon Count mean:", np.mean(env.total_count_list))
    print("Total reward:", reward_sum)
    print("Total steps:", env.step_number)


    env.render(keep_open=True)
