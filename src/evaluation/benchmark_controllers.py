"""Simulator with owned random streams, and the classical controllers of Fig. 4."""
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from numba import njit
from src.rl_env.environment import CavityCoolingEnv, m, kB, hbar, k_852, trap_potential
from src.evaluation.fig4 import SCENARIOS
from src.rl_env.noise_covariance import cavity_rates, noise_at_state


class BenchmarkEnv(CavityCoolingEnv):
    """Each episode seed fixes all of its random draws; the detuning drift restarts every episode."""
    def __init__(self, study_seed=0, **kwargs):
        self.episode_rng = np.random.default_rng(study_seed)
        self.numpy_state = np.random.RandomState(study_seed).get_state()
        with self.owned_numpy():
            super().__init__(seed=study_seed, **kwargs)

    @contextmanager
    def owned_numpy(self):
        outside = np.random.get_state()
        np.random.set_state(self.numpy_state)
        try:
            yield
        finally:
            self.numpy_state = np.random.get_state()
            np.random.set_state(outside)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.episode_rng = np.random.default_rng(seed)
            episode_seed = int(seed)
        else:
            episode_seed = int(self.episode_rng.integers(0, 2**31 - 1))
        self.numpy_state = np.random.RandomState(episode_seed).get_state()
        self.sde_generator = np.random.default_rng(np.random.SeedSequence([episode_seed, 7281]))
        self.cavity_detuning_drift = 0.0
        with self.owned_numpy():
            return super().reset(seed=episode_seed, options=options)

    def step(self, action):
        with self.owned_numpy():
            return super().step(action)


class ConstantController:
    """No feedback: holds the trap at full depth."""
    def predict(self, observation, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


class FilteredDifferentiatorController:
    """A differentiator that requests the depth directly, like the Kalman and particle controllers.

    The normalized count is low-pass filtered, y_t = exp(-dt/filter_time) y_{t-1}
    + (1 - exp(-dt/filter_time)) c_t, the feedback signal is the decrease of the
    filtered count, zeta = y_{t-1} - y_t, and the requested depth is depth + gain*zeta.
    The filter starts from the seven stacked counts of the first observation.
    """
    def __init__(self, depth=0.8, gain=0.75, filter_time=0.0, dt=27.3e-6):
        self.depth, self.gain = depth, gain
        self.beta = 1.0 if filter_time <= 0 else 1 - np.exp(-dt / filter_time)
        self.reset()

    def reset(self):
        self.filtered = None
        self.change = 0.0

    def _update(self, count):
        previous = self.filtered
        self.filtered = previous + self.beta * (count - previous)
        self.change = self.filtered - previous

    def predict(self, observation, deterministic=True):
        if self.filtered is None:
            stacked = [float(observation[i]) for i in range(len(observation) - 27, len(observation), 4)]
            self.filtered = stacked[0]
            for count in stacked[1:]:
                self._update(count)
        else:
            self._update(float(observation[-3]))
        return depth_action(self.depth - self.gain * self.change)


@dataclass
class Calibration:
    dt: float
    exposure: float
    delay: float
    ramp: float
    depth: float
    probe: float
    delta_atom: float
    kappa: float
    cavity_waist: float
    g: float
    trap_waist: np.ndarray
    offsets: np.ndarray
    limits: np.ndarray
    efficiency: float
    count_min: float
    count_max: float
    detuning_noise: float
    position_std: np.ndarray
    velocity_std: np.ndarray
    omega: float

    @classmethod
    def from_scenario(cls, scenario="nominal"):
        """The simulator's parameters at an operating point, as the filters' internal model."""
        env = BenchmarkEnv(**SCENARIOS[scenario])
        temperatures = np.array([env.radial_temperature, env.radial_temperature, env.temperature])
        frequencies = np.array([env.top_trap_frequency_x, env.top_trap_frequency_y, env.top_trap_frequency_z])
        velocity = np.sqrt(kB * temperatures / m)
        result = cls(env.t_step, env.t_step, env.delay_time, env.ramp_time, env.top_trap_U0_max,
                     env.cavity_probe_intensity_max, env.cavity_probe_atom_detuning,
                     env.kappa, env.cavity_waist, env.g_max,
                     np.array([env.top_trap_waist_x, env.top_trap_waist_y, env.top_trap_waist_z]),
                     np.array([env.top_trap_offset_x, env.top_trap_offset_y, env.top_trap_offset_z]),
                     np.asarray(env.atom_trapped_values), env.photon_detection_efficiency,
                     env.expected_min_photon_number, env.expected_max_photon_number,
                     env.detuning_noise, velocity/frequencies, velocity, frequencies[2])
        env.close()
        return result

    def physics(self):
        return (self.depth, self.delay, self.ramp, self.probe, self.delta_atom,
                self.kappa, self.cavity_waist, self.g, self.trap_waist, self.offsets, self.limits)

    def decode_count(self, centered_observation):
        # The environment delivers normalized_counts - 0.5.
        return self.count_min+(float(centered_observation)+.5)*(self.count_max-self.count_min)


def depth_action(depth):
    return np.array([2*np.sqrt(np.clip(depth, .001, 1.))-1], dtype=np.float32), None


class KalmanQuadratureController:
    """Signal-space Kalman filter for the offset and two rotating quadratures of the count.

    The measurement is the bin average of the rotating component, and its variance is the
    Poisson variance of the predicted count. The feedback signal is the rotating component forecast
    to the middle of the action interval, divided by its amplitude plus amplitude_floor, like the
    energy normalization of the particle controller; amplitude_floor=None keeps the unnormalized law.
    """
    def __init__(self, scenario="nominal", frequency_hz=11000., process_noise=.03, phase=0., gain=1., depth=.85,
                 amplitude_floor=None):
        self.cal = Calibration.from_scenario(scenario)
        self.process_noise, self.gain, self.depth = process_noise, gain, depth
        self.amplitude_floor = amplitude_floor
        theta = 2*np.pi*frequency_hz*self.cal.dt
        c, s = np.cos(theta), np.sin(theta)
        self.F = np.array([[1.,0,0],[0,c,-s],[0,s,c]])
        measured_angle = 2*np.pi*frequency_hz*self.cal.exposure
        self.H = np.array([1.,np.sinc(measured_angle/(2*np.pi))*np.cos(measured_angle/2),
                           np.sinc(measured_angle/(2*np.pi))*np.sin(measured_angle/2)])
        advance = 2*np.pi*frequency_hz*(self.cal.delay+self.cal.dt/2)+phase
        self.control = np.array([0.,np.cos(advance),-np.sin(advance)])
        self.reset()

    def reset(self):
        self.tick = 0
        self.x = np.zeros(3)
        self.P = np.eye(3)

    def predict(self, observation, deterministic=True):
        if self.tick:
            counts = float(observation[-3])
            if self.tick == 1:
                self.x[0] = counts
            self.x = self.F @ self.x
            scale = self.cal.count_max-self.cal.count_min
            predicted = float(self.H @ self.x)
            R = max(self.cal.decode_count(predicted), 1.)/scale**2
            Q = self.process_noise*R*np.diag([.1,1.,1.])
            self.P = self.F @ self.P @ self.F.T+Q
            S = float(self.H @ self.P @ self.H+R)
            K = self.P @ self.H/S
            self.x += K*(counts-predicted)
            A = np.eye(3)-np.outer(K,self.H)
            self.P = A @ self.P @ A.T+R*np.outer(K,K)
        self.tick += 1
        drive = float(self.control @ self.x)
        if self.amplitude_floor is not None:
            drive /= np.hypot(self.x[1], self.x[2])+self.amplitude_floor
        return depth_action(self.depth+self.gain*drive)


@njit(inline="always")
def _rates_into(y,t,old_depth,new_depth,detuning,p,out):
    """Allocation-free copy of the simulator drift (src.rl_env.environment._drift) with the filter's cavity detuning."""
    depth,delay,ramp,probe,delta_atom,kappa,waist,gmax,trap,offsets,limits=p
    fraction=min(max((t-delay)/ramp,0.),1.)
    current_depth=depth*(old_depth+(new_depth-old_depth)*fraction)
    f,u_y,u_z,n,shift,loss=cavity_rates(y,delta_atom,waist,gmax)
    rate=kappa/2+loss
    _,Fx,Fy,Fz=trap_potential((y[0],y[1],y[2]),trap,offsets,current_depth)
    dipole=-2*hbar*shift*(n-.5)
    out[0]=y[3];out[1]=y[4];out[2]=y[5]
    out[3]=(-Fx+2*hbar*k_852*loss*(n-.5))/m
    out[4]=(-Fy+dipole*u_y)/m
    out[5]=(-Fz+dipole*u_z)/m
    out[6]=-probe+(shift-detuning)*y[7]-rate*y[6]
    out[7]=-(shift-detuning)*y[6]-rate*y[7]


@njit(inline="always")
def _noise_into(y, p, w, out, scale, counts=None):
    """Apply the simulator's noise factor at y to five independent draws (G columns 3-7)."""
    G = noise_at_state(y, p[4], p[5], p[6], p[7], counts)
    for j in range(8):
        out[j] = 0.
        for k in range(5):
            out[j] += scale*G[j, k+3]*w[k]


@njit(cache=True)
def propagate_particles(states, detunings, old_depth, new_depth, duration, p, wiener, noise_scale, count_start=0.,
                        covariance_counts=None):
    """Propagate particles over one interval: RK4 drift plus stochastic-Heun diffusion.

    Also returns each particle's integrated intracavity (Wigner) photon number after count_start.
    """
    n_particles, substeps = wiener.shape[:2]
    dt = duration/substeps
    out = states.copy()
    integrals = np.zeros(n_particles)
    f1=np.empty(8);f2=np.empty(8);f3=np.empty(8);f4=np.empty(8)
    trial=np.empty(8);deterministic=np.empty(8)
    noise0=np.empty(8);noise1=np.empty(8)
    for particle in range(n_particles):
        y = states[particle].copy()
        for step in range(substeps):
            t = step*dt
            n0=y[6]**2+y[7]**2
            _rates_into(y,t,old_depth,new_depth,detunings[particle],p,f1)
            if noise_scale>0:
                _noise_into(y,p,wiener[particle,step],noise0,noise_scale,covariance_counts)
                for j in range(8):trial[j]=y[j]+dt*f1[j]+noise0[j]
                _noise_into(trial,p,wiener[particle,step],noise1,noise_scale,covariance_counts)
            # Resolve the known actuator kinks exactly. Otherwise a finer regular
            # grid can cross the ramp at a less favorable phase than a coarser one.
            deterministic[:] = y
            now = t
            first = True
            while now < t+dt-1e-18:
                end = t+dt
                for boundary in (p[1],p[1]+p[2]):
                    if boundary > now+1e-18 and boundary < end:
                        end = boundary
                h = end-now
                if not first:_rates_into(deterministic,now,old_depth,new_depth,detunings[particle],p,f1)
                for j in range(8):trial[j]=deterministic[j]+.5*h*f1[j]
                _rates_into(trial,now+.5*h,old_depth,new_depth,detunings[particle],p,f2)
                for j in range(8):trial[j]=deterministic[j]+.5*h*f2[j]
                _rates_into(trial,now+.5*h,old_depth,new_depth,detunings[particle],p,f3)
                for j in range(8):trial[j]=deterministic[j]+h*f3[j]
                _rates_into(trial,end,old_depth,new_depth,detunings[particle],p,f4)
                for j in range(8):deterministic[j]+=h*(f1[j]+2*f2[j]+2*f3[j]+f4[j])/6
                now=end
                first=False
            if noise_scale > 0:
                for j in range(8):y[j]=deterministic[j]+.5*(noise0[j]+noise1[j])
            else:
                y[:] = deterministic
            if t+dt>count_start:
                begin=max(t,count_start)
                n1=y[6]**2+y[7]**2
                at_begin=n0+(n1-n0)*(begin-t)/dt
                integrals[particle] += .5*(at_begin+n1)*(t+dt-begin)
        out[particle] = y
    return out, integrals


def systematic_indices(weights, offset):
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    return np.searchsorted(cumulative,(np.arange(len(weights))+offset)/len(weights))


class ParticleFeedbackController:
    """Bootstrap particle filter for the full nonlinear atom and cavity-field SDE.

    The feedback signal uses the weighted moments of q = z/z_R and p = v_z/(omega z_R) forecast to
    the middle of the action interval, ((q^2 - p^2) cos(phase) + 2 qp sin(phase))/(q^2 + p^2 + floor).
    For a single atom in the harmonic full-depth trap this is cos(2 omega t + phase), the same phase
    convention as the Kalman controller: phase = 0 follows the transmission, pi/2 cools fastest.
    """
    def __init__(self, scenario="nominal", particles=256, substeps=64,
                 filter_seed=41026, phase=0., gain=.3, depth=.8, energy_floor=.05):
        self.cal = Calibration.from_scenario(scenario)
        self.n, self.substeps, self.filter_seed = particles, substeps, filter_seed
        self.phase, self.gain, self.depth = phase, gain, depth
        self.energy_floor = energy_floor
        self.physics = self.cal.physics()
        self.reset()

    def reset(self):
        self.covariance_counts = np.zeros(2, dtype=np.int64)
        self.rng = np.random.default_rng(self.filter_seed)
        self.states = np.zeros((self.n,8))
        self.states[:,:3] = self.rng.normal(size=(self.n,3))*self.cal.position_std+self.cal.offsets
        self.states[:,3:6] = self.rng.normal(size=(self.n,3))*self.cal.velocity_std
        self.states[:,6:8] = self.rng.normal(0.,.5,size=(self.n,2))   # Wigner vacuum, as in the simulator
        self.detunings = np.zeros(self.n)
        self.weights = np.full(self.n,1/self.n)
        self.tick = 0
        self.previous_depth = 1.

    def feedback(self, current_depth):
        lead = self.cal.delay+self.cal.dt/2
        steps = max(4,int(np.ceil(self.substeps*lead/self.cal.dt)))
        forecast,_ = propagate_particles(self.states,self.detunings,current_depth,current_depth,
                                          lead,self.physics,np.zeros((self.n,steps,5)),0.)
        q = (forecast[:,2]-self.cal.offsets[2])/self.cal.trap_waist[2]
        p = forecast[:,5]/(self.cal.omega*self.cal.trap_waist[2])
        X,Y,Z = [float(self.weights @ v) for v in [q*q,q*p,p*p]]
        drive = ((X-Z)*np.cos(self.phase)+2*Y*np.sin(self.phase))/(X+Z+self.energy_floor)
        return depth_action(self.depth+self.gain*drive)

    def predict(self, observation, deterministic=True):
        current_depth = float(np.clip(((float(observation[-4])+1)/2)**2,.001,1.)) if self.tick else 1.
        if self.tick:
            self.detunings = .5*self.detunings+.5*self.cal.detuning_noise*self.rng.normal(size=self.n)
            wiener = self.rng.normal(size=(self.n,self.substeps,5))*np.sqrt(self.cal.dt/self.substeps)
            self.states,integrals = propagate_particles(self.states,self.detunings,self.previous_depth,
                current_depth,self.cal.dt,self.physics,wiener,1.,self.cal.dt-self.cal.exposure,self.covariance_counts)
            means = np.maximum(integrals*self.cal.kappa*self.cal.efficiency,1e-12)
            counts = max(0.,round(self.cal.decode_count(observation[-3])))
            logw = np.log(np.maximum(self.weights,1e-300))+(counts*np.log(means)-means)
            logw[~np.isfinite(logw)] = -np.inf
            if not np.isfinite(logw).any():
                raise FloatingPointError("Particle propagator or likelihood became nonfinite")
            self.weights = np.exp(logw-np.max(logw))
            self.weights /= self.weights.sum()
            if 1/np.dot(self.weights,self.weights) < .5*self.n:
                indices = systematic_indices(self.weights,self.rng.random())
                self.states = self.states[indices].copy()
                self.detunings = self.detunings[indices].copy()
                self.weights.fill(1/self.n)
        self.previous_depth = current_depth
        self.tick += 1
        return self.feedback(current_depth)

