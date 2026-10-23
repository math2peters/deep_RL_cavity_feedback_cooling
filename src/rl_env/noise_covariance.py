"""Atom-cavity rates and momentum/field noise of the semiclassical model, shared by the simulator and the particle filter.

Domokos, Horak and Ritsch, J. Phys. B 34, 187 (2001), Eqs. (8)-(12), with the cavity mode
f(r) = exp[-(y^2+z^2)/w^2] and the quantization axis along the cavity axis x. With the Wigner photon
number n = a_r^2 + a_i^2 and the local (saturated) scattering rate Gamma_a = Gamma_0 f^2, Cov(dY) = G G^T dt with

  field (a_r, a_i):  d1 = (kappa/2 + Gamma_a)/2 each, uncorrelated;
  velocity:          D_vv = 2 max(n - 1/2, 0) Gamma_a (hbar/m)^2 [u u^T + k^2 x x^T + k^2 Q],  Q = diag(2/5, 3/10, 3/10);
  velocity-field:    correlated with the phase quadrature only, Gamma_a sqrt(n) (hbar/m) u (Domokos d3);

where u = grad f / f = (0, -2y/w^2, -2z/w^2). The velocity terms are the dipole-force fluctuation along the
field gradient, the absorption recoil of the running-wave mode (hbar k along the cavity axis per scattered
photon), and the spontaneous-emission recoil of a sigma+ transition, whose emission pattern
(1 + cos^2 theta) about x gives Q. The correlation of absorption recoil with the field amplitude is
omitted: under the Stratonovich integration used here it would add a spurious mean force. The field noise uses fixed Cartesian columns. Below about 0.6
photons the semiclassical covariance loses positivity; there only the velocity-phase correlation is
shortened, keeping every variance. counts records [evaluations, shortened evaluations].
"""
import numpy as np
from numba import njit

MODEL_VERSION = "domokos-corrected-v5"

HBAR = 6.62607015e-34/(2*np.pi)
MASS = 2.21e-25                      # cesium, as in environment.py
WAVENUMBER = 2*np.pi/852e-9
LINEWIDTH = 2*np.pi*5.23e6           # gamma, full linewidth
RECOIL_AXIAL, RECOIL_TRANSVERSE = .4, .3


@njit(cache=True)
def cavity_rates(state, probe_atom_detuning, cavity_waist, g_max):
    """Return (f, u_y, u_z, n, U0_a, Gamma_a) at state: the local dispersive shift and scattering rate."""
    y, z = state[1], state[2]
    f = np.exp(-((y/cavity_waist)**2 + (z/cavity_waist)**2))
    g = g_max*f
    n = state[6]**2 + state[7]**2
    denominator = probe_atom_detuning**2 + (LINEWIDTH/2)**2 + 2*g*g*n
    return (f, -2*y/cavity_waist**2, -2*z/cavity_waist**2, n,
            g*g*probe_atom_detuning/denominator, g*g*(LINEWIDTH/2)/denominator)


@njit(cache=True)
def noise_factor(state, kappa, Gamma_a, u_y, u_z, counts=None):
    """G (8 x 8) at state; columns 3-5 drive the velocities, 6-7 the field quadratures (a_r, a_i)."""
    n = state[6]**2 + state[7]**2
    if not (np.isfinite(n) and np.isfinite(Gamma_a) and np.isfinite(u_y) and np.isfinite(u_z)):
        raise ValueError("Noise state and rates must be finite")
    if Gamma_a < 0 or kappa < 0:
        raise ValueError("Rates must be nonnegative")
    if counts is not None:
        counts[0] += 1
    G = np.zeros((8, 8))
    d1 = .5*(kappa/2 + Gamma_a)
    root = np.sqrt(d1)
    G[6, 6] = G[7, 7] = root

    unit = 2*max(n - .5, 0.)*Gamma_a*(HBAR/MASS)**2
    k2 = WAVENUMBER**2
    uu = u_y*u_y + u_z*u_z
    var_u = unit*(uu + RECOIL_TRANSVERSE*k2)      # along u in the (y, z) plane
    var_e = unit*RECOIL_TRANSVERSE*k2             # perpendicular to u in that plane
    beta = Gamma_a*np.sqrt(n)*HBAR/MASS/root if root > 0 else 0.   # phase correlation per sqrt(d1), along u
    bounded = beta*beta*uu > var_u
    if bounded:
        beta = np.sqrt(var_u/uu)
        if counts is not None:
            counts[1] += 1
    # Phase quadrature (-a_i, a_r)/|alpha| in Cartesian field coordinates.
    per_amplitude = beta/np.sqrt(n) if n > 0 else 0.
    G[4, 6], G[4, 7] = -per_amplitude*u_y*state[7], per_amplitude*u_y*state[6]
    G[5, 6], G[5, 7] = -per_amplitude*u_z*state[7], per_amplitude*u_z*state[6]
    G[3, 3] = np.sqrt(unit*(1 + RECOIL_AXIAL)*k2)   # absorption + emission recoil along the cavity axis
    # Symmetric square root of the (y, z) residual var_e*I + (unit - beta^2) u u^T, smooth through u = 0.
    residual_u = 0. if bounded else var_u - beta*beta*uu
    root_e, root_u = np.sqrt(var_e), np.sqrt(residual_u)
    if root_u + root_e == 0:
        slope = 0.
    elif bounded:
        slope = -root_e/uu
    else:
        slope = (unit - beta*beta)/(root_u + root_e)
    G[4, 4] = root_e + slope*u_y*u_y
    G[5, 5] = root_e + slope*u_z*u_z
    G[4, 5] = G[5, 4] = slope*u_y*u_z
    return G


@njit(cache=True)
def noise_at_state(state, probe_atom_detuning, kappa, cavity_waist, g_max, counts=None):
    """G(Y) from the state alone, for the simulator's solvers and the particle filter."""
    _, u_y, u_z, _, _, Gamma_a = cavity_rates(state, probe_atom_detuning, cavity_waist, g_max)
    return noise_factor(state, kappa, Gamma_a, u_y, u_z, counts)
