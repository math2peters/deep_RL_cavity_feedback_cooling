"""The semiclassical model against an independent implementation of its theory.

The reference below follows Domokos, Horak and Ritsch, J. Phys. B 34, 187 (2001), Eqs. (8)-(12), in three
dimensions, with gradients of the mode function taken by finite differences; the radiation pressure uses
the running-wave phase exp(ikx). It shares no code with src.rl_env.noise_covariance.
"""
import numpy as np
import pytest

from src.rl_env.environment import CavityCoolingEnv, _drift, _integrate_srs2, draw_srs2_noise, hbar, m, k_852, gamma
from src.rl_env.noise_covariance import noise_at_state
from src.evaluation.benchmark_controllers import Calibration, _noise_into, _rates_into, propagate_particles

Q = np.diag([.4, .3, .3])      # <u u^T> of sigma+ emission about the cavity axis x


def mode(r, w):
    return np.exp(-(r[1]**2 + r[2]**2)/w**2 + 1j*k_852*r[0])


def theory(state, detuning, kappa, w, g_max):
    """Covariance rate in (v_x, v_y, v_z, A_par, A_perp) and the drift forces, from the general formulas."""
    r, n = state[:3], state[6]**2 + state[7]**2
    f = mode(r, w)
    grad = np.array([(mode(r + e, w) - mode(r - e, w))/(2*e.sum()) for e in np.eye(3)*1e-12])
    denominator = detuning**2 + (gamma/2)**2 + 2*g_max**2*abs(f)**2*n
    Gamma0, U0 = g_max**2*(gamma/2)/denominator, g_max**2*detuning/denominator
    grad_abs = np.array([(abs(mode(r + e, w)) - abs(mode(r - e, w)))/(2*e.sum()) for e in np.eye(3)*1e-12])
    D = np.zeros((5, 5))
    q = np.imag(f.conj()*grad)/abs(f)**2          # running-wave phase gradient k x
    D[:3, :3] = 2*Gamma0*max(n - .5, 0)*hbar**2*(np.outer(grad_abs, grad_abs) + abs(f)**2*(np.outer(q, q) + k_852**2*Q))/m**2
    D[:3, 4] = D[4, :3] = Gamma0*hbar*np.sqrt(n)*abs(f)*grad_abs/m
    D[3, 3] = D[4, 4] = .5*(kappa/2 + Gamma0*abs(f)**2)
    force = -hbar*U0*(n - .5)*2*np.real(f.conj()*grad) + 2*hbar*Gamma0*(n - .5)*np.imag(f.conj()*grad)
    return D, force


def quadrature_basis(state):
    """Rows (v_x, v_y, v_z, A_par, A_perp) of an 8-dimensional increment."""
    c, s = state[6:]/np.linalg.norm(state[6:])
    B = np.zeros((5, 8))
    B[:3, 3:6] = np.eye(3)
    B[3, 6:], B[4, 6:] = [c, s], [-s, c]
    return B


@pytest.fixture(scope="module")
def cal():
    return Calibration.from_scenario()


STATES = [np.array([0., 0., 0., 0., 0., 0., 3., 1.]),
          np.array([5e-8, -2e-7, 2.5e-6, .1, 0., -.2, -1.5, 2.]),
          np.array([-1e-7, 3e-7, -6e-6, 0., .3, 0., .4, -4.]),
          np.array([0., 0., 9e-6, 0., 0., 0., 2., 2.])]


@pytest.mark.parametrize("state", STATES)
def test_noise_matches_running_wave_theory(state, cal):
    D, _ = theory(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g)
    counts = np.zeros(2, dtype=np.int64)
    G = noise_at_state(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g, counts)
    B = quadrature_basis(state)
    np.testing.assert_allclose(B @ G @ G.T @ B.T, D, rtol=1e-6, atol=1e-12*np.abs(D).max())
    np.testing.assert_array_equal(G[:, :3], 0.)
    np.testing.assert_array_equal(counts, [1, 0])


@pytest.mark.parametrize("state", STATES)
def test_drift_forces_match_theory(state, cal):
    _, force = theory(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g)
    out = np.empty(8)
    zero = (1e-30, 1e-30, 1e-30)   # trap off
    _drift(state, 0., out, 0., 0., 1e-6, 1e-6, 0., 0., cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g,
           zero, (0., 0., 0.), (1., 1., 1.))
    np.testing.assert_allclose(m*out[3:6], force, rtol=1e-6, atol=1e-9*np.abs(force).max())


def test_correlations_are_shortened_only_near_half_a_photon(cal):
    for n, bounded in ((.3, True), (.6, False), (2., False)):
        state = np.array([0., 0., 4e-6, 0., 0., 0., np.sqrt(n), 0.])
        D, _ = theory(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g)
        counts = np.zeros(2, dtype=np.int64)
        G = noise_at_state(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g, counts)
        C = quadrature_basis(state) @ G @ G.T @ quadrature_basis(state).T
        np.testing.assert_allclose(np.diag(C), np.diag(D), rtol=1e-10, atol=1e-30)   # marginals kept
        assert np.linalg.eigvalsh(C).min() > -1e-9*np.abs(C).max()
        assert counts[1] == bounded
        if n < .5:
            np.testing.assert_array_equal(G[3:6], 0.)


def test_noise_factor_is_continuous_through_the_cavity_axis(cal):
    """A factor that flips sign with the atom's side of the axis would corrupt the SRS2 support terms."""
    above, below = (noise_at_state(np.array([0., 0., z, 0., 0., 0., 2., 1.]), cal.delta_atom, cal.kappa,
                                   cal.cavity_waist, cal.g) for z in (1e-15, -1e-15))
    np.testing.assert_allclose(above, below, rtol=1e-6, atol=1e-6*np.abs(above).max())


def test_sampled_increments_have_the_model_covariance(cal):
    state = STATES[1]
    D, _ = theory(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g)
    G = noise_at_state(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g)
    samples = np.random.default_rng(1).normal(size=(400000, 8)) @ G.T @ quadrature_basis(state).T
    scale = np.sqrt(np.diag(D))
    np.testing.assert_allclose(np.cov(samples.T)/np.outer(scale, scale), D/np.outer(scale, scale), atol=.01)


def test_particle_filter_uses_the_simulator_model(cal):
    p = cal.physics()
    for state in STATES:
        filter_drift, simulator_drift = np.empty(8), np.empty(8)
        _rates_into(state, 3e-6, .5, .8, 0., p, filter_drift)
        depth, delay, ramp, probe, delta_atom, kappa, waist, g, trap, offsets, limits = p
        _drift(state, 3e-6, simulator_drift, depth*.8, depth*.5, delay, ramp, probe, 0., delta_atom, kappa,
               waist, g, tuple(trap), tuple(offsets), tuple(limits))
        np.testing.assert_allclose(filter_drift, simulator_drift, rtol=1e-12, atol=0)
        w, out = np.random.default_rng(9).normal(size=5), np.empty(8)
        _noise_into(state, p, w, out, 1.)
        G = noise_at_state(state, delta_atom, kappa, waist, g)
        np.testing.assert_allclose(out, G[:, 3:] @ w, rtol=1e-12, atol=0)


def test_field_starts_in_the_wigner_vacuum():
    env = CavityCoolingEnv(seed=1)
    try:
        fields = []
        for seed in range(400):
            env.reset(seed=seed)
            fields.append((env.a_r, env.a_i))
        fields = np.array(fields)
        np.testing.assert_allclose(fields.std(axis=0), .5, atol=.06)
        np.testing.assert_allclose(fields.mean(axis=0), 0., atol=.08)
        assert abs(np.corrcoef(fields.T)[0, 1]) < .15
    finally:
        env.close()


def test_episode_random_draws_are_independent_of_the_atom():
    """The atom, the field vacuum, the detuning drift and the noisy imperfections come from separate streams."""
    from src.evaluation.benchmark_controllers import BenchmarkEnv
    atoms, others = [], []
    for seed in range(300):
        env = BenchmarkEnv(study_seed=seed, noisy_measurements=True, verbose=False)   # as in the --noisy sweep
        try:
            env.reset(seed=seed)
            offset = [env.top_trap_offset_x, env.top_trap_offset_y, env.top_trap_offset_z]   # trap centre
            atom = np.concatenate([env.atom.get_position() - offset, env.atom.get_velocity()])
            imperfections = [env.top_trap_offset_y, env.top_trap_offset_z, env.top_trap_U0_max]
            vacuum = [env.a_r, env.a_i]
            env.step(np.array([1.]))
            atoms.append(atom)
            others.append(imperfections + vacuum + [env.cavity_detuning_drift])
        finally:
            env.close()
    correlation = np.corrcoef(np.array(atoms).T, np.array(others).T)[:6, 6:]
    assert np.abs(correlation).max() < .25     # 300 samples: standard error 0.06


def test_zero_coupling_limit(cal):
    state = STATES[1]
    G = noise_at_state(state, cal.delta_atom, cal.kappa, cal.cavity_waist, 0.)
    np.testing.assert_array_equal(G[3:6], 0.)
    np.testing.assert_allclose(G @ G.T, np.diag([0]*6 + [cal.kappa/4]*2), rtol=1e-14)
    out = np.empty(8)
    _drift(state, 0., out, 0., 0., 1e-6, 1e-6, 0., 0., cal.delta_atom, cal.kappa, cal.cavity_waist, 0.,
           (1e-30, 1e-30, 1e-30), (0., 0., 0.), (1., 1., 1.))
    np.testing.assert_array_equal(out[3:6], 0.)


def test_one_solver_step_applies_the_model_covariance(cal):
    """Single SRS2 steps from one state: the increment covariance per unit time equals G G^T."""
    state = STATES[1]
    h = 1/14e6
    tspan = np.array([0., h])
    params = (cal.depth*.8, cal.depth*.8, cal.delay, cal.ramp, cal.probe, 0., cal.delta_atom, cal.kappa,
              cal.cavity_waist, cal.g, tuple(cal.trap_waist), tuple(cal.offsets), tuple(cal.limits))
    dW, J = draw_srs2_noise(20000, 8, h, np.random.default_rng(11))
    ends = np.array([_integrate_srs2(state, tspan, dW[k:k+1], J[k:k+1], 0., 0., False, True, *params,
                                     np.zeros(2, dtype=np.int64))[0][1] for k in range(20000)])
    G = noise_at_state(state, cal.delta_atom, cal.kappa, cal.cavity_waist, cal.g)
    target = (G @ G.T)[3:, 3:]
    measured = np.cov(ends[:, 3:].T)/h
    scale = np.sqrt(np.diag(target))
    np.testing.assert_allclose(measured/np.outer(scale, scale), target/np.outer(scale, scale), atol=.035)


@pytest.mark.parametrize('backend', ['sdeint', 'numba'])
def test_default_probe_turn_on_completes(backend):
    env = CavityCoolingEnv(seed=1, sde_backend=backend)
    env.sde_generator = np.random.default_rng(4)
    try:
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(np.array([1.]))
            assert np.isfinite(observation).all() and np.isfinite(reward)
        assert info['covariance_evaluations'] > info['covariance_bounds'] >= 0
    finally:
        env.close()


def test_particle_propagator_converges_from_zero_field():
    cal = Calibration.from_scenario()
    states = np.zeros((32, 8))
    states[:, 2] = 3e-6
    increments = np.random.default_rng(983).normal(size=(32, 1024, 5))*np.sqrt(cal.dt/1024)
    final = {}
    for steps in (128, 256, 1024):
        draws = increments.reshape(32, steps, 1024//steps, 5).sum(axis=2)
        final[steps], _ = propagate_particles(states, np.zeros(32), 1., .8, cal.dt, cal.physics(), draws, 1.)
    scales = np.array([1e-6]*3 + [.1]*3 + [10.]*2)
    errors = {n: np.sqrt(np.mean(((final[n]-final[1024])/scales)**2)) for n in (128, 256)}
    assert errors[256] < .85*errors[128]
