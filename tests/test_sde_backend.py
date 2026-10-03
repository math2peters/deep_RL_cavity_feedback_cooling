import pytest

np = pytest.importorskip("numpy")
sdeint = pytest.importorskip("sdeint")
pytest.importorskip("gymnasium")
pytest.importorskip("numba")

from scripts.verify_sde_backend import compare, run_episodes
from src.rl_env.environment import draw_srs2_noise


def test_compiled_noise_matches_sdeint_bit_for_bit():
    n_steps, n_wiener, h = 381, 8, 27.3e-6 / 381
    generator = np.random.default_rng(3)
    dW_ref = sdeint.deltaW(n_steps, n_wiener, h, generator)
    _, J_ref = sdeint.Jkpw(dW_ref, h, generator=generator)
    dW, J = draw_srs2_noise(n_steps, n_wiener, h, np.random.default_rng(3))
    assert np.array_equal(dW, dW_ref)
    assert np.array_equal(J, J_ref)


def test_numba_backend_reproduces_sdeint_episode():
    reference, _ = run_episodes("sdeint", [101], noise_seed=7)
    candidate, _ = run_episodes("numba", [101], noise_seed=7)
    all_ok, _ = compare(reference, candidate)
    assert all_ok


@pytest.mark.parametrize('env_kwargs', [
    {'diffusion_on': False, 't_max': 27.3e-6 * 12},
    {'noisy_measurements': True, 't_max': 27.3e-6 * 12},
    {'frame_wait_mode': False, 'atom_capture': True, 't_max': 27.3e-6 * 12},
    {'truncate_if_untrapped': False, 'temperature_input': 2e-3, 't_max': 27.3e-6 * 12},
])
def test_backends_match_other_environment_modes(env_kwargs):
    reference, _ = run_episodes('sdeint', [202], noise_seed=9, env_kwargs=env_kwargs)
    candidate, _ = run_episodes('numba', [202], noise_seed=9, env_kwargs=env_kwargs)
    assert compare(reference, candidate)[0]


@pytest.fixture
def episode_record():
    from scripts.verify_sde_backend import EXACT_KEYS, CONTINUOUS_KEYS
    record = {key: [1.0] for key in EXACT_KEYS + CONTINUOUS_KEYS}
    record['final_temperature'] = 1e-6
    record['terminal_info'] = {'final_temperature': 1e-6, 'trapped_end': True,
                               'mean_ke_z': None, 'episode': {'r': 1.0, 'l': 1}}
    return record


@pytest.mark.parametrize('field', ['final_temperature', 'state', 'trajectory', 'fields', 'photon_sums'])
def test_comparison_rejects_continuous_changes(episode_record, field):
    from copy import deepcopy
    changed = deepcopy(episode_record)
    changed[field] = np.asarray(changed[field]) * 1.01
    assert not compare([episode_record], [changed])[0]


@pytest.mark.parametrize('field', ['initial_obs', 'obs', 'reward', 'counts', 'trapped', 'terminated', 'truncated'])
def test_comparison_rejects_discrete_changes(episode_record, field):
    from copy import deepcopy
    changed = deepcopy(episode_record)
    changed[field] = [0.0]
    assert not compare([episode_record], [changed])[0]


def test_comparison_checks_all_terminal_metrics(episode_record):
    from copy import deepcopy
    changed = deepcopy(episode_record)
    changed['terminal_info']['mean_ke_z'] = 1e-6
    assert not compare([episode_record], [changed])[0]


def test_comparison_rejects_missing_and_empty_episodes(episode_record):
    assert not compare([episode_record], [])[0]
    assert not compare([], [])[0]
    assert not compare([episode_record], [episode_record, episode_record])[0]


def test_comparison_rejects_nonfinite_and_mismatched_states(episode_record):
    from copy import deepcopy
    for value in [[float('nan')], [float('inf')], [1.0, 2.0]]:
        changed = deepcopy(episode_record)
        changed['state'] = value
        assert not compare([episode_record], [changed])[0]


def test_noise_context_preserves_explicit_generator_seeds():
    from scripts.verify_sde_backend import seeded_sde_noise
    expected = np.random.default_rng(51).normal(size=8)
    with seeded_sde_noise(7):
        assert np.array_equal(np.random.default_rng(seed=51).normal(size=8), expected)


def test_backends_match_early_escape_with_trap_off():
    class TrapOff:
        def predict(self, observation):
            return np.array([-1.0]), None

    kwargs = {'frame_wait_mode': False, 'temperature_input': 2e-3}
    reference, _ = run_episodes('sdeint', [404], 27, env_kwargs=kwargs, controller=TrapOff())
    candidate, _ = run_episodes('numba', [404], 27, env_kwargs=kwargs, controller=TrapOff())
    assert reference[0]['truncated'][-1]
    assert compare(reference, candidate)[0]
