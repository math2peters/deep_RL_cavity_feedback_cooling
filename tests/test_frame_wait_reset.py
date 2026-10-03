import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sdeint")
pytest.importorskip("gymnasium")
pytest.importorskip("numba")

from src.rl_env.environment import CavityCoolingEnv

REWARD_SCALE = 3.0


def run_to_end(env):
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.array([0.8]))
    return env.reward_list


@pytest.mark.parametrize("interrupt_after", [1, 4, 7])
def test_reset_during_frame_wait_restores_reward_scale(interrupt_after):
    env = CavityCoolingEnv(seed=5, reward_scale=REWARD_SCALE)
    try:
        env.reset(seed=5)
        for _ in range(interrupt_after):
            env.step(np.array([0.8]))
        assert env.reward_scale == 0  # frame-wait steps zero the scale

        rewards = run_to_end(env)

        assert env.reward_scale == REWARD_SCALE
        assert any(r != 0 for r in rewards[env.frame_stack_number:])
    finally:
        env.close()


def test_reset_after_completed_episode_keeps_reward_scale():
    env = CavityCoolingEnv(seed=5, reward_scale=REWARD_SCALE)
    try:
        run_to_end(env)
        env.reset()
        assert env.reward_scale == REWARD_SCALE
    finally:
        env.close()
