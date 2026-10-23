"""Source data for Fig. 3(c,d): MLP (Sim.) and MLP (Expt.) on the same measured histories.

Both frozen actors receive the policy inputs recorded on the hardware
(data/source_data_fig3/hardware_histories.npz): the pretraining episodes, the last 300
training episodes and the fixed-policy episodes, keeping actions 22-99 of each, after both
moving averages of the count have initialized. For every history the script records both
actors' trap depths and the derivative of each depth with respect to the normalized count
0-6 bins in the past, including its effect on the two moving averages; the command history
is held fixed. Writes data/source_data_fig3/policy_comparison.npz, plotted by reproduce_fig3.py.
"""
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data/source_data_fig3"
FIRST_ACTION, LAST_ACTION, FRAMES = 22, 100, 7
FAST, SLOW = .935, .99   # decay per bin of the fast and slow count averages (CavityCoolingEnv)


def depth_and_gradient(model, inputs):
    """Trap depth U/U_max = ((tanh(mu)+1)/2)^2 of the deterministic actor, and its input gradient."""
    x = torch.tensor(inputs, requires_grad=True)
    mean, _, _ = model.actor.get_action_dist_params(x)
    depth = ((torch.tanh(mean)+1)/2)**2
    return depth.detach().numpy().ravel(), torch.autograd.grad(depth.sum(), x)[0].numpy()


def count_sensitivity(gradient):
    """Derivative of the depth with respect to the count `lag` bins in the past.

    Input columns per frame: command, count, fast average and 2(slow - fast). A count
    entering frame k also moves both averages in every later frame j by (1-beta) beta^(j-k).
    """
    g = gradient.astype(np.float64).reshape(-1, FRAMES, 4)
    result = np.empty((len(g), FRAMES))
    for lag in range(FRAMES):
        k = FRAMES-1-lag
        value = g[:, k, 1].copy()
        for j in range(k, FRAMES):
            fast, slow = (1-FAST)*FAST**(j-k), (1-SLOW)*SLOW**(j-k)
            value += g[:, j, 2]*fast + 2*g[:, j, 3]*(slow-fast)
        result[:, lag] = value
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    with np.load(DATA/"hardware_histories.npz", allow_pickle=False) as data:
        observations, steps, cohort = data["observations"], data["steps"], data["cohort"]
    inputs, episode = [], []
    for e, (rows, n) in enumerate(zip(observations, steps)):
        for action in range(FIRST_ACTION, min(n, LAST_ACTION)):
            inputs.append(rows[action-FRAMES:action].ravel())   # the seven observations before the action
            episode.append(e)
    inputs, episode = np.asarray(inputs, dtype=np.float32), np.asarray(episode)
    depths, sensitivities = [], []
    for name in ("mlp_sim", "mlp_experimental"):
        model = SAC.load(ROOT/f"models/{name}.zip", device="cpu", custom_objects={"lr_schedule": lambda _: 1e-4})
        depth, gradient = depth_and_gradient(model, inputs)
        depths.append(depth)
        sensitivities.append(count_sensitivity(gradient))
    sums = np.zeros((len(steps), 2, FRAMES))
    np.add.at(sums, episode, np.stack(sensitivities, axis=1)**2)
    np.savez_compressed(DATA/"policy_comparison.npz", depth=np.column_stack(depths),
                        episode_states=np.bincount(episode, minlength=len(steps)),
                        episode_sum_squared_kernels=sums, episode_cohort=cohort)
    print(f"Saved {DATA/'policy_comparison.npz'}: {len(inputs)} histories from {len(steps)} episodes")
