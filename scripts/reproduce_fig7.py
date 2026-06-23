from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def sem(values, ddof=1):
    values = np.asarray(values, dtype=float)
    count = np.sum(~np.isnan(values))
    if count <= 1:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanstd(values, ddof=ddof) / np.sqrt(count))


def load_config():
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    with open(script_dir / "params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return package_root, params["fig7"], params["output_dir"]


def load_simulation_data(csv_file):
    df = pd.read_csv(csv_file)
    episodes = df["time/episodes"].values
    timesteps = df["time/total_timesteps"].values
    rewards = df["rollout/ep_rew_mean"].values
    survival_fraction = df["rollout/trapped_end_fraction"].values
    valid_mask = ~(np.isnan(episodes) | np.isnan(rewards) | np.isnan(timesteps))
    episodes = episodes[valid_mask]
    timesteps = timesteps[valid_mask]
    rewards = rewards[valid_mask]
    if len(survival_fraction) > 0:
        survival_fraction = survival_fraction[valid_mask]
        survival_valid_mask = ~np.isnan(survival_fraction)
        survival_episodes = episodes[survival_valid_mask]
        survival_fraction = survival_fraction[survival_valid_mask]
    else:
        survival_episodes = np.array([])
        survival_fraction = np.array([])
    return episodes, timesteps, rewards, survival_episodes, survival_fraction


def load_experimental_data(experimental_csv):
    df = pd.read_csv(experimental_csv)
    df = df[df["replay_buffer_valid"].astype(bool)].copy()
    df = df.sort_values("episode")

    def split_phase(phase):
        phase_df = df[df["phase"] == phase].copy()
        return (
            phase_df["episode"].to_numpy(dtype=float),
            phase_df["total_reward"].to_numpy(dtype=float),
            phase_df["atom_survived"].to_numpy(dtype=float),
            phase_df["num_steps"].fillna(0).to_numpy(dtype=float).sum(),
        )

    pre_eps, pre_rewards, pre_survival, pre_steps = split_phase("pretraining")
    train_eps, train_rewards, train_survival, train_steps = split_phase("train")
    return pre_eps, pre_rewards, pre_survival, pre_steps, train_eps, train_rewards, train_survival, train_steps


def bin_data(episodes, rewards, survival_rates, bin_size):
    if len(episodes) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    max_episode = int(np.max(episodes))
    n_bins = (max_episode + bin_size) // bin_size
    out_episodes = []
    out_rewards = []
    out_rewards_sem = []
    out_survival = []
    out_survival_sem = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        mask = (episodes >= start) & (episodes < end)
        if np.sum(mask) == 0:
            continue
        # The division by 2 centers the bin at its midpoint for plotting.
        # For a bin from [start, end), the center is start + bin_size / 2.
        out_episodes.append(start + bin_size / 2)
        reward_bin = rewards[mask]
        out_rewards.append(np.mean(reward_bin))
        out_rewards_sem.append(sem(reward_bin))
        if len(survival_rates) > 0:
            survival_bin = survival_rates[mask]
            out_survival.append(np.mean(survival_bin))
            out_survival_sem.append(sem(survival_bin))
        else:
            out_survival.append(0.0)
            out_survival_sem.append(0.0)
    return (
        np.array(out_episodes),
        np.array(out_rewards),
        np.array(out_rewards_sem),
        np.array(out_survival),
        np.array(out_survival_sem),
    )


def main():
    package_root, cfg, output_dir_name = load_config()
    output_dir = package_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_episodes, sim_timesteps, sim_rewards, sim_survival_episodes, sim_survival = load_simulation_data(package_root / cfg["mlp_sim_training_csv"])
    pre_eps, pre_rewards, pre_survival, pre_steps, train_eps, train_rewards, train_survival, _ = load_experimental_data(
        package_root / cfg["experimental_csv"]
    )

    split_timestep = cfg["split_timestep"]
    pre_mask = sim_timesteps <= split_timestep
    train_mask = sim_timesteps > split_timestep
    sim_pre_episodes = sim_episodes[pre_mask]
    sim_pre_rewards = sim_rewards[pre_mask]
    sim_train_episodes = sim_episodes[train_mask]
    sim_train_rewards = sim_rewards[train_mask]

    if len(sim_survival_episodes) > 0:
        sim_survival_full = np.zeros(len(sim_episodes))
        for i, episode in enumerate(sim_episodes):
            closest_idx = np.argmin(np.abs(sim_survival_episodes - episode))
            if np.abs(sim_survival_episodes[closest_idx] - episode) <= 10:
                sim_survival_full[i] = sim_survival[closest_idx]
        sim_pre_survival = sim_survival_full[pre_mask]
        sim_train_survival = sim_survival_full[train_mask]
    else:
        sim_pre_survival = np.zeros(len(sim_pre_episodes))
        sim_train_survival = np.zeros(len(sim_train_episodes))

    bin_size = cfg["bin_size"]
    sim_pre_binned = bin_data(sim_pre_episodes, sim_pre_rewards, sim_pre_survival, bin_size)
    pretraining_binned = bin_data(pre_eps, pre_rewards, pre_survival, bin_size)
    sim_train_binned = bin_data(sim_train_episodes, sim_train_rewards, sim_train_survival, bin_size)
    train_binned = bin_data(train_eps, train_rewards, train_survival, bin_size)

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
        "figure.titlesize": 24,
        "lines.linewidth": 2,
        "lines.markersize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.figsize": (11, 5.75),
    })

    colors = {
        "sim_pre": "#56B4E9",
        "sim_train": "#0072B2",
        "pretraining": "#E69F00",
        "train": "#D55E00",
    }
    labels = {
        "sim_pre": "Simulation (Pre-training)",
        "pretraining": "Experimental (Pre-training)",
        "sim_train": "Simulation (Training)",
        "train": "Experimental (Training)",
    }

    fig, ax = plt.subplots()
    datasets = {
        "sim_pre": sim_pre_binned,
        "pretraining": pretraining_binned,
        "sim_train": sim_train_binned,
        "train": train_binned,
    }
    for name, data in datasets.items():
        episodes, rewards, rewards_sem, _, _ = data
        if len(episodes) == 0:
            continue
        ax.errorbar(
            episodes,
            rewards,
            yerr=rewards_sem,
            label=labels[name],
            color=colors[name],
            marker="o" if "sim" in name else "s",
            markersize=8,
            linewidth=2,
            capsize=3,
            capthick=1.5,
            alpha=0.8 if "pre" in name else 1.0,
        )

    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Episode Reward")
    ax.legend(loc="best")
    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1.5)
    plt.tight_layout()
    output_pdf = output_dir / cfg["output_pdf"]
    output_png = output_dir / cfg["output_pdf"].replace(".pdf", ".png")
    plt.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
