"""Source data, frozen learned actors and statistics for Fig. 4.

Plotting needs only NumPy; torch and the simulator are imported only to rerun controllers.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/source_data_fig4"
SCENARIOS = {
    "nominal": {},
    "hot_delta12p5": {"temperature_input": 800e-6, "probe_detuning_input": 2*np.pi*12.5e6},
}
LEARNED = ("mlp", "mlp_large", "affine")   # 16x16 MLP (deployed size), 64x64 MLP, affine actor
FIELDS = ("seeds", "retained", "energy20", "curve_n", "curve_mean", "curve_m2")
STEP_US, WARMUP_BINS, STEP20, FULL_STEPS = 27.3, 6, 26, 106


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    result = json.loads((DATA / "manifest.json").read_text())
    for name, digest in result["files_sha256"].items():
        if sha256(DATA / name) != digest:
            raise ValueError(f"Packaged file checksum mismatch: {name}")
    return result


def validate_record(record, seeds):
    np.testing.assert_array_equal(record["seeds"], seeds)
    if record["retained"].dtype != np.dtype(bool) or record["energy20"].shape != (len(seeds),):
        raise ValueError("Malformed episode data")
    if int(record["curve_n"]) != int(record["retained"].sum()):
        raise ValueError("Curve cohort differs from full-horizon survivors")
    if not np.isfinite(record["energy20"][record["retained"]]).all():
        raise ValueError("Missing energy for a retained episode")


def load_groups(evaluations=None):
    """Records grouped by scenario and family, from the package or a complete rerun directory."""
    info = manifest()
    groups = {name: {} for name in SCENARIOS}
    rerun_version = None
    with np.load(DATA / "evaluations.npz", allow_pickle=False) as bundled:
        for policy in info["policies"]:
            name = policy["id"]
            if evaluations is None:
                record = {key: bundled[f"{name}/{key}"] for key in FIELDS}
            else:
                with np.load(Path(evaluations) / f"{name}.npz", allow_pickle=False) as rerun:
                    if "noise_model" not in rerun:
                        raise ValueError("Unversioned evaluation predates the covariance correction; rerun every policy")
                    version = str(rerun["noise_model"].item())
                    if rerun_version is not None and version != rerun_version:
                        raise ValueError("Cannot combine evaluations from different noise models")
                    rerun_version = version
                    record = {key: rerun[key] for key in FIELDS}
            validate_record(record, np.arange(info["seed_start"], info["seed_start"]+info["episodes"]))
            groups[policy["scenario"]].setdefault(policy["family"], []).append(record)
    return groups


def metric_mean(records, metric, draws=2000, seed=4026):
    """Mean over seeds of the per-seed mean, and its bootstrap standard error.

    Each draw resamples the training or filter seeds and the episodes with replacement.
    """
    if metric == "energy":
        values = np.array([np.where(r["retained"], r["energy20"], np.nan) for r in records])
    elif metric == "retention":
        values = 100*np.array([r["retained"] for r in records], dtype=float)
    else:
        raise ValueError(metric)
    mean = float(np.nanmean(np.nanmean(values, axis=1)))
    rng = np.random.default_rng(seed)
    n_rep, n_ep = values.shape
    boot = [np.nanmean(np.nanmean(values[rng.integers(0, n_rep, n_rep)][:, rng.integers(0, n_ep, n_ep)], axis=1))
            for _ in range(draws)]
    return mean, float(np.std(boot))


def cooling_curve(records):
    """Mean and standard error of the energy of all surviving trajectories, pooled over seeds."""
    records = [r for r in records if int(r["curve_n"]) > 0]
    count = sum(int(r["curve_n"]) for r in records)
    mean = sum(int(r["curve_n"])*r["curve_mean"] for r in records)/count
    m2 = sum(r["curve_m2"] + int(r["curve_n"])*(r["curve_mean"]-mean)**2 for r in records)
    sem = np.sqrt(np.maximum(m2, 0)/(count-1)/count)
    return np.arange(len(mean))*STEP_US/1000, mean, sem


def compact_record(seeds, retained, energies):
    """Per-episode survival and energy at 546 us, plus survivor curve statistics (mean, M2)."""
    curves = energies[retained, WARMUP_BINS:]
    n = int(retained.sum())
    mean = curves.mean(axis=0) if n else np.full(FULL_STEPS-WARMUP_BINS, np.nan)
    record = dict(seeds=np.asarray(seeds, dtype=np.int64), retained=retained, energy20=energies[:, STEP20],
                  curve_n=np.array(n), curve_mean=mean, curve_m2=((curves-mean)**2).sum(axis=0))
    validate_record(record, record["seeds"])
    return record


class FrozenActor:
    """Deterministic actor from an actors.npz file, reproducing SB3's float32 forward pass exactly."""
    def __init__(self, name, path=DATA / "actors.npz"):
        import torch
        from torch import nn
        self.torch = torch
        layers = []
        with np.load(path, allow_pickle=False) as weights:
            n = int(weights[f"{name}/layers"])
            for i in range(n):
                w, b = weights[f"{name}/w{i}"], weights[f"{name}/b{i}"]
                layer = nn.Linear(w.shape[1], w.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(w))
                    layer.bias.copy_(torch.from_numpy(b))
                layers.append(layer)
                if i < n-1:
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
        self.model = nn.Sequential(*layers, nn.Tanh()).eval()

    def predict(self, observation, deterministic=True):
        with self.torch.no_grad():
            value = self.model(self.torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, 28)))
        action = value.numpy().reshape(1)
        return -1.0 + (0.5*(action+1.0)*2.0), None   # SB3's rescaling to [-1, 1], kept for bitwise equality


def actor_weights(checkpoint):
    """The deterministic actor of an SB3 SAC checkpoint, in the format FrozenActor reads."""
    from stable_baselines3 import SAC
    from torch import nn
    model = SAC.load(checkpoint, device="cpu", custom_objects={"lr_schedule": lambda _: 1e-4})
    linear = [layer for layer in model.actor.latent_pi if isinstance(layer, nn.Linear)] + [model.actor.mu]
    weights = {"layers": np.array(len(linear))}
    for i, layer in enumerate(linear):
        weights[f"w{i}"] = layer.weight.detach().numpy().copy()
        weights[f"b{i}"] = layer.bias.detach().numpy().copy()
    return weights


def selected_policies(tuning=None, scenario=None, families=None):
    """Frozen paper policies, optionally using parameters from a new tuning run."""
    policies = [p for p in manifest()["policies"] if (scenario is None or p["scenario"] == scenario)
                and (families is None or p["family"] in families)]
    if tuning is not None:
        selected = json.loads(Path(tuning).read_text())["selected"]
        for policy in policies:
            if policy["family"] not in ("differentiator", "kalman", "particle"):
                continue
            parameters = dict(selected[policy["scenario"]][policy["family"]]["parameters"])
            if policy["family"] == "particle":
                parameters["filter_seed"] = policy["parameters"]["filter_seed"]
            policy["parameters"] = parameters
    return policies


def load_controller(policy, actors=None):
    from src.evaluation.benchmark_controllers import (
        ConstantController, FilteredDifferentiatorController, KalmanQuadratureController,
        ParticleFeedbackController)
    if policy["family"] in LEARNED:
        return FrozenActor(policy["id"], DATA / "actors.npz" if actors is None else actors)
    return {"differentiator": FilteredDifferentiatorController,
            "kalman": KalmanQuadratureController, "particle": ParticleFeedbackController,
            "none": ConstantController}[policy["family"]](**policy["parameters"])
