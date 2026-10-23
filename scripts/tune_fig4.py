"""Tune the differentiator, Kalman and particle controllers of Fig. 4 by direct search.

Every controller and operating point uses the same three-stage search, scored by the mean
total (undiscounted) episode reward, also used to select the SAC checkpoints:

  1. 256 candidates from a scrambled Sobol sequence over the parameter ranges, each run on
     the same 24 episodes (320001-320024);
  2. three rounds of local refinement on 64 new episodes (325001-325064): the 16 best so far
     and four Gaussian perturbations of each, with standard deviations of 0.06, 0.03 and
     0.015 of each normalized range in the three rounds;
  3. the 12 best refined candidates, run on 256 validation episodes (330001-330256).

The search is repeated --replicates times (default 4) with independent Sobol scrambles and
perturbations, and the candidate with the highest validation reward over all searches is
selected. Agreement between the searches measures how well the search has converged.

Search ranges (log-uniform where marked):
  differentiator  mean depth in [0.3, 1], |gain| in [0.05, 8] (log) of either sign,
                  filter time in [1, 316] us (log)
  kalman          frequency f in [4, 16] kHz, process-noise scale q in [1e-4, 20] (log),
                  phase in [0, 2 pi), gain in [0.02, 0.7], mean depth in [0.3, 1],
                  amplitude floor in [1e-4, 1] (log); amplitude-normalized like the particle law
  particle        phase in [0, 2 pi), gain in [0.02, 0.7], mean depth in [0.3, 1],
                  energy floor in [1e-4, 1] (log); 512 particles, 64 substeps, filter seed 41026

Stage records and tuning.json go to --output-dir. Pass that tuning.json to
evaluate_fig4.py with --tuning to evaluate the new selections.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.rl_env.noise_covariance import MODEL_VERSION

KINDS = ("differentiator", "kalman", "particle")
SCENARIOS = ("nominal", "hot_delta12p5")
DIMENSIONS = {"differentiator": 4, "kalman": 6, "particle": 4}
PHASE_DIMENSION = {"kalman": 2, "particle": 0}
STAGES = [("coarse", 320001, 24), ("refined", 325001, 64), ("validation", 330001, 256)]
COARSE_LOG2, LEADERS, PERTURBATIONS, WIDTHS, FINALISTS = 8, 16, 4, (0.06, 0.03, 0.015), 12
CHUNK = 8


def decode(u, kind, scenario):
    u = [float(x) for x in u]
    if kind == "differentiator":
        sign = 1.0 if u[2] >= 0.5 else -1.0
        return {"depth": 0.3 + 0.7*u[0], "gain": sign*0.05*160**u[1],
                "filter_time": 1e-6*10**(2.5*u[3])}
    if kind == "kalman":
        return {"scenario": scenario, "frequency_hz": 4000 + 12000*u[0], "process_noise": 10**(-4 + 5.3*u[1]),
                "phase": 2*np.pi*u[2], "gain": 0.02 + 0.68*u[3], "depth": 0.3 + 0.7*u[4],
                "amplitude_floor": 10**(-4 + 4*u[5])}
    return {"scenario": scenario, "particles": 512, "substeps": 64, "filter_seed": 41026,
            "phase": 2*np.pi*u[0], "gain": 0.02 + 0.68*u[1], "depth": 0.3 + 0.7*u[2],
            "energy_floor": 10**(-4 + 4*u[3])}


def run_chunk(task):
    kind, scenario, parameters, seeds = task
    import torch
    torch.set_num_threads(1)
    from scripts.evaluate_fig4 import run_episode
    from src.evaluation.benchmark_controllers import BenchmarkEnv
    from src.evaluation.fig4 import SCENARIOS as ENV_SCENARIOS, load_controller
    controller = load_controller({"family": kind, "parameters": parameters})
    env = BenchmarkEnv(**ENV_SCENARIOS[scenario])
    try:
        return [float(run_episode(env, controller, seed)[0]) for seed in seeds]
    finally:
        env.close()


def stage(pool, kind, scenario, units, seed_start, episodes, path):
    if path.exists():   # resume: reuse a completed stage with the same candidates, episodes and model
        records = [json.loads(line) for line in path.read_text().splitlines()]
        expected = [{"noise_model": MODEL_VERSION, "unit": [float(x) for x in u],
                     "parameters": json.loads(json.dumps(decode(u, kind, scenario))),
                     "seed_start": seed_start, "episodes": episodes} for u in units]
        if len(records) == len(units) and all({key: r.get(key) for key in e} == e and len(r.get("rewards", ())) == episodes
                                             for r, e in zip(records, expected)):
            return records
    seeds = list(range(seed_start, seed_start + episodes))
    tasks, owners = [], []
    for i, u in enumerate(units):
        parameters = decode(u, kind, scenario)
        for j in range(0, episodes, CHUNK):
            tasks.append((kind, scenario, parameters, seeds[j:j+CHUNK]))
            owners.append(i)
    rewards = [[] for _ in units]
    for i, chunk in zip(owners, pool.map(run_chunk, tasks)):
        rewards[i].extend(chunk)
    records = [{"noise_model": MODEL_VERSION, "unit": [float(x) for x in u], "parameters": decode(u, kind, scenario), "seed_start": seed_start,
                "episodes": episodes, "reward": float(np.mean(r)), "rewards": [float(x) for x in r]}
               for u, r in zip(units, rewards)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def tune(pool, kind, scenario, out, replicate=0):
    from scipy.stats import qmc
    out.mkdir(parents=True, exist_ok=True)
    dimension = DIMENSIONS[kind]
    units = qmc.Sobol(dimension, scramble=True, seed=83126 + replicate).random_base2(COARSE_LOG2)
    started = time.time()
    coarse = stage(pool, kind, scenario, units, *STAGES[0][1:], out/"coarse.jsonl")
    rng = np.random.default_rng(93126 + replicate)
    fine = []
    for round_, width in enumerate(WIDTHS, 1):
        best = sorted(coarse if round_ == 1 else fine, key=lambda r: -r["reward"])[:LEADERS]
        leaders = [np.array(r["unit"]) for r in best]
        candidates = list(leaders) if round_ == 1 else []   # later rounds reuse the leaders' scores
        for leader in leaders:
            for _ in range(PERTURBATIONS):
                u = leader + rng.normal(0, width, dimension)
                if kind in PHASE_DIMENSION:
                    u[PHASE_DIMENSION[kind]] %= 1
                candidates.append(np.clip(u, 0, 1))
        fine += stage(pool, kind, scenario, candidates, *STAGES[1][1:], out/f"refined{round_}.jsonl")
    finalists = [np.array(r["unit"]) for r in sorted(fine, key=lambda r: -r["reward"])[:FINALISTS]]
    validation = stage(pool, kind, scenario, finalists, *STAGES[2][1:], out/"validation.jsonl")
    best = max(validation, key=lambda r: r["reward"])
    summary = {"kind": kind, "scenario": scenario, "replicate": replicate, "selected": best,
               "best_refined_by_round": [max(r["reward"] for r in fine[:LEADERS*(1 + PERTURBATIONS*k)])
                                        for k in range(1, len(WIDTHS) + 1)],
               "minutes": (time.time() - started)/60}
    (out/"summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"tuned": kind, "scenario": scenario, "replicate": replicate, "reward": round(best["reward"], 2),
                      "minutes": round(summary["minutes"], 1)}), flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/fig4_tuning")
    parser.add_argument("--replicates", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "tuning.json"
    selections = json.loads(selection_path.read_text()) if selection_path.exists() else {"selected": {}}
    if selections.get("noise_model") != MODEL_VERSION:
        selections = {"selected": {}}
    selections.update({
        "noise_model": MODEL_VERSION, "objective": "mean total episode reward", "stages": [{"name": n, "seed_start": s, "episodes": e} for n, s, e in STAGES],
        "searches": args.replicates, "candidates": {"coarse": 2**COARSE_LOG2, "refined": LEADERS*(1 + len(WIDTHS)*PERTURBATIONS),
                       "refinement_widths": list(WIDTHS), "validation": FINALISTS},
        "selected": selections["selected"]})
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for kind in args.kinds:
            for scenario in args.scenarios:
                searches = [tune(pool, kind, scenario, args.output_dir/scenario/kind/f"search{r}", r)
                            for r in range(args.replicates)]
                best = max(searches, key=lambda r: r["reward"])
                selections["selected"].setdefault(scenario, {})[kind] = {
                    "parameters": best["parameters"], "validation_reward": best["reward"],
                    "search": searches.index(best),
                    "search_rewards": [r["reward"] for r in searches],
                    "search_parameters": [r["parameters"] for r in searches]}
                selection_path.write_text(json.dumps(selections, indent=1) + "\n")


if __name__ == "__main__":
    main()
