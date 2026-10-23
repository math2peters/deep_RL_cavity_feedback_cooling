"""Pack the best checkpoints of scripts/train_actors.py into the actor file that Fig. 4 is computed from.

  python scripts/export_actors.py            # writes outputs/fig4_actors.npz
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.train_actors import run_directory
from src.evaluation.fig4 import DATA, LEARNED, actor_weights, sha256
from src.rl_env.noise_covariance import MODEL_VERSION


def completed(checkpoint):
    """checkpoint, if it was written by a completed run of the current model."""
    config_path = checkpoint.parent/"config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if "total_timesteps" not in config:
        raise ValueError(f"{checkpoint.parent}: training did not complete")
    if config.get("noise_model") != MODEL_VERSION:
        raise ValueError(f"{checkpoint.parent}: trained with noise model {config.get('noise_model')}, not {MODEL_VERSION}")
    if not checkpoint.exists() or config.get("checkpoints_sha256", {}).get(checkpoint.name) != sha256(checkpoint):
        raise ValueError(f"{checkpoint} does not belong to the completed run")
    return checkpoint


def pack(path, checkpoints):
    arrays = {f"{name}/{key}": value for name, checkpoint in checkpoints.items()
              for key, value in actor_weights(completed(checkpoint)).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Saved {len(checkpoints)} actors to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/fig4_actors.npz")
    args = parser.parse_args()
    manifest = json.loads((DATA/"manifest.json").read_text())
    pack(args.output, {p["id"]: run_directory(p["family"], p["scenario"], p["training_seed"])/"best_model.zip"
                       for p in manifest["policies"] if p["family"] in LEARNED})
