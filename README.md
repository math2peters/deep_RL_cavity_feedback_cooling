# Deep RL Feedback Cooling

This folder reproduces the paper plots for Figs. 1-4, Table I, and supplemental Figures. Fig. 5 is a methods schematic and is not part of this reproduction package.

This is a plot reproduction package built from a subset of unaltered source data. Full datasets and experimental source code are available upon reasonable request.

## License

This repository is released under the MIT License. See `LICENSE`.

## Contents

- `data/source_data_fig1/` contains extracted Fig. 1 shot metadata, trace data, and the inset trace. 
- `data/source_data_fig2/` contains extracted Fig. 2 shot metadata, trace data, the example trace, and the transmission-to-energy calibration. 
- `data/source_data_fig3/` contains the experimental-run detuning and photon-count survival sweep CSVs.
- `data/source_data_fig4/` contains the `mlp_sim`, `mlp_experimental`, and `differentiator` sweep folders, energy traces, and turning-point analysis used for Fig. 4 and Table I.
- `data/source_data_fig6/` contains the fixed-energy and thermal transmission calibration CSVs used for the supplemental transmission figure, along with scripts that regenerate those CSVs.
- `data/source_data_fig7/` contains the MLP (Sim.) training CSV and the extracted experimental episode table.
- `models/` contains the trained MLP policy checkpoints. The differentiator baseline is implemented in `src/rl_env/differentiator.py`; its `.zip` file is retained only for scripts that expect every named controller to have a model path.
- `scripts/params.yaml` contains packaged paths, output names, and fixed inputs.
- `src/` contains the RL environment, training, and evaluation code used to regenerate source data.
- `outputs/` contains generated figures and table files.

## Environment

Python `3.10.12` was used for the validated runs in this package.

Install dependencies from the package root:

```bash
python -m pip install -r requirements.txt
```

The RL evaluation code depends on a pinned `numpy`/`numba`/`torch` stack. If an existing environment already has newer packages installed, reinstalling from `requirements.txt` is recommended before running `src/`.

## Reproduction

Run from the package root:

```bash
python scripts/reproduce_fig1.py
python scripts/reproduce_fig2.py
python scripts/reproduce_fig3.py
python scripts/reproduce_fig4.py
python scripts/reproduce_table1.py
python scripts/reproduce_fig6.py
python scripts/reproduce_fig7.py
```

The scripts write:

- `outputs/Fig1.pdf` and `outputs/Fig1.png`
- `outputs/Fig2.pdf` and `outputs/Fig2.png`
- `outputs/Fig3.pdf` and `outputs/Fig3.png`
- `outputs/Fig4.pdf` and `outputs/Fig4.png`
- `outputs/Fig6.pdf` and `outputs/Fig6.png`
- `outputs/Fig7.pdf` and `outputs/Fig7.png`
- `outputs/table1.csv`, `outputs/table1.tex`, and the detailed performance metric CSVs

## RL Evaluation

The `src/` tree can run RL agents in the simulated environment or regenerate simulation sweeps from the packaged models. For example:

```bash
python -m src.evaluation.evaluate_multiprocess --model mlp_sim --episodes 1 --workers 1 --render 0
python -m src.evaluation.evaluate_multiprocess --model differentiator --episodes 1 --workers 1 --render 0
```

By default, `src.evaluation.analyze_network_boundaries` and `src.evaluation.run_sweep` write regenerated Fig. 4 sweep outputs directly into `data/source_data_fig4/` so they are immediately compatible with the plotting scripts. Use `--output-dir` if you want to write trial runs elsewhere.

The two large Fig. 4 trajectory cache files in `data/source_data_fig4/data_cache/` are omitted from the GitHub repository because they exceed GitHub's per-file size limit. They can be regenerated locally with:

```bash
python -m src.evaluation.collect_atom_trajectories --model models/mlp_sim.zip --episodes 40000 --workers 8 --save-path data/source_data_fig4/data_cache/mlp_sim_40k.pkl
python -m src.evaluation.collect_atom_trajectories --model models/differentiator.zip --episodes 40000 --workers 8 --save-path data/source_data_fig4/data_cache/differentiator_40k.pkl
python -m src.evaluation.convert_trajectories --data-dir data/source_data_fig4/data_cache --max-points 20
```

These commands generate `mlp_sim_40k_turning_points.pkl` and `differentiator_40k_turning_points.pkl`, which are the cache filenames consumed by `scripts/reproduce_fig4.py`.
