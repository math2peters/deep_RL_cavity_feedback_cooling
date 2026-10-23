# Deep RL Feedback Cooling

Code, curated measurements and frozen controllers for *Deep Reinforcement Learning for
Individual Atomic Control and Cooling*. The package redraws figures and reruns the
simulation benchmarks of Figs. 3 and 4. `reproduce_fig6.py` draws the transmission calibration
(Fig. 5) and `reproduce_fig7.py` the training curves (Fig. 7); the methods schematic (Fig. 6) is
drawn in the manuscript. 

## Redraw the figures

Use Python 3.10 in a fresh environment. Run all commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
for figure in 1 2 3 4 6 7; do
    MPLBACKEND=Agg python scripts/reproduce_fig${figure}.py
done
```

Figures and numerical summaries go to `outputs/`. `data/source_data_fig*/` holds the source data of each figure, and
`scripts/params.yaml` sets the paths for the other scripts. `models/` holds the two policies
compared in Fig. 3(c,d), MLP (Sim.) and MLP (Expt.). `data/source_data_fig4/` holds the Fig. 4
actors, per-episode results, selected parameters, operation counts and checksums.


## Rerun the simulation benchmarks

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/verify_sde_backend.py --episodes 5
python scripts/compare_fig3_policies.py
python scripts/evaluate_fig4.py --workers 8
python scripts/reproduce_fig4.py --evaluations outputs/fig4_evaluations
for sweep in detuning photon_number; do
    python scripts/sweep_fig3.py --sweep "$sweep" --workers 8
    python scripts/sweep_fig3.py --sweep "$sweep" --workers 8 --noisy
done
python scripts/reproduce_fig3.py --simulation outputs/fig3_sweeps
```

