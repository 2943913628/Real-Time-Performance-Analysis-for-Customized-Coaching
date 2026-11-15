

## Key results (test sessions, mean of 3 seeds)

| Model | Macro-F1 | PRI R² | NPV (policy) | J |
|---|---|---|---|---|
| **SPRINT (CLEAR-GWO θ*)** | 0.869 | 0.934 | 0.873 | 0.886 |
| HGB (best overall) | 0.879 | 0.944 | 0.931 | 0.908 |
| TARNet-style / LSTM / MLP | 0.882 / 0.884 / 0.882 | 0.933 / 0.934 / 0.931 | 0.912 / 0.905 / 0.905 | 0.902 / 0.902 / 0.900 |

* SPRINT closes ~85% of the gap between logged coach decisions and the oracle, with ~0.3% unsafe advice.
* Short windows (1 min): SPRINT macro-F1 0.787 vs 0.712 for the best baseline; 40% missing channels: 0.785 (MLP+Aug 0.781).
* Leave-one-sport-out: SPRINT best mean macro-F1 (0.759 vs MLP 0.744, TCN 0.680).
* CLEAR-GWO: best geometric-mean rank on 8 benchmark functions (tied with DE); proxy-optimal config is compact (0.24 M MACs, 0.15 ms)
  but generalised slightly worse than larger configurations (reported transparently in the paper).
* On clean data, compact recurrent/descriptor/tree baselines are slightly more accurate than SPRINT — the paper reports and discusses this.

All numbers: `results/tables/` (paper tables exported as `paper_TableNN.csv`).


## Data (CS-PERFORM digital twin) — IMPORTANT

The data are **simulated** by `code/simulate.py` (physiology-grounded digital twin: Banister fitness–fatigue model,
within-session metabolic/neuromuscular/technique dynamics, 12 wearable channels, logged coach interventions with known
propensities and the full set of potential outcomes). Simulation is what allows exact counterfactual evaluation of coaching
policies; the manuscript states this explicitly. Validate on real squads before drawing practical conclusions.

* `athlete_profiles.csv` – observable profile (96 athletes)
* `athlete_latent_traits_GROUND_TRUTH.csv` – latent traits (evaluation/analysis only, never used as model input)
* `sensor_stream_15s_epochs.csv.gz` – 414,720 rows: snapshot, athlete, epoch, 12 channels (NaN = sensor drop-out)
* `sensor_stream_SAMPLE_first2000rows.csv` – small preview
* `window_labels_interventions.csv` – 20,736 athlete-windows: state, PRI, context, logged action, observed outcome,
  potential outcomes `Q_0..Q_4`, logging propensities `prop_0..prop_4`, latent states

## Code

| File | Purpose |
|---|---|
| `simulate.py` | Generates CS-PERFORM (seed 20260519) |
| `autograd.py` | NumPy reverse-mode autodiff engine (gradient-checked; `test_autograd.py`) |
| `data.py` | Loading, cleaning, normalisation, chronological split, causal response memory, physiological prior graph |
| `models.py` | SPRINT (MS-GTA, TAG, descriptor branch, read-out, FiLM, PCGA, CIM) and deep baselines |
| `train.py` | Multi-task IPW training loop, augmentation, metrics, DR off-policy value, safety-constrained recommender, latency |
| `optim.py` | CLEAR-GWO, GWO, I-GWO, PSO, DE, WOA, RS and benchmark functions |
| `hpo.py` | Resource-aware hyper-parameter search of SPRINT |
| `run_exp.py`, `run_sprint.py` | One experiment run (main / cold-start / LOSO / ablation), robustness and interpretability dumps |
| `classical.py` | LR, SVM, RF, HGB and meta-learner baselines |
| `bench_opt.py` | Optimizer benchmark on 8 functions + CLEAR-GWO component ablation |
| `analyze.py`, `paper_build.py`, `paper_text.py`, `paper_refs.py`, `make_docx.js` | Tables, figures, statistics and the manuscript |
| `fig_diagrams.py` | Architecture (Fig. 1) and methodology flow (Fig. 2) diagrams |
| `jobs/`, `run_queue.sh` | The exact job list used to produce all results |

### Reproduce

```bash
cd code
python simulate.py                       # data
python test_autograd.py                  # gradient check
./run_queue.sh                           # HPO + all experiments (≈3–4 h on 2 CPU cores)
python bench_opt.py                      # optimizer benchmark
python fig_diagrams.py && python paper_build.py   # figures, tables, manuscript
```
Requirements: Python ≥ 3.10, NumPy, SciPy, pandas, scikit-learn, matplotlib; Node.js with the `docx` package for the manuscript.
