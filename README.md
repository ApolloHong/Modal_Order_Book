# MODAL Order Book

Simulation and rare-event estimation tools for a MODAL course project on limit
order book dynamics.

The project models queue dynamics with independent birth-death queues and
Hawkes-type excitation, then compares three estimators for rare LOB events:

1. Classical Monte Carlo using Ogata thinning
2. Fixed-Level Splitting
3. Adaptive Multilevel Splitting (AMS)

The current codebase focuses on the models used in the final notebooks:

- `nb1_123_files_poisson_independantes.ipynb`
- `nb2_124_files_hawkes_couplees.ipynb`
- `nb3_125_deuxieme_limite.ipynb`

The older generic Queue-Reactive framework and exploratory QR notebooks were
removed because they were not used by the final workflow.

## Repository Layout

```text
MODAL_ORDER_BOOK/
├── model/
│   ├── __init__.py
│   ├── events.py
│   ├── lob.py
│   ├── hawkes/              # Lowercase Hawkes implementation
│   ├── Hawkes.py            # Compatibility wrapper for old uppercase imports
│   ├── hawkes_4q.py         # Four-queue Hawkes model for second-limit studies
│   ├── hitting_times.py     # Independent Poisson birth-death hitting times
│   ├── ogata.py             # Continuation-capable simulators and trajectories
│   ├── rare_events.py       # Rare-event problems, targets, and score functions
│   ├── splitting.py         # Fixed-Level Splitting and AMS
│   ├── analysis.py          # Estimator comparison helpers
│   └── utils.py             # RNG, timing, and numerical helpers
├── tests/
│   └── test_splitting_validation.py
├── nb1_123_files_poisson_independantes.ipynb
├── nb2_124_files_hawkes_couplees.ipynb
├── nb3_125_deuxieme_limite.ipynb
└── images/
```

## Main Concepts

### Ogata Thinning

Ogata thinning is used to simulate point-process paths for Hawkes-driven order
book dynamics. It generates candidate events from an upper-bound intensity and
accepts or rejects each candidate according to the current true intensity.

In this project, Ogata is the baseline path simulator. It is not itself a
rare-event estimator.

### Rare-Event Splitting

Splitting does not mean simulating each queue separately. It means allocating
more simulation budget to trajectories that move closer to a rare target event.

For example, if the rare event is depletion of `Q+1`, a score can increase as
`Q+1` approaches zero. Trajectories that reach higher score levels are cloned
and continued with independent future randomness.

### Fixed-Level Splitting

Fixed-Level Splitting uses manually chosen score levels, for example:

```python
levels = [0.20, 0.40, 0.60, 0.80, 1.00]
```

At each level, paths that reached the level survive; paths that did not are
discarded. Survivors are resampled from their first level-hitting checkpoints
and then continued independently. The probability estimate is the product of
conditional survival probabilities across levels.

### Adaptive Multilevel Splitting

AMS chooses levels automatically. It simulates a population of particles,
repeatedly kills the worst-scoring fraction, clones from better particles at
the adaptive level, and continues the clones with independent random streams.

A single AMS run returns a probability estimate but not a reliable standard
error. For publication-quality error bars, run several independent AMS
macro-replications and compute the empirical standard deviation.

### Checkpointing and Hawkes Memory

Checkpointing is essential. A Hawkes process cannot be restarted from queue
sizes alone because the current intensities depend on past events through the
excitation state.

The common checkpoint object stores:

- current time
- queue state
- Hawkes excitation state
- current intensity vector
- current score
- metadata needed by the rare-event problem

This is what makes cloning in Fixed-Level Splitting and AMS valid for Hawkes
models.

## Public API

The most useful imports are:

```python
from model import (
    IndependentPoissonSimulator,
    SingleHawkesSimulator,
    CoupledHawkesSimulator,
    FourQueueHawkesSimulator,
    RareEventProblem,
    q1_depletion_problem,
    min_best_depletion_problem,
    q2_after_q1_depletion_problem,
    FixedLevelSplitting,
    AdaptiveMultilevelSplitting,
)
```

Comparison helpers:

```python
from model.analysis import (
    run_mc_baseline,
    run_fixed_level_splitting,
    run_ams,
    compare_estimators,
)
```

Historical Hawkes imports remain available:

```python
from model.Hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
from model.hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
```

## Quick Example

```python
from model import CoupledHawkesSimulator, min_best_depletion_problem
from model.analysis import (
    run_mc_baseline,
    run_fixed_level_splitting,
    run_ams,
    compare_estimators,
)

problem = min_best_depletion_problem(
    T=80.0,
    q1_init=25,
    q_neg1_init=25,
)

simulator = CoupledHawkesSimulator(
    mu_plus=1.5,
    mu_minus=1.1,
    alpha=0.15,
    beta=0.5,
    sign_convention="v4",
)

mc = run_mc_baseline(simulator, problem, n_runs=800, seed=42)

fls = run_fixed_level_splitting(
    simulator,
    problem,
    levels=[0.20, 0.40, 0.60, 0.80, 1.00],
    n_particles=250,
    seed=42,
)

ams = run_ams(
    simulator,
    problem,
    n_particles=250,
    kill_fraction=0.10,
    max_iterations=35,
    seed=42,
)

table = compare_estimators([mc, fls, ams])
display(table)
```

## Comparison Table Columns

The notebooks report tables with columns such as:

- `method`: estimator name.
- `probability`: estimated rare-event probability.
- `std_error`: estimated standard error when available.
- `relative_error`: `std_error / probability`.
- `cpu_seconds`: wall-clock runtime.
- `n_runs`: independent paths for Monte Carlo; particle-level continuation
  budget for splitting methods.
- `n_particles`: interacting particle count for splitting methods; not
  applicable to classical Monte Carlo.
- `n_events`: accepted simulated events.
- `n_candidates`: Ogata candidate events, including rejected candidates.
- `cost_normalized_rel_error`: `relative_error * sqrt(cpu_seconds)`, a rough
  cost-adjusted efficiency measure. Lower is better.

`NaN` values are expected in some places:

- Monte Carlo has no `n_particles`, so this is `NaN`.
- A single AMS run does not expose a trustworthy analytical standard error, so
  `std_error`, `relative_error`, and `cost_normalized_rel_error` are `NaN`.

## Running the Notebooks

From the repository root:

```bash
jupyter notebook
```

Then open:

```text
nb1_123_files_poisson_independantes.ipynb
nb2_124_files_hawkes_couplees.ipynb
nb3_125_deuxieme_limite.ipynb
```

To execute from the command line without overwriting the original notebooks:

```bash
jupyter nbconvert --to notebook --execute nb1_123_files_poisson_independantes.ipynb \
  --output /tmp/nb1_executed.ipynb --ExecutePreprocessor.timeout=900

jupyter nbconvert --to notebook --execute nb2_124_files_hawkes_couplees.ipynb \
  --output /tmp/nb2_executed.ipynb --ExecutePreprocessor.timeout=1200

jupyter nbconvert --to notebook --execute nb3_125_deuxieme_limite.ipynb \
  --output /tmp/nb3_executed.ipynb --ExecutePreprocessor.timeout=1200
```

## Tests

Run:

```bash
pytest -q
```

The test suite checks:

- Poisson tail validation against an exact probability
- Fixed-Level Splitting behavior
- AMS behavior
- deterministic reproducibility with fixed seeds
- Hawkes checkpoint memory preservation
- non-negative Hawkes intensities
- four-queue cross-excitation
- public import smoke tests

## Dependencies

The code uses:

- Python 3.12+
- NumPy
- pandas
- matplotlib
- SciPy, used by existing notebook formulas
- pytest, for tests
- Jupyter / nbconvert, for notebook execution

No heavy simulation framework is required.

## Reproducibility Notes

- All rare-event estimators use explicit `np.random.Generator` streams.
- Splitting clones always receive independent child RNG streams.
- Hawkes checkpoints preserve excitation memory.
- Notebook examples set fixed seeds for reproducible outputs.

## Status

The current implementation is intended for clean course-project experiments,
not for production trading or calibration. The code is organized to make the
Monte Carlo baseline, Fixed-Level Splitting, and AMS comparable under the same
model parameters and rare-event definitions.
