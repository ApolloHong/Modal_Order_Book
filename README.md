# MODAL Order Book

Simulation and rare-event estimation tools for a MODAL course project on limit order book dynamics. This project is conducted by **Lizhan Hong** and **Tom Zhang** under the supervision of Professor **Charles-Albert Lehalle**.
The project models queue dynamics with independent birth-death queues and Hawkes-type excitation, then compares naive Monte Carlo using Ogata thinning and Markovian Conditional Restart Splitting.

1. naive Monte Carlo using Ogata thinning;
2. **Markovian Conditional Restart Splitting**.

The current codebase focuses on the models used in the final notebooks:

- `nb1_123_files_poisson_independantes.ipynb`
- `nb2_124_files_hawkes_couplees.ipynb`
- `nb3_125_deuxieme_limite.ipynb`


## Repository Layout

```text
MODAL_ORDER_BOOK/
├── model/
│   ├── __init__.py
│   ├── events.py
│   ├── lob.py
│   ├── Hawkes.py            # Compatibility wrapper for old uppercase imports
│   ├── hawkes_4q.py         # Four-queue Hawkes model for second-limit studies
│   ├── hitting_times.py     # Independent Poisson birth-death hitting times
│   ├── ogata.py             # Continuation-capable simulators and trajectories
│   ├── rare_events.py       # Rare-event problems, targets, and score functions
│   ├── restart_splitting.py # Markovian Conditional Restart Splitting
│   ├── splitting.py         # Legacy Fixed-Level Splitting and AMS experiments
│   ├── analysis.py          # Estimator comparison helpers
│   └── utils.py             # RNG, timing, and numerical helpers
├── tests/
│   ├── test_restart_splitting.py
│   └── test_splitting_validation.py
├── nb1_123_files_poisson_independantes.ipynb
├── nb2_124_files_hawkes_couplees.ipynb
├── nb3_125_deuxieme_limite.ipynb
└── images/
```

The file `.pytest_cache/README.md` belongs to pytest's local cache and is not part of
the project documentation. The project README is this root-level `README.md`.

## Main Concepts

### Ogata Thinning

Ogata thinning is used to simulate point-process paths for Hawkes-driven order
book dynamics. It generates candidate events from an upper-bound intensity and
accepts or rejects each candidate according to the current true intensity.

In this project, Ogata is the baseline path simulator. It is not itself a
rare-event estimator.

### Markovian Conditional Restart Splitting

**Markovian Conditional Restart Splitting** is the assignment-facing rare-event
method. Splitting does not mean decomposing a Hawkes intensity into baseline and
excitation clocks.

For exponential Hawkes models, the augmented state `(N, S)` is Markov, where
`N` is the queue vector and `S` is the current exponentially decayed excitation
state. The method:

- collects near-boundary checkpoints such as `Q-1 = 1` or `Q+1 = 1`;
- estimates empirical laws such as `Law(S | Q-1 = 1)`;
- restarts local simulations from sampled full Markov checkpoints;
- estimates local rare transitions such as depletion before recovery.

Ogata thinning remains the low-level path simulator.

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

This is what makes Markovian Conditional Restart Splitting valid for Hawkes
models.

### Four-Queue Second-Limit Notation

In `nb3_125_deuxieme_limite.ipynb`, the code uses the state order:

```text
[Q+1, Q-1, Q+2, Q-2]
```

The notebook uses the following interpretation:

- positive index: ask side
- negative index: bid side
- `Q-2 | Q-1 = 0`: bid second limit on the same side as a bid first-limit depletion
- `Q-2 | Q+1 = 0`: bid second limit observed after the opposite first limit depletes
- `Q-2_same = Q-2 | Q-1 = 0`
- `Q-2_opp = Q-2 | Q+1 = 0`

For the four-queue Hawkes model, the notebook-facing excitation vector is:

```text
S = [S^{+1,-}, S^{-1,-}, S^{+1,- -> +2,+}, S^{-1,- -> -2,+}]
```

In the current implementation this is extracted from the checkpoint as:

```text
[H[0], H[1], G[0], G[1]]
```

Here `G[0]` is the ask-side cross excitation from `Q+1` removals to `Q+2`
additions, and `G[1]` is the bid-side cross excitation from `Q-1` removals to
`Q-2` additions. No component is duplicated.

### What nb3 Visualizes

The Section 1.2.5 figures in `nb3_125_deuxieme_limite.ipynb` are generated
from Markovian Conditional Restart Splitting restarts, not from the older
Fixed-Level Splitting or AMS experiments.

The notebook includes:

- empirical conditional excitation summaries for `Law(S | Q-1 = 1)` and
  `Law(S | Q+1 = 1)`;
- marginal histograms of the four-component `S` vector;
- two-dimensional joint distributions of important `S` components;
- restart-content tables showing sampled boundary states, start/end excitation,
  success flags, local hitting times, and final `Q-2`;
- histograms of `Q-2_same`, `Q-2_opp`, `Q-2 | Q-1 = 0`, and `Q-2 | Q+1 = 0`;
- sensitivity plots as a function of the cross-excitation parameter `a_cross`;
- bootstrap confidence intervals for conditional `Q-2` means and
  same-minus-opposite differences;
- a real multilevel MCRS estimator over queue levels `[8, 6, 4, 2, 1, 0]`;
- a matched-budget comparison against naive Ogata Monte Carlo.

Ogata thinning is still used inside the simulator to generate each local path,
but the estimator and the plotted Section 1.2.5 quantities are MCRS-based.

## Public API

The most useful imports are:

```python
from model import (
    IndependentPoissonSimulator,
    SingleHawkesSimulator,
    CoupledHawkesSimulator,
    FourQueueHawkesSimulator,
    RareEventProblem,
    first_limit_depletion_problem,
    q1_depletion_problem,
    min_best_depletion_problem,
    second_limit_activation_problem,
    MarkovState,
    BoundarySample,
    RestartSplittingResult,
    collect_boundary_states,
    default_hawkes_burn_in,
    restart_from_boundary_distribution,
    run_naive_depletion_monte_carlo,
    multilevel_markovian_restart_splitting,
    bootstrap_mean_ci,
    bootstrap_difference_ci,
    summarize_conditional_S,
    run_markovian_conditional_restart_splitting,
)
```

Comparison helpers:

```python
from model.analysis import (
    run_markovian_conditional_restart_splitting,
    run_naive_boundary_mc_comparison,
    compare_restart_results,
    extract_q_neg2_restart_observables,
)
```

Historical Hawkes imports remain available:

```python
from model.Hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
from model.hawkes import simulate_hawkes_queue, simulate_coupled_hawkes
```

## Quick Example

```python
from model import CoupledHawkesSimulator
from model.analysis import (
    run_markovian_conditional_restart_splitting,
    run_naive_boundary_mc_comparison,
    compare_restart_results,
)

simulator = CoupledHawkesSimulator(
    mu_plus=1.5,
    mu_minus=1.1,
    alpha=0.15,
    beta=0.5,
    sign_convention="v4",
)

initial_state = [25, 25]

naive = run_naive_boundary_mc_comparison(
    simulator=simulator,
    initial_state=initial_state,
    queue_index=-1,
    horizon=80.0,
    n_paths=800,
    horizon_local=10.0,
    seed=42,
)

mcrs = run_markovian_conditional_restart_splitting(
    simulator=simulator,
    initial_state=initial_state,
    queue_index=-1,
    horizon=80.0,
    n_boundary_paths=800,
    horizon_local=10.0,
    n_restarts=2_000,
    seed=43,
)

table = compare_restart_results([naive, mcrs])
display(table)
```

## Comparison Table Columns

The notebooks report tables with columns such as:

- `method`: estimator name.
- `probability`: estimated local depletion-before-recovery probability.
- `std_error`: estimated standard error when available.
- `relative_error`: `std_error / probability`.
- `cpu_seconds`: wall-clock runtime.
- `n_boundary_paths`: full paths simulated to collect near-boundary states.
- `n_boundary_samples`: useful checkpoints collected at the boundary.
- `n_restarts`: local simulations restarted from empirical boundary states.
- `n_successes`: restarts that hit zero before recovery.
- `n_events`: accepted simulated events.
- `n_candidates`: Ogata candidate events, including rejected candidates.
- `cost_normalized_rel_error`: `relative_error * sqrt(cpu_seconds)`, a rough
  cost-adjusted efficiency measure. Lower is better.

## Recommended Notebook Usage

- `nb1`: Poisson sanity check. There is no excitation state, so `S` is empty and
  Markovian Conditional Restart Splitting reduces to restarting from a queue
  boundary state.
- `nb2`: coupled Hawkes queues. Use Ogata thinning with the same burn-in policy
  for naive Monte Carlo and Markovian Conditional Restart Splitting, then show
  why preserving `S` matters.
- `nb3`: four-queue second-limit analysis. Use Markovian Conditional Restart
  Splitting for `Q-2_same`, `Q-2_opp`, `Q-2 | Q-1=0`, and `Q-2 | Q+1=0`.
  The notebook also shows the empirical joint distribution of `S` at the
  boundary and the actual restart samples used to produce the second-limit
  figures.

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
  --output /tmp/nb3_executed.ipynb --ExecutePreprocessor.timeout=1800
```

## Tests

Run:

```bash
pytest -q
```

The test suite checks:

- Poisson tail validation against an exact probability
- Fixed-Level Splitting and AMS legacy behavior
- Markovian Conditional Restart Splitting boundary/restart behavior
- conditional `S` extraction, including the four-component four-queue mapping
- deterministic reproducibility with fixed seeds
- Hawkes checkpoint memory preservation
- non-negative Hawkes intensities
- four-queue cross-excitation
- nb3 sign-convention helpers for `Q-2_same` and `Q-2_opp`
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
- Markovian Conditional Restart Splitting restarts receive independent child RNG
  streams.
- Hawkes checkpoints preserve excitation memory.
- Notebook examples set fixed seeds for reproducible outputs.

## Status

The current implementation is intended for clean course-project experiments,
not for production trading or calibration. The final workflow compares naive
Ogata Monte Carlo with Markovian Conditional Restart Splitting under the same
model parameters and boundary definitions. The assignment-facing notebooks now
use the exact method name **Markovian Conditional Restart Splitting**; Fixed-Level
Splitting and AMS remain as backward-compatible research utilities.
