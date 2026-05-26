# Markovian Conditional Restart Splitting

This project is conducted by **Lizhan Hong** and **Tom Zhang** under the
supervision of Professor **Charles-Albert Lehalle**.

**Markovian Conditional Restart Splitting** is the rare-event method used in the
final MODAL order-book notebooks. It is an outer simulation layer around the
existing Ogata thinning path simulator.

It is not an intensity-decomposition method. In particular, it does not split a
Hawkes intensity into baseline and excitation clocks, and it does not use a
branching-process Hawkes simulation.

## Markov State

For exponential Hawkes kernels, the process is Markov if the queue state is
augmented with the current exponentially decayed excitation state:

```text
X_k = (N_k, S_k)
```

Here `N_k` is the queue vector and `S_k` is the Hawkes excitation vector. Between
events, each excitation component decays exponentially. At an event, the
corresponding component receives its jump.

For a scalar exponential Hawkes component:

```text
lambda(t) = mu + S(t)
S(t + dt) = S(t) exp(-beta dt)
S(t_i+) = S(t_i-) + jump
```

The implementation uses the `hawkes_state` stored in Ogata checkpoints as the
authoritative source of `S`. Reconstructing `S` from `lambda - mu` is only a
diagnostic fallback because some model intensities are clipped at zero or
`0.01`.

## Four-Queue Excitation Vector

In `nb3_125_deuxieme_limite.ipynb`, the queue state order is:

```text
[Q+1, Q-1, Q+2, Q-2]
```

The sign convention is:

```text
positive index = ask side
negative index = bid side
```

The notebook-facing four-component excitation vector is:

```text
S_k = (
    S_k^{+1,-},
    S_k^{-1,-},
    S_k^{+1,- -> +2,+},
    S_k^{-1,- -> -2,+}
)
```

The current code stores this as:

```text
H[0] = first-limit removal excitation for Q+1
H[1] = first-limit removal excitation for Q-1
G[0] = second-limit addition excitation for Q+2
G[1] = second-limit addition excitation for Q-2
```

Therefore the exported vector is:

```text
[H[0], H[1], G[0], G[1]]
```

There is no duplicated cross component. `G[0]` is exposed as
`S^{+1,- -> +2,+}` for the ask side, and `G[1]` is exposed as
`S^{-1,- -> -2,+}` for the bid side in the notebook notation.

## Boundary Law

The method first estimates empirical conditional laws at near-boundary levels:

```text
Law(S_k | Q-1 = 1)
Law(S_k | Q+1 = 1)
```

For Hawkes boundary collection, the default burn-in is

```text
10 / (beta * (1 - alpha / beta))
```

when `alpha / beta < 1`. For the nb3 parameters `alpha=0.3` and `beta=0.5`,
this gives a burn-in horizon of `50`.

Each collected boundary checkpoint contains:

- current time;
- queue vector `N`;
- excitation vector `S`;
- raw Hawkes state used by the simulator;
- current intensity vector;
- queue labels, sign convention, and model metadata.

## Restart Step

After boundary collection, the method samples checkpoints from the empirical
boundary law and continues the Ogata simulation from the full Markov state.

For example:

```text
P(Q-1 = 0 before Q-1 >= 2 | Q-1 = 1, S ~ Law(S | Q-1 = 1))
```

The restart must preserve `S`. Restarting from queue sizes alone, or resetting
the intensity to baseline, is mathematically wrong for Hawkes models.

## Notebook Usage

- `nb1`: Poisson sanity check. There is no Hawkes state, so `S` is empty and the
  method reduces to restarting from a queue boundary state.
- `nb2`: Coupled Hawkes queues. The method compares restarts with the correct
  `[H+1, H-1]` state against restarts where `S` is reset to zero.
- `nb3`: Four-queue second-limit model. The method estimates:
  - `Q-2_same`;
  - `Q-2_opp`;
  - `Q-2 | Q-1 = 0`;
  - `Q-2 | Q+1 = 0`.

For `nb3`, `Q-2 | Q-1 = 0` is the bid same-side second-limit distribution, while
`Q-2 | Q+1 = 0` observes the bid second limit when the opposite first queue
depletes.

The notebook also includes a multilevel MCRS estimator over queue levels
`[8, 6, 4, 2, 1, 0]` and a matched-budget comparison against naive Ogata Monte
Carlo. These are estimator-layer additions; the underlying Ogata Hawkes
dynamics are unchanged.

## Limitations

- The empirical boundary distribution must contain enough samples.
- Restart estimates are only as representative as the collected boundary
  checkpoints.
- Hawkes restarts are biased if `S` is not preserved.
- Lambda-only reconstruction of `S` is lossy when intensities are clipped.
- The sign convention must be checked before interpreting same-side and
  opposite-side second-limit distributions.
