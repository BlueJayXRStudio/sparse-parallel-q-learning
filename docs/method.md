# Method and experimental protocol

## Learning rewards

The task is discrete LunarLander-v3 from Gymnasium 1.1.1, with eight
observation components and four actions. Learning rewards are zero on all
nonterminal transitions and equal to the native terminal outcome, `+100` for
landing or `-100` for failure. The same rewards are used for online Q-learning
and reverse replay. Dense environment rewards are recorded only as diagnostic
returns and do not enter the learning targets.

Success is the environment's positive terminal outcome. Gymnasium's native
positive branch detects a sleeping lander; it is not an additional test of
helipad position. Native failure includes crashing and leaving the viewport.
See the [pinned environment source](https://github.com/Farama-Foundation/Gymnasium/blob/v1.1.1/gymnasium/envs/box2d/lunar_lander.py#L656-L662).

## State encoding and local updates

Each observation is multiplied componentwise by `[30, 1, 1, 1, 1, 1, 1, 1]`,
transformed with `tanh`, and quantized by rounding after division by
`[1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]`. The resulting integer tuple is the
state key. A Q-table entry is indexed by `(state, action)` and defaults to zero.
Greedy action ties select the lowest action index.

For an observed transition, the update is:

```text
Q(s,a) <- Q(s,a) + alpha * [r_sparse + gamma * (1 - terminated) * max Q(s_next,.) - Q(s,a)]
```

The reference uses `alpha = 0.05` and `gamma = 0.99`. Discounting gives delayed
terminal outcomes less weight even though their reward magnitude is fixed.
Epsilon starts at `0.03`, decreases linearly to zero at epoch index 100, and
remains zero thereafter. Epochs are indexed from zero; the budget of 264
completed epochs corresponds to indices 0 through 263.

Each worker finishes native episodes until its cumulative interaction count
is strictly greater than 64,000. It retains the most recent 100,000 transitions
in a deque, applies another one-step update to each in reverse chronological
order, clears the deque, and then synchronizes. Extremely long epochs can
discard older replay transitions through the deque's capacity limit. Actual
interaction and replay counts are recorded separately.

This reverse traversal is called "backward induction" in historical code.
It is reverse-order experience replay, rather than an exact finite-horizon
dynamic-programming solution. See [related work](related_work.md).

## Combining workers' Q-tables

Aggregation is performed separately for every state/action key in the union of
workers' Q-tables. Synchronization consists of gathering tables and counts,
aggregating on MPI rank zero, and broadcasting the result to every worker.

The count table starts at one for every key. Online action selections increment
the corresponding count. Replay does not increment it. Counts reset to one
after synchronization, so they describe local interaction in the current epoch.

| Condition | Rule for a state/action pair |
| --- | --- |
| `frequency_wta` | Copy the Q-value from the worker with the largest local count. Equal counts choose the lowest MPI rank. |
| `random_visited` | Uniformly select a worker whose count exceeds one, then copy its Q-value. Use rank zero if no worker qualifies. |
| `count_weighted_mean` | Average all workers' Q-values using weights equal to their counts, including the initial count of one. |
| `uniform_mean` | Average all workers' Q-values with equal weights. |

Missing values are treated as zero and missing counts as one. The averaging
rules therefore include all ranks, even when a rank did not visit the pair in
the current epoch. Selection is per entry; the combined table can contain
values from different workers and does not preserve a complete worker policy.

Random selection uses its own NumPy generator on rank zero. Keys and candidate
ranks are sorted, and the generator is separate from exploration and
environment random-number streams.

## Episode boundaries and evaluation

The training loop ends an episode on `terminated`, not `truncated`. It continues
past Gymnasium's time-limit signal until a native terminal outcome occurs.
No manual timeout reward, artificial failure, or replacement episode cap is
applied. This protocol permits long floating episodes.

Frozen-policy evaluation uses the same encoding and greedy tie-breaking, with
no learning updates or Q-table mutation. It records the first truncation,
native outcome, episode length, and raw environment return. An episode still
unresolved at 100,000 steps is censored. A native termination on that step is
counted as a native outcome.

Reported landing fraction is `successes / planned episodes`. Censored episodes
remain in the denominator and are reported separately. Lower and upper bounds
are `successes / N` and `(successes + censored) / N`. The latter is a censoring
bound, not a confidence interval. The evaluator also reports conditional
episode-level Wilson intervals, which do not measure training-seed variability.

## Reference comparison

The [frozen configuration](../sparse_parallel_q/configs/comparison.json) specifies
four methods, paired training seeds 101, 102, and 103, sixteen workers, and
264 epochs per run. All runs start with empty Q-tables. The ordering of
conditions rotates across seeds. There is no performance-based stopping rule.

Worker seeds use separate streams for environment resets, exploration,
aggregation, NumPy, and PyTorch. The first environment reset is seeded;
subsequent resets continue that worker's environment random-number stream.

Checkpoints are evaluated after completed epochs 32, 64, 96, 128, 160, 192,
224, 256, and 264, using a common 100-episode monitoring seed bank. The final
checkpoint also receives a separate common 200-episode evaluation. Evaluation
is performed after training finishes and does not advance training RNGs.

The budget is fixed in epochs with complete-episode local collection. Actual
sample counts can differ because episodes overshoot the local threshold.
Results must therefore distinguish epochs, interactions, replay updates, and
wall-clock time. Three training seeds provide a descriptive comparison, not a
universal ranking of aggregation methods.
