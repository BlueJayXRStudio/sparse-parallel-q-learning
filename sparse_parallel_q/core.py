"""MPI transport and four tabular Q-value aggregation rules."""

from collections import defaultdict
from mpi4py import MPI

def sync_q_tables_with_aggregation(local_q, local_counts, aggregate_pair, *, sort_keys=False):
    """Run MPI synchronization with a supplied per-state/action aggregation rule.

    Every rank gathers its Q-table and counts, then receives the merged table
    and a fresh count table defaulting to one. Only rank zero calls
    aggregate_pair(q_values, counts), with both lists in MPI rank order.
    Missing Q-values default to zero and counts to one. Root aggregation
    failures are broadcast so every rank raises instead of leaving peers waiting.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    all_qs = comm.gather(dict(local_q), root=0)
    all_counts = comm.gather(dict(local_counts), root=0)

    master_q_data = None
    error = None
    if rank == 0:
        try:
            keys = set().union(*(q.keys() for q in all_qs))
            if sort_keys:
                keys = sorted(keys)
            master_q_data = {}
            for key in keys:
                q_values = [q.get(key, 0.0) for q in all_qs]
                counts = [worker_counts.get(key, 1) for worker_counts in all_counts]
                master_q_data[key] = aggregate_pair(q_values, counts)
            local_q.update(master_q_data)
        except Exception as exc:
            # Complete the collective so peers do not wait forever on root.
            master_q_data = None
            error = f"{type(exc).__name__}: {exc}"

    global_q_data, error = comm.bcast((master_q_data, error), root=0)
    if error is not None:
        raise RuntimeError(f"Q-table aggregation failed on rank 0: {error}")
    return defaultdict(lambda: 0.0, global_q_data), defaultdict(lambda: 1)


def sync_frequency_wta_q_tables(local_q, local_counts):
    """Synchronize Q-tables using frequency-based winner-takes-all aggregation.

    For each state/action key, copy the Q-value from the worker with the
    largest current-round visit count. Equal counts select the lowest MPI rank.
    Counts include the shared pseudocount of one and reset after synchronization.
    """
    def frequency_wta(q_values, counts):
        selected_rank = counts.index(max(counts))
        return q_values[selected_rank]

    return sync_q_tables_with_aggregation(local_q, local_counts, frequency_wta)


def sync_count_weighted_mean_q_tables(local_q, local_counts):
    """Synchronize Q-tables using a count-weighted mean across every MPI rank.

    Weights are counts c=1+visits. Missing Q-values are zero and missing counts
    are one. Counts reset after synchronization.
    """
    def weighted_mean(q_values, counts):
        return sum(q * count for q, count in zip(q_values, counts)) / float(sum(counts))

    return sync_q_tables_with_aggregation(local_q, local_counts, weighted_mean)


def sync_uniform_mean_q_tables(local_q, local_counts):
    """Synchronize Q-tables using an equal-weight mean across all MPI ranks.

    Missing Q-values are zero. Counts are gathered for the same synchronization
    pattern as the other helpers, then reset to one; they do not affect the mean.
    """
    def uniform_mean(q_values, counts):
        return sum(q_values) / float(len(q_values))

    return sync_q_tables_with_aggregation(local_q, local_counts, uniform_mean)


def sync_random_visited_q_tables(local_q, local_counts, *, rng):
    """Synchronize Q-tables by uniformly selecting a visited worker per key.

    Pass a dedicated numpy.random.Generator as rng on rank zero; other ranks
    may pass None. Only rank zero consumes it. Keys are sorted by state tuple
    then action, and eligible ranks (counts > 1) are in ascending rank order.
    Call rng.choice once per key with any eligible workers, even a singleton.
    If no rank visited a key this round, use rank zero's Q-value instead.
    The result is broadcast and counts restart at one, as in frequency WTA.
    """
    def random_visited(q_values, counts):
        eligible = [rank for rank, count in enumerate(counts) if count > 1]
        selected = int(rng.choice(eligible)) if eligible else 0
        return q_values[selected]

    return sync_q_tables_with_aggregation(
        local_q, local_counts, random_visited, sort_keys=True
    )
