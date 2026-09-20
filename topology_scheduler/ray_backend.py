"""Execute one ordinary Ray task per selected GPU in an atomic reservation."""

from collections import Counter
from .policy import Plan, positive


def run(plan: Plan, worker, *, reservation_timeout: float = 60,
        execution_timeout: float = 300, extra_resources=None):
    """Call worker(rank) on each selected node and return results in rank order.

    Caller initializes Ray and supplies a serializable function. Each worker
    reserves one CPU and one GPU. GPU IDs are assigned by Ray, never by this code.
    Both success and failure release the placement group. No automatic replan.

    ``extra_resources`` optionally supplies one resource mapping per rank, added
    to that rank's bundle and task. It exists for callers that must reserve more
    than the node marker, such as a specific device; omitting it leaves the
    default behavior unchanged.
    """
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    positive(reservation_timeout, "reservation_timeout")
    positive(execution_timeout, "execution_timeout")
    if not plan.workers:
        raise ValueError("Plan must contain workers")
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before run()")
    live = [n for n in ray.nodes() if n["Alive"]]
    for key, required in Counter(n.resource_key for n in plan.workers).items():
        matches = [n for n in live if n["Resources"].get(key, 0) > 0]
        if len(matches) != 1:
            raise ValueError(f"{key} must be advertised by exactly one live Ray node")
        resources = matches[0]["Resources"]
        if any(resources.get(r, 0) < required for r in (key, "CPU", "GPU")):
            raise ValueError(f"Insufficient total resources on {key}")
    extra = [dict(item) for item in (extra_resources or [{}] * len(plan.workers))]
    if len(extra) != len(plan.workers):
        raise ValueError("extra_resources must supply one mapping per worker")
    bundles = [{"CPU": 1, "GPU": 1, n.resource_key: 1, **more}
               for n, more in zip(plan.workers, extra)]
    group = placement_group(bundles, strategy="PACK")
    refs = []
    try:
        ray.get(group.ready(), timeout=reservation_timeout)
        remote_worker = ray.remote(worker)
        for rank, node in enumerate(plan.workers):
            refs.append(remote_worker.options(
                num_cpus=1, num_gpus=1,
                resources={node.resource_key: 1, **extra[rank]},
                max_retries=0,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=group, placement_group_bundle_index=rank,
                    placement_group_capture_child_tasks=False),
            ).remote(rank))
        return ray.get(refs, timeout=execution_timeout)
    finally:
        try:
            for ref in refs:
                ray.cancel(ref, force=True)
        finally:
            remove_placement_group(group)
