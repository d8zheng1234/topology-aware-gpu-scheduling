"""Tie a planned GPU identity to the device a worker actually receives.

V1.2 discovers each GPU's UUID and PCI identity, but the task adapter reserves
a node and Ray chooses the device. This module closes the gap it can and names
the gap it cannot:

- Ray's accelerator ids are **indices**, written into ``CUDA_VISIBLE_DEVICES``
  in NVML enumeration order. Ray 2.55.0 never sets ``CUDA_DEVICE_ORDER``, and
  CUDA's default ``FASTEST_FIRST`` ordering may disagree with NVML's, so the
  same number can select a different physical GPU in each ordering. An index
  is therefore not an identity, and this module resolves it to a UUID on the
  node and reports the ordering it saw.
- A ``topology_gpu:<uuid>`` custom resource gives Ray's scheduler one unit per
  physical device, so two tasks cannot hold the same device. That is mutual
  exclusion, not device selection: Ray still assigns the CUDA device itself.

What this provides, therefore, is **verification, not selection**. In
``VERIFY`` mode a worker whose device is not the planned one fails before it
runs, so a placement is honored or refused and never silently moved. Pinning a
process to a chosen device would need
``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES``; see the guide for why that is
deferred.

Importing this module never requires Ray or NVML.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .policy import Plan

DEVICE_RESOURCE_PREFIX = "topology_gpu:"
PCI_BUS_ID = "PCI_BUS_ID"
DEVICE_ORDER_ENV_VAR = "CUDA_DEVICE_ORDER"
# Modes: record what happened, or refuse a placement that was not honored.
OBSERVE, VERIFY = "observe", "verify"
MODES = (OBSERVE, VERIFY)


class DeviceBindingError(RuntimeError):
    """A planned device was not the device a worker received."""

    def __init__(self, message: str, assignments: Sequence["DeviceAssignment"] = ()):
        super().__init__(message)
        self.assignments = tuple(assignments)

    def as_dict(self) -> dict:
        return {"error": str(self),
                "assignments": [item.as_dict() for item in self.assignments]}


def device_resource_key(uuid: str) -> str:
    """The Ray custom resource that represents one physical GPU."""
    if not uuid or not uuid.strip():
        raise ValueError("GPU UUID must be a nonempty string")
    return f"{DEVICE_RESOURCE_PREFIX}{uuid.strip()}"


def device_resources(devices: Iterable) -> dict[str, float]:
    """One unit per device, for a node's ``ray start --resources``.

    Advertising these makes Ray's accounting one-to-one with physical GPUs, so
    no two tasks can reserve the same device even though Ray still decides
    which device each task sees.
    """
    resources: dict[str, float] = {}
    for device in devices:
        key = device_resource_key(getattr(device, "uuid", device))
        if key in resources:
            raise ValueError(f"Duplicate device {key!r} on one node")
        resources[key] = 1
    return resources


@dataclass(frozen=True)
class DevicePlacement:
    """The physical devices a plan asks for, one per rank.

    Kept beside ``Plan`` rather than inside ``Node`` so the planner keeps its
    one-model-per-node schema and a caller can plan without device identity.
    """

    uuids: tuple[str, ...]

    def __post_init__(self):
        if not self.uuids:
            raise ValueError("A device placement needs at least one device")
        for uuid in self.uuids:
            if not isinstance(uuid, str) or not uuid.strip():
                raise ValueError("Every device UUID must be a nonempty string")
        if len(set(self.uuids)) != len(self.uuids):
            raise ValueError("One device cannot serve two ranks")

    def resource_keys(self) -> tuple[str, ...]:
        return tuple(device_resource_key(uuid) for uuid in self.uuids)

    def as_dict(self) -> dict:
        return {"uuids": list(self.uuids), "resources": list(self.resource_keys())}


@dataclass(frozen=True)
class DeviceAssignment:
    """What one rank asked for, what it received, and whether they agree."""

    rank: int
    node_name: str
    requested_uuid: str | None
    assigned_uuid: str | None = None
    assigned_index: int | None = None
    pci_bus_id: str | None = None
    device_order: str | None = None
    problem: str | None = None

    @property
    def matched(self) -> bool:
        return (self.problem is None and self.assigned_uuid is not None
                and self.assigned_uuid == self.requested_uuid)

    def as_dict(self) -> dict:
        return dict(asdict(self), matched=self.matched)

    @classmethod
    def from_dict(cls, value: Mapping) -> "DeviceAssignment":
        fields = {name: value.get(name) for name in cls.__dataclass_fields__}
        return cls(**fields)


def resolve_device(accelerator_ids: Sequence, devices: Sequence, *,
                   device_order: str | None = None) -> dict:
    """Resolve Ray's accelerator ids to one physical device on this node.

    ``devices`` is the node's NVML-ordered device list, as V1.2 discovery
    returns it. A reported ordering other than ``PCI_BUS_ID`` is a problem, not
    a detail: CUDA would then number devices differently from NVML, and the
    index Ray handed out could name a different GPU than the one it reserved.
    """
    if len(accelerator_ids) != 1:
        return {"problem": (
            f"Ray assigned {len(accelerator_ids)} devices to this worker; "
            "device binding requires exactly one GPU per rank")}
    try:
        index = int(accelerator_ids[0])
    except (TypeError, ValueError):
        return {"problem": f"Ray accelerator id {accelerator_ids[0]!r} is not an index"}
    if not 0 <= index < len(devices):
        return {"problem": (
            f"Ray assigned accelerator index {index}, but this node reports "
            f"{len(devices)} devices; the raylet and NVML disagree")}
    device = devices[index]
    resolved = {"assigned_uuid": device.uuid, "assigned_index": index,
                "pci_bus_id": device.pci_bus_id, "device_order": device_order}
    if device_order != PCI_BUS_ID:
        resolved["problem"] = (
            f"{DEVICE_ORDER_ENV_VAR} is {device_order or 'unset'}; set it to "
            f"{PCI_BUS_ID} on every worker so CUDA numbers devices the way NVML "
            "does. Until then an accelerator index cannot identify a device.")
    return resolved


def assignment_for(rank: int, node_name: str, requested_uuid: str | None,
                   resolved: Mapping) -> DeviceAssignment:
    """Combine a request with what the node resolved into one record."""
    return DeviceAssignment(
        rank=rank, node_name=node_name, requested_uuid=requested_uuid,
        assigned_uuid=resolved.get("assigned_uuid"),
        assigned_index=resolved.get("assigned_index"),
        pci_bus_id=resolved.get("pci_bus_id"),
        device_order=resolved.get("device_order"),
        problem=resolved.get("problem"),
    )


def check_assignments(assignments: Sequence[DeviceAssignment], mode: str) -> None:
    """Refuse a placement that was not honored, when the mode asks for it."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if mode == OBSERVE:
        return
    wrong = [item for item in assignments if not item.matched]
    if wrong:
        raise DeviceBindingError(
            "Ray did not assign the planned devices: " + "; ".join(
                f"rank {item.rank} on {item.node_name} asked for "
                f"{item.requested_uuid} and received "
                f"{item.assigned_uuid or 'nothing'}"
                + (f" ({item.problem})" if item.problem else "")
                for item in wrong),
            assignments)


def bundle_resources(plan: Plan, placement: DevicePlacement) -> list[dict]:
    """One device resource per rank, for the adapter's bundles and tasks."""
    if len(placement.uuids) != len(plan.workers):
        raise ValueError(
            f"Device placement names {len(placement.uuids)} devices for "
            f"{len(plan.workers)} planned workers")
    return [{key: 1} for key in placement.resource_keys()]


def preflight_devices(plan: Plan, placement: DevicePlacement,
                      nodes: Sequence[Mapping]) -> None:
    """Reject a placement whose devices no live node advertises.

    Without this the reservation would simply wait for a resource that will
    never appear and fail as a timeout, which hides the real cause.
    """
    bundle_resources(plan, placement)
    live = [node for node in nodes if node.get("Alive")]
    for rank, (node, key) in enumerate(zip(plan.workers, placement.resource_keys())):
        holders = [item for item in live
                   if item["Resources"].get(key, 0) > 0
                   and item["Resources"].get(node.resource_key, 0) > 0]
        if len(holders) != 1:
            raise ValueError(
                f"rank {rank}: {key} must be advertised by exactly one live node "
                f"that also advertises {node.resource_key}. Start that node with "
                f"--resources including this device, as the guide describes.")


def _node_devices() -> tuple:
    """Read this node's NVML-ordered devices inside a worker."""
    import ray._private.thirdparty.pynvml as pynvml

    from .inventory import _read_nvml_snapshot

    return _read_nvml_snapshot(pynvml)[0]


def observe_worker_device(*, devices_provider: Callable[[], Sequence] | None = None,
                          accelerator_ids: Sequence | None = None,
                          environ: Mapping | None = None) -> dict:
    """Resolve the device this worker received. Runs inside a Ray task."""
    import os

    if accelerator_ids is None:
        import ray

        accelerator_ids = ray.get_gpu_ids()
    environ = os.environ if environ is None else environ
    devices = (devices_provider or _node_devices)()
    return resolve_device(accelerator_ids, devices,
                          device_order=environ.get(DEVICE_ORDER_ENV_VAR))


def bind_worker(worker, placement: DevicePlacement, node_names: Sequence[str], *,
                mode: str = VERIFY, devices_provider=None, environ=None):
    """Wrap a worker so it checks its device before doing any work.

    The wrapper returns ``{"device": ..., "result": ...}``; in ``VERIFY`` mode
    it raises before calling the worker, so a wrong device never runs the
    workload at all. ``devices_provider`` and ``environ`` exist so tests and
    examples can exercise this path without NVML or a particular environment.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    uuids, names = tuple(placement.uuids), tuple(node_names)

    def bound(rank):
        resolved = observe_worker_device(
            devices_provider=devices_provider, environ=environ)
        assignment = assignment_for(rank, names[rank], uuids[rank], resolved)
        check_assignments([assignment], mode)
        return {"device": assignment.as_dict(), "result": worker(rank)}

    return bound


def run_with_devices(plan: Plan, placement: DevicePlacement, worker, *,
                     mode: str = VERIFY, devices_provider=None, environ=None,
                     **run_options):
    """Run one worker per planned device and report what each one received.

    Returns ``(results, assignments)`` with results in rank order, exactly as
    the task adapter returns them. In ``VERIFY`` mode a rank that did not get
    its planned device fails the run instead of quietly using another GPU.
    """
    import ray

    from .ray_backend import run

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before run_with_devices()")
    preflight_devices(plan, placement, ray.nodes())
    payloads = run(plan, bind_worker(
        worker, placement, [node.name for node in plan.workers],
        mode=mode, devices_provider=devices_provider, environ=environ),
        extra_resources=bundle_resources(plan, placement), **run_options)
    assignments = tuple(DeviceAssignment.from_dict(item["device"]) for item in payloads)
    check_assignments(assignments, mode)
    return [item["result"] for item in payloads], assignments
