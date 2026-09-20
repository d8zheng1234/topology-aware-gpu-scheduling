# Device identity binding

See [current implementation, validation status, and versions](current-status.md)
for the shared support summary and evidence boundaries.

[V1.2 discovery](v1.2-topology-discovery.md) records each GPU's UUID, PCI
ancestry and NVLink counts, but the task adapter reserves a *node* and Ray
chooses the device. Every device-level observation was therefore unusable for
placement: scoring on an NVLink edge would promise something the execution
layer does not deliver.

[device_binding.py](../topology_scheduler/device_binding.py) closes the part of
that gap which can be closed, and states precisely the part that cannot. It
provides **verification, not selection**: a rank that did not receive its
planned device fails before it runs, so a placement is honored or refused and
never silently moved.

## Why an index is not an identity

Ray's accelerator ids are indices. It writes them into `CUDA_VISIBLE_DEVICES`
in NVML enumeration order, and **Ray 2.55.0 never sets `CUDA_DEVICE_ORDER`**
(checked against the pinned source tree). CUDA's default ordering is
`FASTEST_FIRST`, which need not agree with NVML's, so the same number can
select a different physical GPU in each ordering.

That is why this module resolves an index to a UUID on the node, and why it
treats any ordering other than `PCI_BUS_ID` as a problem rather than a detail.
Start every worker with:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

Until that holds, an assignment is recorded but never counted as verified —
even when the resolved UUID happens to equal the requested one, because the
evidence for that equality is not trustworthy.

## What Ray can and cannot guarantee

| Option | What it gives | Why it is or is not used |
| --- | --- | --- |
| Per-device custom resource `topology_gpu:<uuid>` | One unit per physical GPU, so two tasks cannot hold the same device | **Used.** Mutual exclusion only; Ray still picks which device each task sees |
| Resolve and verify on the node | The rank learns its real UUID and refuses a mismatch | **Used.** This is the guarantee the adapter offers |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` plus adapter-set visibility | True selection: the process sees exactly the chosen device | **Deferred.** It is explicitly experimental and moves device accounting out of Ray, so any workload on the cluster that does not follow the same discipline could claim the same GPU. That precondition needs a maintainer decision before it is safe |
| One GPU per node | The mapping is trivial | Supported as a degenerate case; it is not a general answer |

The combination in use is honest about its limit: the planned devices are
reserved as a *set*, each rank reports which one it actually holds, and a wrong
pairing fails instead of running. The [smoke example](../examples/ray_device_binding_smoke.py)
demonstrates exactly this, including a run where Ray's pairing is recorded
rather than assumed.

## Advertise the devices

Each node advertises one resource per physical GPU alongside its node marker:

```python
from topology_scheduler import device_resources, discover_ray_gpu_inventory

for inventory in discover_ray_gpu_inventory():
    print(inventory.node_name, device_resources(inventory.devices))
```

Use the result in that node's `ray start --resources`, together with its
existing `topology_node:<name>` marker. Without it the reservation would wait
for a resource that never appears; `run_with_devices()` checks first and says
which device is missing from which node.

## Run with device identity

```python
import ray
from topology_scheduler import DevicePlacement, run_with_devices

ray.init(address="auto")
placement = DevicePlacement(("GPU-aaa", "GPU-bbb"))   # one device per rank
results, assignments = run_with_devices(plan, placement, worker)
for item in assignments:
    print(item.rank, item.requested_uuid, item.assigned_uuid, item.matched)
```

`DevicePlacement` sits beside `Plan` rather than inside `Node`, so the planner
keeps its one-model-per-node schema and a caller can still plan without naming
devices. One device cannot serve two ranks.

Modes:

- `verify` (default) — a rank that did not receive its planned device raises
  before the workload runs. The message names the rank, its node, the requested
  and received UUIDs, and the reason.
- `observe` — the assignment is recorded and the workload runs anyway. Use it
  to measure how often Ray's pairing matches before relying on verification.

Each `DeviceAssignment` records the rank, node, requested and assigned UUID,
the accelerator index and PCI address it resolved through, the observed
`CUDA_DEVICE_ORDER`, whether it matched, and any problem. It serializes to
plain types for run records.

A verification failure happens inside the Ray task, so it reaches the driver
wrapped in Ray's task error; the underlying message is the one above.

## Run it without a GPU

```bash
python -m unittest tests.test_device_binding -v
python -m examples.ray_device_binding_smoke
```

The tests cover resolution, ordering, mismatches, preflight, both modes, and
the one-GPU-per-node case with constructed devices. The smoke runs the real
reservation and task path on one local Ray node with **simulated** logical GPUs
and stand-in device identities, so it exercises everything except NVML itself.

## Limits

- This is verification, not selection. Ray still decides which device a task
  receives; the adapter only refuses a result it cannot vouch for.
- Nothing here has run against physical GPUs. Confirming that a resolved UUID
  is the device a process actually computes on needs hardware, and that
  evidence does not exist yet.
- Placement scoring still does not consume device-level relationships. Until
  selection exists, an NVLink pair can be observed and verified after the fact,
  not requested, so scoring on it would still overstate the guarantee.
- Multi-GPU ranks, MIG, and fractional devices are out of scope; each rank
  holds exactly one device.
