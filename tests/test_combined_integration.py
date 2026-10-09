"""Integration-only checks for the five pending PRs in the same checkout."""
import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

import topology_scheduler as public
from topology_scheduler import (
    DeviceAssignment, DevicePlacement, LinkCost, LinkCostSource, Node, PolicyName,
    RayNodeInventory, TopologyGraph, TopologyVertex, TraceJob, Workload,
    collect_host_topology, collect_nic_inventory, run_matched_trace,
)
from test_host_topology import ETH, device, sysfs


class CombinedIntegrationTests(unittest.TestCase):
    def collect(self, layout):
        reader = sysfs(layout)
        gpu = device("00000000:17:00.0", "GPU-a")
        inventory = RayNodeInventory("node-a", "a", "topology_node:a", 1, (gpu,))
        nic = collect_nic_inventory(sysfs=reader, node_id="node-a", node_name="a")
        host = collect_host_topology((gpu,), sysfs=reader, inventory=nic,
                                     node_id="node-a", node_name="a")
        graph = TopologyGraph.from_observations(inventory, nic_inventory=nic, host_topology=host)
        return inventory, nic, host, graph

    def test_current_collector_models_round_trip_with_affinity_ties(self):
        inventory, nic, host, graph = self.collect([
            (("pci0000:00", "0000:00:01.0", "0000:17:00.0"), 0, {}),
            (("pci0000:00", "0000:00:01.0", "0000:18:00.0"), 0, {"eth0": ETH, "eth1": ETH}),
        ])
        encoded = TopologyGraph.from_observations(
            json.loads(json.dumps(asdict(inventory))),
            nic_inventory=nic.as_dict(), host_topology=host.as_dict())
        self.assertEqual(graph.to_json(), encoded.to_json())
        self.assertEqual(graph.to_json(), TopologyGraph.from_json(graph.to_json()).to_json())
        nearby = graph.nics_near_gpu(TopologyVertex("gpu", "node-a", "GPU-a").id)
        self.assertEqual([v.attributes["name"] for v in nearby], ["eth0", "eth1"])
        raw = {n["name"]: n for n in host.as_dict()["nics"]}
        for vertex in nearby:
            self.assertEqual(vertex.attributes["host_topology"], raw[vertex.attributes["name"]])

    def test_missing_gpu_path_cannot_become_known_through_graph_adapter(self):
        _, _, host, graph = self.collect([
            (("pci0000:00", "0000:17:00.1"), 0, {"eth0": ETH}),
        ])
        self.assertIsNone(host.gpus[0].nearest_nic)
        self.assertFalse(graph.nics_near_gpu(TopologyVertex("gpu", "node-a", "GPU-a").id))

    def test_collected_uuid_request_survives_trace_with_link_provenance(self):
        inventory, _, _, graph = self.collect([
            (("pci0000:00", "0000:17:00.0"), 0, {}),
            (("pci0000:00", "0000:18:00.0"), 0, {"eth0": ETH}),
        ])
        uuids = tuple(v.key for v in graph.vertices if v.kind == "gpu")
        self.assertEqual(uuids, (inventory.devices[0].uuid,))
        placement = DevicePlacement(uuids)
        assignment = DeviceAssignment(0, "a", uuids[0], uuids[0], identity_source="injected")
        with patch("topology_scheduler.device_binding.run_with_devices",
                   return_value=(["done"], (assignment,))) as execute:
            records = run_matched_trace(
                [Node("a", "X", 1, 80), Node("b", "X", 1, 80)],
                [TraceJob("job", Workload(1, 1, {"X": 1}), lambda rank: rank)],
                {("a", "b"): 2}, accelerator_type="X", device_placement=lambda plan: placement,
                link_costs={("a", "b"): LinkCost(2, LinkCostSource.MEASURED, 1000)})
        self.assertEqual(execute.call_count, len(PolicyName))
        for record in records:
            value = json.loads(json.dumps(record.as_dict()))
            self.assertEqual(value["status"], "succeeded")
            self.assertEqual(value["device_verification"]["requested_uuids"], list(uuids))
            self.assertTrue(value["device_verification"]["assignments"][0]["matched"])
            self.assertEqual(value["inputs"]["bandwidth_sources"]["a|b"]["source"], "measured")

    def test_all_public_exports_resolve_without_duplicates(self):
        self.assertEqual(len(public.__all__), len(set(public.__all__)))
        for name in public.__all__:
            self.assertTrue(hasattr(public, name), name)
        for name in ("TopologyGraph", "collect_host_topology", "DevicePlacement",
                     "run_with_devices", "DynamoService", "run_matched_trace", "measure_ray_links"):
            self.assertIn(name, public.__all__)
