"""Deterministic hardware/model/workload signatures.

Phase 8 indexes accumulated knowledge by this signature. It must be stable for a
given hardware + model + workload combination and human-readable.
"""

from __future__ import annotations

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.workload import WorkloadProfile


class Signature(ContractBase):
    hardware: str
    model: str
    workload: str

    def key(self) -> str:
        return f"{self.hardware}||{self.model}||{self.workload}"


def hardware_signature(profile: HardwareProfile) -> str:
    gpus = "|".join(f"{gpu.vendor.value}:{gpu.name}" for gpu in profile.gpus)
    return (
        f"cpu={profile.cpu.name};"
        f"cores={profile.cpu.physical_cores}/{profile.cpu.logical_cores};"
        f"ram={profile.memory.total_bytes};"
        f"gpu={gpus};"
        f"os={profile.os.version}.{profile.os.build}"
    )


def model_signature(profile: ModelProfile) -> str:
    return (
        f"{profile.name};"
        f"arch={profile.architecture};"
        f"quant={profile.quantization.value};"
        f"layers={profile.n_layers};"
        f"heads={profile.n_attention_heads};"
        f"kv_heads={profile.n_kv_heads};"
        f"ctx={profile.trained_context_length}"
    )


def workload_signature(profile: WorkloadProfile) -> str:
    return (
        f"{profile.workload_id};"
        f"kind={profile.kind.value};"
        f"seed={profile.seed};"
        f"ctx={profile.expected_context_length};"
        f"concurrency={profile.concurrency}"
    )


def make_signature(
    hardware: HardwareProfile,
    model: ModelProfile,
    workload: WorkloadProfile,
) -> Signature:
    return Signature(
        hardware=hardware_signature(hardware),
        model=model_signature(model),
        workload=workload_signature(workload),
    )
