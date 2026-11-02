"""Pure Stage-A observers over retained b11139 startup, /slots and native inventory.

No configuration request, launch command or old study record supplies an observation.
Missing or conflicting evidence stays unknown. Historical logs are parser fixtures only.
"""
from __future__ import annotations

import re
from typing import Any

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.hardware.cpu import build_cpu_profile
from inferenceos.tune.lookup_qualification import MachineBinding


def observe_effects(text: str, slots: Any) -> dict[str, Any]:
    values: dict[str, Any] = RuntimeConfiguration().model_dump(mode="json")
    sources: dict[str, list[str]] = {}
    problems = []

    def read(name: str, pattern: str, convert: Any = int) -> None:
        found = list(re.finditer(pattern, text, re.MULTILINE))
        observations = [convert(m.group(1)) for m in found]
        if observations and all(v == observations[0] for v in observations):
            values[name] = observations[0]
            sources[name] = [m.group(0) for m in found]
        elif observations:
            problems.append("contradictory runtime evidence: " + name)

    # system_info is emitted after common CPU parameter postprocessing; the actual
    # pool message supplies an additional generation-thread consistency check.
    read("threads", r"(?:system_info: |threadpool init, )n_threads\s*=\s*(\d+)")
    read("threads_batch", r"system_info: n_threads\s*=\s*\d+\s*\(n_threads_batch\s*=\s*(\d+)\)")
    read("batch_size", r"llama_context: n_batch\s*=\s*(\d+)\s*$")
    read("ubatch_size", r"llama_context: n_ubatch\s*=\s*(\d+)\s*$")
    read("gpu_offload_layers", r"load_tensors: offloaded (\d+)/\d+ layers to GPU")
    read("context_length", r"load_model: initializing, n_slots = \d+, n_ctx_slot = (\d+)")
    read("use_mmap", r"load_tensors: loading model tensors[^\n]*\(load_mode = (mmap|none)\)", lambda v:v == "mmap")
    read("flash_attention", r"resolve_fused_ops: Flash Attention (enabled|disabled)", lambda v:v == "enabled")
    # auto in llama_context is an unresolved request, never a true/false observation.
    read("device", r"llama_prepare_model_devices: using device (Vulkan\d+) \(", str)
    read("kv_cache_type_k", r"llama_kv_cache[^\n]*?\bK \(([a-z0-9_]+)\):", str)
    read("kv_cache_type_v", r"llama_kv_cache[^\n]*?\bV \(([a-z0-9_]+)\):", str)
    for name in ("threads", "threads_batch", "batch_size", "ubatch_size", "context_length"):
        if values[name] is not None and values[name] <= 0:
            problems.append("nonpositive runtime setting: " + name)
    initialized_slots = list(re.finditer(r"load_model: initializing, n_slots = (\d+), n_ctx_slot = \d+", text))
    if not initialized_slots or any(int(m.group(1)) != 4 for m in initialized_slots):
        problems.append("four initialized slots not observed or startup slot counts conflict")
    else:
        sources["server_slots"] = [m.group(0) for m in initialized_slots]
    # An explicit alias/override is not observable from command absence. This first
    # proposal requires null requested aliases/overrides and separately observed sides.
    if (not isinstance(slots, list) or len(slots) != 4 or any(not isinstance(s, dict) for s in slots)
            or any(type(s.get("id")) is not int for s in slots)
            or {s.get("id") for s in slots} != {0, 1, 2, 3}
            or any(s.get("is_processing") is not False for s in slots)
            or any(type(s.get("n_ctx")) is not int or s.get("n_ctx") != 4096 for s in slots)):
        problems.append("four distinct idle 4096-capacity slots not observed")
    return dict(configuration=values, sources=sources, problems=problems)


def slot_activation_problems(slots: Any, arm: str) -> list[str]:
    """Treatment proof is separate from the base compared between the two arms."""
    if arm not in ("control", "lookup"):
        return ["unknown slot activation arm"]
    expected = arm == "lookup"
    if (not isinstance(slots, list) or len(slots) != 4
            or any(not isinstance(s, dict) or type(s.get("speculative")) is not bool for s in slots)):
        return ["literal speculative activation booleans required for every slot"]
    if any(s["speculative"] is not expected for s in slots):
        return ["slot speculative activation contradicts the arm/startup requirement"]
    return []


def machine_from_raw(raw: Any, device: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(device, str) or re.fullmatch(r"Vulkan\d+", device) is None:
        raise ValueError("raw machine observation and Vulkan association required")
    cim = raw["cim"]
    processors = cim["processors"]
    if not isinstance(processors, list) or len(processors) != 1:
        raise ValueError("one unambiguous physical CPU inventory required")
    cpu_raw = processors[0]
    for name in ("Name", "Manufacturer", "Architecture", "NumberOfCores", "NumberOfLogicalProcessors"):
        if cpu_raw.get(name) is None:
            raise ValueError("missing CPU source field: " + name)
    if any(type(cpu_raw[k]) is not int or cpu_raw[k] <= 0 for k in ("NumberOfCores", "NumberOfLogicalProcessors")):
        raise ValueError("explicit positive CPU topology required")
    for flags in (raw["cpuid"],raw["os_flags"]):
        if not isinstance(flags, dict) or any(not isinstance(k,str) or type(v) is not bool for k,v in flags.items()):
            raise ValueError("explicit CPU feature observations required")
    cpu = build_cpu_profile(processors, cpuid=raw["cpuid"], os_flags=raw["os_flags"])
    vulkan = raw["vulkan"]
    if not isinstance(vulkan, list) or len(vulkan) != 1 or vulkan[0].get("device_type") not in (1, 2):
        raise ValueError("one usable integrated/discrete Vulkan device required")
    gpu = vulkan[0]
    index = int(device.removeprefix("Vulkan"))
    if type(gpu.get("index")) is not int or gpu["index"] != index:
        raise ValueError("startup/native Vulkan device association differs")
    controllers = [v for v in cim["video_controllers"] if v.get("Name") == gpu.get("name")]
    if len(controllers) != 1 or not controllers[0].get("DriverVersion"):
        raise ValueError("selected GPU needs a unique OS driver association")
    os_raw = cim["os"]
    os_build = "/".join(str(os_raw.get(k, "")) for k in ("Caption", "Version", "BuildNumber", "OSArchitecture"))
    if any(not os_raw.get(k) for k in ("Caption", "Version", "BuildNumber", "OSArchitecture")):
        raise ValueError("complete original OS build inventory required")
    power = raw["power_plan"]
    if not isinstance(power, str) or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", power):
        raise ValueError("live power-plan GUID required")
    vendor = {0x1002:"AMD", 0x8086:"Intel", 0x10DE:"NVIDIA"}.get(gpu.get("vendor_id"))
    if vendor is None:
        raise ValueError("known selected GPU vendor required")
    binding = MachineBinding(hostname=raw["hostname"], os_build=os_build, architecture=cpu.architecture or "unknown",
                             cpu_name=cpu.name, cpu_vendor=cpu.vendor or "unknown", physical_cores=cpu.physical_cores,
                             logical_cores=cpu.logical_cores, instruction_sets=tuple(cpu.instruction_sets),
                             ram_bytes=raw["memory"]["total_physical_bytes"], gpu_name=gpu["name"], gpu_vendor=vendor,
                             gpu_vram_bytes=gpu["device_local_bytes"], driver_version=controllers[0]["DriverVersion"],
                             pci_device_id=controllers[0].get("PNPDeviceID"), usable_gpu_count=1,
                             device_name=gpu["name"], device_index=index, power_plan=power, power_source=raw["power_source"])
    return binding.model_dump(mode="json")
