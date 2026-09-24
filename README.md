# InferenceOS

Autonomous adaptive local LLM optimization system. The authoritative specification is
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

**Phase 0 — Contracts & Environment: complete.**
**Phase 1 — Hardware & System Characterization: complete.**
**Phase 2 — Model & Backend Abstraction: complete.**
**Phase 3 — Baseline Profiling & Telemetry: complete.**
**Phase 4 — Deterministic Bottleneck Detection: complete.**
**Phase 5 — Reasoning Interface & Local LLM Engine (Deep-Diagnosis Mode): complete.**

Phase 0 establishes the vocabulary and interfaces every later phase builds against,
and the real environment baseline (llama.cpp + Vulkan on the AMD Radeon, a real GGUF
model, logging, config, and the audit-record schema).

Phase 1 populates the `HardwareProfile` and live `SystemState` contracts from real
Windows probes: CPUID-derived CPU features, real Vulkan device enumeration plus
CUDA/ROCm presence checks, `GlobalMemoryStatusEx` RAM, physical-disk/media
classification, and live per-process CPU/RSS, hard page-fault and GPU engine
telemetry.

Phase 2 adds real GGUF metadata introspection and the concrete llama.cpp backend
adapter: it reads the actual GGUF metadata (architecture, layers, attention/KV
heads, head dimension, context length, quantization, parameter count) into
`ModelProfile`, computes an architecture-derived KV-cache/footprint estimate that is
correct for GQA/MQA and MHA, and runs real inference through `BackendAdapter`.

Phase 3 adds the baseline profiling harness: a deterministic, versioned workload
suite, real repeated runs through the backend adapter with per-phase timing (load,
TTFT, prefill, decode), throughput, token counts, peak RSS, CPU/GPU/page-fault
telemetry sampled over the inference interval, explicit metric provenance, and
persistent self-contained baseline reports with observed run-to-run variance.

Phase 4 adds deterministic bottleneck detection: it consumes a real Phase 3
`BaselineReport`, groups comparative observations, applies relative-significance and
comparative-evidence rules (model footprint vs available RAM, memory pressure ↔
paging, paging ↔ latency, CPU/GPU execution vs offload configuration state, stage
attribution), and produces an evidence-preserving `BottleneckDiagnosis`. Resource
consumption alone is never a bottleneck; "no meaningful optimization opportunity"
and "insufficient evidence" are first-class outcomes. No LLM, no recommendations.

Phase 5 adds a local reasoning layer for deep diagnosis when the deterministic
evidence is insufficient or ambiguous. A deterministic, inspectable escalation gate
decides whether to invoke a local LLM behind the existing `ReasoningEngine`
interface; the model receives a compact structured context and must return strict
JSON that is schema-validated, evidence-grounded and rejection-checked (no invented
measurements, no optimization recommendations). Reasoning failures fall back safely
to the deterministic diagnosis. No candidate generation.

## Layout

```
src/inferenceos/
  contracts/      core data contracts (HardwareProfile, ModelProfile, ModelFootprint,
                  SystemState, Observation, Bottleneck/BottleneckDiagnosis, ...)
  interfaces/     abstract ReasoningEngine and BackendAdapter interfaces
  reasoning/      ReasoningEngine implementations: deterministic stub + the local
                  LLM engine, providers, structured context/prompt, strict parsing,
                  grounding validation, escalation gate and deep-diagnosis lifecycle
  native/         platform-specific probes (WMI/CIM, CPUID, Vulkan, Win32 memory,
                  Windows performance counters) kept behind one module
  hardware/       host-independent characterization (CPU/GPU/RAM/storage/OS)
  models/         real GGUF reader + architecture/quantization normalization +
                  KV/footprint estimator + ModelProfile loader
  profiling/      deterministic benchmark workloads + repeated-run harness +
                  aggregation statistics + baseline reports
  diagnosis/      deterministic comparative bottleneck rules + confidence model +
                  evidence-preserving diagnosis artifacts
  telemetry/      persistent Windows performance-counter sampler + live SystemState
  backends/       real llama.cpp adapter (+ low-level CLI toolchain/timing/memory parsing)
  audit/          audit-record schema (Phase 8/9 persistence + rollback)
  config.py       runtime configuration (INFERENCEOS_* env overrides)
  logging_config.py
  selftest.py     contract round-trips + reasoning interface self-test
  cli.py
tests/            pytest suite (deterministic + real-machine integration)
tools/            llama.cpp Vulkan build + app-local VC++ runtime
models/           real GGUF models (gitignored)
reports/          human-verifiable verification reports (JSON)
```

## Environment baseline (this machine)

- Windows Boot Camp, Intel i7-8850H (6C/12T), 16 GB RAM, Apple SSD.
- GPU: **Radeon Pro 560X**, AMD driver 31.0.21921.1000 (Adrenalin 24.9.1), Vulkan 1.3.260.
- llama.cpp: `tools/llama.cpp` (release **b11139**, `llama-*-bin-win-vulkan-x64`).
- Microsoft VC++ 2015–2022 x64 runtime deployed app-local in `tools/llama.cpp`
  (extracted from the official `vc_redist.x64.exe`; the Boot Camp image lacked
  `vcruntime140_1.dll`). Reference copy in `tools/vcredist_x64/`.
- Model: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Human-verifiable Phase 0 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase0
```

This loads the real model, runs one prompt through the real Vulkan-enabled
llama.cpp binary, and prints:

1. the **generated text**,
2. the **backend/device** that executed it (Vulkan0 / Radeon Pro 560X, layers offloaded),
3. **GPU utilization** observed via Windows GPU Engine counters during the call,

then writes a JSON report to `reports/`. It exits non-zero unless every acceptance
check passes.

Additional checks:

```powershell
.\.venv\Scripts\python.exe -m inferenceos contracts-check      # contract JSON round-trips
.\.venv\Scripts\python.exe -m inferenceos reasoning-selftest   # stub ReasoningEngine round-trip
.\.venv\Scripts\python.exe -m mypy                             # all contracts type-check
.\.venv\Scripts\python.exe -m pytest                           # unit + deterministic tests
```

## Human-verifiable Phase 1 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos profile
```

Prints the full real `HardwareProfile` and a live `SystemState` snapshot for this
machine — CPU identity/cores/caches/instruction sets, detected GPUs and usable
backends, physical RAM and modules, physical storage with SSD/HDD media type, the
storage backing the model file, top processes by RSS with CPU%, hard page-fault
rate, and GPU engine/adapter-memory telemetry — then writes a JSON report to
`reports/`. Cross-check the CPU/cores and RAM against Task Manager, and the GPU
against Device Manager.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase1
```

Runs the real Phase 1 acceptance: it builds the profile, captures live telemetry,
cross-checks the CPU/RAM figures against `psutil` (the OS), identifies the storage
backing the model, and runs a short real llama.cpp Vulkan inference while observing
real Windows GPU Engine utilization. It prints every check as PASS/FAIL and writes
`reports/phase1_verification_*.json`; it exits non-zero unless all checks pass.

## Phase 1 detection notes / limitations

- **CPU features** come from a real `CPUID` probe (cross-checked against Windows
  `IsProcessorFeaturePresent`). AVX-family features are only reported when the OS
  has enabled the extended state. Cache sizes are as reported by CPUID leaf 4 for
  the calling processor (L1/L2 per-core, L3 physical), with a WMI fallback.
- **GPU usability** is decided by real Vulkan physical-device enumeration; a GPU
  visible in WMI but not enumerated by Vulkan is recorded as detected-but-not-usable.
  CUDA/ROCm are reported as *detected* when their runtimes are loadable; their
  device usability is not validated in Phase 1.
- **GPU total VRAM** is only available from Vulkan device-local heaps; the Windows
  `GPU Adapter Memory` counters expose *dedicated usage* but not a total, so the
  live `gpu_memory[*].memory_total_bytes` is `None` rather than a fabricated value.
- **Hard page faults** are read from `\Memory\Pages Input/sec` (real hard faults);
  Windows has no swap/swappiness equivalent, so commit ratio and hard-fault rate are
  the pressure signals.
- **Storage media** (SSD/HDD) is read from the Storage management namespace
  (`MSFT_PhysicalDisk`); direct physical-drive IOCTL classification was rejected
  because it requires elevation (standard-user default).
- **GPU `device_index`** in live telemetry follows Windows LUID ordering, which is
  not necessarily the same ordering as Vulkan device indices.

## Human-verifiable Phase 2 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos model-info
```

Parses the real GGUF metadata of every model in `models/` and prints the normalized
`ModelProfile` (format, architecture, quantization, parameter count, context length,
layers, attention heads, KV heads, head dimension, embedding/vocab) plus the KV-cache
and total-footprint estimates at several context lengths. `--json` emits a
machine-readable form; `--model <path>` targets one file.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase2
```

Runs the real Phase 2 acceptance over at least two actual GGUF models:

1. parses each model's real metadata into `ModelProfile` and reports the
   architecture-derived KV/footprint estimates;
2. exercises the concrete `BackendAdapter` (real `llama-cli`), prints the backend
   capabilities, and generates real text through the **Vulkan/Radeon** path with
   layer offload;
3. measures **predicted vs measured** memory across 2–3 context lengths on the CPU
   (so the KV cache is in system RAM) and compares the measured peak-RSS delta
   against the predicted KV delta, alongside the KV size the backend itself reports.

It prints every check PASS/FAIL and writes `reports/phase2_verification_*.json`;
it exits non-zero unless all checks pass.

## Phase 2 model/estimation notes / limitations

- **KV formula** (per sequence position):
  `n_layers * n_kv_heads * (key_length + value_length) * element_bytes`, with
  `element_bytes = 2` for the default f16 cache. The **KV-head** count is used, not
  the attention-head count, so GQA/MQA models are not overestimated.
- **`weight_bytes` is an estimate** derived from the real (already-quantized) file
  size; it is deliberately kept distinct from `file_size_bytes` (the fs size), the
  KV estimate, and from **measured** process RSS. Phase 2 only performs the limited
  measurements needed to validate the estimate; it does not build the Phase 3
  profiling harness.
- **Backend execution model:** the acquired llama.cpp build is a one-shot CLI, so
  the adapter's `load` pins the model + runtime configuration (and probes real
  devices) and `generate` runs a real `llama-cli` process. Residency is per
  invocation; a resident-server adapter can implement the same `BackendAdapter`
  interface later without changing callers.
- **KV-cache type:** the estimator assumes an f16 cache; `ModelProfile`/
  `ModelFootprint` scale linearly for other element widths, but quantized KV-cache
  types (block scales) are not modelled and remain estimates.
- **Real models used for acceptance:** `qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (Qwen2, GQA, 28 layers / 12 heads / 2 KV) and `stories15M-q4_0.gguf`
  (Llama, standard MHA, 6 layers / 6 heads / 6 KV). Models are gitignored and not
  committed.

## Human-verifiable Phase 3 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos baseline --repetitions 3
```

Runs the deterministic benchmark suite (`short-short`, `long-prefill`,
`short-long-decode`) against every model in `models/` through the real backend and
prints a per-run table plus aggregated statistics (mean/stddev/min/max/coefficient
of variation), then writes a self-contained `reports/baseline_*.json`. Options:
`--model`, `--workload` (repeatable), `--repetitions`, `--mode cold|warm|both`,
`--context`, `--gpu-layers`, `--no-telemetry`, `--out`.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase3
```

Runs the real Phase 3 acceptance: it profiles both structurally different real
models (GQA + MHA) through the **Vulkan/Radeon** path, captures per-phase timings
(load, TTFT, prefill, decode), throughput, token counts, peak RSS and interval
CPU/GPU/hard-fault/memory telemetry, demonstrates repeated-run reproducibility
against agreed variance bands, performs a cold run and a profiler-overhead
comparison, and prints every check PASS/FAIL. It writes
`reports/phase3_verification_*.json` and exits non-zero unless all checks pass.

## Phase 3 measurement notes / limitations

- **Only measurements** are produced; there is no diagnosis, ranking, or automatic
  configuration change (those are Phase 4+).
- **Cold vs warm semantics are honest:** every measured run is a fresh `llama-cli`
  process. `COLD` = no preceding warm-up (model likely not in the OS page cache);
  `WARM` = measured after an explicit warm-up so the model file is OS-page-cache
  resident. `WARM` does **not** mean a resident model process or reused KV/weights;
  resident-process warm inference is not supported by this adapter and is reported
  as `resident_warm_supported: false`.
- **TTFT is derived** (prefill + one decode step) because llama.cpp exposes no
  first-token timestamp; it is labelled `derived` in each run's metric-quality map.
- **Unavailable telemetry is explicit:** GPU total VRAM is not exposed by Windows
  counters and is reported as `None` with quality `unavailable`.
- **Profiler overhead** is quantified in acceptance (decode tokens/sec with external
  counter sampling on vs off; observed ~1–5%).
- **COLD runs are performed once per model/workload** because the OS page cache
  cannot be evicted without elevation; repeating them would not be genuinely cold.

## Human-verifiable Phase 4 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos diagnose
```

Consumes the newest Phase 3 report in `reports/` (or `--report PATH`), runs the
deterministic detector, prints a human-readable evidence chain, and writes a
machine-readable `reports/diagnosis_*.json`. `--json` prints the artifact;
`--strict` exits non-zero when no meaningful opportunity is found.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase4
```

Runs the real Phase 4 acceptance:

1. ingests a real Phase 3 report and diagnoses it;
2. confirms the **false-positive guard** — a workload with sustained ~96% GPU
   utilization is *not* reported as a GPU bottleneck, only as an explained
   non-meaningful observation, and the overall result is "no meaningful
   optimization opportunity";
3. runs a **controlled real induced-memory-pressure** comparison (control vs. a
   comparable run while physical memory is deliberately occupied) and confirms the
   detector reports a genuine, evidence-supported **paging** bottleneck (hard
   faults materially above reference, elevated commit pressure, degraded latency).

It prints every check PASS/FAIL and writes `reports/phase4_verification_*.json`
plus per-scenario diagnosis artifacts.

## Phase 4 diagnosis notes / limitations

- **Resource consumption is not a bottleneck.** Thresholds/ratios only raise
  *candidate* signals; a finding becomes meaningful only with corroborating and,
  where possible, comparative evidence (e.g. the same model/workload under a
  different condition performing materially better). High GPU, high RSS, or high
  CPU alone do not produce a bottleneck.
- **Evidence is preserved separately from interpretation:** each `Bottleneck`
  carries supporting and contradicting `EvidenceItem`s, a rationale, the affected
  inference stage, unavailable measurements, and the detector version.
- **Confidence is explainable, not calibrated:** `confidence = clamp((support −
  0.5·contradict)/3, 0, 1) · quality_factor`, where the quality factor is reduced by
  insufficient repetitions or high variance (unless high-magnitude evidence is
  present). Categories are `none`/`low`/`medium`/`high`.
- **First-class terminal outcomes:** `NONE` (no meaningful opportunity) and
  `INSUFFICIENT_EVIDENCE` (below-threshold or conflicting signals →
  `escalation_required`, the Phase 5 entry point). No optimization is recommended.
- **The induced-pressure scenario is a controlled real measurement**, not a
  synthetic fixture; it is labelled as such in its diagnosis artifact. It holds a
  bounded amount of memory (leaving a safety margin) during otherwise-identical
  real runs.
- GPU total VRAM and per-process lists remain unavailable in the real baseline and
  are reported explicitly as unavailable.

## Human-verifiable Phase 5 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos deep-diagnose
```

Loads the newest Phase 4 diagnosis (or a Phase 3 report / `--diagnosis` / `--report`),
applies the deterministic escalation gate, and — only if escalation is justified —
invokes the configured local reasoning model. It prints the full deep-diagnosis
outcome (escalation reason, raw structured result, lifecycle cost, final root cause)
and writes `reports/deep_diagnosis_*.json`. `--no-reasoning` keeps it deterministic.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase5
```

Runs the real Phase 5 acceptance: a real Phase 3 report → real Phase 4 diagnosis →
deterministic escalation → **real local reasoning model** → strict validation and
evidence grounding → final deep diagnosis, plus a strong-deterministic case that is
correctly *not* escalated, a provider-failure fallback, a target-health re-measurement
and a reproducibility comparison. It prints every check PASS/FAIL and writes
`reports/phase5_verification_*.json` and a `phase5_deep_diagnosis_*.json` artifact.

## Phase 5 reasoning notes / limitations

- **The LLM is never the source of telemetry.** It reasons over a compact structured
  context and must return strict JSON referencing supplied evidence ids; it may not
  invent measurements, and optimization recommendations are stripped/rejected.
- **Escalation is deterministic and inspectable** (`EscalationDecision.reason_code`):
  strong deterministic diagnoses are never escalated, no-signal cases are never
  escalated, and missing telemetry is not escalated (reasoning cannot resolve it).
- **Local only.** The reasoner runs through the same local llama.cpp binary with
  JSON-schema constrained generation (`--no-jinja -j`); there is no network or cloud
  fallback and no silent model substitution.
- **Default reasoner:** the locally available `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (configurable via `INFERENCEOS_REASONING_MODEL_PATH` / `reasoning_provider`); if it
  is absent the engine reports itself unavailable and the deterministic diagnosis is
  retained.
- **Non-resident lifecycle:** the backend is process-per-generation, so there is no
  resident target to release; the lifecycle records reasoning latency and peak RSS.
- **Confidence categories are evidence-derived**, not calibrated probabilities;
  reasoning output is diagnostic only (Phase 6 owns candidate generation).

## Human-verifiable Phase 6 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos candidates
```

Loads the newest Phase 5 deep diagnosis (or `--deep` / `--diagnosis` / `--report`),
maps the validated bottleneck to registered actions through an explicit causal layer,
evaluates preconditions against real hardware/model/workload context, estimates
benefit/cost/risk, gates and ranks the candidates, and prints/writes
`reports/candidates_*.json`. It **executes nothing**: candidates are only prepared for
Phase 7 consideration. `--reasoning` additionally lets the local reasoner *propose*
registered actions when deterministic mapping is insufficient; every proposal is
resolved against the registry, schema-validated, and evidence-grounded.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase6
```

Runs the real Phase 6 acceptance: (A) a real induced-memory-pressure diagnosis yields
relevant registered candidate(s) with real evidence references, evaluated
preconditions, and a transparent multi-factor ranking — with no execution and no
environment mutation; (B) the real healthy baseline produces no speculative
candidates and an explicit terminal reason; (C) an adversarial reasoning proposal
(unregistered action, hallucinated evidence, fabricated measurement) is rejected, so
the LLM cannot bypass deterministic safety. It prints every check PASS/FAIL and writes
`reports/phase6_verification_*.json`.

## Phase 6 candidate-generation notes / limitations

- **No execution, no mutation.** Phase 6 only inspects context and ranks options. It
  never changes runtime parameters, OS settings, processes, or model files, and it
  contains no subprocess/command construction. Phase 7 owns experiments.
- **Registered vocabulary only.** Candidates may only reference `CandidateAction`
  ids in the action registry; unknown/unregistered actions are rejected. The initial
  vocabulary covers v1 surfaces (context/KV right-sizing, mmap, GPU offload, threads)
  plus explicitly approval-gated broad surfaces (system pagefile, external process)
  and a non-reversible model-requantization action that policy rejects.
- **Causal, not resource-based.** An action is only generated when its registered
  `supported_bottlenecks` contains the *validated* effective diagnosis; high
  utilization with no meaningful diagnosis produces no candidates.
- **Uncertainty-aware benefit.** KV-derived context savings are reported as bounded
  estimates; effects that cannot be responsibly estimated are explicitly `known=false`
  and must be measured in Phase 7. No speedup percentages are fabricated.
- **External/system-wide/elevated work is always approval-gated** and never selected
  for autonomous consideration; non-reversible/high-risk actions are rejected.
- **Optimizer overhead is measured** (`OptimizerOverhead`) and is part of the
  candidate cost; ranking decomposes benefit, uncertainty, cost, overhead, risk,
  reversibility, confidence, causal strength, scope, approval burden, and precondition
  certainty.
- **Pressure induction is load-sensitive:** `verify-phase6` may fall back to the most
  recent real meaningful `phase4_diagnosis_pressure_*.json` when a fresh induced run
  does not reach the meaningful threshold.

## Human-verifiable Phase 7 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos experiment
```

Turns the newest eligible Phase 6 candidate (or `--plan` / `--diagnosis` / `--deep` /
`--report` / `--candidate-id`) into a real controlled experiment: it revalidates the
candidate immediately before execution, captures a real baseline through the Phase 3
harness, applies exactly that candidate's runtime change through the registered
action surface, measures real inference again, validates output quality, compares on
a statistically transparent basis, decides keep/reject, and restores the protected
baseline. It prints the full result and writes `reports/experiment_*.json` plus an
audit record `reports/experiment_audit_*.json`. If there is no eligible candidate it
executes nothing and says so.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase7
```

Runs the real Phase 7 acceptance: a real Phase 4 induced-pressure diagnosis becomes an
eligible registered candidate; the candidate is applied to the real llama.cpp backend
with real before/after measurements, quality validation, a decision, and verified
baseline restoration; the real healthy baseline executes no experiment; an
approval-required/system-wide candidate is blocked before execution; and a controlled
quality failure blocks a performance improvement. It prints every check PASS/FAIL and
writes `reports/phase7_verification_*.json`.

## Phase 7 experiment / quality notes / limitations

- **Apply → measure → validate → restore.** Phase 7 owns the first real mutation, but
  it mutates only the target model's *runtime configuration* (an inference-level,
  reversible surface) and always restores and verifies the protected baseline in a
  `finally` block. It never persists a deployment.
- **Registry-gated execution.** Only registered actions with a real, safe runtime
  surface are executable (`runtime.right_size_context`, `runtime.enable_mmap`,
  `runtime.reduce_gpu_offload_layers`, `runtime.reduce_threads`). External-process,
  system-wide, elevated, and non-reversible actions have no execution surface and are
  rejected. Candidate parameters are re-validated against the schema; no command is
  ever constructed from candidate data.
- **Revalidated immediately before execution.** A candidate that was eligible during
  Phase 6 is re-resolved through the registry, its preconditions re-run against the
  current context, its environment assumptions checked, and its freshness confirmed.
- **Honest, diagnosis-aware decisions.** The decision keys on the metric(s) that
  correspond to the diagnosed bottleneck (not an unrelated side effect), and requires
  no quality/objective/resource regression. `no_meaningful_improvement` is a normal,
  first-class outcome; thresholds are configurable policy, never tuned post hoc.
- **Quality first.** Performance gains cannot mask empty, malformed, truncated, or
  materially divergent output. The deterministic benchmark suite supplies a real
  reference output; an unknown quality impact is reported as `unknown`, not assumed
  good.
- **Noise-aware and bounded.** Repetition policy, the `max_total_runs` budget, the
  time limit, and per-phase minimum valid samples bound every experiment; measurement
  failures and invalid runs are recorded, not hidden.
- **Overhead is separate.** Experiment/optimizer overhead is timed separately from the
  backend's own performance counters.
- **No Phase 8+ behaviour.** Phase 7 emits a structured `ExperimentResult` and an
  `AuditRecord` only; it holds no long-term knowledge, no learned rankings, no
  rollback database, and never chains another experiment.

## Human-verifiable Phase 8 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos knowledge --health
.\.venv\Scripts\python.exe -m inferenceos knowledge --action runtime.right_size_context --applicability --detail
```

Inspects the durable knowledge store (`data/knowledge.db`): stored optimization
records, historical candidate sequences, rollback records, and — with
`--applicability` — an explainable verdict on whether each historical result still
applies to the current real machine/model/workload context. `--json` emits the same
information machine-readably. The command never starts an optimization loop.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase8
```

Runs the real Phase 8 acceptance using the actual Phase 7 `ExperimentResult`: persists
it into the durable store, reinitializes the store and retrieves it, reads it from a
separate process, verifies duplicate recording is idempotent, evaluates real
applicability, shows that a materially different context requires revalidation /
is not applicable, verifies the rollback record's integrity, confirms negative
(non-improving) knowledge is retained, detects a tampered record, and measures
knowledge-store overhead. It prints every check PASS/FAIL and writes
`reports/phase8_verification_*.json`.

## Phase 8 knowledge / rollback notes / limitations

- **Conditional evidence, never universal truth.** Historical results are stored with
  a reproducible context fingerprint (identity vs. transient conditions). A past
  success is classified as directly applicable, conditionally applicable, requiring
  revalidation, not applicable, or insufficient context — it is never blindly reused.
- **Durable and immutable.** Knowledge lives in a real SQLite database with a schema
  version, per-record checksums, append-only lifecycle events, and `experiment_id`
  uniqueness. Records are never silently overwritten; duplicate recording is a no-op
  and a differing payload under the same id is reported as a conflict.
- **Honest persistence failures.** A store failure is surfaced as `failed` /
  `unavailable`, distinct from "persisted" and from "history unavailable"; the
  optimizer never fabricates knowledge it could not write or read.
- **Rollback foundation only.** Rollback records store the prior/resulting runtime
  configuration and the exact change as *values* plus an integrity verification
  (registry + parameter + prior-value + context). Phase 8 never executes a rollback.
- **Not a learned optimizer.** No learned success predictors, ranking weights, policy
  updates, reinforcement learning, or cross-machine transfer. The store is evidence +
  explicit applicability reasoning.
- **Not a deployment loop.** Phase 8 records and retrieves; it never applies an
  optimization or consumes history to deploy one. That is Phase 9.
- **Knowledge-store overhead** is measured (write/read/applicability latency and
  database size) and reported; the optimizer operates correctly with no history.
- **Schema/version mismatch** is rejected explicitly rather than silently migrated.
