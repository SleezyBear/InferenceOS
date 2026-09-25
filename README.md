# InferenceOS

Autonomous adaptive local LLM optimization system. The authoritative specification is
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). This README describes what is
**actually implemented**, how to reproduce its acceptance evidence, and where the
evidence is real versus controlled.

> **Completion status (implementation phases 0–11 plus the R remediation rounds and
> the final cross-phase remediation round): complete.**
> No Phase 12 exists; no kernel/code generation, graph rewriting, arbitrary
> requantization, system-wide OS modification, remote/cloud deployment, or
> external-process mutation is implemented.

## 1. Purpose and governing principle

InferenceOS observes a local `llama.cpp` inference workload, determines whether a
*meaningful* bottleneck exists, reasons about the cause (deterministically, escalating
to a local LLM only when justified), generates causally-relevant candidate
interventions, evaluates them with real controlled experiments, validates quality,
keeps or reverts, records the outcome durably (with rollback), runs the same
reasoning at runtime under conservative constraints, and can surface read-only
recommendations about external processes.

The governing principle, applied at every layer:

> **Every claimed capability must correspond to a real implementation that can be
> exercised, measured, quality-checked, persisted where appropriate, and rolled back
> safely.**

Corollaries enforced throughout:

* Resource consumption is **never** equated with a bottleneck.
* "No meaningful optimization opportunity — retain the current configuration" is a
  first-class, successful outcome.
* The optimizer never invents candidates for unregistered or unsupported actions.
* Controlled/fixture evidence is labelled as such; it is never presented as
  uncontrolled real-world evidence.

## 2. Architecture

```
src/inferenceos/
  contracts/    typed, immutable data contracts (pydantic) for every phase
                (hardware, model, observation, diagnosis, optimization, experiment,
                 quality, knowledge, loop, runtime, external, fresh/deployment)
  interfaces/   abstract ReasoningEngine and BackendAdapter interfaces
  native/       platform probes (WMI/CIM, CPUID, Vulkan, Win32 memory, perf counters)
  hardware/     host-independent characterization and live SystemState
  telemetry/    persistent Windows performance-counter sampler
  models/       GGUF reader, architecture normalization, KV/footprint estimator,
                independent KV validation
  profiling/    deterministic workloads, repeated-run harness, baseline reports
  diagnosis/    deterministic comparative bottleneck rules + evidence model
  reasoning/    stub + local LLM engines, escalation gate, deep-diagnosis lifecycle
  optimization/ action registry, context, strategies, eligibility, ranking, generator
  experiment/   executor, measurement, quality, comparison, decision, restoration
  knowledge/    durable SQLite store (Phase 8) + rollback record/executor
  loop/         Phase 9 orchestrator + Phase 9R fresh-run provenance/deployment
  runtime/      Phase 10/10R monitor, resident workload, approval, interventions
  external/     Phase 11R read-only external-process observation + recommendation
  backends/     llama-cli adapter, resident llama-server backend + adapter, process
                control (Job Objects), peak-RSS tracking
  audit/        audit-record schema
  deployment.py persisted DeploymentState validation + minimal `serve`
  remediation.py cross-phase remediation acceptance
  config.py     runtime configuration (INFERENCEOS_* env overrides)
  cli.py, selftest.py, logging_config.py
tests/          pytest suite (deterministic + real-machine integration)
tools/          llama.cpp Vulkan build + app-local VC++ runtime
models/         real GGUF models (gitignored)
reports/        human-verifiable verification reports (JSON)
```

### 2.1 Optimization loop (Phase 9 / 9R)

```
OBSERVE → DIAGNOSE → (DEEP DIAGNOSE when justified) → UNDERSTAND INTERACTIONS
  → GENERATE CANDIDATES → ASSESS BENEFIT/COST/RISK → CONSULT CONTEXTUAL KNOWLEDGE
  → CHECK SAFETY → EXPERIMENT OR RECOMMEND → REQUEST APPROVAL WHEN REQUIRED
  → APPLY → MEASURE → VALIDATE QUALITY → KEEP OR REVERT → VERIFY
  → PERSIST KNOWLEDGE → CONTINUE OR TERMINATE
```

### 2.2 Runtime adaptive loop (Phase 10 / 10R)

The active-runtime monitor samples lightweight telemetry (cheap psutil samples, not
the full Phase 3 counter harness), temporally qualifies deviations, reuses the Phase 4
diagnosis engine, evaluates explicit intervention economics, and only acts on a
persistent condition with explicit approval. Disruptive adaptation (pause/reload) is
never automatic; a checkpoint is taken before pausing and the in-flight generation is
drained, never killed.

### 2.3 External-process recommendations (Phase 11 / 11R)

```
observe (read-only) → attribute (proportional) → require a diagnosed bottleneck
  → generate structured recommendation → present → USER DECISION → record
```

There is **no execution stage**. Phase 11R is recommendation-only: it never
terminates, suspends, reprioritises, changes affinity of, or reconfigures an external
process, and it never requires elevation. A recommendation that the user "accepts" is
recorded as user intent only.

## 3. Phase-by-phase status

| Phase | Scope | Real acceptance command | Status |
|---|---|---|---|
| 0 | Contracts & environment (llama.cpp + Vulkan) | `verify-phase0` | PASS |
| 1 | Hardware/system characterization | `verify-phase1` | PASS |
| 2 | Model/backend abstraction + KV estimation | `verify-phase2` | PASS |
| 3 | Baseline profiling & telemetry harness | `verify-phase3` | PASS |
| 4 | Deterministic bottleneck detection | `verify-phase4` | PASS |
| 5 | Reasoning interface (legacy, non-resident) | `verify-phase5` | see note §8 |
| 5R | Resident target lifecycle & deep diagnosis | `verify-phase5r` | PASS |
| 6 | Candidate generation & controlled optimization | `verify-phase6` | PASS |
| 7 | Experiment execution & quality validation | `verify-phase7` | PASS |
| 7R | Resident experiment/quality machinery | `verify-phase7r` | PASS |
| 8 | Persistent knowledge & rollback foundation | `verify-phase8` | PASS |
| 8R | Persistent knowledge & **real** rollback | `verify-phase8r` | PASS |
| 9 | Pre-deployment autonomous loop | `verify-phase9` | PASS |
| 9R | Fresh autonomous run + deployment state | `verify-phase9r` | PASS |
| 10 | Runtime adaptive monitoring | `verify-phase10` | PASS |
| 10R | Resident runtime adaptive monitoring | `verify-phase10r` | PASS |
| 11R | External-process recommendations (recommendation-only) | `verify-phase11r` | PASS |
| remediation | Candidate surface, peak RSS, KV validation, `serve` | `verify-remediation` | PASS |

`verify-phase11` is retained as a deprecated alias for `verify-phase11r`.

## 4. Candidate surfaces actually supported

Only actions that are registered **and** have a real, safe, reversible runtime
surface can be executed. The registry and the executor are kept in lockstep by tests.

### Executable (target-model runtime configuration only)

| Action | Surface | Direction / trigger |
|---|---|---|
| `runtime.right_size_context` | context/KV management | reduce context under memory pressure/paging |
| `runtime.enable_mmap` | memory management | enable mmap under memory pressure/paging |
| `runtime.reduce_gpu_offload_layers` | GPU offload | reduce offload under VRAM limits |
| `runtime.increase_gpu_offload_layers` | GPU offload | increase offload under diagnosed GPU underutilization |
| `runtime.reduce_threads` | runtime params | right-size threads under CPU contention |

Every executable action flows through the full lifecycle: diagnosis → applicability →
candidate → executor apply → real measurement → quality validation → keep/revert →
knowledge/rollback provenance (Phase 6/7/8/9 machinery).

### Recommendation-only / gated (no executor)

| Action | Why |
|---|---|
| `external.request_close` | belongs to the user; Phase 11R recommends only |
| `external.reduce_priority` | belongs to the user; Phase 11R recommends only |
| `external.suspend_process` | belongs to the user; Phase 11R recommends only |
| `system.increase_pagefile` | system-wide, requires elevation; never automatic |
| `model.requantize` | destructive/non-reversible; rejected by policy |

### Deferred / unsupported candidate surfaces

These are recognised backend knobs or plan-listed surfaces that are deliberately
**not** registered as executable candidates in this v1. They are omitted from the
registry (and therefore from all policy code) so nothing implies executable support
that does not exist.

* **Batch/ubatch tuning** (`batch_size`, `ubatch_size`): the runtime accepts the
  fields and llama.cpp exposes `-b`/`-ub`, but no deterministic diagnosed bottleneck
  currently maps to batch tuning with a grounded causal rationale, so no candidate is
  registered. Adding it would require establishing a real causal bottleneck first.
* **Flash attention** (`flash_attention`) and **KV-cache element type**
  (`kv_cache_type`): backend knobs, not required by the v1 candidate-surface contract.
  They were previously referenced in the quality-tolerance policy as if executable;
  those forward references have been removed. Implementing them honestly would require
  a full execution + quality-validation path.
* **Cheap perplexity/perplexity-proxy quality gate**: see §7.

The plan's Phase 6 "v1 candidate surfaces" list is a **scope vocabulary**; the
registry above is the accurate statement of what can really be optimized today.

## 5. Measurement semantics

### 5.1 Peak vs final RSS (remediation)

* The **resident** `llama-server` generation samples real target-process memory during
  the generation interval (one cheap psutil snapshot per 100 ms) and folds in the
  authoritative Windows peak working set (`peak_wset`) where available. It reports
  `peak_rss_bytes` (max of samples + OS peak) **and** `final_rss_bytes` (last sample),
  plus `rss_sample_count`. These are distinct fields; peak is never aliased to final.
* The **process-per-generation** `llama-cli` path samples RSS over the whole process
  lifetime (load + generation) and records peak and final from the same sampler.
* The resident target PID is used (never the monitor process); if the process exits,
  sampling stops and the last valid values are retained.

### 5.2 KV-cache validation (remediation)

Phase 2 validates the backend-reported KV allocation against an **independent**
derivation from GGUF architecture metadata only:

```
expected_kv = n_layers * n_kv_heads * (key_length + value_length) * element_bytes * context
```

The derivation uses the **KV-head** count (not the attention-head count), so GQA/MQA
models are not overestimated. The explicit relative tolerance
(`KV_CACHE_RELATIVE_TOLERANCE = 0.05`) covers allocator alignment / per-context
overhead; it is a property of the derivation, not tuned per run. Validation refuses to
compare one llama.cpp-reported number against another, marks an unavailable
backend value as *unavailable* (never a silent pass), and additionally proves that the
plausible-but-wrong attention-head derivation does **not** agree with the backend.

### 5.3 Measurement validity — power/thermal/frequency confounds (B1)

A transient battery/power/thermal/frequency condition must not be silently read as
evidence that a runtime configuration should be optimized.

**Observed (real Windows APIs, isolated behind `native/windows.py`):**
* power source via `GetSystemPowerStatus` → `PowerState.AC | BATTERY | UNKNOWN`
  (`UNKNOWN` is never coerced to AC);
* live processor clock via `powrprof.CallNtPowerInformation(ProcessorInformation)` →
  `cpu_current_mhz`, `cpu_max_mhz`, `cpu_limit_mhz`, and
  `cpu_performance_ratio = current / (power-state ceiling or nominal max)`.

These fields are added to `SystemState` and populated by the profiling harness, the
live system sampler, and the lightweight runtime sampler, so every `Observation`
carries them.

**Deterministic confound policy (`measurement_validity.py`; thresholds configurable via
`phase_validity_*`):**
* a sample is *degraded* when `cpu_performance_ratio < phase_validity_cpu_ratio_floor`
  (default `0.60`) — below 60% of the available clock ceiling;
* a degraded state is a **confound** only when persisted for
  `phase_validity_persistence_samples` consecutive samples (default `2`) **and**
  corroborated by battery power or a material drop
  (`phase_validity_cpu_ratio_deviation`, default `25%`) from the best ratio observed in
  the same session;
* `BATTERY` alone does not invalidate a run; a stable low-but-consistent clock is not
  treated as throttling; a single transient sample never invalidates;
* unavailable telemetry is `UNKNOWN` (never assumed healthy); by default it does not
  block, and `phase_validity_unknown_blocks=true` makes it fail closed.

**Safety behavior:** when confounded, `BottleneckDiagnosis.measurement_validity` is
`DEGRADED`, no primary bottleneck is selected, findings are preserved as
observations, escalation is suppressed (`measurement_confounded`), and candidate
generation produces no executable candidate. A confounded measurement therefore never
silently drives an autonomous optimization.

**Evidence:** `inferenceos verify-validity` (real power/clock telemetry plus the
deterministic policy and suppression).

## 6. Resident server lifecycle, rollback, approval, serve

* **Resident lifecycle (Phase 5R):** a real `llama-server` is started with a
  per-process secure API key (temporary key file), a real authenticated readiness
  check, and OS Job-Object containment (`KILL_ON_JOB_CLOSE`) so unload/timeouts cannot
  orphan the process tree. Load/unload/reload are measured; TTFT is streamed, not
  derived.
* **Rollback (Phase 8R):** every applied target-runtime change is recorded with its
  exact prior value and an integrity check; a rollback is a real reload of the prior
  configuration, verified by re-measurement, not by reading back a Python object.
* **Runtime approval (Phase 10R):** disruptive active-runtime changes require
  explicit, contextual approval bound to the proposal + context + process identity.
  Stale/mismatched approval is downgraded to `STALE` and never authorizes an action.
  Timeout never authorizes execution.
* **External recommendations (Phase 11R):** observe → recommend → user decision, with
  no execution stage (see §2.3).
* **Serve (remediation):** `inferenceos serve --deployment <file>` consumes a
  persisted `DeploymentState`, fails closed if it is invalid/unsigned/tampered/stale/
  incompatible, starts the **existing** resident `llama-server` with the retained
  runtime configuration, verifies process identity and the effective configuration,
  serves real local inference, and shuts down cleanly (no orphan). It is not an
  orchestration platform.

## 7. Perplexity proxy — decision

The plan says a cheap proxy metric (e.g. perplexity on a fixed corpus) **may** be used
for early rejection; it is explicitly not mandatory and the plan also warns that final
retention decisions must use task-relevant quality evaluation. Therefore:

* No perplexity/perplexity-proxy subsystem is implemented, and **no code claims one
  exists**.
* Quality validation is the existing deterministic, task-relevant Phase 7R suite
  (non-empty/alphabetic output, exact-reference match, semantic similarity, content
  coverage, length band) with an adaptive tolerance band.
* A cheap perplexity-style early-rejection gate remains a **deferred** capability. If
  implemented later it must use a versioned corpus, deterministic tokenization, real
  token-level likelihoods, an explicit baseline/tolerance, and must act only as an
  early rejection gate—never as the authoritative quality metric.

## 8. Known limitations and evidence boundaries

* **Legacy `verify-phase5`** exercises the older non-resident reasoning path with the
  small local model. Because the 1.5B model does not always emit schema-valid JSON,
  this legacy acceptance is intermittent (it failed on earlier attempts and passed in
  the final remediation regression). It is unrelated to, and superseded by,
  **Phase 5R**, which is the authoritative resident reasoning/deep-diagnosis
  lifecycle.
* **Real vs controlled evidence.** Scenarios that require a specific diagnosis (e.g.
  GPU underutilization, external memory pressure) may construct a controlled
  diagnostic condition using real Phase 3/4 artifacts and/or real controlled helper
  processes. These are labelled in the reports. Real external-process telemetry, real
  process identities, real resident targets, real backend-reported KV allocations, and
  real before/after measurements are used wherever the report says "real".
* **Load-sensitive acceptances.** `verify-phase4`, `verify-phase6`, `verify-phase8R`
  and similar induced-pressure/rollback scenarios depend on the live machine; a first
  run may fall below a threshold. Re-run is the documented remedy; thresholds are never
  weakened to force green.
* **GPU telemetry limits.** Windows GPU Engine counters expose utilization and
  dedicated usage but not total VRAM; total VRAM is reported as `None`. GPU engine
  telemetry and storage-contention signals remain pre-deployment/profiling signals
  rather than part of the lightweight runtime monitor.
* **CPU-only operation.** The system is designed to run CPU-only; GPU acceleration is
  used when Vulkan enumerates a usable device. GPU-offload candidates are only
  generated for diagnosed GPU underutilization/VRAM limits.
* **Measurement validity (B1) is inferred, not a thermal sensor.** Power source is
  directly observed, and CPU performance is directly observed as a clock ratio. No
  direct thermal-throttling sensor is read, and the implementation never labels
  something "thermal throttling" — only a power/performance confound. Whether the
  platform's reported `CurrentMhz` tracks dynamic thermal throttling is
  hardware/driver-dependent; on this Boot Camp image it reports the nominal ceiling
  under the observed conditions, so the battery-corroborated and relative-drop
  criteria are what make the signal actionable. `verify-validity` proves the telemetry
  path and the deterministic policy; it does **not** prove the machine can be
  artificially throttled.
* **Quantized KV cache types** are not modelled analytically (block-scale overhead is
  unaccounted); the estimator assumes f16 and scales linearly for other element
  widths. This is why `kv_cache_type` is not an executable candidate.
* **Reasoning model** defaults to the locally available
  `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`. If absent or if it returns invalid
  output, the deterministic diagnosis is retained.
* **Controlled comparison option.** Attribution supports a controlled observational
  comparison, but the external engine does not automatically construct one; a dominant
  but correlation-only process is at most an informational recommendation.

## 9. Verification commands and what they prove

Run from the repository root with the project venv
(`.\.venv\Scripts\python.exe`). All commands write a JSON report under `reports/` and
exit non-zero unless every check passes.

### Foundation

```powershell
.\.venv\Scripts\python.exe -m inferenceos contracts-check     # every contract round-trips JSON
.\.venv\Scripts\python.exe -m inferenceos reasoning-selftest  # stub ReasoningEngine round-trip
.\.venv\Scripts\python.exe -m mypy                            # type-check src + tests
.\.venv\Scripts\python.exe -m pytest                          # unit + integration tests
```

### Phase acceptances

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase0 --n-predict 32
.\.venv\Scripts\python.exe -m inferenceos verify-phase1
.\.venv\Scripts\python.exe -m inferenceos verify-phase2
.\.venv\Scripts\python.exe -m inferenceos verify-phase3 --repetitions 3
.\.venv\Scripts\python.exe -m inferenceos verify-phase4
.\.venv\Scripts\python.exe -m inferenceos verify-phase5r
.\.venv\Scripts\python.exe -m inferenceos verify-phase6
.\.venv\Scripts\python.exe -m inferenceos verify-phase7r
.\.venv\Scripts\python.exe -m inferenceos verify-phase8r
.\.venv\Scripts\python.exe -m inferenceos verify-phase9r
.\.venv\Scripts\python.exe -m inferenceos verify-phase10r
.\.venv\Scripts\python.exe -m inferenceos verify-phase11r
.\.venv\Scripts\python.exe -m inferenceos verify-remediation
.\.venv\Scripts\python.exe -m inferenceos verify-validity
```

What each proves (abridged):

* **phase0** — real llama.cpp inference with Vulkan offload and observed GPU
  utilization; generated text and device printed.
* **phase1** — real HardwareProfile/SystemState cross-checked against the OS.
* **phase2** — real GGUF parsing (GQA + MHA), real Vulkan inference, **independent**
  architecture-derived KV validation (positive + wrong-source negative) and
  predicted-vs-measured RSS deltas.
* **phase3** — real repeated runs with per-phase timings, variance bands, cold/warm
  semantics, profiler-overhead comparison.
* **phase4** — real diagnosis with the false-positive guard (GPU saturation is not a
  bottleneck) and a controlled real induced-memory-pressure paging diagnosis.
* **phase5r** — real resident target lifecycle, streaming TTFT, unload/reload with a
  new verified identity, entry economics refusing deep diagnosis, adversarial
  containment.
* **phase6** — a real induced-pressure diagnosis yields ranked registered candidates
  with evaluated preconditions and no execution; healthy baseline yields none;
  adversarial reasoning is contained.
* **phase7r** — real resident baseline, retained candidate with real quality +
  restoration, quality regression rejected and restored, unsafe candidate refused.
* **phase8r** — persisted rollback applied to a real target and verified by
  re-measurement, idempotency, stale/incompatible rejection, integrity failure
  refusal, negative knowledge.
* **phase9r** — real fresh autonomous run (no-op and optimization paths) with
  provenance and a deployment state.
* **phase10r** — real monitored resident workload, workload shift detection, approvals
  (reject/approve), in-flight drain, failure containment, stale-approval rejection.
* **phase11r** — real external-process observation with PID-reuse-safe identity,
  diagnosis-aware recommendation, irrelevant-resource non-recommendation, rejection,
  acceptance-without-execution, stale process/provenance, no-bottleneck non-action.
* **verify-remediation** — (R1) the GPU-underutilization action is registered,
  executable, and generates a real increase candidate; no unregistered action is
  referenced in policy code; (R2) peak and final RSS are distinct measurements, with a
  real generation where peak exceeds final; (R4) independent KV derivation matches the
  real backend and the wrong derivation does not; (R5) a valid persisted
  `DeploymentState` starts a real resident server, serves, verifies identity/config,
  shuts down cleanly, and tampered/stale state is rejected.
* **verify-validity** — real Windows power state and CPU clock performance observed
  into `SystemState`; measured observation overhead; the deterministic confound policy
  (normal AC, battery-without-degradation, persisted battery/relative-drop degradation,
  unknown, transient, boundary); and end-to-end proof that a confounded measurement
  yields no primary, no executable candidate and no escalation, while a valid
  measurement is unaffected.

## 10. Human-verifiable command entry points (non-acceptance)

```powershell
.\.venv\Scripts\python.exe -m inferenceos profile            # real hardware + live SystemState
.\.venv\Scripts\python.exe -m inferenceos model-info         # real GGUF ModelProfile + KV estimates
.\.venv\Scripts\python.exe -m inferenceos baseline           # deterministic benchmark suite
.\.venv\Scripts\python.exe -m inferenceos diagnose           # deterministic diagnosis artifact
.\.venv\Scripts\python.exe -m inferenceos deep-diagnose      # escalation + local reasoning
.\.venv\Scripts\python.exe -m inferenceos candidates         # ranked candidates (no execution)
.\.venv\Scripts\python.exe -m inferenceos experiment         # one real controlled experiment
.\.venv\Scripts\python.exe -m inferenceos knowledge --health
.\.venv\Scripts\python.exe -m inferenceos optimize           # full autonomous loop + deployment state
.\.venv\Scripts\python.exe -m inferenceos monitor --duration 120 --approval interactive
.\.venv\Scripts\python.exe -m inferenceos external --duration 10 --approval reject
.\.venv\Scripts\python.exe -m inferenceos serve --deployment reports/deployment_<ts>.json
```

## 11. Environment baseline (this machine)

* Windows Boot Camp, Intel i7-8850H (6C/12T), 16 GB RAM, Apple SSD.
* GPU: **Radeon Pro 560X**, AMD driver 31.0.21921.1000, Vulkan 1.3.260.
* llama.cpp: `tools/llama.cpp` (release **b11139**, `llama-*-bin-win-vulkan-x64`),
  with the VC++ 2015–2022 x64 runtime deployed app-local.
* Models (gitignored): `qwen2.5-1.5b-instruct-q4_k_m.gguf` (Qwen2, GQA) and
  `stories15M-q4_0.gguf` (Llama, MHA).

## 12. Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

No commit or push is performed by the implementation. Models and reports are
gitignored as appropriate.
