# InferenceOS

InferenceOS looks for a measurably faster, safe setup for running a local LLM on your own machine. It profiles the hardware, watches a running `llama.cpp` server, works out what is actually slowing it down, and tries changes that could help. A complete selected setup must pass fresh performance, safety and quality checks; if no improvement is demonstrated, the current setup is retained. Finding the absolute best combination is not required or guaranteed. Applied changes require the existing approval and rollback rules.

## At a glance

- **Hardware-aware.** Reads the CPU through CPUID, GPUs through WMI plus real Vulkan device enumeration and CUDA/ROCm runtime checks, memory through `GlobalMemoryStatusEx`, and live load through Windows performance counters. It also measures memory bandwidth and can turn on the Vulkan backend's per-operation timers to see where GPU time goes.
- **Model-aware.** A dependency-free GGUF reader gives layer counts, KV heads and tensor sizes. A GQA-aware KV-cache estimator is checked against what the engine actually allocates (within 5%).
- **Placement search.** Searches GPU layer offload and per-tensor CPU/GPU placement (`--override-tensor`), such as keeping MoE expert tensors or the output layer on the CPU, and confirms each placement from the engine's own load log.
- **Fair measurement.** Current and candidate setups are measured from fresh loads in alternating blocks (A B, B A, …), so steady drift of the machine cancels out. Runs confounded by battery power or clock throttling are rejected instead of being read as evidence.
- **Safe by construction.** The engine runs inside a Windows Job Object so nothing is orphaned, process identity is checked by PID and creation time, a crashed server is recovered at most once, and every applied change is recorded with its exact rollback.
- **Tested.** More than 3,000 tests and strict `mypy` type checking.

## Quick start

Windows, Python 3.11+, and a llama.cpp build with the Vulkan backend (see §11 for the exact environment).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Look for a faster setup, confirm the complete result, approve and save it.
.\.venv\Scripts\python.exe -m inferenceos tune --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf --context 4096

# Run the model with its saved setup until Ctrl+C.
.\.venv\Scripts\python.exe -m inferenceos start --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf
```

## How this README is organized

The current standalone `tune` inventory is distinct from the older runtime card
registry described in section 4:

| Tuning path | Unbounded | Practical | Evidence status |
| --- | --- | --- | --- |
| GPU layer count | Yes | Yes | Confirmed gains on real models |
| CPU generation threads | Yes | Yes | Confirmed gains on some models |
| Prompt batch size | Yes | With a long workload | Confirmed long-prompt gains in Unbounded |
| Prompt-reading threads | Yes | No | Implemented; no retained gain established |
| Flash attention | Yes | No | Implemented; measured trade-offs, no automatic retained gain established |
| Build-specific engine switches | Where applicable | No | Tested switches on this laptop were slower |
| Individual tensor placement | Where applicable | No | Activation demonstrated; no retained tune gain established; grouped hybrid placement remains disabled |
| KV-cache memory format | Where applicable | No | Activation demonstrated; useful tune gain still needs validation |

These paths share dispatch and explicit dependency rechecks, followed by confirmation
of the complete configuration. The search is bounded and can miss jointly beneficial
combinations. Practical still searches only its three declared categories; historical
ordering proposals remain shadowed by default. Broader Practical admission needs
development-time evidence for the model/hardware family.

Prompt reuse and [lookup decoding](docs/lookup-decoding.md) are development capabilities,
not automatic tune paths. Reuse studies found a Gemma wrong answer on one native prompt
pair. The accepted focused job129 study found that failure absent with verified full
cache for that pair and setup; this does not establish general reuse safety or justify
a default change. Lookup decoding now has opt-in backend support, a matched evaluation,
simulation, offline audit and a prepared study; no lookup hardware gain is established.
No helper model or persistent lookup cache is used.

Remaining work includes memory-residency/loading tuning, broader drafting methods,
operating-system interventions with crash-safe undo, thermal/battery goals, additional
engine connectors, and stronger measured path-interaction/admission evidence. Existing
software tests and replay equality establish regression coverage, not universal
performance, factual answer correctness or global optimality.

The rest of this document describes in detail what is implemented, how to reproduce each acceptance check, and where evidence comes from a real run versus a controlled one. The original specification is archived at [`docs/archive/IMPLEMENTATION_PLAN.md`](docs/archive/IMPLEMENTATION_PLAN.md), and the development log at [`docs/archive/project-scope-and-roadmap-2026-10-08.md`](docs/archive/project-scope-and-roadmap-2026-10-08.md).

> **Completion status (implementation phases 0–11, the R remediation rounds, the
> final cross-phase remediation round, B1 measurement validity, B2 resident-server
> bounded recovery, H2 log/report retention & disk protection, H4 optimizer
> self-overhead / net-benefit accounting, and the final production-hardening pass
> H3/H1/B3/H6): complete.**
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

## 3. Phase history

The system was built in phases 0 to 11R, each proven by its own `verify-phaseN`
acceptance command on the dev laptop. Those commands were removed in the Sep 2026
cleanup (git history keeps them); the phase documents and the original plan are in
[`docs/archive/`](docs/archive/). The engine code they exercised is still in use.
The remaining acceptance commands are listed in §9.

## 4. Optimization cards and supported changes

Every kind of optimization is described by one **card** in
`src/inferenceos/cards/builtin/` (one file per card). The rest of the system reads
cards only: the action registry is built from the cards, the candidate generator asks
a card whether it applies and which value to try, the executors apply and undo a card's
change through its per-program setting, and the audit, quality and safety code read the
card's fields. Card files are found automatically, so adding a card needs no other code
change (proven by `tests/test_cards.py`).

A card states, in plain words: what it changes, where it acts, which programs it works
with and how each sets it, when it may be tested, whether it needs a restart, when it
applies, the values to try, what it should improve and might worsen, which cards it
works with or clashes with, its quality risk, how it is undone, what needs approval,
whether it needs admin rights, and that it is context safe.

**Automatic context-safety check.** A card that is not context safe, or whose setting
or parameter is the context size, is refused when it is registered. Every applied
change is also checked: if the configuration after the change has a different context
size, the change is refused, and the resident target refuses any configuration update
that would change the running model's context size.

### Applied cards (target-model runtime configuration only)

| Card | Surface | Direction / trigger |
|---|---|---|
| `runtime.enable_mmap` | memory management | enable mmap under memory pressure/paging |
| `runtime.reduce_gpu_offload_layers` | GPU offload | reduce offload under VRAM limits |
| `runtime.increase_gpu_offload_layers` | GPU offload | increase offload under diagnosed GPU underutilization |
| `runtime.reduce_threads` | runtime params | right-size threads under CPU contention |

Every executable action flows through the full lifecycle: diagnosis → applicability →
candidate → executor apply → real measurement → quality validation → keep/revert →
knowledge/rollback provenance (Phase 6/7/8/9 machinery).

### Suggest-only cards (never applied by InferenceOS)

| Card | Why |
|---|---|
| `external.request_close` | belongs to the user; recommended only |
| `external.reduce_priority` | belongs to the user; recommended only |
| `external.suspend_process` | belongs to the user; recommended only |
| `system.increase_pagefile` | system-wide, requires elevation; never automatic |

There is no card for context size (the optimizer never changes or suggests context
changes) and none for re-compressing the model (quantization is never changed). The
old `runtime.right_size_context` and `model.requantize` entries were deleted in step 3;
the requantize definition survives only as a test fixture proving the policy refuses it.

### Deferred / unsupported candidate surfaces in the older runtime loop

This registry is separate from standalone `tune`. Its missing batch, attention and
KV-cache cards below do not mean the corresponding eight-path tune implementation is
missing; the current mode inventory and evidence limits are listed above.

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

### 5.4 Resident-server crash detection & bounded recovery (B2)

A resident `llama-server` can die (or a PID can be reused by an unrelated process)
mid-operation. B2 hardens the **existing** resident lifecycle; it is not a second
server architecture and not a high-availability/failover system.

* **Identity.** The workload captures the target's `ProcessIdentity` (PID **and** OS
  creation time, plus name/executable/command hash). Identity is read against the live
  OS process through the resident adapter (`verify_process_identity`); a PID alone is
  never accepted, so a new process that later occupies the same PID is treated as a
  loss, not as the original target.
* **Detection.** Target loss is detected when the adapter reports the process gone or
  when the captured identity no longer matches. A generation that fails because the
  server died flows into the same mechanism — there is no separate implicit reload path.
* **Bounded recovery.** The authority is `ResidentActiveWorkload`: under one lock it
  performs **at most one** controlled recovery per failure event (`MAX_RECOVERY_ATTEMPTS
  = 1`). It cleans the failed target (Job Object handle + temporary API-key file, no
  orphan), reloads via the existing resident adapter/configuration/readiness/API-key
  machinery, then verifies **process identity, readiness, effective configuration,
  model and backend** before updating the authoritative target identity. A replacement
  that serves a real request earns the right to recover from a later, independent
  failure; a replacement that dies before serving anything becomes unrecoverable
  instead of looping.
* **Unrecoverable handling.** If the single attempt fails any verification, the target
  is marked `TARGET_UNRECOVERABLE`, the failed/new process is removed, and callers get
  a deterministic failure. There is no silent fallback to another backend/model/config
  and no unbounded restart loop.
* **Events & evidence.** `ResidentTargetRecovery` records `TARGET_LOST`,
  `TARGET_RECOVERED`/`TARGET_UNRECOVERABLE` transitions with old/new PID+identity, the
  loss reason, attempt count, each verification result, cleanup and orphan status. The
  runtime monitor records these in `RuntimeMonitorRun.target_recoveries` and adds
  `target_lost`/`target_recovered`/`target_unrecoverable` to its state timeline; healthy
  operation adds no continuous polling beyond the existing monitor samples.

**Evidence:** `inferenceos verify-resident-recovery` — starts a real resident server,
serves real inference, kills the actual server process, detects the loss, performs
exactly one controlled recovery, verifies the replacement's identity/readiness/
configuration/model/backend, serves real inference again, proves `old identity != new
identity`, proves a controlled identity mismatch is treated as loss, and proves an
injected reload failure reaches an explicit unrecoverable state without a second
attempt (and with no orphan).

### 5.5 Log/report retention & disk protection (H2)

Unattended operation must not eventually exhaust the disk through InferenceOS's own
generated artifacts. H2 adds a small, explicit retention policy; it does not redesign
logging and is not a generic storage-management subsystem.

**Artifact classes**

* **Operational logs** (`logs/*.log`): bounded by the standard
  `logging.handlers.RotatingFileHandler` — `log_max_bytes` per file and
  `log_backup_count` rotations, newest preserved. A logging setup failure degrades to
  stderr-only and never prevents a command running.
* **Generated reports** (allowlisted, timestamped evidence directly under `reports/`):
  disposable, removed oldest-first. Examples: `phase*_verification_*.json`,
  `phase*_run_*.json`, `phase*_monitor_*.json`, `baseline_*.json`, `diagnosis_*.json`,
  `candidates_*.json`, `experiment[_audit]_*.json`, `hardware_profile_*.json`,
  `deep_diagnosis_*.json`, `*_verification_*.json`, `serve_*.json`.
* **Durable state — never automatically deleted:** every SQLite database under `data/`
  (knowledge, rollback/audit history), persisted deployment state (`deployment_*.json`,
  `deployment_cli_check.json`), configuration, source, and models. Any file whose name
  is not on the generated-report allowlist is treated as durable.
* **Temporary files:** atomic-write temporaries (`.*.tmp`) are never evidence; only
  stale ones (older than 1 hour) are removed.

**Default limits (override with `INFERENCEOS_*` env vars or `.env`)**

| Setting | Default | Meaning |
|---|---|---|
| `retention_enabled` | `true` | master switch |
| `retention_report_max_files` | `1000` | max disposable report files |
| `retention_report_max_bytes` | `512 MiB` | max disposable report bytes |
| `retention_min_free_bytes` | `1 GiB` | free-space floor (cleanup below) |
| `retention_max_deletions_per_pass` | `500` | bounded deletions per sweep |
| `log_max_bytes` | `8 MiB` | per-log-file rotation size |
| `log_backup_count` | `3` | rotated backups per log name |

*Rationale:* the repository accumulated 459 reports / ~43 MB and ~1.9 MB of logs in
about two days of heavy testing (~215 files and ~21.5 MB per day). 1000 files covers
~4–5 days at that heaviest observed cadence while retaining the most recent ~1000
evidence artifacts; 512 MiB covers several weeks; the 1 GiB free-space floor protects
the drive with margin on this machine. Logs are bounded per phase name.

**Precedence and trigger.** Cleanup occurs when **any** limit is exceeded (file count
**OR** byte budget **OR** free space below the floor). Oldest disposable artifacts are
removed first until every applicable limit is satisfied or the per-pass deletion bound
is reached. Cleanup runs once per process start (`run_startup_retention` in `main()`,
non-fatal — a permissions error or individual deletion failure is recorded and never
prevents startup), on log rotation, and opportunistically before persisting a
disposable report when space is low. There is no background thread and no per-request
directory scan.

**Disk-pressure behavior.** Free space below the floor is `LOW`: a retention pass runs
and the space is re-checked. Below half the floor it is `CRITICAL`: a disposable report
write is refused and returned as `degraded` (`persisted=false`); the caller never claims
it was persisted, and the core runtime continues. Unknown/unmeasurable space never
blocks.

**Atomic writes.** Disposable evidence is written with `atomic_write_text` (temporary
file → flush/fsync → atomic `os.replace`), so a crash leaves either the previous state
or nothing at the final path; a partial temporary can never be mistaken for a completed
report. The H2 acceptance report uses this path.

**Evidence:** `inferenceos verify-storage` — controlled count/byte retention (oldest
removed, newest and durable/unknown files preserved), malformed-file preservation, path
safety (outside-directory refusal and symlink non-deletion), disk-pressure cleanup +
refused/degraded write + recovery, atomic success/interruption, bounded log rotation,
  non-fatal startup cleanup, a real Windows scan of the actual reports/logs directories
  (only allowlisted files would ever be removed; durable `data/` preserved) with a small
  real inference afterward, and a measured retention-scan latency well under budget.
* **verify-accounting (H4)** — controlled beneficial / overhead-dominant /
  insufficient-baseline / missing-overhead / material-distortion / incomparable-metric
  net-benefit assessments, proof that nested timers are not double-counted, a real
  resident-workload run through the runtime monitor that produces baseline target
  metrics, measured optimizer overhead, post-intervention metrics, measurement
  distortion and a final net-benefit status embedded in the run evidence, and the
  measured (sub-millisecond) overhead of the accounting itself.
* **verify-hardening (final pass)** — (H3) stale InferenceOS-owned key sweep plus the
  real key lifecycle (present during the server lifetime, removed on stop, no leftover);
  (H1) real `knowledge.db`/`external_knowledge.db` read-only inspection (journal mode,
  row counts) and a controlled close/reopen persistence round-trip proving durable audit
  events are not auto-deleted; (B3) controlled deterministic factual/arithmetic/
  structured quality checks and a real model output demonstrating preservation and
  loss detection; (H6) optimizer self-PID exclusion by the engine and the real process
  observer, with a legitimate external process still selectable.

### 5.6 Optimizer self-overhead & net-benefit accounting (H4)

H4 consolidates **already-measured** evidence into one conservative answer to "did the
optimization benefit outweigh the optimizer's own cost?". It adds no timing to the
inference hot path and is a *derived* accounting layer over existing `RuntimeOverhead`,
`LoopOverhead`, `ResourceImpact` and experiment/baseline metric maps.

**Boundary.** *Optimizer overhead* = time InferenceOS spends observing/measuring,
diagnosing, deep-reasoning, generating/ranking candidates, knowledge-querying,
checkpointing, validating and persisting **because of** optimization. It never includes
target inference time. *Target benefit* = improvement in a target-outcome metric.
*Measurement distortion* = whether observing/optimizing perturbed the target; reported
separately and never netted out of benefit.

**Accounting model (no double counting).** `OverheadAttribution.inclusive_overhead_s`
is authoritative (the runtime `optimizer_overhead_s` / loop exclusive-phase sum) and is
**not** recomputed as the sum of components. Components are recorded for attribution
only; if their sum differs from the inclusive value the record is flagged `nested=True`
and explained. Pause/experiment/resume are target-interruption costs and are excluded
from optimizer overhead. Target workload time enters only as a denominator
(`overhead_fraction = optimizer_overhead_s / baseline_target_workload_s`), never as
overhead itself.

**Comparable metrics only.** Auto-selection order over target-outcome metrics is
throughput → latency/TTFT → target memory (`decode_tokens_per_sec`,
`prompt_tokens_per_sec`, `total_latency_s`, `ttft_s`, `peak_rss_bytes`,
`final_rss_bytes`). Host-state signals (`memory_pressure`, `available_ram_bytes`, hard
page faults) are deliberately **not** treated as target benefits. A memory optimization
is represented in bytes, not forced into a throughput formula. Unknown/incomparable
metrics yield a structured insufficient-evidence result; no scalar quality score is
invented.

**Net-benefit status.** `benefit_ratio` (dimensionless, positive = improvement) and
`overhead_fraction` (dimensionless) are the only quantities compared — heterogeneous
units are never subtracted (`net_value` stays `None`). Thresholds reuse existing policy
where possible: `benefit_min_relative` = existing Phase 7 `min_effect` (5%);
`overhead_fraction_max` = 20% (smallest defensible bound consistent with the existing
Phase 6/10 overhead budgets); `distortion_tolerance` = existing runtime deviation
threshold (25%). Status:
* `BENEFICIAL` — benefit ≥ 5% **and** overhead fraction ≤ 20%;
* `OVERHEAD_DOMINANT` — overhead fraction > 20% and not outweighed by benefit;
* `NEUTRAL` — comparable, within bounds, but below the beneficial criteria;
* `INSUFFICIENT_EVIDENCE` — no valid baseline, no overhead measurement, incomparable
  units, or material measurement distortion. Uncertainty is never converted into a
  positive result.

**Enforcement vs reporting.** H4 is **reporting/qualification-only** (`enforceable =
false`): it does not veto optimizations. Existing overhead budgets/gates are unchanged
and are not weakened. The assessment is attached to the existing runtime monitor run
(`RuntimeMonitorRun.net_benefit`) and the autonomous loop run
(`OptimizationRun.net_benefit`) as machine-readable evidence, so an auditor can see the
baseline metrics, optimization metrics, optimizer overhead, measurement distortion,
benefit relationship and net-benefit status together.

**Limitations.** The runtime `baseline_target_workload_s` is the observed
target-active window (`monitoring_wall_s - optimizer_overhead_s`); the loop uses
`wall_clock - optimizer_overhead`. Neither is a separately instrumented "would-have-run"
counterfactual. Where the architecture cannot separate two components, the record says
so (`nested`, `notes`) rather than inventing precision. H4 does not prove that every
optimization is profitable.

**Evidence:** `inferenceos verify-accounting` — controlled beneficial,
overhead-dominant, insufficient-baseline, missing-overhead, material-distortion,
incomparable-metric and nested-timing (no double count) cases; a real resident-workload
scenario through the runtime monitor with a real intervention producing baseline target
metrics, measured optimizer overhead, post-intervention measurement, distortion and a
final net-benefit assessment; and the measured overhead of H4's own accounting.

### 5.7 Final production hardening (H3 / H1 / B3 / H6)

Assessed against the actual repository; only genuine gaps were implemented.

**H3 — temporary API-key hygiene (implemented).** The resident server uses a
`secrets.token_urlsafe(32)` key written to a uniquely-named `inferenceos-key-*.txt`
system-temporary file, passed to `llama-server` via `--api-key-file` (never on the
command line) and never logged. The file is deleted on normal stop, startup failure and
B2 recovery/cleanup. A bounded sweep at each server start additionally removes
InferenceOS-owned key files older than 24 h (exact `inferenceos-key-*.txt` match,
direct regular non-symlink files only); fresh keys and unrelated temporary files are
never touched. Since the 15 Oct review (R5), each key's name records the owning
InferenceOS process (id and start time), and a key whose owner is still running is
never removed, however old: `start` can run for days and clients read that file to
connect. If the owner's start time cannot be read when the key is made, the name
records the process id with an unknown start time, and the key is kept while any
process has that id. Only keys whose owner is gone are removed by age; an owner that
cannot be checked keeps its key. Known limit: old-style keys without an owner (made
before this change) are still removed by age alone, so a server started by an older
version and left running for over 24 hours can lose its key file. A key file that
cannot be deleted at stop (for example another program holds it open) is reported by
`start` ("clean shutdown: False" plus a warning naming the file) and retried at the
next cleanup. **ACL hardening was assessed as low
value** on the target (single-user machine; the file already lives in the per-user
`%TEMP%`) and deliberately not implemented to avoid a fragile custom ACL subsystem;
this is a documented limitation, not an unaddressed exposure.

**H1 — SQLite WAL / retention (assessed; no change).** InferenceOS has two SQLite
stores: the durable knowledge/audit store (`data/knowledge.db`: optimization records,
rollback records, append-only events) and the external-recommendation knowledge store
(`data/external_knowledge.db`). Both are single-writer with short transactions and no
concurrent readers, so **WAL provides no concrete benefit** and would add `-wal`/`-shm`
sidecars and checkpoint concerns. The `phase8_retention_days` field defaults to `0`
(keep forever) and is intentionally not auto-enforced: the tables are exactly the
durable audit/revert history that must not be automatically deleted. No retention was
added. `verify-hardening` inspects the real stores read-only (journal mode, row counts,
size) and round-trips a record to prove persistence.

**B3 — deterministic task-relevant quality (implemented).** The existing Phase 7R suite
gains two hard, deterministic cases: `required_content` (exact factual/arithmetic
content the baseline demonstrated must be preserved; comparison is case/whitespace
normalized so formatting variation is accepted) and `required_fields` (required JSON
keys for structured workloads). A phrase is only required when the baseline itself
produced it, so a model limitation is never a candidate failure — but a candidate that
loses demonstrated correctness is a hard `QUALITY_REGRESSION` (`validated=False`,
`unacceptable_output=True`), so an invalid candidate cannot be retained merely because
its performance improved. `QUALITY_LONG_WORKLOAD` declares the `ZEPHYR-9137` identifier
it already contained.

**H6 — optimizer self-target exclusion (implemented).** B2 handled identity continuity;
the remaining gap was that the production external engine did not pass its own PID, so
the optimizer's own process could appear as a candidate. `ExternalRecommendationEngine`
now defaults `own_pid` to `os.getpid()`, and `WindowsProcessObserver` excludes that
process (unless it is explicitly the monitored target). **Children are not excluded** —
the resident server is a legitimate child of the optimizer and often the monitored
target. Real evidence: the optimizer PID is absent from the observed process table, a
legitimate external process remains selectable, and the target is preserved.

**H5 is already sufficient** and required no change.

**Evidence:** `inferenceos verify-hardening` — controlled stale-key sweep, real
key-file lifecycle (present during server lifetime, removed on stop, no leftover),
real database inspection + persistence round-trip, controlled and real model
deterministic quality checks (preservation and loss detection on real output), and
optimizer self-PID exclusion (engine + real observer).

### 5.8 User optimization preference, adaptive economics & long-search resume

The user decides **how much optimization effort is acceptable**; InferenceOS still
decides **which configurations are technically safe and valid**. The preference changes
only economics/search effort — never OOM/VRAM/quality/health/checkpoint/rollback/restore
safety.

* **Program-level preference** (`optimization_mode`): `balanced` (default),
  `performance_first`, `minimize_optimization_cost`, or `custom`. Persisted through the
  existing `Settings` surface (`INFERENCEOS_OPTIMIZATION_MODE` / `.env`; CLI flags
  override for one invocation). Recorded in `RuntimeMonitorRun.optimization_preference`
  and in each `GpuOffloadSearchResult.optimization_mode`.
* **Adaptive urgency** (`optimization/`): the target's performance state
  (NORMAL/DEGRADED/SEVERELY_DEGRADED/UNKNOWN) is classified from the existing
  baseline/workload metrics (decode/prompt throughput, latency, TTFT) against the
  established baseline and configurable thresholds. Degraded performance lowers the
  required worthwhile improvement and expands the bounded search budget; insufficient
  evidence yields `UNKNOWN` and the policy baseline is applied without guessing.
* **Worthwhile-improvement economics**: the runtime economics gate now compares the
  candidate's *empirically measured relative performance improvement* against the
  policy's minimum for the current state, separate from the technical candidate decision
  and every approval/safety classification.
* **Bounded search budget**: mode/state select the maximum search time (bounded by
  `optimization_max_search_time_hard_cap_s`; there is no unlimited search). Estimated vs
  actual search time, configured maximum, candidates evaluated, retries, mode and state
  are recorded on the search result.
* **Long-search resume fix**: an explicit
  `optimization_search_active → resume_requested → resume_verifying` lifecycle replaces
  the fixed 45 s check. The verification timeout is a bounded base allowance plus a capped
  share of the *actual* search duration, always within a hard maximum, so a 4–5 minute
  search no longer fails merely because it exceeded 45 s. Genuine failures (target gone,
  unrecoverable, identity mismatch, never ready within the window) still reach the
  existing `safety_terminal`.

**Evidence:** `inferenceos optimization-policy` prints the resolved policy;
`tests/test_optimization_policy.py` covers preference/urgency/budget/resume/integration;
and `artifacts/ab_effectiveness/gpu_optimization_policy_<ts>.md|json` records real
Windows runs where 124.6–175.9 s searches resumed and verified with no `safety_terminal`.

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
* **Resident crash recovery (B2):** a lost resident target is detected (PID + creation
  time, not PID alone), recorded, and recovered with **at most one** controlled reload
  through the existing resident architecture; the replacement is accepted only after
  identity/readiness/configuration/model/backend verification, otherwise the operation
  fails closed as unrecoverable. See §5.4.
* **Log/report retention (H2):** disposable generated reports and operational logs are
  bounded by explicit count/byte/rotation limits and cleaned oldest-first at startup
  (non-fatal), with disk-pressure protection that refuses a disposable write rather
  than crash the runtime. Durable SQLite/deployment/audit state is never auto-deleted.
  See §5.5.
* **Optimizer overhead accounting (H4):** a derived, conservative assessment of target
  benefit vs the optimizer's own measured overhead, with measurement distortion kept
  separate; attached to existing runtime/loop run evidence. Reporting-only; existing
  overhead budgets are unchanged. See §5.6.

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

* **Real vs controlled evidence.** Scenarios that require a specific diagnosis (e.g.
  GPU underutilization, external memory pressure) may construct a controlled
  diagnostic condition using real Phase 3/4 artifacts and/or real controlled helper
  processes. These are labelled in the reports. Real external-process telemetry, real
  process identities, real resident targets, real backend-reported KV allocations, and
  real before/after measurements are used wherever the report says "real".
* **Load-sensitive acceptances.** Induced-pressure and recovery scenarios (for example
  `verify-resident-recovery`) depend on the live machine; a first
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

### Acceptance commands

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-remediation
.\.venv\Scripts\python.exe -m inferenceos verify-validity
.\.venv\Scripts\python.exe -m inferenceos verify-resident-recovery
.\.venv\Scripts\python.exe -m inferenceos verify-storage
.\.venv\Scripts\python.exe -m inferenceos verify-accounting
.\.venv\Scripts\python.exe -m inferenceos verify-hardening
```

What each proves (abridged):

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
* **verify-resident-recovery (B2)** — a real resident server is crashed mid-run; the
  loss is detected and recovered exactly once through the existing architecture; the
  replacement's identity/readiness/configuration/model/backend are verified and real
  inference resumes; a controlled identity mismatch is treated as target loss; and an
  injected reload failure reaches an explicit unrecoverable state with no second
  attempt and no orphan.
* **verify-storage (H2)** — count/byte retention removes oldest disposable reports while
  preserving newest, durable and unknown files; malformed artifacts are never deleted;
  paths outside the owned reports directory and symlink escapes are refused; disk
  pressure triggers cleanup and a still-critical disposable write is refused as
  `degraded` (runtime unaffected) then succeeds once space is restored; atomic writes
  leave no partial artifact; file logs rotate within bounds; startup cleanup is
  non-fatal; the real Windows reports/logs directories are scanned with only allowlisted
  files ever removable and real inference succeeding afterward; retention latency is
  measured.

## 10. Human-verifiable command entry points (non-acceptance)

### Everyday commands

```powershell
# Find the best setup for a model, confirm it, answer yes/no for each change, save it.
.\.venv\Scripts\python.exe -m inferenceos tune --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf --context 4096
# Run the model with its saved setup (this computer only) until Ctrl+C.
.\.venv\Scripts\python.exe -m inferenceos start --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf
```

* `tune` checks the machine with no model loaded, starts from your saved setup (or as
  many graphics card layers as the model file says fit), asks you to leave the computer
  alone, searches graphics card layers then processor threads within the effort mode's
  time limit, confirms the result against the starting setup by alternating runs, and
  asks about each change. Saved setups are kept per model, context size and power
  source (plugged in / battery). The context size is never changed.
* `start` does a quick check first. If the graphics driver, llama.cpp version, power
  plan or hardware changed since tuning, it says so and starts anyway. A replaced model
  file is refused. A new password is made at every start; only the location of its
  owner-only file is printed, and the file is deleted when the model stops.

### Advanced single-step commands

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
.\.venv\Scripts\python.exe -m inferenceos optimization-policy
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
