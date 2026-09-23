Implementation Plan — Autonomous Adaptive Local LLM Optimization System

Target environment (v1): Windows 11 via Boot Camp, Intel MacBook Pro (i7-8850H), AMD Radeon discrete GPU + Intel integrated GPU. Standard-user privileges by default; explicit elevation only per-operation. Backend: llama.cpp, GPU acceleration via discovered/available Windows backend (expected Vulkan; CUDA unavailable on this hardware). Reasoning: deterministic engine + local LLM behind an abstract reasoning interface, invoked selectively via a non-resident deep-diagnosis mode.

This plan is approved at the architecture level. Platform, backend, reasoning architecture, and macro phase structure (0–11) are settled and must not be reopened except for a genuine implementation-blocking contradiction discovered during build.

Operating Instructions for the Implementation Agent

These rules govern how you (the implementation model) execute this plan. They apply to every phase without exception.

Work autonomously within a phase

You must implement, run, test, diagnose failures, fix problems, and iterate without stopping, for all of the following (this list is illustrative, not exhaustive — the principle is: normal implementation friction is yours to resolve):

a test fails

a build fails

an API/library behaves unexpectedly

you hit an implementation difficulty

you need to read documentation or source to understand a dependency

you discover an earlier assumption in this plan was wrong

you need to refactor existing code, including code from an earlier phase

an experiment fails or produces an unexpected result

performance is initially worse than expected

you need several iterations to get something working

None of the above are blockers. Keep working through them.

Stop only for one of two reasons

Case 1 — Phase complete. All phase scope, the phase's real functional acceptance criteria, its human-verifiable evidence requirement, and the regression requirement (below) are satisfied.

Case 2 — Genuine blocker. Progress cannot continue without human input or an external condition you cannot safely resolve yourself. Examples: a required hardware capability is genuinely unavailable on this machine; a required external resource (license, credential, physical access) cannot be obtained or changed by you; the plan intentionally leaves an open choice that requires a human decision; continuing would require an unsafe or destructive action needing explicit authorization.

"This is hard," "I'm not sure yet," "the test is currently failing," or "this will take a while" are explicitly not blockers.

When you hit a genuine blocker, stop and report: (1) exactly what you attempted, (2) what failed, (3) evidence proving it's a real blocker, (4) why you can't resolve it yourself, (5) exactly what decision or input you need. Otherwise, keep working.

Phase completion standard

A phase is complete only when all three hold:

Real functional acceptance — automated tests that exercise the actual implementation against the real subsystem, not a mock or placeholder. Synthetic/fixture-based unit tests are appropriate and expected for deterministic logic (e.g., diagnosis rules against controlled inputs), but wherever a phase depends on a real subsystem (real llama.cpp inference, real Vulkan telemetry, real GGUF parsing, real memory measurement, real model load/unload, real reasoning-model invocation, real candidate application, real before/after measurement, real quality evaluation, real rollback, real audit reconstruction), that real path must be exercised and must pass. A test that can pass with the underlying functionality missing or mocked out does not count toward phase completion.

Human-verifiable evidence — a reproducible way for a human to independently confirm the phase works: a UI to inspect if one exists at that point, or otherwise an executable verification command/report/artifact whose output clearly shows the real result (not a log line claiming success). You must run this verification yourself before declaring the phase complete, and the plan for how to reproduce it must be stated in your completion report.

No regression — functionality delivered by prior phases still works, demonstrated by re-running their real functional acceptance checks (not re-deriving them from scratch).

Do not declare a phase complete on the basis of passing unit tests alone.

The anti-triviality principle applies globally, every phase

The optimizer must never equate resource consumption with a bottleneck, and must never manufacture an intervention merely because the system is running. The causal chain, everywhere it appears in this plan (explicitly in Phases 4, 6, 9, but the principle governs Phases 5–11 as a whole):

OBSERVE
  → determine whether a meaningful bottleneck exists
  → DIAGNOSE / root-cause reasoning
  → understand relevant interactions
  → identify the relevant optimization surface
  → generate candidates
  → estimate benefit / cost / risk
  → discard low-value or non-causal interventions
  → experiment / recommend / apply
  → measure
  → quality-check
  → keep / revert
  → record
  → continue

"No meaningful optimization opportunity — retain current configuration" is a valid, successful, first-class terminal outcome at every phase where a decision is made. A phase's acceptance criteria are not satisfied by a system that always finds something to change; they require that it can also correctly find nothing.

Phase 0 — Contracts & Environment

Objective: Establish the vocabulary and interfaces every later phase builds against, before writing optimization logic.

Scope:

Define core data contracts (as code) HardwareProfile, ModelProfile, WorkloadProfile, SystemState, Observation, Bottleneck, Candidate, ExperimentResult, ReasoningRequest, ReasoningResponse.

Define the abstract ReasoningEngine interface (deterministic and LLM implementations both conform from day one).

Define the BackendAdapter interface (llama.cpp is the only implementation; the optimizer never calls it directly).

Acquire or build llama.cpp with the Vulkan backend enabled, verified against the AMD Radeon device.

Establish repo structure, logging, config, and the audit-record schema (per Phase 8/9 needs).

Real functional acceptance:

The actual built llama.cpp binary runs a real inference request against a real model file on this machine, with Vulkan GPU offload enabled, not a stub.

All contracts type-check and a stub ReasoningEngine round-trips — this stub use is acceptable here specifically because Phase 0's job is the interface, not the reasoning logic itself.

Human-verifiable evidence: A command that loads a real model, runs one prompt, and prints (a) the generated text, (b) confirmation of which backend/device executed it, (c) GPU utilization observed during the call (e.g. via a Windows GPU monitoring tool or API, captured and shown, not just asserted).

Failure/rollback condition: If Vulkan cannot be made to offload to the AMD GPU after genuine troubleshooting, this is a Case 2 blocker — report exactly what was attempted, the errors encountered, and stop for a human decision on whether to fall back to CPU-only rather than silently redefining the project's scope.

Phase 1 — Hardware & System Characterization

Objective: Deterministically populate HardwareProfile and live SystemState, Windows-native.

Scope:

CPU: core count, cache sizes, instruction-set capabilities — detected dynamically (never assumed) via a real CPUID-equivalent probe.

GPU: enumerate available backends by actually querying them (real Vulkan device enumeration; real check for ROCm/HIP and CUDA presence) — not asserted from known hardware.

RAM: total, available, current pressure via the real Win32 GlobalMemoryStatusEx (or equivalent) call.

Storage: real detection of which physical drive the model file resides on (internal SSD vs external HDD), with the HDD case explicitly flagged.

Live system state: real per-process RAM/CPU, hard page fault rate, GPU engine utilization, read from actual OS counters (Windows has no swap/swappiness equivalent — committed-vs-available memory and hard page fault counters are the real pressure signal here, not a Linux analogy).

Real functional acceptance:

Profiler run against the actual dev machine produces a HardwareProfile that correctly reports: CPU instruction sets actually present, Vulkan available, CUDA absent, and RAM/storage figures matching Windows Task Manager's own numbers within reasonable tolerance.

Repeated runs on an idle real system are stable within an agreed variance band.

Human-verifiable evidence: A command that prints the full real HardwareProfile/SystemState for the current machine, side-by-side comparable against Task Manager / Device Manager so a human can visually cross-check CPU, GPU, RAM, and storage entries.

Phase 2 — Model & Backend Abstraction

Objective: Populate ModelProfile and wrap llama.cpp behind BackendAdapter.

Scope:

Parse real GGUF metadata: parameter count, architecture, quantization format, context length, layer count, attention-head count, KV-head count, head dimension.

Compute analytical memory estimates: weight size from real file size; KV-cache size derived from the model's actual attention architecture read from metadata — the formula must not assume standard MHA. For GQA/MQA models the KV-head count (not the attention-head count) drives the calculation. This estimator needs dedicated tests against at least one real GQA model and one real standard-MHA model, since causal memory diagnosis in Phase 4 onward depends on this being right, not merely plausible.

BackendAdapter exposes real load/unload/generate/set-runtime-params/telemetry read-back against the real llama.cpp binary from Phase 0.

Real functional acceptance:

Load at least two real GGUF models with different quantization levels and, ideally, different attention architectures (standard MHA and GQA if both are obtainable) through the adapter; generate real output from each.

For each, compare the analytical KV-cache estimate against the actually measured RSS delta when varying real context length on an otherwise-idle system, and show the estimate is within a stated tolerance — this is a real-measurement comparison, not a unit test against a hand-computed expected value.

Human-verifiable evidence: A report (table or printed output) showing, per real model tested: declared architecture/KV-head count from GGUF, predicted KV-cache size at 2–3 context lengths, and the actually measured RSS delta at those same context lengths, so a human can see the estimator's real accuracy directly.

Phase 3 — Baseline Profiling & Telemetry Harness

Objective: Build the controlled measurement harness that all later phases' comparisons depend on.

Scope:

Fixed prompt sets (short/long, code/prose), fixed seeds.

Cold-start vs warm-cache measurement modes, explicitly labeled.

Real per-run metrics: model load time, TTFT, prefill latency, decode tokens/sec, total latency, peak RSS, GPU utilization, hard page fault count — all captured from the real system during real inference runs, not simulated.

Baseline persistence: every real run's full context (hardware/model/workload state) stored with its metrics.

Real functional acceptance:

The harness executes real inference runs against the real backend from Phase 2 and produces real measured metrics for each.

Two consecutive identical real runs on an idle system fall within an agreed variance band; if they don't, this is treated as a harness defect to fix before Phase 4 begins, not tolerated.

Human-verifiable evidence: A generated baseline report (file or printed table) from a real run against a real model on this machine, showing all captured metrics per prompt, per mode (cold/warm), readable and inspectable by a human without needing to read code.

Phase 4 — Deterministic Bottleneck Detection

Objective: Implement the causal-reasoning layer's cheap, high-confidence branch.

Scope:

Rule-based checks using relative-significance logic (proportional comparison, not absolute thresholds): model footprint vs available RAM; memory pressure correlated with real page faults; page faults correlated with real observed latency; GPU utilization vs offload configuration state.

Each rule outputs a Bottleneck with a confidence score and the specific evidence used.

Escalation rule: below-threshold confidence emits a ReasoningRequest instead of guessing.

Real functional acceptance:

Retain the synthetic causal fixtures from before (model-dominates / external-process-dominates / no-pressure / ambiguous) as unit tests — these remain valid and required.

In addition, run the diagnosis pipeline against real telemetry captured from Phase 3's harness under at least one real induced-pressure scenario (e.g., a real large-context run on this machine that genuinely produces measurable page-fault activity) and confirm the real diagnosis output matches the expected causal conclusion for that real condition.

Human-verifiable evidence: A report showing, for a real captured telemetry sample, the full diagnosis chain: observed evidence → confidence per rule → final Bottleneck (or explicit "no meaningful bottleneck") with the reasoning stated in the same terms a human reviewer could check against the raw numbers.

Phase 5 — Reasoning Interface & Local LLM Engine (Deep-Diagnosis Mode)

Objective: Implement the ReasoningEngine interface with a real local model, as a non-resident diagnostic mode.

Scope:

ReasoningRequest → ReasoningResponse contract with structured (not free-text) output.

Local LLM selection behind the interface, sized using Phase 4's real escalation-rate data where available.

Deep-diagnosis state machine: target model resident → entry decision → pause/checkpoint target → unload target → load reasoner → run diagnosis → unload reasoner → reload target → resume.

Entry decision computed from: expected reasoning benefit (from Phase 4's confidence gap), measured load/unload cost (from real timings, not assumed), interruption cost (context-dependent), UX, and severity/uncertainty — logged whether or not the mode is entered.

Reasoner's peak RAM/VRAM requirement checked as a real feasibility precondition (it still has to actually fit once the target is unloaded), in addition to load/unload time already accounted for.

Real functional acceptance:

A real end-to-end lifecycle on this machine: a real target model is loaded and generating, a real trigger causes entry into deep-diagnosis mode, the real target model is genuinely unloaded (verify via real memory measurement, not a flag), the real reasoning model is loaded and produces a real, schema-valid ReasoningResponse for a real ambiguous scenario, the reasoning model is genuinely unloaded, and the target model is reloaded and verified — via real measurement — to be in a state indistinguishable from before the pause (same config, able to generate correctly again).

A real request arriving during the diagnosis window is queued or rejected per the chosen contract, observed directly, not asserted.

A case where computed cost exceeds computed benefit correctly results in the mode not being entered, verified against real measured load/unload costs from this machine.

Human-verifiable evidence: A timestamped real execution log/report showing each state-machine transition with real timings (target unload time, reasoner load time, diagnosis duration, reasoner unload time, target reload time) and the real ReasoningResponse produced, so a human can see the full real lifecycle end to end.

Phase 6 — Candidate Generation & Controlled Optimization Engine

Objective: Turn a Bottleneck into ranked, evaluable Candidate interventions with expected-value gating.

Scope:

v1 candidate surfaces: runtime parameters (threads, batch/ubatch, GPU offload layers, mmap on/off), context/KV management (context-length right-sizing, KV-cache footprint reduction). Kernel/graph-level candidates remain explicitly out of scope.

Expected-value scoring: (expected benefit × confidence) / experiment cost, using Phase 3's real harness for cost estimation and Phase 4/5's real diagnosis for benefit/confidence.

Rejected candidates logged as considered-and-rejected, for auditability.

Real functional acceptance:

Feed the engine a real diagnosis produced by Phase 4 (or Phase 5) against real captured telemetry — not only a hand-built fixture — and confirm it generates a ranked, non-trivial candidate list with a discard step that actually excludes low-value candidates rather than only demonstrating a ranking.

Fixture-based tests remain valid as a supplement for deterministic edge cases (e.g., forcing a "no viable candidate" case).

Human-verifiable evidence: A report showing, for a real diagnosis input: all candidates considered, each one's computed benefit/confidence/cost/expected-value, which were kept vs discarded and why, in a form a human can audit line by line.

Phase 7 — Experiment Execution & Quality Validation

Objective: Run selected candidates against the real harness and validate real quality impact before applying.

Scope:

Pre-deployment mode: full real harness run (multiple real prompts, cold+warm) per candidate.

Quality evaluator uses a fixed, reproducible benchmark suite appropriate to the model/workload. A cheap proxy metric (e.g. perplexity on a fixed corpus) may be used for early rejection, but final retention decisions must use task-relevant quality evaluation — an optimization touching quantization, context, or representation can degrade instruction-following, reasoning, coding, or factual accuracy in ways perplexity alone doesn't reliably capture. The exact initial benchmark suite is an implementation/research decision to finalize before building this phase, not fixed by this plan.

Adaptive quality tolerance band, driven by Phase 4's real diagnosis severity.

Real functional acceptance:

Apply at least one real candidate to a real model on this machine, measure real before/after performance via Phase 3's harness, run the real quality evaluation suite before and after, and make a real keep/revert decision based on the real tolerance band — end to end, not simulated.

Demonstrate both outcomes for real: one candidate that is correctly retained (improves performance within quality tolerance) and one that is correctly rejected (either no real benefit or a real quality regression beyond tolerance), if the hardware/model combination allows constructing both cases.

Human-verifiable evidence: A before/after report for the real applied candidate(s): real performance metrics, real quality scores, the tolerance band used and why, and the final retain/revert decision — inspectable without reading code.

Phase 8 — Persistent Knowledge & Rollback

Objective: Make accepted/rejected decisions durable and reversible.

Scope:

Per-decision audit records covering the full chain: baseline → diagnosis → candidate → result → decision.

Rollback mechanism: every applied real runtime/context change must be re-appliable to its exact prior value from the stored record.

Conditional retrieval indexed by hardware+model+workload signature; past knowledge is a prior to re-validate, not blindly reapplied.

Real functional acceptance:

Take a real candidate applied and retained in Phase 7, trigger a real rollback, and verify via real re-measurement (not just reading back the stored config) that the system is restored to the original real baseline performance/configuration.

Confirm the full decision chain for that candidate is reconstructable from the audit log alone, by a process that only reads the log — no reliance on in-memory state from the run that produced it.

Human-verifiable evidence: A reconstructed audit trail for one real decision, printed from the persisted log only, plus a real before-rollback vs after-rollback measurement comparison.

Phase 9 — Pre-Deployment Autonomous Loop

Objective: Wire Phases 1–8 into the full unattended pre-deployment flow.

Scope: characterize → baseline → diagnose → generate candidates → evaluate → experiment → validate quality → keep/revert → record → best validated state.

Real functional acceptance:

Run the complete loop, unattended, against a real fresh model on this real machine, start to finish, with no manual intervention between steps.

The loop terminates (does not run indefinitely) and produces a real deployable configuration.

The anti-triviality principle is demonstrably respected: if the real loop encounters a genuinely well-optimized state at some point, it must be able to conclude "no further meaningful optimization" rather than manufacturing a low-value change — this should be checked as part of acceptance, not assumed.

Human-verifiable evidence: The full real audit log for one complete unattended run, plus the final recommended/applied configuration, readable end to end by a human to confirm the reasoning chain is coherent and evidence-based.

Phase 10 — Runtime Adaptive Monitoring (conservative mode)

Objective: Extend the loop to run during active use, under stricter conservative constraints.

Scope:

Lightweight continuous telemetry (always-on, cheap) vs. the full harness (expensive, pre-deployment-only), clearly separated.

Real workload-change detection triggering re-evaluation.

Safe-vs-risky classification: only changes confined to the inference process's own resource usage are auto-appliable; anything touching another process, requiring elevation, or entering deep-diagnosis mode is always recommend-and-approve, never automatic.

Real functional acceptance:

With a real model actively serving real requests on this machine, induce a real workload shift (e.g., a real short-context session followed by a real long-context session) and confirm the monitor genuinely detects it and triggers re-evaluation, without disrupting a real in-flight generation (verified by confirming the in-flight generation completes normally).

Confirm a real candidate requiring elevation, and a real deep-diagnosis-mode entry proposal, both surface as approval prompts and are never auto-applied, in a live running session — not just asserted from code inspection.

Human-verifiable evidence: A real session log showing the workload shift, the detection event, the re-evaluation, and (if applicable) an approval prompt shown to a human with its cost/benefit breakdown, with the in-flight generation's completion timestamp shown to demonstrate no disruption occurred.

Phase 11 — External Process Recommendations (staged, minimal)

Objective: Implement the smallest viable version of external-process awareness.

Scope:

Read-only relative-significance assessment of whether an external process is a plausible dominant contributor to real memory pressure (proportional comparison, not absolute-size ranking).

Recommendation-only in v1 (no auto-termination), surfaced as an approval-gated suggestion with stated expected benefit.

Real functional acceptance:

Validate against real system conditions where practical: on this real machine, with real applications running alongside the real target model, confirm the assessment correctly does not flag a minor real process when the model genuinely dominates memory, and correctly does flag a real process when it can be made to genuinely dominate (e.g., a real memory-heavy application launched deliberately for the test).

Retain the Case A / Case B controlled fixtures as supplementary deterministic edge-case tests; they do not replace the real-condition validation above.

Human-verifiable evidence: A real report from a live run showing actual processes, actual memory figures, the relative-significance computation, and the resulting recommendation (or explicit non-recommendation), matched against what Task Manager shows at the same moment so a human can sanity-check it directly.

Deferred / explicitly out of scope for this plan

Kernel generation/modification, graph-transformation search, model requantization pipelines.

Any system-wide OS reconfiguration requiring persistent elevation (pagefile size, global power/CPU governor settings).

Cross-machine / community knowledge sharing.

Multi-backend support beyond llama.cpp (the BackendAdapter interface leaves room, but no second backend is implemented here).

Open item carried forward

The reasoning model's specific weights/size for Phase 5 are an implementation decision made once Phase 4's real escalation rate is known from actual use — not pinned in this plan, per earlier direction not to lock to one model prematurely.