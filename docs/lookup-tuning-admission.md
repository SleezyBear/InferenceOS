# Lookup tuning admission: proposal for review

Design version: `lookup-admission-design-2`, 31 October 2026.

Revision 2 resolves the independent review's missing final performance-role contract.
The new final workload-role plan below makes lookup's primary workload `better` in
both modes, including lookup-only runs, and governs reservation through final verdict.
Revision 1 and its independent reproductions remain in the preserved review artifacts.

This document specifies proposed work. It implements no path, flag, registry,
profile, request format, judge, scheduler or serving endpoint. The maintained
implementation handoff remains `reviewer-implementation-handoff-current.md`.
No additional hardware study is prepared or authorized by this proposal.

## 1. Objective and present decision

The objective is a demonstrated improvement in the complete configuration for an
explicit use case, while retaining required correctness and safety. Global optimality
is not required. A fast incorrect answer, an incomplete task, or an isolated path gain
that disappears in the complete setup does not meet that objective.

Job 130 is independently accepted as a bounded performance study. It is evidence for
choosing development qualification work, not an admission certificate. **No current
model/workload combination is automatically eligible for lookup tuning or saving.**

| Job-130 workload | Llama 3.2 3B Q4_K_M | Gemma 3 1B Q4_K_M | Proposed consequence |
| --- | --- | --- | --- |
| Prose copying | 32.0% less request time | 27.3% less request time | Both are candidates for correctness qualification |
| Code copying | 33.3% less request time | 25.0% less request time | Both are candidates for correctness qualification |
| Row extraction | 15.8% less request time | -4.6%, interval includes gain and loss | Qualify Llama; no positive Gemma scheduling prior |
| Original prose | No accepted draft work | No accepted draft work | No lookup admission from this evidence |

The five positive comparisons met the frozen rule. Gemma extraction is inconclusive,
not proof of a loss and not a gain. None of the cases establishes general task
correctness: all 80 requests generated 128 tokens, potentially stopping before the
requested task was complete. In particular, Gemma's prose-copy reference omitted
requested record labels. Exact agreement preserved that error.

Evidence sources are the accepted job-130 raw records, revision-5 rules and independent
review `reviewer_independent_job130_results_20261031_checks.json` in `_jobs/out`.
The original artifacts and their bounds remain unchanged. Full findings and limits
are in the existing handoff and [capability documentation](lookup-decoding.md).

## 2. Constraints found in the current implementation

| Existing surface | Relevant constraint |
| --- | --- |
| [RuntimeConfiguration](../src/inferenceos/contracts/experiment.py) | No lookup field; the development option exists only on backend RuntimeParams |
| [ProposedChange and SavedSetup](../src/inferenceos/contracts/tune.py) | Changes use scalar values; saved profiles have no request/workload scope |
| [Strict contract base](../src/inferenceos/contracts/base.py) | Unknown fields are forbidden; old readers cannot be assumed to accept a new grouped setting |
| [Path declarations](../src/inferenceos/tune/declarations.py) | Eight paths; Practical layout is exactly placement/layers, threads and batch |
| [Planner](../src/inferenceos/tune/planner.py) | Three-value Setup and evidence scopes; adding a declaration does not add a fourth dimension safely |
| [Practical selection](../src/inferenceos/tune/budgeted.py) | Complete earlier readings, choose an actual complete base, plan one fresh final check, then approve/save |
| [Whole-setup confirmation](../src/inferenceos/tune/pipeline.py) | Found workloads must improve; guard workloads must not lose beyond existing policy; strictest merged quality requirement applies |
| [Ordinary start](../src/inferenceos/tune/start.py) | A single saved process configuration serves clients; it cannot infer or enforce a copy-workload restriction |
| [History identity](../src/inferenceos/tune/learning.py) | Version 4 describes requested-fresh ordinary measurements; it has no lookup execution contract |
| [Measurement and fair comparison](../src/inferenceos/tune/measure.py), [fair.py](../src/inferenceos/tune/fair.py) | Existing quality, power/disturbance, planned-look, guard, cleanup and stop owners remain authoritative |

Job 130 used one active native request at a time, not a one-slot server. Its load logs
show four slots and context 4096 per slot. They also identify an i7-8850H CPU and
Radeon Pro 560X Vulkan device, with the GPU ID printed as unknown. The record is not
a complete versioned hardware qualification object. Neither a current hardware scan
nor a saved setup's older machine record may be backfilled as historical run evidence.

## 3. Admission states: different kinds of permission

| State | Required evidence | Permission |
| --- | --- | --- |
| Unqualified | Missing, negative, inconclusive or incompatible evidence | Keep lookup out; record why |
| Performance-supported | Accepted development gain for a narrow cell | Prioritize a new qualification proposal only |
| Qualified for a local comparison | Reviewed complete-task quality, operational identity, workload/configuration scope and measured cost evidence | Offer one opt-in scoped comparison when budget permits |
| Provisional candidate | Fresh matched comparison on the captured actual base passes | Consider it in fresh whole-setup confirmation; no save yet |
| Whole setup confirmed | Actual combined configuration passes all original and additional checks | Report the scoped result; deployment permission is still separate |
| Scoped deployment permitted | An explicitly reviewed consumer enforces the same request/lifecycle scope, supports the schema and has demonstrated end-to-end benefit | Enable only that scoped consumer after the existing approval/save rules |

Today the five positive job-130 cells reach only Performance-supported. There are no
qualified admission artifacts. A missing field is unknown, not a wildcard or a pass.
Quality failure, identity refusal, unaffordable work and absent performance gain are
distinct outcomes. An untested model is not labelled technically unsupported.

Qualification must remeasure both complete-task performance and costs under its new
prospective protocol. Adding an oracle to old job130 rows cannot mint permission:
their capped outputs, instrumentation and costs belong to the original experiment.
If complete tasks show no gain, lookup remains out even when their prefixes were fast.

This separates development-time path selection from runtime acceptance. Practical
does not run discovery probes to decide whether lookup might be useful. Its candidate
permission and cost evidence must already exist before tune planning. Fresh tuning
measurements are still required to demonstrate a gain in the user's actual setup;
historical gains never substitute for that confirmation.

## 4. Proposed evidence contract

A future, immutable `LookupQualification` artifact would bind all of the following:

- Original record/audit/protocol hashes, explicit independent-review disposition,
  artifact schema, corpus, oracle and cost-model versions, and revocation state.
- Full model/tokenizer hash; executable selection/path and engine/library hashes;
  effective backend/device; Windows, driver, hardware and power-plan identities.
  Reuse the existing hardware and runtime identity readers where applicable, without
  treating their missing fields or multi-device limitations as verified support.
- Requested and observed context, slot layout, effective thread/default resolution,
  complete base configuration and any ambient configuration/environment effects.
- Algorithm `ngram-simple`, N=12 and M=16 only. No parameter search, helper model,
  persistent cache or transfer to a different drafting method in the first version.
- Versioned workload templates, deterministic typed input grammar, full prompt and
  expected-output corpus hashes, allowed input/output length envelope and holdout IDs.
- Native completion wire format, one in-flight request, fresh process/load per arm,
  `cache_prompt=false`, no engine reuse option, seed 42, temperature 0, natural EOS,
  output ceiling, streaming and complete generated-ID instrumentation.
- The exact benchmark/serving lifecycle and instrumentation. Return-token IDs and
  diagnostic logging are not silently removed at deployment while reusing their
  measured gains. A changed format, sampling policy or warm/chat lifecycle is new work.
- Measured cost distributions for quality, identity, both arms, guards, cleanup and
  final confirmation within the declared configuration envelope. Forecast uncertainty
  is retained; five observed maxima are not a guaranteed deadline or a percentile.

Static hardware compatibility can establish that a knob exists; it cannot establish
a gain. A CPU/GPU name, architecture label or keyword such as "copy" is insufficient
to match a performance cell. Initial matches are exact reviewed identities and
explicitly qualified configurations, not a learned family-transfer rule.

The initial matcher is machine-local: recorded hostname plus a versioned stable
HardwareProfile projection, effective selected device and runtime/driver hashes.
The current human-readable hardware_signature alone is too coarse. The projection
must specify its required fields before qualification: CPU identity/core topology,
instruction-set evidence, total RAM, GPU/vendor/VRAM, driver and OS build, power plan,
and the observed backend/device association. Exclude volatile free-memory figures,
timestamps and clock samples from identity; retain them as measurement context.
PCI identity is included when exposed, with explicit missing metadata otherwise;
an "unknown id" string never becomes a stable ID. Initial eligibility requires one
unambiguously matched usable GPU; multi-device/ambiguous mappings remain unqualified.
This describes compatibility on an observed machine, not tamper-proof identity or a
claim that same-name machines are interchangeable. No new hardware scan is performed
in this documentation batch.

The entire RuntimeConfiguration content participates, including fields beyond the
eight path-owned settings: context, mmap, device, batch defaults and shared/per-side
KV precedence. Requested null defaults must also match the observed effective state
where it matters. A new selected base outside that envelope skips lookup. This may
make initial Practical coverage small; that is preferable to inventing a cost or gain.

Unknown model/hardware/build/scope, a revoked artifact, missing cost, or incomplete
proof produces a reported skip with lookup absent. There is no runtime bootstrap
probe, hidden download or "similar GPU" fallback. Broader family transfer requires a
separately frozen development/holdout design before data, then independent review.

Evidence age is recorded, but no arbitrary expiry interval or confidence is invented
here. Identity change or explicit revocation invalidates it. Unchanged identities
still require fresh local performance confirmation; they do not prove conditions
remain representative.

## 5. Qualification must judge complete tasks

The first implementation work should be software-only corpus/oracle construction,
reviewed before any further model study. No tuning policy is trained or enabled yet.

| Family | Independent expected answer | Required rejection examples |
| --- | --- | --- |
| Labelled prose copy | Complete input-derived records, including every requested label and order | Missing labels, dropped/duplicated records, extra explanation, fabricated text, incomplete tail |
| Code copy | Complete requested source as data; explicit newline policy | Missing lines/indentation, added code fences, changed literals, extra code, truncated function |
| Row extraction | Exact selected rows computed from typed input and requested predicate before inference | Odd/unrequested row, missing row, duplicates, changed number, wrong order, extra text |

Use an explicit whole-output grammar, not substring matching or a model judge. Freeze
any CRLF/LF equivalence and terminal-newline policy; do not erase labels, digits,
indentation, Unicode, signs or arbitrary punctuation to make failures pass. Treat code
as text, never execute it. Parser fixtures include empty answers, echoed prompts,
refusals, reordered/duplicate records, number/identifier lookalikes and malicious text.

Prompts must ask for complete tasks that fit the frozen output/context envelope.
Expected answers are generated independently before inference. Natural completion
and the full oracle must pass; a limit-truncated prefix is insufficient even when
both arms match. The existing 128-token ceiling is not evidence for other lengths or
for completion. Tokenizer/context checks belong to the later explicit qualification
study; do not start a model simply to fill in this design.

Freeze calibration and held-out content/length/repetition strata before data. Distinct
holdouts must prevent merely memorizing job-130 prompts, and include low-opportunity
cases within the intended input envelope. A template name is not a performance oracle.
Every in-scope mandatory case counts; wrong/insufficient/uncertain cases are retained.

For each mandatory qualification case, each required control and lookup answer must
be task-correct, complete and operationally usable. Control wrong means **task not
qualified**, not candidate-caused regression and not permission to drop the case.
Candidate wrong on a correct control disqualifies that configuration. No fitted
judge/threshold changes, favourable subset selection or unrecorded prompt repairs.
Revisions require a new prospective protocol and fresh qualification data, preserving
all earlier records. A failed cell remains failed until that separate process.

This new qualification adds an absolute task gate specifically for lookup. It does
not silently change the existing baseline-relative quality rules for other paths.
Existing known-answer/precision requirements and strictest-merged checks remain in
force. Passing a small corpus never certifies arbitrary generated explanations.

## 6. Proposed first integration: lookup last, on one captured base

Do not add a fourth coordinate to Practical's Planner or put lookup into its current
three integer slots. Do not pretend it is another prompt-batch value. Do not keep an
editable capability copy in the coordinator. Use the shared coordinator for a bounded
terminal phase after actual base selection and before the one fresh final confirmation.

1. Existing paths run with lookup absent. Their schedulers, measurements, selection
   of earlier complete setups, safety flow and stop behaviour remain authoritative.
2. Capture immutable base B, its exact identity and the original confirmation baseline
   S0. Only enter on a normal continuation with a qualified workload/cell. Never enter
   after cancellation, error, unresolved cleanup, failed required safety/quality work,
   or time exhaustion. No-change branches require an explicit hook; they cannot be
   reached by continuing after a failed confirmation.
3. Validate the artifact and compile the explicit scoped final workload-role plan
   below, including lookup's primary `better` role. Reserve the full optional phase
   and that plan's final-check budget. With no suitable envelope/cost/reserve, skip
   lookup and continue existing behaviour on B. No calibration probe is run to
   manufacture a forecast.
4. Compare B with explicit algorithm none against the **same B** with ngram-simple
   12/16. Both arms share offline flags, requested work, instrumentation, identities
   and applicable quality checks. This is one binary comparison, at most once per
   tune, not a parameter sweep or another scheduler. Capture B once; never rebuild
   the second arm from a changed best configuration.
5. Require complete-task correctness and matched text/IDs/endings for this lookup
   off/on pair. Use the existing mode's fair/keep rules; do not import job 130's effect
   size or development alpha into local tune acceptance. Insufficient/no-gain work
   leaves lookup absent. Clean no-change transitions are still recorded once.
   Report request latency as well as the existing goal score; decode throughput or
   accepted-token ratios alone cannot justify a request-latency claim. If the requested
   goal/workload has no applicable measured qualification, do not invent new weights.
6. If provisionally selected, freshly confirm S0 against the actual combined B+lookup
   configuration using that same explicit plan: lookup's primary workload must pass
   `better`, including when B equals S0. Keep every original found/guard workload and
   required scoped task check. Preserve existing minimum-gain/payback, guard,
   precision, disturbance, planned-look and memory rules. The scope must cover every
   required measurement role, including guards; otherwise skip lookup rather than
   replace or drop a workload. Guard-only qualification confers no permission to
   serve that task family. A guard pass alone cannot confirm lookup.
7. Do not reuse local lookup comparison rows as final proof. Final confirmation is
   the normal single fresh check. If it fails, follow the existing baseline outcome;
   do not start an unplanned second final check of B to rescue the result. That could
   discard useful unconfirmed core changes, but avoids a new fallback/multiple-look
   policy and unbudgeted work. A different fallback design needs separate review.
8. When saving later becomes supported, approval changes the actual saved configuration.
   Declining lookup or changing any base setting invalidates the combined proof:
   remove the provisional lookup result
   and use the existing owner to re-confirm the actual lookup-off approved subset,
   if permitted and affordable. Never silently drop the option but keep its gain.
   No second optional lookup comparison is started in this version.

Exact token agreement is between lookup off/on at B. It is **not** required between
all earlier configurations: a precision path may legitimately change a correct answer's
wording under its existing quality policy. The final setup must satisfy the strictest
existing requirements plus lookup's independent task qualification, not substitute
one for another. A safety-only save carries no lookup speed claim.

### Final workload-role plan: required contract for both modes

D3 must introduce one explicit, immutable `ScopedFinalPlan`, owned by the existing
tuner and consumed by both mode routes. It describes required evidence, not a second
search queue or a synthetic scalar `ProposedChange`. The current helpers do not
implement this contract: Practical derives roles from three changed coordinates,
and the shared checker derives them from ordinary changes. Reusing those derivations
unchanged can yield an empty plan or a guard-only pass for lookup-only work.

Construct the plan as follows, before dispatching the optional lookup comparison:

1. Capture every existing final improvement (`better`) and guard (`not_worse`) role
   on the actual selected B, including the main workload's existing fallback guard
   where required. Record the original role and its source; lookup does not delete
   any original requirement. The ordinary no-core-change plan may be empty.
2. Add the qualified lookup primary workload as `better`, even if there are no
   ordinary changes and B equals S0. Its identity, requested objective and applicable
   qualification are fixed before measurement, not selected from favourable final
   results. Key entries by full workload/request/policy/scope identity, not just a
   display name or a short/long coordinate. A reused workload ID with contradictory
   content or controls refuses the plan rather than aliasing different work.
3. Merge identical entries with `better` dominating `not_worse`. Thus an existing
   guard used as lookup's primary is **promoted to `better`**, not left guard-only or
   measured twice. Retain its original provenance, quality and per-measure guard
   requirements. Other original improvement and guard roles remain unchanged. Keep
   mandatory oracle/precision/effect checks as additional required gates; a quality
   check alone never supplies performance improvement evidence.
4. Order the lookup primary first as the designated performance/result anchor, then
   other `better` entries in their original relative order, then remaining guards in
   their original relative order. This order is fixed prospectively, not chosen from
   measured gains. Report its measured gain as the scoped headline, with each other
   result retained; do not pool gains or let a guard become the headline.
5. Bind S0 explicitly off, actual B+lookup, qualification/scope digests, every workload
   definition, objective/quality requirements, modes, planned looks/attempt caps and
   cost provenance into one plan digest. Use the existing selected mode's planning
   rules with applicable scoped inputs; do not reuse a core-only effect estimate as
   the S0-to-B+lookup effect or add the two screening gains. Unavailable evidence stays
   unknown. An existing documented planning fallback may remain a labelled forecast,
   never measured evidence or a claim of achieved power. If no valid bounded plan and
   qualified both-side cost forecast can be produced before entry, skip lookup.
6. Use this **same sealed plan** for the optional-phase reserve, final cost/sizing,
   planned looks, fresh execution, progress/ledger records and final verdict. Charge
   both sides of every entry, allowed attempts, all required task/quality gates and
   cleanup. Practical's current three-coordinate `_final_plan`, `_final_cost` and
   `_plan_confirmation` cannot silently rebuild or omit the added role. Its scoped
   adapters and the shared checker's explicit-plan input are required D3 work. Freeze
   roles and look schedules before optional dispatch; never demote, add, resize or
   reselect them after seeing optional or final results. A plan/digest disagreement
   refuses execution. If the optional comparison is not kept, discard this prospective
   combined plan and continue the unchanged ordinary core route, without lookup.

Both modes must use a common scoped confirmation result with the same success rule:
the primary `better` comparison satisfies `fair.kept` (including the existing minimum
gain/payback rule), every other `better` entry satisfies `fair.kept`, every remaining
guard satisfies `fair.not_worse`, and all required quality, observed-effect, identity,
power and cleanup gates pass with complete fresh evidence at permitted looks. The
primary confirmation must itself be `confirmed=True`; all other required gates can
only remove that success. Missing entries, incomplete work, a failed gate or a plan
without its primary `better` role can never confirm the scoped result.

In particular, neither mode may treat the existing checker's `passed` boolean as this
scoped verdict: guard-only comparisons can return `passed=True` while their
`ConfirmationResult.confirmed` is false. The new common scoped `confirmed` verdict
must be the sole success signal in both callers. A positive local screen is not
final proof. Flat or slightly negative primary final performance must fail even if
it passes a guard margin; a qualifying positive result can pass only with all the
other required checks. This changes no legacy feature-off success contract.

Required D3 caller-level regressions cover **both modes**: lookup-only B=S0, an added
primary absent from core roles, a guard promoted to `better`, and an already-`better`
primary merged without duplicate measurements. For each, a qualifying positive final
result passes, flat and approximately 1% negative final results fail; a positive
primary plus a failed/incomplete guard fails. Check minimum-gain/payback rejection,
original role preservation, identical plan digests from reserve to verdict, correct
look/attempt/cost accounting, and conflicting workload identities causing zero dispatch.
These are future implementation requirements, not claims that runtime admission now
exists or that passing documentation probes has implemented these mode callers.

### Interactions with all eight existing paths

| Current path | Why its change invalidates a prior lookup comparison |
| --- | --- |
| layers | CPU/GPU division, buffers and verification work change |
| threads | Draft lookup and model verification compete for CPU work |
| reading_threads | Prompt processing cost and thread allocation change |
| prompt_batch | Prefill and verification batching/resource use change |
| flash_attention | Attention implementation, memory and numerics can change |
| engine_switches | Effective kernels/backend execution can change |
| placement | Per-piece compute, transfers and memory layout change |
| kv_cache | Representation, numerical results and allocation can change |

These are conservative dependency reasons, not measured interaction strengths.
Any changed base field invalidates lookup evidence, even outside this table. The
first design avoids cycles by keeping lookup absent throughout core exploration and
running it last. It does not claim those core choices remain individually optimal
after enabling lookup. Full combined confirmation establishes the required measured
improvement on the tested objective, not a joint optimum. Never add separate gains.

Bidirectional retuning with lookup active is deferred. It requires pre-registered
interaction experiments, a finite revisit/budget rule and tests of both mode policies.
Likewise, prompt reuse, cache-ram changes, SWA variants, chat templates, warm state and
concurrency cannot inherit this permission. Job 129's single-pair full-cache finding
is not a general mitigation or proof of a safe lookup/reuse combination.

## 7. Ownership and modes

| Owner | Proposed responsibility |
| --- | --- |
| Existing Planner | Its three coordinates, choices, queue and core evidence scopes; unchanged in the first integration |
| Existing BudgetedTune / Tuner | Actual selected B, budgets/ledger, histories, confirmation, approval, save and stopping |
| ScopedFinalPlan, owned by that tuner | One immutable ordered role/look/cost contract, including lookup primary `better`, shared by reservation, execution and the scoped verdict |
| Mode policy + shared coordinator | One explicit terminal phase and trace; no second cache, scheduler, selected setup or cleanup owner |
| Scoped immutable trial description | Reference to B plus one typed LookupDecodingParams value and scope digest; execution description, not another editable configuration |
| Backend and adapter | Current-generation application, required observed effects, cleanup and restoration; refuse unsupported CLI backends |
| Qualification matcher | Read-only artifact validation/eligibility with explicit reasons; never dispatches a probe |
| Existing measurement/quality owners | Real paired work, new optional evidence transport, absolute task gate and existing stricter checks |

Practical needs empirical estimates for both arms, qualification/identity overhead,
guards and final work before entry. Reserve final proof first; optional lookup cannot
consume its budget. A forecast is not a guarantee. Use the existing soft target,
started-pair extension and watchdog boundaries; do not cut an allowed pair in half or
add a mid-pair policy check. All elapsed overhead and disturbance repeats count.
An unaffordable step is skipped without probing or changing the current Planner.

Unbounded uses the same eligibility and acceptance gates. More time may allow deeper
existing searches, not unsupported lookup families, missing quality proof, free
hardware discovery or relaxation of unknown identities. It is still finite and uses
the existing deadline/revisit limits. Both modes compare on their actual captured B.

## 8. Contracts, measurement transport and history

First deliver a report-only, explicitly opt-in rehearsal. The current runtime and
SavedSetup schemas remain unchanged. A typed scoped trial can adapt B and the existing
backend group for measurement, but cannot be serialized as an ordinary selected setup.
This preserves old profiles and prevents accidental activation by ordinary start.

Proposed D3 transport is an immutable effective trial description containing B, the
typed lookup group and a scope digest. The selected base remains owned by the tuner;
this description does not mirror it in another mutable planner. New explicit start
and candidate execution options must travel through runtime_params_from, the real
harness and fair/whole-setup comparison. Both final arms must use the same request
controls and instrumentation, with explicit lookup off for S0. Current conversion
does not carry these options: wrapping a configuration or toggling ambient adapter
state alone is not an implementation. The old _settings_to_verify/strictest(changes)
logic cannot see a hidden backend group; add scoped effect verification and combine
the ENGINE quality floor plus the mandatory task gate with the existing strictest
requirement. Precision's known-answer checks cannot be downgraded.

Report the grouped capability in a versioned scoped result, not a fabricated scalar
ProposedChange. An experimental terminal coordinator action can represent it without
adding an ordinary ninth declaration or violating Practical's validated three-slot
layout. Future catalog/atomic-profile support must be reviewed explicitly before
calling it a normal executable tune path. Feature-off action lists stay unchanged.

Before any saved path, a separately versioned effective-configuration and atomic
grouped-change contract must carry the complete lookup group and scope end to end:
conversion, adapter reload/restart/rollback, measurement, report, approval, store,
start and failure recovery. ProposedChange's current scalar/related maps cannot
silently encode the group as an ad-hoc string. An unsupported consumer refuses rather
than ignoring it. Legacy saved-file bytes must survive unsupported/new entries under
the existing exact-preservation and store-lock rules. No blanket rewrite/migration.

Measurement transport is a hard prerequisite: complete generated token IDs, raw native
settings/counters, endings, process/log/library proof and scope must survive the real
harness, ExperimentMeasurement and report round trips. Backend-only last_response
counters do not prove that tune retained them. Add an explicitly versioned optional
evidence field for scoped requests, not lossy strings in an untyped extra map. Missing
proof makes that request unusable; no extra inference is run just to populate a trace.

The scoped harness must actually enforce the qualified fresh first-request lifecycle.
The ordinary harness's warm-up and repeated warm requests are not interchangeable
with it; attaching new evidence fields to that unchanged execution is insufficient.
The final plan binds the execution controls on both sides as well as workload roles.

All scoped cache/cost/history identities include B, algorithm/N/M, workload/template
and oracle versions, full request work, execution lifecycle/instrumentation and observed
runtime identity. Never key these rows only by the existing three-value Setup or
workload ID. Exact setup snapshots, freshness and within-step captured bases remain.

Proposed scoped history format is a new version, not a v4 request with an unrecorded
flag. No v5-to-v4 collapse, no inferred "old means explicitly off", no importing job130
as confirmed tune history, and no candidate-selection feedback from failed/incomplete
or unaccepted proposals. Initially keep rehearsal records in the development namespace.
Lookup-off ordinary runs keep their current v4 identities/serialization and behaviour.
Any later optional field must not add a null key to every legacy request silently or
drop all existing nulls to compensate. Explicit legacy projections need parity tests.

Unresolved stop/cleanup, superseded adapter operations or unsupported effective settings
still prohibit further dispatch/save. Preserve the existing generation/admission gates
through halt, resume, unload, cleanup, restoration and failed reload; one backend owns
resources. The identity gate shared with the auditor must run before generation and
again before the next load. No generic retry or new cleanup framework is proposed.

## 9. Deployment boundary

A general SavedSetup plus ordinary start is insufficient: lookup is process-wide,
while job130 describes fresh native tasks and not arbitrary cached chat. A UI label or
client-supplied "copy" tag cannot enforce that scope. Enabling it in a shared server
would apply it to unqualified traffic and warm/parallel states.

The first integration therefore publishes a scoped **report/recommendation only**.
Lookup-containing configurations are not saved into the normal profile store. If only
ordinary core settings are to be saved, confirm their actual lookup-off configuration;
do not claim the lookup-on measurement applies. Older profiles/start keep behaving as
before, and no runtime flag bypasses the deployment gate.

Keep the rehearsal result separate from TuneResult.final_configuration and any
ordinary SavedSetup.confirmed_gain. The former still describes a representable
lookup-off setup; the latter may describe only its own fresh lookup-off confirmation.
A successful scoped check cannot be relabelled as that ordinary proof. During the
first rehearsal, the opt-in caller must choose report-only operation before dispatch;
it must not perform a combined final check and then silently save B without lookup.
Any future core-only save flow needs its own explicit, budgeted existing-owner
confirmation branch; it cannot follow a failed combined check as a rescue attempt.

An eventual minimum enforceable consumer could be a separately reviewed typed native
task executor: it constructs the qualified prompt from validated data, owns a private
worker for one bounded fresh task, requests the qualified instrumentation, checks the
complete oracle and releases through the current backend. It does not expose that
worker's URL/key as a general chat endpoint. Unknown templates, reuse, sampling,
lengths or warm/parallel use are refused or routed to a separately owned lookup-off
profile before dispatch, never executed under the wrong permission. This is not a
security claim against a local owner who deliberately bypasses the managed consumer.

That consumer does not exist today. It needs its own review and measured end-to-end
benefit including any load, inspection, parsing, response and cleanup costs the user
actually pays. Job130 excluded those costs. It cannot simply import the 25-33% request
figures as deployment gains. A resident warm/chat service needs different evidence and
scope enforcement. No silent request retry is introduced after incorrect output.

## 10. Delivery order and review gates

| Stage | Concrete next deliverable | Exit condition / remaining permission |
| --- | --- | --- |
| D0: this batch | Proposal, accepted-result documentation, source/evidence inventory and same handoff | Independent design review; no executable admission |
| D1: software qualification | Deterministic typed corpus/oracles, explicit artifact matcher/schema and fake-engine reports | Adversarial oracle/identity/cost/serialization tests; separately review/freeze a development protocol before data |
| D2: development qualification | Only a separately owner-authorized model study under that reviewed protocol | Correct complete tasks, all pre-registered mandatory/holdout/negative cases, operation/cost evidence and independent results acceptance |
| D3: report-only integration | Opt-in terminal coordinator action in both modes, full evidence transport and explicit shared final-role plan from reserve through verdict | Feature-off strict parity; both-mode lookup-only/guard-promotion performance, combined-tune and failure regressions; no ordinary save/start activation |
| D4: scoped deployment | Versioned atomic configuration/profile plus an enforceable consumer and end-to-end study | Independent software/protocol/results review and separate hardware authorization; existing user approval/save remains required |
| Later: expand coverage | Measured interaction grid, new families/hardware, warm/chat/reuse/concurrent designs | Fresh prospective evidence; never extrapolate from a model name or the original five cells |

No new hardware wrapper, queue file, executable eligibility registry or numeric
admission threshold is delivered here. D2 and D4 require separate owner authorization;
accepting this document does not authorize them. Existing mode thresholds/look plans
stay unchanged. Details of a future qualification population and its statistical
analysis must be frozen and reviewed before data, not improvised from its results.

## 11. Required adversarial coverage before activation

- Wrong control and equally wrong lookup, especially Gemma's omitted labels; late
  truncation after a correct first line; malicious/echoed input; Unicode/identifier
  and duplicate-row traps; complete holdouts and no dropping inconvenient cases.
- Forged/missing/revoked artifacts; changed model, driver, hardware, actual executable,
  mapped library, base/default resolution, template, decoder group or corpus/oracle;
  unknown is never compatible. Identical unregistered executable paths remain refused.
- Two arms from one captured B after a core choice changes; each of the eight core
  changes invalidates lookup proof; context/mmap/device/batch defaults do too. Preserve
  layer-to-thread and batch-to-thread behaviour, exact caches, fresh repeats and an
  earlier complete core configuration winning over the planner's latest choices.
- Practical's normal three slots, missing long question, weak-repeat/cache semantics,
  no-time/unknown-cost skips, final reserve, allowed late pair completion, no extra
  measurements, no adaptive final look and no rescue after failed confirmation.
- Lookup-only/no-core-change normal entry versus error/cancel/deadline/safety-failure
  exits; original same-exception propagation, partial reports, one cleanup owner and
  no later request/save when release or observed identity is unresolved.
- Lookup primary `better` in both modes, including promotion from guard; no empty or
  guard-only success; positive, flat and slightly negative final evidence; all original
  roles retained; one sealed role/look/cost plan from reserve through final verdict.
- Atomic approve/decline, derived approved subsets, confirmed-gain provenance,
  reload rollback, delayed publications, halt/resume permission preservation, exact
  store preservation/lock errors and unsupported older readers/CLI adapters.
- Native evidence through harness/history/report round trips; no off/on cache alias,
  v4 compatibility fiction, lost token IDs, hidden warm/reuse state or raw-to-chat
  substitution. Ordinary start must not activate a report-only scoped candidate.
- Feature-off before/after Practical 231, Unbounded 328/323916 and history cost/candidate
  studies, with unchanged populations/normalization and raw original references kept.
  These are planned implementation gates, not new test runs claimed for this document.

These cases guard the scoped admission decision and its complete configuration.
Passing them is not evidence of a universal gain, exhaustive interaction search or
general semantic correctness. Independent design review is the next action.
