# D2 lookup qualification protocol: proposal for review

Protocol: `lookup-qualification-protocol-2`, 1 November 2026.

## Prospective amendment from version 1

The Stage-A revision-2 review supports distinguishing client warm-up requests from
the pinned engine's internal initialization. Version 2 applies that clarification
before any Stage-A data or live manifest. The original accepted version-1 document,
proposal and packages remain preserved; no historical result is reinterpreted.

| Contract | Version-2 rule |
| --- | --- |
| Extra client warm-up, priming or discovery | Forbidden; no extra completion/chat request in Stage A |
| Native startup empty run | Retain the pinned engine's default initialization and its complete current startup log |
| Stage-B cycle time | Include all startup/initialization work; never subtract warm-up time or reuse a process between answers |
| Identity | New protocol/document/proposal hashes; version-1 manifests are not accepted by the amended collector |

No population, model/configuration pin, task, judge, output cap, order, statistical
rule, cost role or permission boundary changes. Stage-A repair/software review and
any later manifest review remain required; this amendment grants no hardware run.

This is a prospective design, not an executable registration or hardware approval.
D1 software is independently accepted at owner commit `0bb1a3e`. Its corpus, oracle,
draft schema and numeric/provenance repairs remain unchanged. The governing
[admission design](lookup-tuning-admission.md) remains `lookup-admission-design-2`.
The only maintained implementation handoff is
`reviewer-implementation-handoff-current.md`.

No collector, auditor, live wrapper, qualification certificate, tuning admission,
profile or deployment consumer is implemented by this document. No job is placed in
`later` or a queue. A reviewed protocol alone is insufficient to run either stage.

## 1. Question and permission boundaries

For each of the two pinned models and each of the three D1 task families, ask:

> On one exact base configuration, does fixed ngram-simple lookup preserve complete
> correct answers and reduce the measured fresh-request cycle time for the frozen
> held-out task suite, without an unacceptable loss on its calibration suite?

There are six candidate cells. Job130 motivates testing them; its timings, capped
answers and configuration assumptions are not new quality, timing or cost samples.
Gemma extraction has no positive prior. Testing it here is not a claim of benefit.
No architecture name or runtime discovery probe selects a Practical path.

| Gate | Required before proceeding | Permission after passing |
| --- | --- | --- |
| Protocol review | This design and its immutable proposal package accepted | Build and review software; no model work |
| Stage-A software review and owner authorization | Fit-only collector, adversarial tests, source/binary/model pins, output and lifecycle protections reviewed | Only the four fit/identity loads below |
| Stage-A results review | All required observations available and independently accepted | Instantiate a Stage-B manifest; still no answers |
| Stage-B software/manifest review and owner authorization | Complete collector/auditor reviewed; manifest and analysis frozen | Only the registered answer study |
| Results review | Raw answers, timing, costs, identities and exclusions independently accepted | A narrowly scoped evidence recommendation; no automatic tune/save/start activation |

Authorizations are separate for A and B. There is no A-to-B automatic continuation.
Failed A observations cannot be repaired with a generation probe or guessed default.
A revised population, base, cap, judge, analysis or observer contract requires a new
prospective version and review before affected data, preserving earlier evidence.

## 2. Fixed assets and initial scope

Use the accepted `lookup-task-corpus-1` / `lookup-task-oracle-1` corpus:
`363d3870ad21b072a18cc269fda1cf325a641e28e5fb3a37039910d3840c21ea`.
All 42 cases are mandatory: 14 per family, 21 calibration and 21 holdout overall.
Each family has seven cases in each split: six length/repetition combinations and
one short-output case. The latter is low-opportunity by construction, not a promise
of zero accepted drafting. The public holdout is a frozen analysis split, not secret
data. Do not substitute job130 prompts or generate more favourable content.

The proposal package includes the complete ordered prompts, expected answers, typed
inputs, IDs and hashes from D1. Neither task inputs nor the oracle changes here.

| Model asset | SHA256 | Proposed requested base |
| --- | --- | --- |
| `models/study/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` | Job130 frozen base, setup source `setup-7c62902e00` |
| `models/study/google_gemma-3-1b-it-Q4_K_M.gguf` | `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` | Job130 frozen base, setup source `setup-0e54fdd01e` |

Full 14-field requested configurations and all 43 proposed engine-file pins are in
the immutable proposal JSON. They come from accepted job130 revision-5 rules, whose
SHA256 is `273b217c6243d500adbd1c40b2a1d7cc474163a74b471ddeb26f99ca37601f2e`.
These are prospective pins, not a claim of inspecting current model/engine bytes.
Stage A must verify the actual files and saved setup's model identity before backend
construction. Read saved setups only; never rewrite, retune or use a stale entry by
path alone. A changed setup refuses; it is not silently adopted as the new base.
Resolve and bind the actual data directory and saved-file paths as well. This first
proposal uses the repository's canonical `data` root; an alternate configured root
or linked store requires a newly reviewed manifest, even if copied entries match.
Protect the configured root and repository data from output writes in every case.

Keep context 4096 per slot, four observed slots, one in-flight request and one fresh
process/load per answer. Fresh engine state does not imply cold OS/file caches.
Both arms use native streaming completion, seed 42, temperature 0, n_predict 512,
ignore_eos false, no custom stops, cache_prompt false and complete generated IDs.
The lookup arm uses only ngram-simple N=12/M=16; the control is explicitly none.
No helper model, lookup-parameter search, cache-reuse, SWA variant, extra client
warm-up, priming, discovery, chat or concurrent request is permitted. The pinned
engine's internal empty startup run remains part of initialization, with its
complete current log retained and confirming the empty startup run. Explicit
warm-up enable/disable overrides are not permitted. Stage A permits zero completion/chat requests.
Stage B includes initialization in full-cycle accounting; it must not subtract
startup warm-up or reuse the process between answers. Return-token IDs and diagnostic
logging remain part of the measured scope, including log verbosity 4.

The embedded tokenizer asset is the same complete GGUF file: the tokenizer asset
hash equals that file's SHA, with kind `embedded-gguf-whole-file-1`. This binds its
bytes without pretending to compute a tokenizer-only metadata hash. Stage-A token
IDs additionally bind its observed tokenization. External tokenizer files or a
different interpretation require a new protocol.

## 3. Stage A: fit and observations, no generated answers

The proposed order is Llama/control, Llama/lookup, Gemma/lookup, Gemma/control.
There are four separately released loads, no /completion or chat request. Calls are
limited to existing readiness/inspection, tokenizer and lifecycle surfaces. A spy
test must prove zero generation dispatches. A complete process/key release is
required before every next load and between models.

For each load:

1. Enforce actual selected executable path/hash and model/setup identity before
   construction/start; reject alternate executable copies as well as wrong bytes.
2. Capture current PID/start identity, complete startup log, effective layout/base,
   selected Vulkan device and required mapped-library evidence. Use the live/audit
   identity rule shared by job130; missing required libraries refuse before work.
   Every HTTP slot must expose a literal speculative-activation boolean consistent
   with the arm and startup evidence. The initialized slot count must agree with
   all four observed HTTP slots. A process PID/create-time pair already observed
   in an earlier load refuses before tokenization, followed by owned release and
   explicit not-run accounting for remaining loads.
3. Tokenize each full prompt with the native request's add-special setting. Tokenize
   the independent expected answer with add_special false, plus canonical CRLF/LF
   and optional terminal-newline representations. Save the full ID arrays and hashes,
   not just lengths. Deduplicate identical byte strings, retaining their form labels.
4. Require each prompt's positive token count P and `P + 512 + 16 + 64 <= 4096`.
   Require the largest of these canonical expected counts E to satisfy `E + 32 <= 512`.
   The 16 is draft capacity, 64 a fixed input/context margin, and 32 output reserve.
5. Verify token IDs match between arms for every identical string on the same model.
   Record power before/after, release snapshots and credential/process exit evidence.

These canonical counts are a capacity screen, not a mathematical bound on every
possible encoding or on EOS placement. Even task-correct text may have a different
tokenization. Stage B still requires actual natural EOS, no truncation and <=512 IDs.
No failure is rescued by dropping a case, shortening input, raising the cap or taking
512 tokens as a valid prefix. All 42 cases and both models must pass A before B.

Collect a new live `lookup-machine-local-1` projection using existing readers:
hostname, OS build/architecture, CPU identity/topology/instructions, RAM, GPU identity/
VRAM, driver, Vulkan device association and power plan/source. One unambiguously
matched usable GPU is required under D1's schema. PCI absence may be explicit null;
an unknown string is not an ID or wildcard. Record observation sources and original
inventories. Do not backfill job130 with this newer machine observation.

Keep all 14 requested fields explicit, including nulls. Independently observe the
required effective values with their raw sources; request/command echoes alone do
not prove default resolution. In particular, thread/batch defaults, per-slot context,
K/V types, offload, mmap, attention and selected device cannot be guessed. Shared-KV
alias absence is distinct from the observed per-side types. Unobserved required
effects, contradictory sources, ambiguous devices or unverified activation prevent B.
Generation settings available only in a completion are checked on each B answer;
do not generate an A answer just to populate them.

After independent A review, freeze a separate immutable B manifest containing this
proposal hash, accepted A record/audit hashes, all fit arrays, exact live machine and
effective bases, actual paths/file identities, configuration/suite hashes, source
pins for the reviewed collector/auditor, full plan, limits and analysis constants.
No missing value becomes a wildcard. The current 527 source pins describe the design
baseline, not an executable collector that does not yet exist.

## 4. Stage B population and order

Define two finite suite workloads for each model/family cell:

| Role | Cases and order | Development requirement |
| --- | --- | --- |
| Primary/final-work rehearsal | Its seven holdout cases, in D1 corpus order | Cycle reduction lower bound >5% |
| Calibration/guard-work rehearsal | Its seven calibration cases, in D1 corpus order | Cycle reduction lower bound >-2% |

These roles specify the proposed execution primitives, not completed production
confirmation or existing tune guard coverage. New prompts, other content or different
suite ordering are outside this initial finite scope. There is no general copying
certificate, family/model transfer rule or permission for arbitrary grammar inputs.

Run eight macroblocks, numbered 0..7. At each block:

- Model order is Llama/Gemma for even blocks, reversed for odd blocks.
- Rotate the canonical family order (prose, code, rows) left by block modulo three.
- For each cell, run its primary role, then its calibration role.
- For role index r=0/1, canonical model index m=0/1 and family index f=0/1/2, use
  control then lookup when `(block + m + f + r) % 2 == 0`, otherwise reverse.
- Execute all seven cases of one suite arm before the other arm, using a new load
  for EVERY case. Preserve the same case order in both arms. There is no shared KV
  state or warm request within a suite.
- Rest 30 seconds after each cell's two role pairs. Rest is scheduled, not a response
  to a favourable timing, clock speed or temperature. Retain clocks/conditions as
  context; do not discard candidate-caused heat or select blocks by speed.

Thus each cell/role has four AB and four BA blocks. Model positions balance four/four;
family rotation is not claimed to be perfectly balanced in eight blocks. Full plan:
6 cells x 8 blocks x 2 roles x 2 arms x 7 cases = **1,344 answer loads**, in 192 suite
arms / 96 paired role blocks. With Stage A, the maximum is **1,348 loads**. Normal
completion has 2,688 B power checks plus eight A checks. No extra repeat or pilot is
allowed. The 48 scheduled cell rests total 24 minutes in a complete run.

The immutable proposal contains the full 1,344-row order, not just this algorithm.
A later run's source/run/model/cell/block/role/arm/case IDs must match it exactly.
Duplicate, missing, unknown, reordered or mixed-run observations never select a
favourable member. Every planned slot remains executed, refused or explicitly not
run with its cause; missing rows do not silently shrink an average.

## 5. Native evidence and whole-answer gates

Every attempted answer needs current ready and release observations; explicit actual
generation settings; complete text and incremental generated IDs; terminal stop:true
with stop_type eos; truncated false; no protocol error; positive valid durations;
and a valid slot ID. Both native prediction counts must agree with the complete ID
list under the reviewed b11139 contract. A terminal empty tokens array is not an
empty total list. Preserve chunks/terminal fields needed to audit that distinction.

Both prompt-work counts must equal the separately tokenized prompt, and cache_n must
be integer zero. Occupancy (`tokens_cached`) is not reused work. A requested fresh
flag, completed HTTP connection, answer text or timings cannot replace these proofs.
Check effective seed, temperature, output cap, ignore_eos and actual speculative
type; preserve job130's treatment of inert none in its supported type representation.
One-sided/malformed draft counters are invalid. Missing both does not prove positive
draft work; each primary macroblock must contain observed accepted drafting before
that cell can receive an attributed lookup-benefit recommendation.

Use D1's unchanged whole-answer oracle on EVERY answer. Natural EOS and explicit
no-truncation are mandatory. There is no first-line judge or prefix comparison.
Additionally require byte-for-byte text equality and exact generated-ID equality
between the two arms of every case/block, even if both answers separately pass D1's
newline equivalence. Different correct outputs fail this conservative preservation
criterion. No new semantic inference, prompt repair or output normalization is used.

On the first observed task-incorrect answer, stop that cell after its current load
is safely released; mark all its remaining slots not-run/futility. A wrong control
means task unqualified. A wrong lookup with a known correct paired control is a
candidate regression. If lookup ran first, record control-unobserved rather than
inventing a regression attribution or running an extra diagnostic. Pair equivalence
failure also stops that cell. Other registered cells may continue after clean release.
No failed cell can qualify from its surviving rows or be retried in this run.

An operational/evidence breach is different: identity contradiction, unknown required
work proof, malformed counters, unexpected prompt reuse, power failure/unknown, stream
failure, cancellation or unresolved process/key cleanup stops the ENTIRE study. No
later cell/model dispatches. All cells remain ineligible from that incomplete run,
including favourable earlier prefixes. Known task-futility terminal states alone
do not invalidate complete evidence from another independently registered cell.

## 6. Timing, statistical unit and one final decision

Record integer monotonic nanoseconds for all execution boundaries. Never round input
values or replace simulated/missing time with wall-clock observations. Preserve wall
timestamps separately. `request_ns` spans submission through completed native reply.
`cycle_ns` spans the per-load gate through verified release, including start/readiness,
ready identity/effect checks, prompt tokenization, generation, individual task judging
and post-release power verification. Start includes native initialization and its
internal empty warm-up; none of that work may be subtracted. It is not just the
engine's decode time.

The primary measured scope is this instrumented fresh-request cycle after run-level
protocol validation. Initial whole-job hashing/collection, pair comparison, output
publication, rests and offline auditing remain separately timed costs. They are not
hidden, amortized into an invented production gain, or claimed free. No uninstrumented
ordinary-start or complete deployment-speed claim follows.

For cell c, role r and macroblock b, use the sum of its seven `cycle_ns` values per
arm: `x[c,r,b] = 1 - sum(lookup) / sum(control)`. All seven positive complete values
are required. This is a fixed one-of-each task mix; it does not reweight whichever
cases completed. Cases are not independent statistical replicates. The eight complete
macroblock effects are the units, with no rounding or partial-block pooling.

There are twelve registered one-sided statements: six primary improvements and six
calibration noninferiority checks. Use existing `stats.block_bounds` with
`alpha = 0.05 / 12` for each, df=7, once after all eight blocks. Primary lower must
be STRICTLY above 0.05; calibration lower STRICTLY above -0.02. Equality fails.
No additional look, extra blocks, alpha recycling from failed cells or post-hoc
model/family/case subset is permitted. Zero sample variance is inconclusive for
qualification; it is not certainty from repeated rounded timings.

As additional conservative screens, every case's eight performance-pair mean cycle
reduction must be >=-0.02, and the primary suite's pooled reduction must be positive
in both preassigned AB and BA strata. These are registered descriptive screens, not
confidence guarantees for each case or order. Report every case's raw effects and
costs; no per-case significance, noise-baseline or general noninferiority claim is
made. Adding screens cannot rescue a failed main bound.

Use exact ratios of the integer nanoseconds for these descriptive screens: compare
the mean rational case effect with -1/50, and compare pooled order-arm totals as
integers. Equality at -2% passes that point screen; a single nanosecond beyond it
does not disappear through rounding. Retain rational block effects and convert to
finite floats only for the pinned `block_bounds` calculation. Confidence thresholds
remain strict comparisons of that published algorithm's unrounded bounds.

The t bounds assume suitable independent, approximately normal block effects.
Counterbalancing and rests reduce some drift; they do not establish independence or
distribution-free 95% coverage. Report effect distributions/order/clock diagnostics.
An independent review may find assumptions unsupported and leave results inconclusive;
it must not repair them by deleting blocks or searching for a favourable estimator.

These are development criteria, not replacements for any effort mode's minimum gain,
payback, quality, guard, planned-look or final-confirmation rule. D3 must still execute
the same accepted explicit final plan in both modes. Flat, negative, below-threshold,
incomplete or uncertain development results do not grant local-comparison eligibility.

## 7. Cost evidence and its limits

Use the SAME reviewed scoped suite executor for main work and its guard/final-work
rehearsal tags. Measure actual role-arm start/end spans around their seven released
loads, plus per-load and job-level costs. Do not create another measurement or run a
probe solely to manufacture a cost field. The eight role observations are genuinely
measured spans, not aliases of the first task or copied 1.0-second fixtures.

| Cost evidence | Actual source | Interpretation |
| --- | --- | --- |
| Identity | Timed original file/scope/machine validation and ready/release checks | Keep one-time and per-load observations separate |
| Oracle | Individual full-answer validation and paired equality/audit work | Include failures; no guessed constant |
| Control / lookup | Each inclusive released load cycle, with native request/start/stop subspans | Exact model/base/case/arm/lifecycle |
| Guard work | Measured calibration-suite arm span | Only that finite seven-case scope |
| Final work | Measured primary-suite arm span | A lookup-only S0=B rehearsal, not actual tune confirmation |
| Cleanup | Actual release span and owner evidence | Retain failures/censoring; unresolved is not a finite clean sample |

Inclusive cycle and suite spans contain subspans. Store containment/sample IDs and
intervals so forecasting never adds them twice. Global validation/output/rest costs
remain in the total ledger. Failed or right-censored operations are retained, never
turned into short completed costs. The performance calculation uses only the
registered cycle values; the full ledger still reconciles to observed run time.

D1 `CostSamples` describes individual cases and cannot faithfully identify a seven-
case role, arm or nested timing interval. The future D2 evidence envelope therefore
needs a reviewed versioned role-cost record with ordered suite IDs/case hashes,
arm/base/request/executor/source hashes, raw boundaries and measured/derived status.
Do not name a suite sample after its first case or copy derived numbers into
apparently measured `guard`/`final` fields. D1's component-presence flag is not proof
of a forecast or production budget coverage.

Publish distributions/counts/min/median/max and the full raw samples, including large
costs. Eight observations do not certify a percentile or worst-case deadline. No
forecast confidence or safety factor is invented here. A future D3 reserve needs a
separately reviewed estimator with uncertainty and all actual role/caller overhead.
Missing comparable evidence makes that operation unaffordable/unknown, not free.

This study covers one base per model and lookup-only S0=B role work. A changed core
base, different S0, another guard/workload, ordinary warm harness or changed executor
is outside the measured envelope. D3 must skip unsupported plans, preserve every
existing workload role, and reserve actual final work before optional dispatch. D2
cannot certify unimplemented mode planning or caller overhead. Cost insufficiency can
withhold any eligibility recommendation despite accepted quality/performance results.

## 8. Operational limits, preservation and stops

Use one native backend resource owner for the whole run; no second cleanup owner,
resident adapter workaround, overlapping model or unmanaged process kill. Preserve
pending PID/credential ownership even on failed/late starts and interrupts. Existing
backend cleanup/confirmation may retry held resources after a stop, but never retry
generation or resume dispatch. Clean means no current/unconfirmed owned process and
no held/undeleted key. A process-only clean flag cannot hide a password file.

Require accepted Windows AC evidence before AND after every attempted load, including
Stage A and unsuccessful attempts. Initial battery/unknown may wait without creating
a backend; after a load has begun it stops the run. Power waiting is not a successful
trial or a request to resume after failure. Freeze helper/source versions in software
review; do not replace unknown with an inferred mains state.

Propose 120s monotonic deadlines for readiness and for the complete generation call.
An HTTP socket inactivity timeout is not that wall deadline: progress chunks must
not extend it indefinitely. The future collector must bound I/O by remaining time,
reject late replies and deliver expiry through the existing backend cancellation/
stop owner. Software tests must cover a dribbling stream and a late valid EOS reply.
Keep accepted release ownership and its timeout settings (unwind 30s, exit 10s,
previous-gone 30s), recording exact values in the reviewed software manifest. No
competing cleanup owner or direct PID kill is added. A timer never establishes actual
process exit or credential removal; unresolved resources remain held and stop dispatch.

Stage B has a 12-hour soft wall target, including waits/rests/overhead, checked before
each paired role block. A started block may finish; a hard stop/evidence breach still
interrupts it under the existing owner. Target expiry prevents the next block and
makes the run incomplete, with no statistical look on a prefix. No claim of a hard
12-hour deadline is made. Runtime is unmeasured: the 1,344-load schedule is many times
job130's 80 loads and may require several hours. No 2-3-hour promise is imported.

Checkpoint a unique owner run ID, phase, immutable governing hashes, exact next slot,
all replies/evidence/costs and terminal cause. Refuse existing output and mixed/prior
records; never append/resume, reuse earlier conclusions or overwrite files. Ctrl+C
retains partial data through the existing report owner and is re-raised. No dispatch
follows a stop. Missing requested slots remain visible with their reason.

Retain the actual before/after relative-path, size, mtime and SHA maps of the protected
data folder, plus saved-setup file bytes/hash/identity, not just a boolean fingerprint.
Results/output cannot lie in or reach protected roots through links. Known input
contradictions refuse before construction; ready contradictions before generation;
released contradictions before another load. Check frozen generations at boundaries
and full hashes at initial/final verification. Preserve all raw startup logs, selected
commands without secrets, image/library observations, process identities and exits.
File checks are bounded observations, not locking or loaded-instruction attestation.

## 9. Required software review and eventual conclusions

Before either live authorization, require caller-level tests of wrong/stale model or
saved identity and alternate executable paths (zero constructions/starts), incomplete
or contradictory ready libraries (zero generation), released contradictions (no next
load), changed source/input generations, and the current lifecycle race regressions.
Also require no-generation A, tokenizer/cap/context boundaries, all source/fit hashes,
EOS/chunk/count/native-settings contracts, future suite-cost serialization and interval
accounting, duplicate/missing/mixed-run rows, both orders, output preservation,
interruption/power/cleanup boundaries and the one final analysis look. Re-run D1's
omission/foreign-model/numeric regressions. Replay old job128/129/130 conclusions and
ordinary tune/history parity when shared executable code changes.

The A software package must include a field-by-field observer/source matrix and
missing/contradictory-default tests before hardware authorization. Historical logs
may test parser behaviour; they cannot supply this run's live effects or machine
facts. An unimplemented observer is a software readiness gap, not a reason to launch
an exploratory answer to discover it. Include alternate data/store roots and
dribbling/late HTTP replies in zero-dispatch/deadline caller tests.

Do not monkeypatch job130's TOKENS=128, weaken its limit/EOS rule, change D1's oracle,
or reinterpret old records as complete-task qualification. A new collector/auditor
and versioned evidence contract need independent software review. The proposal's
model-free plan/reference checks are design checks, not those implementations.

Each cell reports every gate separately: operational completion, task/control status,
exact off/on preservation, accepted draft work, primary/calibration statistics,
descriptive screens, cost coverage and review disposition. DONE alone is not success.
Wrong controls remain task-unqualified, wrong candidates remain failures, missing
work remains incomplete, and statistical uncertainty remains inconclusive. Do not
make any favourable historical/tuning record from these states.

Even a fully accepted positive result only recommends this exact finite local scope
for a future opt-in comparison, conditional on adequate costs and implemented matching.
All current draft permission flags stay false. D3 still needs both mode callers,
real native evidence transport, shared final workload-role plan/reserves/verdict and
fresh actual-setup confirmation. D4 still needs a scope-enforcing consumer and an
end-to-end deployment study. Nothing here activates ordinary tune, save or start.
