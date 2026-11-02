# D2 Stage-B native collector software candidate, revision 4

The Stage-A results are independently accepted. This batch repairs the rejected
software core and implements the prospective native collector for review. It creates
**no live manifest or hardware job** and runs no model. Software acceptance, frozen
input review and separate explicit owner authorization remain prerequisites to any
native study. There is no new tune path, saving route, default or automatic admission.
The sole maintained handoff is `reviewer-implementation-handoff-current.md`.

## Repaired evidence rules

Every required operation's outcome is reconciled. Failed pair comparisons,
publication, rests or other operational leaves invalidate completion and suppress
positive analysis. A suite failure is permitted only when derived from actual cell
futility. Each rest consumes a unique, complete, ordered interval of at least 30 seconds.
Admissions follow all preceding assigned pair/rest work, including skipped cells;
stale timestamps cannot admit work after the 12-hour soft target. The native origin
is captured before initial output/validation, rather than reset at the first answer.

Recorded stop/analysis/publication failure markers cannot support completion, even
when a forged status says `done` or a marker is empty/null. A genuine soft-target
stop remains auditable as incomplete with zero analysis looks.

Intermediate native settings, counters, slots and endings cannot contradict final
work. The pinned engine's absent/-1 partial slot is unknown; concrete slots agree
with the terminal. Accepted draft counts are not given an invented emitted-ID bound:
b11139 accounts an accepted batch before walking it through EOS. Missing evidence
never supplies a gain. Invalid audits return incomplete, zero-look results.

## New native components and ownership

- `lookup_answers_manifest.py` defines strict `lookup-answer-manifest-1`. Six exact
  original artifacts bind accepted protocol/proposal, A manifest/record/public audit
  and independent acceptance. The original A audit is recomputed with its original
  source pins. Current B source pins are separate: they may truthfully describe
  reviewed changes, renames or removals of historical source files. There is no
  A-to-B source-hash equality requirement or special exempt-file list. A derived
  adapter feeds the existing Preflight, which fully validates the current source
  population and bytes, then retains its existing per-load file-generation and
  population checks. Final preservation rehashes the full map. These observations
  do not lock files or attest resident instructions. Stale historical pins cannot substitute for
  execution pins. Actual model/saved-setup/selected-executable/source/environment
  and all six historical artifacts are checked before construction and before loads.
  Nested explicit-field provenance is retained. A pinned external review statement
  is not a cryptographic signature or owner authorization created by the parser.
  The shared fake-native fixture now changes an existing source after capturing A,
  so parser, preflight, runner, journal and offline audit exercise this distinction.
  Historical A artifacts and source pins are never rewritten to satisfy B.
- `lookup_answers_native.py` composes the accepted LiveFitEngine. The existing
  backend remains the sole process/key cleanup owner. Held resources, cancellation,
  concurrent operations and late startup close admission. No new PID killer,
  lifecycle owner, priming request or measurement-only probe exists.
  One absolute startup deadline is captured before resource inspection and passed
  into the existing owner. Both prechecks and startup consume its remaining time;
  late workers use the same deadline and cancellation flag. No fresh 120-second
  allowance starts after the checks. The shared owner's optional deadline is the
  only pre-core source change; no-deadline Stage-A calls retain their former bounds.
- `lookup_answer_transport.py` bounds observations/tokenization at 30 seconds and
  native requests at 120 seconds total including credentials, connect, headers and
  all chunks. Literal authenticated loopback `/completion` only; SSE framing,
  response sizes/counts, JSON, EOF and HTTP framing are checked. Remaining-time
  sockets and shutdown prevent dribbling from extending deadlines.
- Read workers enqueue bounded immutable messages. Only the calling thread delivers
  chunks, so abandoned workers cannot mutate completed evidence. Arbitrary blocking
  Python callbacks cannot safely be preempted: the caller remains inside them and
  rejects a late return before cleanup. The real callback only captures raw chunks.
  Callback/read exceptions retain their original identity across ordinary close errors.
- The pinned b11139 native endpoint requires terminal JSON, not OAI's `[DONE]`.
  Official source bytes and their hashes are preserved in review artifacts. No
  engine availability or real native request was established by those source reads.

The fixed 1,344-slot executor, 192 seven-case arms, 96 paired roles, 48 rests, complete
D1 oracle, natural EOS, fresh counters, paired exact text/IDs and final eight-block
screens remain unchanged. Whole-job targets include initial hashing/preflight.
A permitted pair may finish across the soft target; operational failures still stop.

| Owner | State and responsibility |
| --- | --- |
| Original accepted Stage-A artifacts | Historical model/base/fit observations and their original source pins |
| `AnswerPreflight` | Current frozen input checks; delegates actual environment/model/executable checks to the existing preflight |
| Existing `LiveFitEngine` and resident backend | Process, password file, pending exit and cleanup evidence; sole release/retry owner |
| `NativeAnswerEngine` and transport | Serial request admission, bounded reads, cancellation and caller-thread chunk delivery |
| `collect_answers` | Fixed suite/pair traversal, complete-task evidence, futility, rest and target accounting |
| `run_native` | Initial validation/power, outer caller intervals and delegation to the existing cleanup owner |
| `OwnedAnswerOutput` | Exclusive result paths, delta journal, same-run memory diagnostics and publication receipts |
| Offline auditors | Pure reconstruction and validation; no engine, saved setup or permission changes |

The existing tuning planners, histories, acceptance helpers and saving routes remain
outside this development collector. There is no second editable tuning queue or cache.

## Durable publication and measured caller boundaries

`lookup_answer_publication.py` reserves fresh result/journal/closure paths exclusively.
There is no resume mode. Hash-chained row/interval deltas flush/fsync after each load,
not 1,344 copies of the growing history. Open intervals can close only with the same
identity/start; closed intervals cannot be rewritten. Output generation and disk
margin are checked before dispatch. Final record and separate closure bind each other.

Completed rows are immutable, including older rows and same-dict mutations. Pending
rows may append observations/errors and fill named lifecycle placeholders, while
every earlier raw observation remains unchanged. The writer retains detached
snapshots before I/O. Writer and replay use the same transition rules: prior stop/
error metadata and admissions cannot disappear, rest/futility identity remains fixed,
and even an open cost interval cannot change an observed failure into success.
Numeric types and signed zero remain distinct.

Replay checks incoming replacements without traversing private prefix rows which
the incoming packet cannot supply or change. The live writer still checks every
detached prior row/cost against mutable producer state. That safety check is linear
per checkpoint and quadratic over the fixed schedule; its actual cost remains
visible in caller timings. Software microbenchmarks are not native cost evidence
or a production forecast.

The same run's memory diagnostic is retained before checkpoint I/O. An interrupted
write can therefore preserve complete raw work in the failure report. It does not
become durable evidence or a successful closure when the journal disagrees. The same
interrupt propagates after delegated cleanup. Before callback entry or when output
cannot be written, only earlier durable checkpoints/memory may remain; no success is claimed.
Ordinary journal-close failures retain an earlier interrupt. A new interrupt during
close retains the earlier failure as context; an ordinary close failure returns a
publication failure code. It cannot silently become a successful caller return.

`lookup-answer-caller-costs-1` measures an exact integer interval tree: initial output,
validation, construction, execution, final cleanup, preservation, main publication
and heavy closure verification. Final analysis/checkpoint are measured children of
execution. Inclusive cycle/suite and caller parents must not be added twice.

**The closure receipt's own publication is explicitly right-censored**, outside the
observed job boundary, rather than invented as zero. There is no certified full
production reserve, percentile or forecasting method. This finite boundary needs
review before live inputs; D3 still requires a reviewed estimator with caller costs.
No uninstrumented deployment-speed claim follows from this instrumented study.

Real A snapshots alone project above 64 MiB for full B. Native record reads/writes are
bounded at 512 MiB and journals at 1 GiB, with a 64 MiB free-disk margin. Size/quota/disk
failure stops dispatch and preserves available diagnostics. Limits do not guarantee
that every pathological engine response can finish. No native budget is inferred
from fabricated simulation costs.

## Entry points for review

Software-only commands remain:

```powershell
.venv\Scripts\python.exe -B scripts/evaluate_lookup_answers.py --simulate --out _jobs/out/FRESH_SIMULATED.json
.venv\Scripts\python.exe -B scripts/audit_lookup_answers.py _jobs/out/FRESH_SIMULATED.json --out _jobs/out/FRESH_SIMULATED.audit.json
```

Prospective native arguments are `--rules`, `--rules-sha256`, `--authorized-stage-b`
and `--out`. No prepared live arguments or manifest are delivered. Native offline
analysis also needs the original rules, journal and closure receipt. It recomputes
original A provenance, preservation, caller costs and every core decision. All
qualification/admission/deployment flags remain false, even for positive development
screens. Independent results review and D3/D4 consumers are still required.

Tests use small fake assets/owners, a complete 1,344-request fake-native caller run,
corruptions, interruptions and local fake HTTP endpoints. Actual Stage-B answer
correctness, EOS, speed and costs remain unmeasured. Existing protocol/proposal bytes
are unchanged. All earlier artifacts, including the first core and its limits, remain
preserved below and in the handoff.

<details>
<summary>Historical revision1 software-only core (superseded availability/implementation status)</summary>

# D2 Stage-B software evidence core

Version: `lookup-answer-core-1`; interval format: `lookup-role-intervals-1`.

Stage-A results are independently accepted. The [version-2 protocol](lookup-qualification-protocol.md)
and its original proposal remain unchanged. This batch implements a **software-only
core**, not a live Stage-B collector, execution manifest or tuning admission path.
Live dispatch is explicitly refused before any engine operation. No model, saved
setup, default, threshold, corpus, oracle, history consumer or current tune loop changes.

The new role-cost format needs review before a native adapter can build on it. The
only maintained implementation handoff is `reviewer-implementation-handoff-current.md`.
Frozen Stage-A sources/results and earlier review artifacts stay preserved.

## Ownership and fixed work

[lookup_answers.py](../src/inferenceos/tune/lookup_answers.py) reconstructs the accepted
six suites and complete 1,344-slot order. Golden digests match the accepted proposal;
normal product tests do not depend on ignored files. There are eight macroblocks,
six model/family cells, primary then calibration, two seven-case suite arms per
role, and the registered AB/BA rotations. All 48 fixed rests remain scheduled even
after cell futility. No rest is selected by a favourable measurement or heat reading.

[lookup_answers_collect.py](../src/inferenceos/tune/lookup_answers_collect.py) owns
only traversal, evidence rows and the interval ledger. Its explicit engine interface
owns start/observe/tokenize/answer/release; it has no native factory in this delivery.
Every case gets a fresh load and release. The same suite executor produces both role
tags; a suite sample is not copied from its first case or an estimated constant.

The soft 12-hour target is checked and recorded before every paired role block.
An admitted pair may finish across that target; it does not bypass individual
operation deadlines. This core's elapsed origin is its collector-body job span;
the future outer caller must include earlier validation and all caller overhead
when enforcing the protocol's whole-job target. That boundary is not implemented.
Late completed starts/observations/tokenizations/answers refuse
at the core boundary. A future adapter must additionally bound blocking/dribbling IO
and cancel through the existing backend owner; this core cannot make a blocking
engine implementation safe, which is one reason native dispatch remains disabled.

## Evidence, task futility and whole-study stops

The evidence scope retains both explicit 14-field requested/effective bases, all
200 fit-array rows per model, model/engine identities, machine projection and executor
identity. Required observed settings cannot be null; aliases/custom overrides are
outside this initial scope. Simulation invents these values and claims no native meaning.
This scope is not the future Stage-B execution manifest or a signed authorization.

Each raw chunk and its integer receipt time are retained before interpretation.
Incremental IDs are accumulated, including a legitimately empty terminal ID array.
No ID array at all cannot be rescued by prediction counts. Both native prediction
counts must equal the returned list. Prompt counts equal separately verified fit
tokenization; cache work is integer zero. Intermediate counter contradictions cannot
be hidden by a valid final chunk. Effective seed, temperature, 512-token cap,
ignore-eos and speculative type are checked, including the pinned engine's inert
`none` prefix. Only natural EOS with explicit no-truncation passes. Missing both
draft counters is not positive drafting; one-sided, null or contradictory counters
refuse. Draft counts are never invented from activation or a requested flag.

An explicitly counted empty EOS answer can be task-wrong when its other work and
positive timing evidence is valid. Missing IDs/counters and an uncompleted stream
remain evidence failures. D1's unchanged whole-answer oracle judges every valid
answer. Both text bytes and generated IDs must also match its current paired arm;
two individually correct but differently formatted answers fail preservation.

A wrong answer or pair mismatch closes that cell after its current load is safely
released. A wrong control is task-unqualified; a wrong lookup is a regression only
when its current paired control is known right. Lookup-first failure reports paired
control unobserved, not an invented attribution. Later slots stay explicitly not run
with their cause. Other cells retain the frozen order and may continue.

Evidence/identity/reuse/power failures, user interruption or unresolved cleanup stop
the whole study. Completed and partial rows remain; no later measurement runs. The
same KeyboardInterrupt/SystemExit propagates after existing cleanup, unless a new
interrupt takes precedence. Cleanup errors do not turn a previous interruption into
a normal success. An interruption or error during final analysis still attempts the
terminal checkpoint, retains all raw work and marks analysis incomplete with no look.
No second process/key cleanup owner or unmanaged PID kill is added.

## Measured costs and explicit unfinished boundaries

[lookup_role_cost.py](../src/inferenceos/tune/lookup_role_cost.py) stores append-only
intervals with unique run/sample IDs, actual integer monotonic boundaries, parent
IDs, outcomes, scope and case references. A role scope contains model content and
engine-file-set hashes as well as model label, family, role,
arm, seven ordered case IDs/hashes, requested/effective/machine/request/executor
hashes and suite hash. It cannot be identified by a single case or a scalar duration.

Cycle scope includes gate/power, start, ready checks, prompt tokenization, request,
individual judging, before-stop checks, cleanup and post-release power. Pair judging
is outside the cycle and separately measured. Seven case cycles sit inside one
measured suite-arm span. Parent/child containment, sibling order, connected ancestry
and complete-case accounting are audited. A completed cycle cannot hide a failed
required operation, and a pair cannot stand in for multiple scheduled pairs.
Inclusive spans must not be added twice;
their unassigned gaps remain visible. Failed and running work is retained, never
turned into a short successful role sample. Distribution min/median/max and raw
samples remain available; medians use exact rational integer nanoseconds.

**No complete job-cost or forecast is claimed.** Initial scope validation, final
analysis and final publication are explicitly right-censored caller boundaries in
this core. The job-root interval covers the collector body, not those missing tails.
The final checkpoint cannot truthfully contain its own completed write interval;
the future outer publisher must supply reviewed closure evidence. Nothing is filled
with zero, copied from another role or treated as free. `forecast_available` remains
false. This incomplete full-job/caller accounting prevents treating these software
records as production budget coverage or qualification, even if task/timing math passes.

## Final analysis and offline reconstruction

[lookup_answers_audit.py](../src/inferenceos/tune/lookup_answers_audit.py) reconstructs
raw work, judges, pairing, futility, exact population, scoped cost ownership and the
final analysis. It does not trust stored right/wrong labels or winners. A stopped
study takes no statistical look and cannot use a favourable prefix. No native record
can be relabeled as an accepted result through this simulation-only auditor.

The final math is the registered eight-block, twelve-statement rule, alpha .05/12,
primary lower >5%, calibration lower >-2%, nonzero variance, exact -2% case screens,
positive primary AB and BA strata and accepted drafting in every primary block.
Case screens use rational effects of original integer durations. Simulation may
pass these mathematical screens; actual qualification/admission/deployment remain
false. These are not tests of statistical assumptions or hardware performance.

Software-only commands, always with fresh output paths:

```powershell
.venv\Scripts\python.exe -B scripts/evaluate_lookup_answers.py --simulate --out _jobs/out/FRESH_answers_SIMULATED.json
.venv\Scripts\python.exe -B scripts/audit_lookup_answers.py _jobs/out/FRESH_answers_SIMULATED.json --out _jobs/out/FRESH_answers_SIMULATED.audit.json
```

No model/engine asset or process is opened by simulation. Native factories, bounded
stream transport, Stage-A acceptance/provenance transport into a real B manifest,
full-job/caller cost closure, live output/checkpoint ownership and actual owner
hardware authorization remain prerequisites for any live Stage-B work. They are
not delivered or waived here. No new hardware job or queue entry exists.

</details>
