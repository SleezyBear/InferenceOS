# D2 Stage-A fit-only software

Software: `lookup-fit-2`, manifest: `lookup-fit-manifest-2`.

This implements the four-load Stage A with the prospective version-2 amendment to
the [D2 protocol](lookup-qualification-protocol.md). The original version-1 protocol
was accepted; the reviewer accepted the prospective warm-up clarification and the
slot/process-freshness repairs. The remaining live command-capture repair awaits
independent software re-review. No live
manifest or job is delivered, no engine has been run for this batch, and hardware
authorization remains separate. Stage B, its answer collector, role-cost format and
forecasting method are future work. No qualification, tuning or deployment permission
can come from a Stage-A record or its audit.

## Entry and ownership

[collect_lookup_fit.py](../scripts/collect_lookup_fit.py) reads two externally pinned
JSON inputs: the version-2 proposal and a later reviewed Stage-A manifest. The manifest
must bind the proposal/document/corpus digests, both exact model/configuration/setup
identities, all current Python sources and engine files. It cannot change the corpus,
base or engine pins from the proposal. The proposal must retain the exact version-2
initialization policy, including JSON types; omitted or contradictory policy refuses.
Version-1 manifests/records are not silently reinterpreted. Every configuration field, including
nulls, remains explicit. D1's accepted recursive provenance check runs before model
serialization, including on constructed/copied inputs.

Real collection additionally requires the operator's `--authorized-stage-a`
acknowledgement of **separate owner authorization**. This option is not itself review
or permission. No actual live invocation or manifest is supplied here. The collector
has no Stage-B option and cannot continue into generated answers.

Preflight checks both model hashes and saved model/setup identities, the actual
selected executable path/hash, complete Python population, engine files, governing
documents, canonical data root and absence of ambient engine configuration before
constructing a backend. Identical executable copies at other paths are refused.
Unknown inherited engine overrides, alternate data roots and input links/junctions
are refused. Sources/models/store/documents are checked again at boundaries; full
hashes and retained data maps bracket the run. These are bounded observations, not
file locking or resident-instruction attestation.

One native backend owns every process and credential across all four loads. The
collector owns only traversal and the report. It exposes `start`, `observe`,
`tokenize` and `release`; there is no generation method. Its new HTTP inspection
client admits only `/slots` and `/tokenize`, with explicit add-special semantics.
Health requests remain in the existing backend's startup lifecycle. No prompt is
submitted to completion/chat and no client warm-up or discovery answer is dispatched.

**Prospective warm-up amendment:** version 2 forbids extra client warm-up, priming
and discovery requests, retains the pinned engine's internal startup empty run and
its complete current log, and requires Stage-B cycle time to include initialization.
It cannot subtract startup warm-up or reuse a process between answers. The launch
default is unchanged. A missing current empty-run log marker or an explicit
`--warmup`/`--no-warmup` override refuses live and offline. The API spy establishes zero completion/chat requests, not zero
internal decode work. Historical evidence has 160 snapshots across 80 job130 loads;
it is not a current Stage-A observation. The original version-1 document/proposal and
all earlier packages remain preserved. New protocol/document/proposal hashes bind
this amendment before any live manifest; hardware authorization remains absent.

## Live observer matrix

The raw startup text, loss metadata, process identity, mapped backing files, slots
and native inventories are retained. Stored projections/pass labels are recomputed
by the auditor and cannot stand in for the sources below.

Command evidence contains the actual launch executable followed by the existing
credential-redacted arguments. The collector copies the backend's launch command
once inside a bounded read; it never substitutes the selected path for missing or
contradictory launch evidence. The ready gate compares that captured executable
with the independently selected executable before tokenization. The shared display
redactor keeps its original argument-only contract for all other callers. Tests
exercise the production command builder, live observer and collector together with
fabricated IO, including both arms and absent/wrong executable evidence.

| Field | Required current source | Unknown/contradictory treatment |
| --- | --- | --- |
| Generation threads | Postprocessed `system_info` and actual threadpool-init log | Conflicting counts refuse; no CPU-topology estimate |
| Prompt threads | Postprocessed `n_threads_batch` in `system_info` | Missing refuses; no copy of requested `-tb` |
| Batch / microbatch | `llama_context` initialization's `n_batch` / `n_ubatch` | Positive consistent counts required |
| Offloaded layers | Actual `load_tensors: offloaded ...` log | Request/command is not an observation |
| Context and slot count | Startup initializes four slots plus four distinct `/slots` entries, IDs 0..3 | Counts must agree; every slot idle and 4096; missing/conflicting/busy refuses |
| Speculative activation | Startup proof plus each slot's literal `speculative` boolean | True only in lookup, false in control; missing/numeric/string/mixed/contradictory values refuse live and offline |
| mmap | Tensor loader's actual `load_mode` | No inference from `--no-mmap` absence or memory use |
| Flash attention | Resolved fused-operation enabled/disabled message | `flash_attn = auto` alone is unknown |
| Device | Actual model-device selection log, matched to one native Vulkan device | Ambiguous devices or mismatched indexes refuse |
| K / V formats | Every actual KV-cache allocation log | Mixed formats conflict; no requested-format echo |
| Optional aliases/overrides | This proposal requests null shared-KV/placement/environment overrides; per-side settings are observed separately | Custom aliases/overrides require another reviewed proposal |
| Machine | New native CIM, CPU feature flags, physical RAM, Vulkan enumeration, selected GPU's uniquely matched OS driver, active power-plan GUID and AC evidence | No old inventory, guessed topology, unknown driver or model-name transfer |
| Engine identity | Current PID/create time and complete required mapped-library hashes, selected path, current complete startup log and explicit offline/algorithm command | A previously observed PID/create-time pair refuses at readiness before tokenization; missing/lost/stale/contradictory evidence also stops |

The slot contract was checked against pinned upstream
[b11139 server context](https://github.com/ggml-org/llama.cpp/blob/b11139/tools/server/server-context.cpp):
its slot JSON exposes the slot's ID, capacity and boolean processing state. The
collector accepts idle boolean `false`, never numeric zero or a missing field.
The same upstream slot JSON exposes `speculative` through `can_speculate()`, which
checks the slot's speculative context while idle too. This proves initialized
capability, not accepted draft work or a performance gain.
Activation is validated separately from the base configuration: its intentional
difference between arms must not look like unwanted settings drift. Process
freshness uses earlier retained snapshots, with no second mutable identity registry.
Historical job130 logs informed parser tests only; no historical observation fills
a current missing value. Real default/inspection availability remains unproven until
independent software review and an authorized Stage-A run.

## Bounds, cancellation and preservation

Readiness has a 120-second monotonic deadline; individual inspection/tokenization
operations have 30-second limits. HTTP reads also enforce a total deadline, remaining
socket timeout, response-size ceiling and a socket shutdown backstop. Redirects,
unknown endpoints and malformed JSON are refused. Late valid replies cannot pass.
Credential-file and process-identity reads are inside the inspection deadlines too.
Governing JSON reads are capped at 16 MiB; record/output reads at 64 MiB. Decimal and
exponent numbers that would lose value during float decoding are refused explicitly.
The collector waits up to two seconds for an already-written readiness log marker,
without sending an extra request.

OS/file/native observation calls run in bounded read workers. A timed-out daemon may
finish its read later; Python cannot forcibly interrupt an OS call safely. It cannot
publish late evidence or control an engine, and expiry stops the whole study. Startup
uses the existing backend's atomic admission predicate with both cancellation and
deadline checks. A late startup worker cannot obtain new admission after expiry.
Abort uses the same backend's persistent halt and release ownership, never a direct
PID kill or second cleanup owner. Unwind/exit/previous-gone timeouts are unchanged.

Every attempted load retains pre/post AC checks, partial replies and release evidence.
The protocol's eight A power checks are the load boundaries. This collector also
reads AC while collecting each of its eight machine inventories, plus one admission
check before construction: **17 power readings on a complete real CLI run**. The
record/auditor distinguish boundary, inventory and admission counts; these extra
safety observations never replace a boundary reading or select a timing subset.
The first unknown/contradictory observation, capacity failure, changed tokenization,
power failure, cleanup failure or interruption stops further dispatch. A failed
release measurement closes admission even when liveness looks clean. Process exit,
unconfirmed identities, current password files and undeleted passwords are all checked.
Final held-resource retry uses only the existing backend's cleanup/confirmation;
successful retry does not turn a failed load into a completed one. Interrupts retain
their exception object after reporting/release; a new interrupt may interrupt cleanup.
An ordinary cleanup or post-load power error cannot swallow a pending `SystemExit`
or `KeyboardInterrupt`. Both causes remain recorded and later loads remain not run.
Release evidence requires literal boolean `true` for both process and credential
release; numeric `1` or `1.0` is invalid in the collector and offline auditor.

Output is exclusively reserved, then atomically checkpointed. Existing output and
detected replacement of owned output refuse further publication. Output is protected
from reaching data, models, inputs, source or documents through an alias. A `done`
checkpoint is withheld until preservation checks finish. A write/serialization failure
cannot hide an already captured interrupt. Output/path checks are observations, not a
filesystem transaction against a hostile concurrent writer.

Actual before/after data maps and saved-setup bytes are retained. The offline auditor
recomputes map equality, byte/hash consistency, chosen saved entry/model/configuration,
frozen population and selected executable binding. A boolean `unchanged` is insufficient.
Before/after data maps must match as typed canonical JSON; Python's equality between
`false` and `0`, or `true` and `1`, cannot stand in for unchanged evidence.
An unrecorded transient write cannot be ruled out by equal endpoint maps.

## Population, simulation and audit

Each load tokenizes all 42 prompts with add-special true and each distinct canonical
expected-answer form with add-special false. Deduplication retains form labels.
The unchanged corpus currently produces **200 requests per load / 800 total**.
All ID arrays and hashes are kept. Capacity checks use the protocol's fixed margins;
between-arm token IDs must match on the same model. This is a capacity screen, never
proof of natural EOS or correct generated answers.

Software-only example (choose a fresh output):

```powershell
.venv\Scripts\python.exe -B scripts/collect_lookup_fit.py --simulate --out _jobs/out/FRESH_fit_SIMULATED.json
```

Simulation fabricates processes, machine inventory and token IDs. It constructs no
backend and measures neither real token fit nor performance. Its audit remains
incomplete for live evidence. Exit 0 describes completed fixture collection only.

[audit_lookup_fit.py](../scripts/audit_lookup_fit.py) is a separate read-only tool.
It requires the original record plus byte-pinned manifest/proposal, refuses existing
output, reconstructs the exact four-load/200-request populations and independently
checks raw observations, token fit, process freshness, both-arm equivalence and
preservation. Duplicate/missing/reordered/mixed-run evidence is never selected away.
Inputs are re-read before publication; a changed record, manifest or proposal refuses
the audit. The record also lists every remaining slot as not run when a stop occurs.
It creates no Stage-B manifest and every permission flag stays false, including on a
complete audit. Independent acceptance of actual A results is still required.

The adversarial [tests](../tests/test_lookup_fit.py) use fake engines, synthetic GGUF
files and local fake HTTP responders only. Existing tune/search/history execution is
unchanged. Broad software checks, type checking and replay evidence are recorded in
the same maintained implementation handoff and the sealed batch artifacts.
