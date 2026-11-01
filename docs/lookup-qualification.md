# Lookup qualification: D1 software inputs

D1 implements complete-task software fixtures, strict development-draft bindings and
a scripted report. It implements no hardware collector, accepted qualification,
automatic tune path, final-mode integration, saved profile or deployment consumer.
The governing [accepted admission design](lookup-tuning-admission.md) is unchanged.
The only maintained handoff is `reviewer-implementation-handoff-current.md`.

## Complete tasks and exact answers

`lookup_corpus.build_corpus()` constructs 42 deterministic typed cases: labelled prose
copying, code copying as text, and even-row extraction. Each family has 14 cases;
calibration and holdout each have 21. The main grid covers three lengths and repeated
versus varied content; six additional short-output cases exercise intended low lookup
opportunity. Those labels are input strata, not observed engine behaviour. All cases
are mandatory in the software report. The corpus is public, not a secret holdout.

Corpus version: `lookup-task-corpus-1`. Oracle: `lookup-task-oracle-1`.
Full ordered corpus SHA256:
`363d3870ad21b072a18cc269fda1cf325a641e28e5fb3a37039910d3840c21ea`.
Hashes include typed inputs/metadata, the exact rendered prompt, complete expected
answer and oracle version. Duplicate IDs or prompts, including across splits, refuse
construction. Changed content or case order produces a different hash. Job130 prompts
are not reused as this corpus.

Expected answers derive from the input, not a model's baseline answer:

| Family | Complete expected answer |
| --- | --- |
| Prose | Every numbered Record label and payload, in order |
| Code | Entire supplied source, preserving indentation, literals and Unicode |
| Rows | Every even-numbered input row, unchanged and in input order |

Input is rendered as a JSON string explicitly treated as data. Source is never
executed. Typed validation refuses ambiguous record boundaries, duplicate row IDs,
control characters and row delimiter injection. This version requires at least one
selected row; an empty-result task is outside its grammar. Code data uses LF with no
terminal newline. These bounded types are a task definition, not a general code or
natural-language parser.

The judge compares the whole answer. CRLF and LF are equivalent, and one optional
terminal newline is allowed. It does not strip whitespace, labels, code fences,
digits, signs, Unicode or punctuation. A second terminal newline fails. So do missing
labels/rows/tails, repeated records, refusal and explanatory text. Natural `eos` and
explicit no-truncation evidence are required even if a limit-ended answer matches.
The caller supplies those observations; D1 has no real native evidence collector.

No tokenizer has verified these tasks or the draft's illustrative 512-token ceiling.
D2 must prospectively check input/output/context fit and actual low-opportunity
strata, then review and freeze its own population and analysis before data. Forty-two
software fixtures are not a hardware sample-size prescription or correctness proof
for other tasks. A wrong control is reported as task unqualified, never discarded or
accepted because lookup repeats its mistake.

## Draft schema and matcher

`lookup_qualification.QualificationDraft` is strictly **development-draft**, with
independent review **pending**. An accepted disposition cannot be encoded. Nested
records use frozen models, immutable tuples, strict numbers/types, forbidden unknown
fields and complete explicit configuration fields, including null defaults.
The loader refuses duplicate JSON keys, nonfinite numbers and numbers that would be
rounded by this format. At float-valued fields it checks integer conversion before
Pydantic can round it: `9007199254740993` refuses, while exactly representable large
integers remain valid. Integer-valued identity fields stay exact integers. Python
numeric wrappers such as Decimal/Fraction are not accepted as float inputs; JSON
integer/float values are required. Decimal/exponent JSON tokens retain the existing
lossless decimal round-trip check. Hashes use complete validated serialization,
without rounding or depth truncation. There is no profile migration or
accepted-certificate reader.

Use `parse_draft()` for raw saved JSON. A generic JSON/Pydantic decoder can already
erase duplicate keys or numeric precision before these checks receive a model;
matching and hashing cannot recover that discarded information.

The binding describes supplied facts about:

- Exact model/tokenizer, full corpus/oracle and complete requested/effective base.
- Machine-local CPU/topology/instructions/RAM, GPU/VRAM, selected Vulkan device,
  driver, OS build, hostname and power plan. Initial shape requires one unambiguous
  usable GPU; PCI identity is explicitly absent when unavailable, never invented.
- Actual selected executable and all five engine components required by the existing
  lookup auditor, with exact paths/hashes. Duplicate library names refuse. Pin order
  is canonicalized; alternate executable paths remain different even with equal bytes.
- The native, fresh-load, one-request-per-arm lifecycle; one active request, bound
  slot/context layout, explicit off/reuse/seed/sampling/output/instrumentation controls,
  and fixed lookup lengths 12/16. Chat, warm state, SWA variants and concurrency do
  not fit this draft format. Four slots do not mean four active requests.
- Provenance hashes and raw positive cost samples by component and task. A missing
  protocol/record/audit hash remains missing; the fixture does not invent real ones.

`match_draft()` is a pure comparison of supplied facts; it performs no scan, process
operation, file inspection, model work or estimation. It and the draft hash revalidate
even models made through unchecked copy/construct methods. Before any serialization,
they check the original nested configuration's explicit-field provenance on both
inputs: omitted fields refuse, including omitted null defaults. A complete explicitly
supplied null remains valid. The check also covers a foreign model exporting the
same configuration fields; changing the model class cannot supply omitted defaults.
Original numeric values and forbidden extra assertions
are retained for validation rather than silently coerced or dropped by a model
serializer. A caller that has already discarded provenance by exporting an unchecked
object cannot recover it from those exported values; this remains supplied-data
validation, not authentication of origin. Its outputs deliberately separate:

| Field | Meaning |
| --- | --- |
| `same_scope` | Supplied scope values and full corpus identity match exactly |
| `supplied_effects_complete` | Required supplied effective values are present and do not contradict explicit requests; not an attestation of real observation |
| `has_each_cost_component` | Each named component has an in-corpus sample; not adequate coverage, a cost forecast, percentile or budget guarantee |
| `tuning_allowed`, `deployment_allowed` | Always false |

Shared/per-side KV precedence uses the existing configuration semantics. Requested
null defaults are bound separately from their supplied resolved effects. Unknown
effects and ambient configuration are reported, not treated as wildcard compatibility.
Revocation and costs for a different task are visible problems. All bindings can still
be fabricated by a caller: the matcher does not authenticate their origin or inspect
resident instructions. The distinction is intentional; D2 needs reviewed raw evidence,
live identity gates and an independent audit. No current model/cell becomes eligible.

## Software report

```powershell
.venv\Scripts\python.exe -B scripts/evaluate_lookup_qualification.py --simulate --out _jobs/out/NEW_lookup_qualification_SIMULATED.json
.venv\Scripts\python.exe -B -m pytest tests/test_lookup_corpus.py tests/test_lookup_qualification.py
```

The runner exposes only `--simulate` and `--out`; there is no model/engine/real-run
option. It scripts both arms from the independent expected output and uses a clearly
fabricated machine/file/cost binding. It measures no latency, collects no native IDs
or counters, and makes no performance conclusion. Its report explicitly denies
hardware qualification, token-fit verification, tuning admission and deployment.

All 84 scripted rows and 42 cases remain in the result. Wrong controls/candidates,
duplicate/missing/unknown/reordered rows and changed case identities make the report
fail; a favourable duplicate is never selected. Two task-correct but raw-different
answers are reported as different, not a matched pair. This preserves the distinction
between absolute task correctness and the later exact native off/on requirement.

Publication serializes first and uses exclusive creation. Existing or concurrently
created output files are never overwritten; no partial file is resumed. Exit 0 means
all software fixture pairs pass, 2 means fixture failure, 4 refusal and 7 publication
failure. A failed write can leave an incomplete diagnostic file; it is not qualification
evidence. Use fresh names, including after an interrupted write.

## What follows

Independent software review comes next. No D2 hardware wrapper or protocol is prepared
by this batch. D2 must freeze the actual complete-task population, tokenizer envelope,
correctness/performance/cost rules, identities, power/cleanup and data-preservation
protocol before separately owner-authorized data collection. No job130 gain is imported.

D3 still needs the accepted explicit `ScopedFinalPlan`, both-mode caller tests, real
first-request transport, reserve/planned-look accounting and the common final verdict.
An ordinary warm-up/repeated-warm harness cannot simply receive a lookup flag.
Saving or serving lookup remains D4 work with an enforceable scope and demonstrated
end-to-end benefit. This software does not establish any of those later permissions.
