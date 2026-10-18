# InferenceOS: full scope, current state and every remaining step (8 Oct 2026; updated 16 Oct 2026)

**18 Oct status:** the owner's current HEAD is `cd56c46`; the whole-project audit
and repairs are uncommitted. Follow-up repairs correct cross-API GPU attribution,
GPU-capability inputs to path selection, automatic-setting comparisons and actual
baseline restoration proof. A reproduced deadline-test race is also repaired. Final
Windows gates pass; the earlier 45-decision snapshot and failed/intermediate runs
are preserved separately. Independent review remains. The maintained local record is
[reviewer-implementation-handoff-current.md](reviewer-implementation-handoff-current.md).
It separates implemented repairs, reproductions, validation and remaining design
limits. Earlier job 128 v8 verification covers the study, not this broader audit.
Job 128 remains in `later`; no hardware run is part of the current audit. Final
Windows gates: **3,071 passed, 3 skipped, 1 deselected**, mypy clean on 460 files,
Practical **231/231 identical**, Unbounded **328 scenarios / 323,916 events / zero
differences** against the preserved entry source with the same corrected synthetic
profile/file-path binding (original mismatch retained; no wider exclusions). The
single handoff records 50 findings/decisions, reproductions, complete review artifacts,
and the distinction between this self-audit and independent acceptance. The audit covers all
existing optimization paths and their interactions; it does not claim that the
bounded search proves a global optimum or that additional Practical paths have
been admitted by development evidence.

**16 Oct status (authoritative; supersedes every older "uncommitted", "not committed" or
"remain unfixed" sentence below, which is kept as dated history):** the owner's commits,
include `582661d` (step 4) and `163a9ab` (step 5) on 1 Oct, then `7a47d3c` (C3-6),
`df01ea3` (C3-7), `47b8d15` (C4-1), `f9d79ca` (C4-2), `b7d1e23` (C4-3) and `329aad8`
(15 Oct: the R1-R5 repairs, accepted by the reviewer after two Windows review rounds;
owner smoke test 126 on a real `start` passed). Nothing listed in this document as
implemented is waiting for a commit at `329aad8`. Work after `329aad8` is described in
the dated handoff `reviewer-prompt-reuse-2026-10-16.md` and is uncommitted until the
owner commits it. The remaining-path inventory (what is implemented, wired into tune,
activation-verified and shown useful) is in that handoff, section 2.

**15 Oct owner commit and cumulative handoff:** owner commit `b7d1e23` contains
C4-3 and its review fixes, following `f9d79ca`, `47b8d15` and `df01ea3`. The complete
local handoff is [reviewer-implementation-handoff-2026-10-15.md](reviewer-implementation-handoff-2026-10-15.md):
all 27 changed files since C3-6, every dated decision ledger, rationale, superseded
choice, validation boundary and retained artifact. This supersedes older
uncommitted/pending-implementation status below. The completed
[whole-project adversarial self-review](reviewer-project-adversarial-2026-10-15.md)
requests changes for five reproduced persistence/serving/reporting defects. They
remain unfixed; owner commits and passing tests are not independent acceptance.

**15 Oct fixes for R1-R5 (supersedes "remain unfixed" above):** all five findings were
reproduced and fixed by a new implementer, delivered to the laptop and not yet committed.
R1: a save refuses to replace a saved-setups file it could not read in full (the file
keeps its bytes; `tune` writes a failure report); unreadable single entries are carried
over unchanged. R2: `start` always stops the engine, even after an error in its startup
report or after the engine died, and reports "clean" only from the stop's own evidence.
R3: legacy `serve --deployment` compares against engine evidence only and lists settings
it could not confirm. R4: generated tune report names carry a random suffix and are
published atomically. R5: key files record their owning process; a live owner's key is
never swept. Checks (Linux cloud copy): 24 new tests (21 fail on `b7d1e23`); broad suite
2,038 passed with the same 22 Windows-only failures as before; mypy clean on 447 files;
Practical 231/231 and Unbounded 328 scenarios / 323,916 events unchanged against `b7d1e23`
(both sides Linux). No Windows run or hardware job yet. Details: ignored
`reviewer-project-fixes-2026-10-15.md`.

**15 Oct C4-3 review corrections (supersedes the delivery status below):** historical
candidate ordering now defaults to **shadow mode**: normal Practical runs record the
proposed order but keep the planner's original order. Application is available only
through an explicit internal development-evaluation flag; there is no user setting
or automatic activation. Activation criteria and demonstrated budget benefit remain
future development-time study work. The known losses remain reproducible when
application is explicitly enabled; shadow mode preserves the control's saves.

The corrected consumer uses only the latest attempt for a comparison within each
run, so failed repeats cannot revive older positive readings. Clean negative
screenings from unconfirmed/interrupted runs can now suppress an old preference;
those run outcomes cannot contribute a positive promotion margin. Three eligible
independent runs remain required for promotion. The ranking rule is version 2;
stored comparison schema version 1 and historical records remain unchanged.
Validation: 514 relevant tests pass, type checks are clean on 446 files, and all
231 no-history Practical replays match the preserved C4-2 reference. The fixes and
validation are recorded in ignored `reviewer-c4-3-fixes-2026-10-15.md`.
Prior audit reports and raw evidence remain preserved. C4-3 is still uncommitted.

**14 Oct C4-3 delivery (supersedes the C4 status below):** the owner committed C4-2
as `f9d79ca`. Practical now uses compatible, goal-labelled historical comparisons
to reorder only the values already admitted by its existing planner. Three independent
recent runs are required. The captured comparison base, candidate set, path order,
revisit rules, current-run measurement cache and fresh final confirmation remain
authoritative. Legacy comparisons without goal identity cannot supply hints.

The initial C4 recording, cost-forecast and candidate-hint consumers are implemented.
This is not a proven generally better planner: in eight paired simulations the old
favourite is measured earlier and both groups save confirmed configurations; in two
tight-budget cases with changed/slower performance, hinted ordering loses a save
opportunity that default ordering retains. These negative results are preserved.
No hardware benefit or production-default change is claimed. The local ignored
`reviewer-c4-3-implementation-audit-2026-10-14.md` records decisions, tests, exact
commands and limitations. C4-3 remains uncommitted and available for independent audit.
Next comes the remaining-path work and development-time studies for the broader
interaction strategy and Practical path admission, in the agreed order.

**14 Oct owner direction and C4-2 update:** the owner committed C4-1 as `47b8d15`.
Continue the original sequence: compatible historical cost forecasts (C4-2), then
freshly measured candidate hints (C4-3), then additional optimization paths and the
broader interaction strategy. Future Practical path admission must be designed from
development-time studies of models, hardware and important combinations, not guesses
or preliminary exploratory tests on each user's machine. Those studies are distinct
from the measurements and final confirmation performed inside admitted paths. Both
future modes should use the same interaction strategy over their admitted paths.

C4-2 is implemented and locally validated, ready for independent audit. It forecasts the cost of a
specific setup/question before starting screening pairs, weak repeats or completion
readings. Three compatible prior runs are required; fresh costs for the same request
take precedence. It reads only the configured history store and never uses historical
scores as acceptance evidence. New observations include actual watchdog allowances
and hashes of the observed loaded Windows engine modules plus the local executable/DLL
bundle. Legacy incomplete observations remain ineligible. Hardware reuse is initially
restricted to an observed setup on a single-GPU Windows host; absent/inaccessible
identity falls back. A separate format-3 cost rule permits differing watchdog caps
only when the total completed measurement fits below every old and new cap. It does
not alter any watchdog or parity-comparison exclusion. The future admission strategy,
C4-3 ordering hints, final-side history reuse and new optimization paths are not in
this batch. See the local ignored `reviewer-c4-2-implementation-audit-2026-10-14.md`
for decisions, reproduction commands, evidence and limitations. Final checks: 251
focused tests pass, mypy is clean on 444 files, and all 231 Practical replays match.
Before the final identity-validation hardening, the full suite passed 2,018 tests
with 3 skips and the Unbounded replay matched 328 scenarios / 323,916 events. The
final hardening has focused regression coverage; the full suite was not repeated.
Earlier dated entries
below describe their historical state; this update supersedes their C4 status.

**9 Oct update:** the first practical-mode hardware evaluation (C2) is complete (jobs 123 and 124, section 8.2). Its follow-up, confirmation planning, is built and simulation-validated; its hardware check (job 125) completed in 14.6 minutes with a passing short-question check (section 8.3); the reporting corrections are done (9 Oct). C3 has started: every tune path has a declaration, and the practical search reads its three paths through them, with behaviour unchanged. In the second slice (9 Oct) the declarations are validated on import, the layers and threads cards carry them in an optional `tune` field, and Unbounded's static tables (`AFFECTS`, `MEMORY_PATHS`, the card-to-quality `REQUIREMENTS`) are computed from them, again with behaviour unchanged. In the third batch (9 Oct, late) the reviewer's validation gaps were closed and each path's engine option and applicability rule now come from its declaration through an explicit registry of the existing rules, with every outcome and reason unchanged. In the fourth batch (10 Oct) a runtime catalog (`tune/catalog.py`) binds every declared path to its existing code, with duplicated bindings removed and behaviour unchanged (section 9.1). On 11 Oct the catalog builder was fixed to check the entries it is given with the same binding rules. The same day (C3-6) Unbounded's option-path loop was moved behind a shared coordinator with a mode-owned `UnboundedPolicy`; the original 431-tune comparison used rounding and timing exclusions later corrected in C3-7. Practical's extraction is now implemented (12 Oct update below). The owner's direction for practical mode is a 10–15 minute soft target. The production default is still undecided.

**12 Oct implementation update:** C3-7 is implemented and awaiting independent audit. Both modes now execute their search dispatch through the shared coordinator: Unbounded uses `UnboundedPolicy`, and Practical uses `PracticalPolicy`. Their ordering, eligibility, evidence rules and acceptance remain separate and unchanged. Practical's 231-scenario before/after check passes with only the new exploration trace and documented volatile fields set aside. The corrected full-precision Unbounded gate passes 328 simulated scenarios plus 107 test-driven tune traces against the preserved C3-6 source. The Windows suite passed: 1,910 tests, with 3 skipped (the existing suite and new tests were run separately). C4, persistent measured costs and tried-value learning, is next after independent review. No hardware was run or queued. The implementation decisions, commands, raw recordings and limits are documented in the local ignored audit handoff.

**13 Oct correction update:** the C3-7 self-attack found and fixed failure-path trace export, missing per-run model/context recording, and missing report-round ordering checks. The stronger format-3 replay matches all 231 Practical scenarios and all 328 simulated Unbounded scenarios against the preserved owner baseline; Unbounded also matches the pre-C3-6 source. The real-clock test-driven recordings exposed one nondeterministic 0.4-second deadline case, reproduced on unchanged source. Separate replays controlling only that test's clock match all 107 owner-baseline and 103 pre-C3-6 tune traces; the original mismatches remain preserved. After-fix checks: 201 focused Windows tests pass, type check clean on 437 files. The 1,910 passed / 3 skipped full-suite result above predates these corrections. Both dispatch policies remain implemented and ready for independent audit; C4 has not started. No commits, pushes or hardware runs were made. The local ignored audit handoff records decisions F01-F08, original failures, commands and artifacts.

**13 Oct C4 design update:** the owner committed C3-7 as `df01ea3` and requested the next step. C4 now has a code-level design; runtime learning is not implemented yet. The finite batches are: (1) durable Practical observations with exact identity and scratch-store isolation, (2) compatible historical cost forecasts with current-run measurements taking precedence, and (3) historical candidate ordering hints that are always remeasured. Unbounded and acceptance remain unchanged. The local ignored handoff `reviewer-c4-learning-design-2026-10-13.md` records decisions C4-D01 through C4-D11, integration points, ownership, missing identity evidence and pass conditions. C3-7 independent review remains pending; the owner's commit is not described as an independent acceptance.

**Who this is for:** the owner, the implementation agent and the independent reviewer. The goal is that all three share one picture of the project:

- what InferenceOS is meant to become;
- what exists today and how well it is proven;
- how the optimization paths fit together, now and in the end;
- what remains to be built, in what order, and what decisions are still open.

**How to read it:**

- Sections 1–2: what the project is and the product the owner has decided on.
- Sections 3–6: what is built today, how a tune works end to end, every optimization path, and the measurement rules.
- Section 7: how the paths affect each other. This is the integration story.
- Sections 8–10: everything that remains, the open decisions, and the target design.
- Sections 11–13: risks, working rules, a glossary of plain terms and code names, and an index of the other project documents.

**Status labels used throughout:**

| Label | Meaning |
| --- | --- |
| **Proven on hardware** | Built, and shown to work on the owner's laptop by a real run |
| **Built, cloud-tested** | Built and covered by automated tests on the cloud copy, but never run on real hardware |
| **Built, partly proven** | Some parts proven on hardware, others not (each case says which) |
| **Designed** | Written down and agreed in principle; not built |
| **Not started** | In the plan; no design or code yet |
| **Open decision** | Needs the owner (often with the reviewer's advice) before anything is built |

**Sources of truth, in order:**

1. the code in the repository;
2. hardware run results (`_jobs\out\` on the laptop);
3. the dated project documents (index in section 13).

Where this document and the code disagree, the code wins, and this document should be corrected.

---

## 1. What InferenceOS is

### 1.1 The purpose

InferenceOS makes local language models run as well as the physical machine allows. It does this by finding and removing avoidable waste in software. The hardware is measured, never changed.

"Better" is not only speed. It covers:

- time to the first word, and writing speed;
- reading speed for long questions;
- loading time;
- memory use and paging (the computer moving memory to and from disk);
- the computer staying usable while the model runs;
- heat, fan noise and battery drain;
- reliability: fewer crashes and clean recovery.

### 1.2 It is not a settings tuner

Settings are one small tool. The intended loop is:

1. **Observe:** what is the machine doing right now?
2. **Profile:** the machine, the model, the workload, and the current system state.
3. **Find the real cause of the waste.** Examples: memory speed, calculation speed, data moving between processor and graphics card, paging, other programs, heat, loading.
4. **Act at the level where the cause lives.** Possible levels:
   - the calculation code (alternative, rebuilt, modified or generated);
   - where each piece of the model runs;
   - data movement between processor and graphics card;
   - memory: what stays resident, duplicates, allocation, swapping;
   - scheduling;
   - the operating system;
   - other programs;
   - storage and loading;
   - combinations of these.
5. **Measure** fairly.
6. **Check answer quality.**
7. **Keep or undo.**
8. **Remember why.**
9. **Keep watching.**

Two principles shape every decision:

- **Situational.** There is no fixed rule like "always save memory" or "always maximize speed". The question is always what is limiting *this* model, on *this* machine, with *this* workload, right now. "Nothing worth fixing" is a valid and good answer.
- **Evidence-proportional.** If one cause is obvious, fix it directly; don't run a huge search. The cost of searching matters as much as the gain.

### 1.3 What is out of scope (fixed rules)

Only three things are excluded:

1. **The model's quantization** (its compression format). It is never changed.
2. **Context size** (how much conversation the model can hold). Nothing may change it, suggest a change to it, or change it indirectly. A separate project will handle context injection; until that is finished, this rule is absolute.
   - **Allowed, because neither changes context size:** reusing the already-processed start of a repeated prompt, and saving and restoring the model's progress around an approved test.
   - **Enforced in code:** every tried setup is checked against the starting context size (`cards/safety.py`, `check_context_unchanged`).
3. **Physical hardware.** Using existing hardware better is allowed; changing it is not.

Everything else that can improve inference through software is in scope. That includes operating-system changes, which need the owner's approval to be kept, with admin rights asked for one change at a time.

### 1.4 The most important architectural requirement

InferenceOS must work as **one coordinated system**, not a pile of separate optimizers.

- A change in one area can invalidate another area's assumptions. For example, moving model layers to the processor changes the best thread count.
- Two changes that each help on their own may not help together.
- So every combination must be confirmed as a combination, and earlier decisions must be reconsidered when something they depend on changes.

The decided mechanism is the **optimization card** (section 2.4). Section 7 describes how far the code has got toward this integration and what is still missing. The honest summary: the measurement and acceptance machinery is shared, but the paths are still ordered by hand-written rules, not by a general coordinator reading card declarations.

---

## 2. The product the owner has decided on

These decisions date from 30 Sep 2026. Later decisions override earlier ones; the full list is in `claude/build-plan.md` and the project memory.

### 2.1 Shape

| Aspect | Decision |
| --- | --- |
| Scope | One machine, purely local. Nothing is shared or sent anywhere |
| Form | An app with a screen. Window or browser page is decided at step 11 |
| Systems | Windows first, Linux later |
| Programs | llama.cpp, Ollama and LM Studio. Models are **always started through InferenceOS**; it never attaches to a model started another way |
| Models | Text chat models, and models that also read images (images come later) |
| Graphics | AMD through Vulkan is required. NVIDIA is added if possible, otherwise later |

### 2.2 Two situations

**Before a model loads: an intense search.** This is the `tune` command today.

1. Test the machine alone.
2. Read the model file.
3. Check past results.
4. Measure the starting point.
5. Search the applicable optimizations.
6. Confirm the result by alternating old and new runs.
7. Ask the owner to approve each change.
8. Save the setup. `start` then uses it.

Tests run freely inside the effort mode's limit and always undo themselves. Only kept changes need approval. On later starts the saved setup is used automatically, after a quick check that nothing important has changed.

**While a model runs: a watcher.** Not built yet; this is step 9.

- **It only watches.** It never changes anything and never runs anything that could slow the model. Its own cost must stay under 2%.
- **What it watches:** paging, memory, processor and graphics use, temperature, responsiveness and request speed, compared against the saved setup.
- **It only recommends fixes,** with one-click apply.
- **When the machine is struggling,** it asks the owner to allow an in-depth test. The request is dropped if ignored.
- **If the owner says yes:**
  1. it saves the model's progress;
  2. it runs a test sized by the effort mode and aimed at the goal mode;
  3. it restores the model;
  4. it asks the owner to approve the results.

### 2.3 Two modes

- **Effort mode** decides only how much time and test cost may be spent: balanced, performance first, minimize cost or custom.
  - It also sets the smallest gain worth keeping: 5%, 1.5% and 15% respectively.
  - A **payback check** applies on top. The expected gain multiplied by expected use (about 10 hours for a setup found before loading, about 1 hour for a change while running) must outweigh the measured test time.
- **Goal mode** decides what counts as better.
  - **Ready-made goals:** balanced, fastest answers, computer stays usable, quiet and cool, save battery. A custom goal lets the owner rank the measures.
  - **Built so far:** balanced, fastest answers and custom. The other three need step 6's new readings.
  - Measures not ranked first act as **guards** that must not get worse.

### 2.4 How optimizations plug in: cards and connectors

**Cards.** Every optimization is meant to be described by one **card**. A card states:

- what it changes, and where (model settings, the program, the operating system, other programs);
- which programs it works with, and how it is set in each;
- when it applies, and whether it needs a restart;
- the values to try;
- what it should improve and what it might worsen;
- which cards it **works with** (their best values shift together) or **clashes with**;
- its quality risk, and how it is undone;
- whether it needs approval or admin rights;
- that it is context safe;
- its measured test cost.

**Connectors.** Each program connects through one **connector**. A connector can:

- start and stop the program;
- read its settings;
- apply and undo a setting;
- read per-request speed numbers;
- discover new options from the installed version. An option it finds is never tested until a card exists for it.

### 2.5 Safety rules that never bend

1. Never touch context size, quantization or hardware.
2. Every test undoes itself, including after a crash. The **undo journal** for operating-system changes is not built yet (step 8).
3. Every kept change needs the owner's yes. A card that pauses, closes or slows another program also asks **before** it tests.
4. Answers must stay good:
   - every change gets a wording check against the original answers;
   - quality-risk changes also get questions with known right answers.
5. Admin rights are asked for per change and never kept.
6. "No change" is a good result.
7. Past results are a starting point, never proof. Every reused setup is measured again.

### 2.6 Learning

- **Remember values,** not just kinds of change. For example: "24 layers helped and 28 hurt".
- **Near matches** are starting guesses, always re-measured.
- **Newest results first.**
- **Record true outcomes.** A change that measured better but wasn't worth keeping is remembered as "worked, not worth it", never as a bad result.
- **Learn how long tests take** on this machine; this feeds the payback check and time planning.
- **Old results** are converted to the new format and kept.

## 3. Where the project is today (8 Oct 2026; practical-mode items updated 9 Oct)

### 3.1 The order of work and its status

| Step | What | Status |
| --- | --- | --- |
| 1 | Repo cleanup (old phase checks deleted, old documents archived, single-step commands kept as advanced commands) | Done, committed |
| 2 | Known bugs fixed (rejected change recorded as improved, failed undo labelled restored, graphics use over-counted, approval prompts never timing out) | Done, committed |
| 3 | The card format (first cards; context-safety check) | Done, committed |
| 4 | Effort mode everywhere, goal mode (balanced / fastest answers / custom), payback check | Done and checked on the laptop; committed (`582661d`, 1 Oct) |
| 5 | `tune` and `start` as everyday commands | Done and proven on the laptop; committed (`163a9ab`, 1 Oct); R2 shutdown repairs committed in `329aad8` |
| 6 | Scorecard additions (usable timer, heat, fans, battery, power; the other three goals) | **Not started** |
| 7 | More model settings, tuned together | **Partly built** (section 5): batch, flash attention, reading threads, engine switches, per-piece placement and conversation-memory format exist. Keep-in-memory and word drafting do not; prompt reuse has a capability and a development evaluation (16 Oct), not a tune path. Tuning together exists only as hand-written ordering and rechecks |
| 8 | Operating-system cards and the undo journal | Not started (three suggest-only cards exist) |
| 9 | The watcher | Not started (an older `monitor` command exists; see 3.5) |
| 10 | Ollama and LM Studio connectors | Not started |
| 11 | The app | Not started |
| 12 | Linux | Not started |
| Later | Image models with sample images; drive, memory and core speed tests in the machine check | Not placed yet (a memory-speed test exists for diagnosis) |

On top of the plan, a large body of work since 3 Oct went into making decisions **trustworthy** and **affordable**. It falls into three areas, each covered in its own section:

- the fair comparison and acceptance rules (section 6);
- a time-limited practical search mode (section 4.4);
- engine stop/start safety and a precise paging monitor (sections 4.6 and 4.7).

**Git:** the owner commits and pushes. The authorized cleanup was completed earlier. **Current baseline (16 Oct): owner commit `329aad8`.** Historical (12 Oct): the baseline then was owner commit `7a47d3c84733b896fc1900cb5526ea87e6277ffa` (C3-6) and C3-7 changes were uncommitted; C3-7 was later committed as `df01ea3`. This implementation made no commit, push or history change, and does not make a fresh claim about remote state. Reviewer/audit files (`/reviewer-*.md`) remain ignored and untracked.

### 3.2 The machine and models

**The laptop:**

- Intel i7-8850H: 6 physical cores, 12 logical.
- 16 GB of memory.
- AMD Radeon Pro 560X with 4 GB of graphics memory.
- Windows; llama.cpp build b11139 with the Vulkan back end; Python 3.12.
- Repo at `C:\Users\madha\Projects\InferenceOS`. Do not use the Desktop or OneDrive copies.

**Study models:**

| Model | Notes |
| --- | --- |
| Llama 3.2 3B | Dense; fits fully on the card |
| Granite 3.1 MoE | Mixture of experts |
| Granite 4 H Tiny | Hybrid: attention plus recurrent blocks |
| Phi-4 mini | Dense; its old saved setup became unsafe for memory |
| Qwen3.5 9B | Only partly fits on the card |
| LFM2 1.2B | Hybrid convolution |
| Gemma 3 1B | Small; nothing helped |
| Qwen2.5 1.5B | Reference for the "fits fully" case |

**NVIDIA machine:** an 8 GB NVIDIA machine was planned for two days (`claude/nvidia-machine-plan.md`). It has not been used. When the owner says it is available, that plan runs. It needs updating first (section 9.10).

### 3.3 How work is done

- **Current implementation workflow (12 Oct):** the owner authorized the former reviewer to implement C3-7 directly in the Windows workspace. Decisions and reproducible evidence are recorded in an ignored audit handoff for another model to review. Earlier batches used a cloud copy and byte-for-byte laptop delivery.
- **Hardware runs** happen only through a job queue. The implementer drops PowerShell scripts into `_jobs\`, and `agent-runner.ps1` (started by the owner) runs them one at a time.
  - Output goes to `_jobs\out\`.
  - Prepared but unapproved jobs wait in `_jobs\later\`.
  - Every job waits until the laptop is plugged in.
- **Independent review:** the next model should inspect the source and retained before/after evidence. Implementation test results are not independent approval.
- **Hardware jobs since the redesign:**

| Job | What it did | Result |
| --- | --- | --- |
| 120 | Engine stop/start, monitor cost, shared load | Every part passed |
| 121 | Paging attribution, monitor overhead | Passed |
| 122 | Boundary bounds, model paging | Step 1 passed; step 2 inconclusive (paging only at load) |
| 123 | Practical mode, Llama 3.2 3B, 15 min target, job 101 start, isolated stores | 7.6 min; +4.1% estimated short-question score, unsure; nothing saved; 199 real files unchanged |
| 124 | Practical mode, Granite 3.1 MoE, 15 min target, job 102 start, isolated stores | 7.7 min; 0 layers + 10 threads confirmed (combined short-question score +144%); saved only to scratch; 199 real files unchanged |

### 3.4 What is proven on hardware vs only in the cloud

**Proven on the laptop:**

- the layer, threads and prompt-batch paths, with real kept gains in tunes (section 5);
- `tune` then `start` on real models;
- all six study-model tunes (jobs 099–107) under the fair comparison and final check;
- answer-quality checks, applied-settings verification on fresh loads, and the memory-safety correction (Phi);
- per-piece placement identity and the conversation-memory format being applied by the engine (job 113). These were never kept by a tune;
- engine stop during a request: about 0.2 s. During loading: about 2.3 s (job 120). Recovery 11–14 s;
- the precise paging monitor's attribution of independently caused paging (jobs 121 and 122, step 1);
- the monitor's overhead: about +1 s per measurement, against +12.8 s for today's per-answer monitor;
- the shared-load saving (about 3% on Llama; not enabled);
- model-file hashing time (about 5 s per 2 GB);
- the Windows single-writer lock on saved setups;
- the practical (time-limited) search mode, **twice** (jobs 123 and 124, 15-minute target, isolated scratch stores): the strongest early placement was measured within two minutes in both; Granite's large gain was confirmed in under eight minutes; Llama's small gain ended "unsure" with about half the time left. Neither run measured the long question, so no long-question gain, batch gain or batch cost is measured.

**Built, cloud-tested only:**

- the practical mode's confirmation planning (9 Oct; simulated and hardware-checked once in job 125; reporting corrected 9 Oct);
- the path declarations (C3, 9 Oct): read by the practical search through an adapter (first slice); validated on import, linked from the layers and threads cards, and the source of Unbounded's three static tables (second slice). Third batch: each `OptionPath`'s engine option and applicability rule are read from its declaration (`tune/applicability.py` registry); 0 of 241,920 matrix checks changed. Behaviour unchanged in 231 simulated practical tunes and 160 Unbounded tunes. Fourth batch (10 Oct): a runtime catalog binds each declared path to its existing code; on 11 Oct its builder was fixed to check the entries it is given. C3-6 (11 Oct): Unbounded's option-path loop runs through the shared coordinator (`tune/coordinator.py`, `tune/unbounded_policy.py`), the original 431-tune comparison had normalization limits; the C3-7 full-precision replay supersedes that gate;
- the stop-during-load lifecycle against real Windows process trees, beyond job 120's cases;
- the scoped paging monitor under genuine model paging during answers (job 122 couldn't produce it);
- trade-off offers in a real approval flow. A trade-off was found in job 097 but refused by the paging guard.

**Most recent full-suite run (Windows, 12 Oct, before the self-attack fixes):** 1,910 passed, 3 skipped, no failures. This is 1,877 existing tests plus 33 new tests, run separately with JUnit records. Type checking is clean on 437 source/test files. Older cloud counts are historical; the current Windows run does not have the 21 platform-dependent cloud failures.

### 3.5 Repository map (main parts only)

Paths are under `src/inferenceos/`.

| Area | Where | What it does |
| --- | --- | --- |
| Everyday commands | `tune/commands.py`, `tune/start.py`, `cli.py` | `tune` (search, confirm, approve, save) and `start` (run the saved setup). Many advanced commands remain in `cli.py` |
| Tune orchestration (Unbounded) | `tune/pipeline.py` (`Tuner`) | The full search: layer round, diagnosis, paths, revisits, final check, approval, save |
| Practical mode | `tune/budgeted.py` (`BudgetedTune`), `tune/planner.py` | Time-limited search with one final check. Internal only |
| Paths | `tune/paths.py` (`OptionPath` objects), `tune/layer_search.py`, `tune/layer_top.py` | Threads, reading threads, flash attention, prompt batch, engine switches, per-piece placement, conversation-memory format; the layer search |
| Model and machine facts | `tune/model_facts.py`, `tune/machine.py`, `tune/placement.py`, `tune/fit.py` | Reading the model file, the engine's placement rule, memory fit estimates |
| Diagnosis | `tune/diagnose.py`, `tune/engine_probe.py`, `tune/memory_speed.py` | Processor vs card share of each word, memory speed ceiling, which paths to try first |
| Measurement | `tune/measure.py`, `profiling/runner.py` | Fresh load, warm-up, measured answers, disturbance repeats, phase timings |
| Fair comparison | `tune/fair.py`, `tune/stats.py` | Alternating blocks, planned looks, guards (including the bounds-aware paging guard) |
| Quality | `tune/known_answers.py`, `tune/quality_risk.py`, `experiment/quality*.py` | Wording check by word probabilities; known-answer questions |
| Paging monitor | `telemetry/counters.py`, `telemetry/attribution.py` | Per-answer sampler (default) and the experimental measurement-scoped monitor |
| Engine control | `backends/resident_server.py`, `backends/llama_server_toolchain.py`, `backends/llama_cpp_resident_adapter.py`, `runtime/resident_workload.py` | Start, stop, cancel, halt and resume of `llama-server` |
| Saved setups | `tune/profiles.py` | Per model file, context, power source and program; atomic, guarded commit |
| Cards | `cards/` (`base.py`, `catalog.py`, `safety.py`, `builtin/`) | Card format and the first cards. **Note the gap in section 7.6:** tune's paths are not yet cards |
| Older system | `optimization/`, `loop/`, `experiment/`, `knowledge/`, `reasoning/`, `diagnosis/` | The earlier diagnose-and-experiment pipeline behind `optimize` and `monitor`, the learning store and the helper model. Kept as advanced commands; partly superseded by `tune` |
| Scripts | `scripts/` | Hardware probes (`check_verify_cost.py`, `check_monitor_attribution.py`), simulations (`simulate_search_order.py`, `simulate_tune.py`, `simulate_fair.py`, `simulate_long_runs.py`), studies |

## 4. How a tune works today, end to end

### 4.1 One measurement

A **measurement** of a setup on a question works like this:

1. A fresh start of the engine with that setup.
2. One warm-up answer, never scored.
3. N measured answers. N is chosen once per question to give the most certainty per minute, clamped to 5–20.
4. Each metric is reduced to the median.

**Disturbance:** if other programs use more than 25% of the processor, the measurement is repeated with a fresh load, at most twice. After that it is marked unusable.

**Questions used:**

- **Short question:** sensitive to first-word time and writing speed.
- **Long question:** built from repeated passages and sized per model, so one answer takes about 20 s. It is sensitive to prompt reading, which is where the prompt batch matters. On Llama it is about 1,850 tokens.

**Metrics read per answer:**

- writing speed;
- time to first word;
- prompt reading speed;
- total latency;
- peak memory;
- paging (hard page faults per second);
- per-phase timings (load, monitor, request, unload).

### 4.2 The goal score

The default goal (balanced) scores a change as the 50/50 mean of the relative improvements in writing speed and first-word time. Other measures are guards. Fixed weights per goal replaced the old "biggest improvement counts in full" rule.

### 4.3 Unbounded: the full search (the default `tune`)

There is no time limit. Steps:

1. **Measure the start.** Find the layer **top**: the most layers that load, found by load-only tries.
2. **Memory-safety correction,** if the start no longer fits. The start's layer count is lowered to what loads, with its own check and its own yes/no. It makes no speed claim.
3. **Layer round:** the whole safe range, fewer as well as more layers.
   - Both extremes come first.
   - Then a search that keeps splitting gaps near the best.
   - Counts within 0.5% count as equal, and the smaller wins.
4. **Diagnosis** at that placement: how much of each word is processor work vs card work, and how close the processor part is to the memory-speed ceiling. It orders and filters the paths.
5. **Paths in a fixed order:** threads → reading threads → flash attention → prompt batch → engine switches → per-piece placement → conversation-memory format.
   - Each path screens candidates, then runs a **light check** (one look at 4 blocks, 20% error allowance). The light check only chooses.
6. **Revisits:** affected paths are tried again after a kept change, at most 2 extra times each.
7. **One fresh final check** of the whole combined setup against the start, at a 5% error rate (section 6).
8. **Approval** per change, then **save**.

**Real results:** ten Performance-first tunes took 25.4 hours in total; details in section 5. Where the time went:

- 40% on paths that kept nothing;
- 10% on revisits that kept nothing;
- 17% on light checks of candidates whose first reading was under 4%.

None of the candidates in that last group held up.

The winning layer count was always tried within about 3 minutes, but the big batch gains came only at 70–145 minutes, because of the fixed path order. That is why the owner prioritised a time-limited mode and kept Unbounded as an explicit choice with no default duration.

### 4.4 Practical mode: time-limited search (internal, experimental; evaluated on hardware twice)

**Status:** it runs only when `tune_experimental_budget_s` > 0. There is no command-line flag and no default duration (owner's rule). It uses the same acceptance rules as Unbounded.

**Exploration** comes from the planner (`tune/planner.py`). The planner is a pure decision object: it says what to measure next and in which exact setup. It never measures and never accepts.

1. **Placement first:** 0 layers (processor only), 1 (only the output step on the card) and the top.
   - Dense models try the top first; mixture-of-experts models try 0 first.
   - **Intermediate counts** are tried when the model only partly fits, or when the result is ambiguous.
   - Then ±1 around the best.
2. **Choose a placement** only if its reading is **clear**: more than 2 noise widths above zero, and at least the smallest worthwhile gain.
3. **Order threads and batch by the processor's share of the work** at that placement:

| Processor share | Order |
| --- | --- |
| ≥ 50% and large batches usable | threads, then batch |
| 20–50%, or large batches not usable | batch, then threads |
| < 20% | batch only; threads deferred |

4. **Candidates come from the hardware, never from model names:**
   - threads: two counts between the physical and logical core counts (8 and 10 on this laptop), then ±1;
   - batch: up to two larger sizes that the long question can use and that fit in card memory, then ×2 and ÷2.
5. **Weak readings** (positive, but within noise) are deferred, and may be repeated up to twice. **Missing readings never count as gains.**
6. **Evidence scoping:** a reading counts only in the exact setup it was measured in. When another setting changes, the stale choice is rechecked once in the new setup.

**Time model:**

- target = budget;
- a bounded, explained extension = max(20% of the budget, 120 s);
- a 30 s save reserve.

Costs come only from this tune's own measurements. Exploration stops **before** it would leave too little time for the final check of the best setup so far.

**Choice:** candidates are the best-so-far setup, its subsets (some settings back at the start), up to 6 earlier complete setups measured in this tune, and repeated weak setups. The most valuable eligible candidate whose final check fits the remaining time is chosen.

**One final check** at a block count fixed in advance (4–16). Selection judges affordability with the selection-time count. Since 9 Oct, just before the check starts, the count is **re-planned** (`_plan_confirmation`):

- the count for 80% planned power under the screening noise width times an uncertainty allowance (1.75 with the fallback load-to-load spread, 1.25 with a learned one), using the test's own t critical value;
- used only if the whole check (both sides, every question, re-attempt reserve, saving and cleanup) is predicted to finish by the **soft target**; otherwise the largest count that does; never fewer than the selection count;
- the plan (measured inputs and predictions kept apart) is recorded in the report; nothing is added after the check starts.

It is paced at block boundaries and may stop "incomplete". A passing check is always kept.

**Stops:** there are three distinct stops:

| Stop | What happens |
| --- | --- |
| Target reached | Planning stops; nothing running is interrupted |
| User stop | Sticky: nothing may start again; nothing new is saved unless the save had already committed |
| Stuck operation | A watchdog stops the engine (non-sticky); the tune ends exploration with its evidence so far |

**Hardware evidence (jobs 123 and 124, 9 Oct, 15-minute target, isolated stores):**

| | Llama 3.2 3B (job 101 start) | Granite 3.1 MoE (job 102 start) |
| --- | --- | --- |
| Time | 7.6 min | 7.7 min |
| Strongest candidate | 27 layers (+5.5% screening) at 106 s; 28–29 did not keep the free-memory margin | 0 layers (+126.5%) at 64 s |
| Final check | 4 blocks, unsure: +4.1% (−1.9% to +10.1%) | 4 blocks, better: writing +221%, first word −68%, score +144% |
| Saved | nothing | scratch store only |

Llama's check was sized from a screening width of 1.98% while its blocks varied by 5.1 points. Neither run measured the long question, so batch costs stay predictions and no long-question result exists. This is not evidence of "95% retention".

**Simulation evidence** (invented model-like worlds, not measurements) is in `claude/search-order-v2-2026-10-06.md`, `claude/review9-fixes-2026-10-06.md` and, for confirmation planning, `claude/confirmation-planning-2026-10-09.md`.

### 4.5 Approval and saving

- **Approval:** each change is shown with its measured gain. In practical mode it is all-or-nothing; Unbounded re-checks the rest if one is declined.
- **Saving:** atomic (prepare a temporary file, then one guarded replace). A user stop is either before the commit (nothing saved, previous file byte-identical) or after it (saved and reported). There is no partial state.
- **Saved setups** are keyed by model file (hash), context length, power source and program.

### 4.6 Engine control and stop safety

The engine (`llama-server`) runs as a child process with a start/stop state machine (idle → starting → ready → stopping):

- **Stopping during a load works:** the process is owned from creation, the readiness wait can be interrupted, and a cancelled start is never published.
- **A user stop is sticky:** nothing restarts until it is explicitly allowed. A stuck-operation stop is not sticky, so the tune can continue.
- **No new start** while an old process is not confirmed gone.
- **Hardware (job 120):** stops took 0.21–0.27 s during a request and about 2.3 s during loading (the cause of the slower case is unknown). Recovery took 11–14 s.

### 4.7 The paging monitor

Paging (pages read from disk to resolve memory faults) is a guarded measure. A setup that makes the machine page more must not be kept just because it is faster.

**Default: the per-answer sampler.** It starts a Windows counter reader for each answer and waits up to 6 s for its first reading.

- It is reliable, but it costs about 2 s per answer: +12.8 s per short-question measurement on Llama (job 121).

**Experimental: the measurement-scoped monitor.** It is off by default (`tune_experimental_measurement_monitor = False`).

- **How it reads:** one .NET raw-count reader per measurement, reading every 0.1 s, with every reading bracketed by clock times.
- **How it attributes:** pages in intervals wholly inside an answer are certain. Pages in intervals crossing an answer's edge may or may not belong to it, so they become **bounds** rather than unknown.
- **What the guard does with bounds:** pass only if the worst case passes, breach only if the best case breaches, otherwise "not settled", which blocks acceptance.
- **Cost:** about +1 s per measurement instead of +12.8 s.
- **Hardware validation:**
  - Job 121: 24/24 bursts confirmed, 0 wrong attributions.
  - Job 122 step 1: the same, with no unknowns left, thanks to the bounds.
  - Job 122 step 2: inconclusive. The model paged only while loading, so genuine paging during answers was never exercised. One warm-up answer was unknown because it began 31 ms after the monitor's first reading.
- **Whether to enable it** is an open question (section 9.2). In job 121 the main worry was unknown answers spreading into "unsure" comparisons. The bounds have since reduced that.

## 5. Every optimization path

A **path** is a self-contained area with its own search, like the graphics-card layer path. Each entry below says:

- what the path changes;
- what it mainly moves;
- its status and evidence;
- how it is searched today;
- what it depends on, and what it changes for other paths (the interaction facts used in section 7);
- what remains to be done.

Paths 5.1–5.7 are built. Paths 5.8 onward are not built, or exist only as suggest-only cards.

### 5.1 Graphics-card layers (placement count)

- **Changes:** how many model layers run on the graphics card.
  - Engine rule in b11139: 1 layer = only the output step on the card; N layers = the output step plus the last N−1 blocks; blocks + 1 = everything on the card.
- **Moves:** writing speed and first-word time, often by a lot. It also changes how much work the processor does, and how much card memory is left.
- **Status:** **proven on hardware.**
- **Evidence:**

| Model | Layers | Gain |
| --- | --- | --- |
| Llama | 23 → 29 | short +14% |
| Granite MoE | 33 → 0 | +106% on its own |
| Granite 4 H | → 0 | +3–5% |
| Qwen | 8 → 10 | +3.5% (needed 16 blocks to verify) |

- **Searched by:**
  - Unbounded: a whole-range layer search with a fair check.
  - Practical: extremes first, then intermediate counts for partly-fitting models.
- **Depends on:** free card memory, which is itself affected by the conversation-memory format, per-piece placement, prompt-batch buffers and other programs on the card. Also on the memory-safety margin (about 0.8 GB kept free on the 4 GB card).
- **Changes for others:**
  - the processor share, which changes the threads optimum;
  - the card memory left, which changes which batch sizes fit;
  - the sizing of the long question (it is sized at the chosen placement).
- **Remaining:** none essential. On the NVIDIA machine the curves must be re-learned, not assumed.

### 5.2 Processor threads (writing threads)

- **Changes:** the number of processor threads used while writing.
- **Moves:** writing speed and the processor part of reading, when the processor carries much of the work.
- **Status:** **proven on hardware.**
- **Evidence:** kept twice: 10 threads on Granite MoE, 9 on Granite 4 H.
  - Counts below the physical core count were always worse (−4% to −70%).
  - 12 threads was worse on three models.
  - Several first readings of +2–4% did not hold up.
- **Searched by:** candidates between the physical and logical core counts. It is skipped or deferred when the card carries most of the work.
- **Depends on:**
  - placement (the processor share);
  - prompt batch: an interaction is possible. A simulated example: 10 threads helped at batch 512 and hurt at 1024;
  - other programs' load;
  - core choice (5.11).
- **Changes for others:** possibly the batch optimum.
- **Remaining:** threads together with "which cores" (an operating-system card, 5.11).

### 5.3 Reading threads (prompt-processing threads)

- **Changes:** threads used while reading the question.
- **Status:** **built, never kept.** The best first reading was +2.0% (Granite 4 H), and it did not hold up.
- **Searched by:** Unbounded only.
- **Depends on:** placement and batch.
- **Remaining:** stays in Unbounded. It is a candidate for "later session" lists in practical mode.

### 5.4 Prompt batch (ubatch)

- **Changes:** how many prompt tokens are processed at once.
- **Moves:** prompt reading speed and the first word on long prompts. It needs a prompt longer than the batch to have any effect.
- **Status:** **proven on hardware.**
- **Evidence:**

| Model | Batch | Long-question gain |
| --- | --- | --- |
| Granite MoE | 1024 | +11–14% |
| Phi | 2048 | +14% |
| Granite 4 H | 2048 | +15% |
| Llama | 1024 | +1.7–1.9% (weak) |
| Qwen 9B | 256 (3 Oct) | first word about 38% sooner on long prompts |

  On Qwen, 512 was −34%. The lesson is "keep a good current batch and respect memory".
- **Searched by:** sizes the long question can use and that fit in card memory, by a working-buffer estimate. The short question gets a "not worse" guard.
- **Depends on:**
  - the long question's length;
  - free card memory (after placement, memory format and pieces);
  - threads (the processor part of reading).
- **Changes for others:** it uses card memory for working buffers, which can lower the layer top. It can shift the threads optimum.
- **Open:** the 2% "not worse" guard on the short question costs 7–16 blocks on noisy models. Whether a cheaper guard is acceptable is an open decision.

### 5.5 Flash attention

- **Changes:** a different attention calculation in the engine.
- **Moves:** speed and memory; the effect differs by model.
- **Status:** **built, never kept automatically.**
- **Evidence:** it is a trade-off on Gemma. Turning it off gave long question +23.9%, short question −2.4% (job 095). Job 097 found it again, but it paged heavily in 2 of 4 blocks, so it was refused.
- **Searched by:** Unbounded only. Trade-offs are offered separately and kept only with an explicit yes.
- **Depends on:** the conversation-memory format (the engine refuses quantized values with flash attention off) and batch size (paging seen with batch 1024).
- **Remaining:**
  - a usage-weighted trade-off rule (open);
  - the cause of the paging with flash off (not investigated).

### 5.6 Engine switches

- **Changes:** build-specific engine options (calculation-path choices).
- **Status:** **built, never kept.**
- **Evidence:** on this card, only "fusion off", "graph reordering off" and "host memory" visibly take effect, and all of them are slower. A switch whose effect can't be observed is skipped as "activation unverified; not tested".
- **Rules:** switches come only from a per-build, per-hardware list. Each must be shown to take effect and to keep answers the same.
- **Remaining:** NVIDIA needs its own list. Unbounded only.

### 5.7 Per-piece placement and conversation-memory format

**Per-piece placement:**

- **Changes:** which specific pieces of the model go on the card. Examples: the expert pieces of mixture-of-experts models on the processor; the output step on the processor; grouping by block kind for hybrids.
- **Status:** **built, partly proven.**
  - **Proven (job 113):** the engine applies exactly the requested pieces.
  - **Not proven:** never kept by a tune. Grouped hybrid placement was much slower in single runs and stays disabled (`tune_grouped_placement_probed = False`).
- **Verification:** each candidate's identity is read from a separate debug-level load of that setup; measured runs are checked by buffer totals.
- **Guard:** a placement must not lose on the long question. Pieces on the processor can slow long-prompt reading.

**Conversation-memory format (KV cache):**

- **Changes:** storing keys and/or values at 8-bit (q8_0) instead of 16-bit. This roughly halves that memory, which can let more layers fit.
- **Status:** **built, partly proven.**
  - **Proven (job 113):** the engine applies it.
  - **Not proven:** never kept by a tune. Job 115 (Qwen) was stopped by the owner before any check finished.
- **Quality risk:** it can change answers, so it needs known-answer questions (policy known-answers-v3). Context size never changes.

**What both do to other paths:** they free or move card memory and processor work. The layer top must be found again, then placement, then batch fit.

**Remaining:**

- a real tune that exercises both;
- deciding whether they belong in practical mode (not yet).

### 5.8 Memory residency and model file loading (not built as a path)

- **Changes:** keep the model in memory so Windows doesn't page it out; how the model file is loaded (memory mapping, locking, prefetch).
- **Moves:** paging, loading time, and stalls while answering under memory pressure.
- **Status:** a memory-mapping card exists from the old system (`cards/builtin/memory_mapping.py`). There is no tune path.
  - **Job 122 finding:** on this laptop llama.cpp reads the whole model at load, so there is no paging during answers when memory is plentiful. The path matters most when memory is tight (large models, other apps).
- **Depends on and affects:** other programs, operating-system cards (keep in memory, large pages), and the watcher's paging readings.
- **To build (step 7/8):** a card with a quality check of none, the restart rule, and a memory-pressure test scenario. That scenario is the same one needed to validate the paging monitor (section 9.2).

### 5.9 Prompt reuse (capability and development evaluation built 16 Oct; not a tune path)

- **Changes:** reuse the already-processed start of a repeated prompt (llama.cpp's prompt cache, cache reuse).
- **Moves:** first-word time for repeated prompts. It does not change context size, so it is explicitly allowed.
- **Engine facts (b11139 help text in `tests/fixtures/llama_server_help_b11139.txt`):** per-request
  `cache_prompt` is **on by default** in llama-server, so a model run with `start` already reuses
  prompt starts for its clients; `--cache-reuse N` (off by default) also reuses matching chunks after a
  mismatch by shifting them; `--cache-ram` (8 GiB by default) keeps idle slots' work in memory;
  `--ctx-checkpoints`/`--checkpoint-min-step` govern reuse on hybrid and sliding-window models.
  Tune has always sent `cache_prompt: false`, so every tune measurement is a fresh reading.
- **Built (16 Oct, uncommitted):** explicit `cache_prompt` on requests (default False) and reuse counters
  on results; `--cache-reuse` plumbing for the evaluation only (no RuntimeConfiguration field, so tune and
  saved setups never set it); tune request identity `tune-request-4` with an explicit fresh prompt-reuse
  block (a cached request is never comparable with history); the development evaluation
  (`tune/prompt_reuse.py`, `scripts/evaluate_prompt_reuse.py`, simulated engine for tests); prepared job
  127 (`_jobs/later/`, not queued). Details: `reviewer-prompt-reuse-2026-10-16.md`; revised after the
  16 Oct review (immediate validity and cleanup stops, strict work evidence, real cancellation evidence,
  checkpointed results, fail-closed mains power, a shifted-chunk positive control):
  `reviewer-prompt-reuse-fixes-2026-10-16.md`; revised again after the second review (one cleanup
  owner per run with confirmed process identity, native Windows power status, client-cutoff-only
  cancellation, no reuse of earlier results files, required slot identity, validity gates on every
  reading, and E7 split by whether the engine kept `--cache-reuse` on):
  `reviewer-prompt-reuse-fixes-r3-2026-10-16.md`; and after the third review (password-file and
  process identity kept on record until their release is confirmed, malformed slot replies never
  count as idle, E7 judged load by load): `reviewer-prompt-reuse-fixes-r4-2026-10-16.md`; and after
  the fourth review (overlapping cleanups take turns and never replace a failed start's error; the
  password file is owned from the moment it is created): `reviewer-prompt-reuse-fixes-r5-2026-10-17.md`;
  and after the fifth review (the password file is prepared under the cleanup lock, so a cleanup can
  never be followed by a re-created file): `reviewer-prompt-reuse-fixes-r6-2026-10-17.md`.
- **Job 127 (17 Oct; details `reviewer-job127-results-r2-2026-10-17.md`):** all four runs completed with clean operational checks (cleanup, power, data folder, cancellation).
  Not every research rule passed:
  - **E1:** all priming and fresh readings reported zero reuse, for these prompts, setups and engine conditions. This is not a guarantee for other conditions or an audit of past history.
  - **Reuse size:** reusing the whole earlier prompt cut first-word time by 89-99%.
  - **Partial reuse varied by model:** all of the shared start on Llama, about half on LFM2, none on Granite.
  - **E3 failed on Gemma:** partial reuse gave a different answer in all 4 repeats of one prompt pair.
  - **E7 (`--cache-reuse`) not testable:** the engine sent dissimilar prompts to another slot (4 slots by default, `--slot-prompt-similarity` 0.10), and Gemma's engine turned the option off. The E7 classifier was corrected (`e7-status-2`), and the records were re-analysed into new files; the originals are unchanged.
  - **No formal noise floor:** the isolation control was unusable under the slot rule.
- **Job 128 (prepared 17 Oct, not run; pre-registration `reviewer-job128-prereg-2026-10-17.md`):** a known-answer
  study of whether partial reuse changes answers. It covers 9 prompt pairs (including job 127's failing one) on Gemma, with and
  without `--swa-full` (effective setting read from each load's log), plus Llama as a control and an ordinary chat-client
  exposure check. No default is changed by it. Amended after review (r2,
  `reviewer-job128-prereg-r2-2026-10-17.md`): a written answer grammar with right, wrong and unscorable; the
  `--swa-full` comparison only on pairs informative under both variants; chat replies need complete usage evidence;
  a matched, identically primed chat reference; immediate chat stops. Amended again (r3,
  `reviewer-job128-prereg-r3-2026-10-17.md`): numbers read whole with their signs, unscorable answers never establish a
  removed failure (per-pair partial-reuse evidence), and interruptions at power readings keep their evidence. Amended
  again (r4, `reviewer-job128-prereg-r4-2026-10-17.md`): the judge reads only a fixed list of number forms, compared as
  exact values; anything else with a digit is unscorable. The amendment in
  `reviewer-implementation-handoff-current.md` (current study `reuse-correctness-8`) normalizes formatting before spelling/sign
  rules, preserves punctuation and standalone operators, rejects ambiguous signs, and includes all documented magnitude
  plurals. The 18 Oct self-audit corrections use full Unicode numeric discovery and preserve the own-identifier colon
  boundary before signed values. Software validation and independent review remain separate; job 128 is still not run.
- **Not yet shown:** `--cache-reuse` behaviour (slot-pinned study), whether the Gemma result generalises or reaches
  `start` users (a known-answer study; `--swa-full` untested), and answer correctness beyond identity between arms.
- **Before it could become a tune path:** a RuntimeConfiguration setting (a history-format change), a
  workload with repeated prefixes judged by its own rules, and evidence from job 127 across models.

### 5.10 Word drafting (speculative decoding; not built)

- **Changes:** cheaply guess upcoming words, either copied from the question or from a small helper model, and let the main model check them.
- **Moves:** writing speed.
- **Costs:** card memory for a helper model, which competes with layers and the memory format.
- **To build (step 7, later):** a card; helper-model selection (no new downloads without approval); an acceptance-rate measurement; the quality rule (output must match the main model's own).

### 5.11 Windows / operating-system cards (step 8; not built)

- **Cards planned:**

| Card | Admin rights | Asks before testing | Notes |
| --- | --- | --- | --- |
| Program priority | Not for a small raise | No | |
| Which cores | No | No | Pairs with threads |
| Power plan | Usually not | No | |
| Keep model in memory | Sometimes | No | |
| Large memory pages | Yes | Yes | |
| Background-program priority/pause | Depends | Yes | |
| Page file size | Yes, plus a restart | Recommend only | Suggest-only card exists |
| Virus-scan exclusion | Yes | Recommend only | |

- **Already exists:** suggest-only cards for page file, other-program priority, close and pause (`cards/builtin/`).
- **Needs first:** the **undo journal**. Before any operating-system change, write the old value and how to restore it; after a crash, the next start undoes open entries.
- **Depends on and affects:** threads (core choice), paging, measurement validity.

### 5.12 Background programs (step 8; suggest-only today)

- **What it is:** other programs competing for memory, processor or card. Anything affecting another program asks before testing.
- **Today:** the measurement only detects disturbance and repeats.

### 5.13 Calculation code and builds (not built)

- **Changes:** other versions of llama.cpp's calculation code, or llama.cpp built for this exact machine; later, modified or generated code.
- **Owner's wish:** a general capability, so that on the NVIDIA machine it only needs testing, not rewriting.
- **Depends on:** the engine version (5.15) and the switch lists (5.6).
- **Not designed yet.**

### 5.14 Heat and power (partly built)

- **Built:**
  - processor-speed reading per run;
  - a heat report at the end of `tune`;
  - an end-of-tune heat check that warns when answers really got slower.
- **Owner's rules:**
  - with no throttling, heat must never interfere;
  - heat a setting causes counts against it;
  - rounds disturbed by other programs are repeated;
  - say so when heat changed a decision.
- **Not built:**
  - temperature, fans, battery drain and power readings (step 6);
  - the "quiet and cool" and "save battery" goals;
  - a path that keeps speed steady as the laptop heats up.

### 5.15 The engine itself (not built)

- **Changes:** other llama.cpp versions; later, Ollama and LM Studio through connectors (step 10).
- **Changes for others:** everything. Every result is tied to the engine build, and a new build means re-measuring.

### 5.16 Paths by where they act

| Level (from the core idea) | Paths |
| --- | --- |
| Where each piece runs | 5.1, 5.7 |
| Processor scheduling | 5.2, 5.3, 5.11 (cores, priority) |
| Data moving / batching | 5.4, 5.7 |
| Memory | 5.7 (memory format), 5.8, 5.11 (large pages, keep in memory) |
| Calculation code | 5.5, 5.6, 5.13 |
| Operating system / other apps | 5.11, 5.12 |
| Storage and loading | 5.8 |
| Workload-level | 5.9, 5.10 |
| Heat and power | 5.14 |
| Engine | 5.15 |

---

## 6. Measurement and acceptance rules (the contract every path obeys)

These rules are shared by Unbounded and practical mode, and every future path must use them.

1. **The only acceptance route is the fair comparison** (`compare_fairly`).
   - It runs alternating blocks; each block measures the start and the candidate from fresh loads (AB, BA, ...).
   - Decisions happen only at planned looks, on exactly the planned number of usable blocks.
   - The statistic is a one-sided t-bound of the mean block score.
2. **False-keep rate:** at most 5% (1 in 20), split across looks.
   - "Better" needs the lower bound > 0, plus the effort mode's smallest worthwhile gain and the payback check.
   - "Not worse" (guards) needs the lower bound > −2%. This tolerance is the same in every effort mode.
   - "Unsure" fails.
   - Running out of attempts or time gives "unsure". An expired budget never triggers an extra acceptance check.
3. **Looks by effort mode:**

| Effort mode | Looks |
| --- | --- |
| Minimize cost | One look at 4 blocks |
| Balanced | 4 and 8; its final must-not-lose check looks at 8 and 16 |
| Performance first | 4 and up to 16 |
| Practical mode | One look at a fixed N per question, chosen before the check |

4. **Whole-setup final check:** each question a change was found on must show "better", and each question a change was only guarded on must show "not worse".
   - Layers and threads are checked on the short question.
   - Batch is checked "better" on the long question and "not worse" on the short one.
   - In Unbounded, per-piece placement must also not lose on the long question, and the memory format is judged on the long question and guarded on the short one.
   - The reported gain is the one measured on the first question, never multiplied together.
5. **Guards:**
   - **Paging:** the median paired difference must be ≤ 100 faults/s, and severe episodes (≥ 1,000/s) are counted. With bounded readings, pass only in the worst case and breach only in the best case; otherwise "not settled", which blocks acceptance.
   - Any guarded measure missing from a counted block blocks acceptance.
6. **Applied-settings verification:** every fresh load used as evidence must show, from the engine's own output, that the requested settings were applied, on both sides.
7. **Answer quality:**
   - every change gets a wording check against the start's answers (word probabilities within tolerances);
   - quality-risk changes also need known-answer questions;
   - the strictest requirement of any change in a combination applies;
   - unusable or unfinished quality checks count as incomplete, never as a pass.
8. **Screening readings only choose; they are never evidence.**
9. **Trade-offs:** a change better on one question but losing more than 2% on another is never kept automatically. It is offered with its measured benefit and uncertainty, and kept only with an explicit yes.
10. **Memory-safety correction:** it has its own check and its own yes/no, and makes no speed claim. If it can't be verified, nothing is saved and the unsafe start is not restored.
11. **Disturbed measurements are repeated.** If still disturbed, they are not used.
12. **Heat** is never used to drop rounds. It is only noted and reported.

## 7. How the paths interact

This is the heart of the "one coordinated system" requirement. The section has six parts:

- 7.1: the physical reasons settings interact;
- 7.2: the interaction map as written in the code today;
- 7.3: how Unbounded and practical mode handle interactions;
- 7.4: the ordering principle that follows from 7.1;
- 7.5: when stored results may be reused;
- 7.6: the gaps between today and the target.

### 7.1 Why settings interact

Almost every interaction goes through one of four shared resources.

1. **Card memory.** These all draw on the same graphics memory:
   - layers on the card (5.1);
   - pieces on the card (5.7);
   - the conversation memory and its format (5.7);
   - prompt-batch working buffers (5.4);
   - a drafting helper model (5.10);
   - other programs.

   Anything that frees memory can let more layers fit. Anything that uses memory can push the layer top down.
2. **The processor's share of the work.** Placement decides how much of each word runs on the processor. That decides whether threads matter at all, and which thread count is best. Core choice and priority (5.11) change how much processor the model really gets.
3. **Prompt reading.** Batch size, reading threads, flash attention and pieces on the processor all change how the question is read, so they change first-word time on long prompts.
4. **System memory and paging.** Memory residency, other programs, large pages and the page file decide whether the model stalls on disk reads.

Heat touches all speed readings. A setting that makes the laptop hotter pays for it in measured speed, by the owner's rule.

### 7.2 The interaction map in the code today

`tune/diagnose.py` holds the `AFFECTS` table: which paths to look at again after a change is kept on another. It mirrors how llama.cpp works. Since the C3 second slice (9 Oct) the table is computed from each path declaration's `invalidates` (`tune/declarations.py`); the contents below are unchanged.

| Kept change on | Look again at |
| --- | --- |
| layers | threads, reading threads, prompt batch, per-piece placement, memory format |
| threads | reading threads, prompt batch |
| reading threads | threads, prompt batch |
| prompt batch | threads, reading threads, flash attention, engine switches, per-piece placement |
| flash attention | prompt batch, threads, reading threads, engine switches, per-piece placement, memory format |
| engine switches | threads, reading threads, prompt batch, flash attention |
| per-piece placement | threads, reading threads, prompt batch, flash attention, engine switches, memory format |
| memory format | per-piece placement, prompt batch, flash attention, threads |

A kept change on per-piece placement or the memory format also makes the **layer count** worth searching again (`MEMORY_PATHS`).

### 7.3 How interactions are handled today

**Unbounded:**

1. **Fixed order:** layers first, then the paths. The order roughly follows "settle what changes resources first", except that per-piece placement and the memory format come last today. Section 7.4 says why that is backwards.
2. **Diagnosis** orders and filters the paths by the processor/card split. A path is skipped only with strong evidence (unsupported, or clearly inapplicable). Uncertain paths get cheap probes.
3. **Revisits** follow `AFFECTS` after each kept change: at most 2 extra attempts per path, within the time left, never twice on the same setup.
4. **One final check of the combined setup.** Independently successful changes are never assumed to combine. The combination itself must pass, and gains are never multiplied together.

**Practical mode:**

1. **Placement is settled first, by rule.**
2. **Threads and batch are ordered by the processor share** at the chosen placement.
3. **Evidence scoping:** a reading counts only in its exact setup. If another category's choice changes, the category is stale: its choice and runner-up are rechecked once in the new setup.
4. **Earlier complete setups and subsets stay candidates.** This exists because of a reviewer's case: an earlier combination (processor only + 10 threads at batch 512, +42%) beat the later one, because batch 1024 turned 10 threads into a loss.
5. **One final check** of the chosen combination, as in Unbounded.

**Real interaction evidence:**

| Case | What was seen | Strength |
| --- | --- | --- |
| Granite MoE | Placement to 0 layers made threads matter (10 threads kept) and batch then helped at that placement | Measured |
| Granite 4 H (104 → 107) | After 9 threads were kept, a later tune found 0 layers + batch 2048 on top | Measured |
| Gemma | Flash attention off with batch 1024: a trade-off, and paging | Measured |
| Phi | A stored layer count became unsafe when free card memory changed (job 103), so stored results can expire | Measured |
| Threads × batch | No real tune measured this directly | Only by reasoning and simulation |

### 7.4 The ordering principle (what the full system should follow)

Settle settings in this order, re-checking downstream settings whenever an upstream one changes:

1. **Resource-shaping settings** decide what fits where. Each needs a fresh layer top.
   - conversation-memory format (quality-gated);
   - per-piece placement;
   - later, a drafting helper's memory.
2. **Placement count** (layers), given the memory that is now free.
3. **Work-allocation settings,** ordered by the processor share at that placement:
   - threads (and later, which cores);
   - prompt batch (fit checked against the memory left);
   - reading threads.
4. **Calculation-path settings:** flash attention, engine switches, later calculation-code variants. Each is shown to take effect, and its interactions with batch and memory format are checked.
5. **System settings** (step 8): priority, power plan, keep in memory, large pages, background programs. Approval and the undo journal apply.
6. **Workload-level settings:** prompt reuse, word drafting.
7. **One confirmation of the whole combination,** with the strictest quality requirement of any part.

Rules that go with it:

- **Evidence-proportional:** a path is entered only if diagnosis or cheap probes say it might help. "Nothing worth fixing" ends the search early.
- **Targeted rechecks, not restarts:** when an upstream setting changes, recheck only the dependants named by the interaction map, using the current choice and the runner-up.
- **Trade-offs are offered, never kept on their own.**
- **Marginal readings are deferred, not cut.** They are offered in a later session or as a single-component check when time allows.

**Today's code differs from this principle in two places:**

- Unbounded runs per-piece placement and the memory format after the processor paths, then relies on revisits.
- Practical mode does not include those two paths at all.

Reordering Unbounded is **not** planned, because the owner said to keep it unchanged. The principle applies to the practical mode and to the future coordinator (section 10).

### 7.5 When stored results may be reused (retunes)

A stored result may skip re-screening a setting only if all of these still hold:

- the same model file (hash), context length, engine build and back end, and graphics driver;
- the same power source and a similar free-card-memory band;
- every setting it depends on is unchanged:

| Setting | Depends on |
| --- | --- |
| Threads | placement, batch, processor |
| Batch | placement, card-memory headroom, context |
| Placement | free card memory, memory format, piece placement |

- **and** a fresh start measurement matches the stored one within noise.

Otherwise that setting is re-screened with a few readings, not a full search.

**Status:** designed (`claude/search-order-v2-2026-10-06.md`, section 5); **not built**. Today `start` reports changes and starts anyway (owner's decision). `tune` begins from the saved setup and searches again.

### 7.6 Integration gaps (honest list)

1. **Tune's paths are not yet cards.**
   - Today there are two parallel mechanisms: the card system (`cards/`: format, catalog, context-safety check, 8 built-in cards from the old system) and tune's paths (`tune/paths.py` `OptionPath` objects plus the layer search).
   - The paths carry a `card_id`, but several have no card file: flash attention, prompt batch, reading threads, engine switches, per-piece placement and memory format.
   - The cards' "works with" and "clashes with" fields are not what drives tune. The `AFFECTS` table (computed from the path declarations since 9 Oct) and the planner's rules do.
   - **Since 9 Oct (C3):** every path has a declaration; the layers and threads cards link to theirs through an optional `tune` field that the older optimizer ignores. The six paths without card files are reached only through the declarations; they are not legacy cards.
   - The plan's step-3 promise ("adding an optimization means writing a card, nothing else changes") holds for the old system, not for `tune`.
2. **Shared dispatch is wired for both modes (C3-7, awaiting independent audit).** Declarations, runtime bindings and the coordinator are implemented. Unbounded retains its existing option-path policy; Practical retains its Planner and workload-specific evidence. Initial layer/safety work, later evidence completion, confirmation, approval and saving remain in their existing owners. The catalog records specialized layer entry points; it does not make every search generic. New paths still need specialized code. Learned costs, persistent tried values and new ordering policy are subsequent C4 work.
3. **Cost history is now connected narrowly (C4-2, 14 Oct).** The live ledger still owns current-run costs. Separate durable Practical observations can forecast an unmeasured exact setup/workload under compatible conditions. Missing identity, legacy records and failed/incomplete measurements cannot supply a prior. Final confirmation still uses its existing current-run cost machinery.
4. **Candidate ordering hints are implemented in shadow mode (C4-3, corrected 15 Oct).** Practical's isolated history proposes a permutation of an existing Step's eligible values using compatible comparisons with the same scoring goal and setup. Normal runs retain the original order; applying proposals requires an explicit internal development flag. Latest failed attempts suppress older readings from that run, and clean contrary readings count regardless of whole-run outcome. It cannot add paths/values, replace the captured base, skip measurements using history or accept an old score. Fresh confirmation remains required. General planner benefit, activation criteria and broader interaction design are unproven.
5. **The watcher and `tune` don't share a profile model yet.** The watcher (step 9) must compare against the saved setup and feed in-depth tests back through the same acceptance rules.
6. **Operating-system and background-program settings can't be tested safely** until the undo journal exists (step 8).
7. **Ollama and LM Studio** have no connectors (step 10). Every path today is llama.cpp-specific.

## 8. The immediate next steps (updated 9 Oct 2026)

### 8.1 Job 122: decided

Job 122 (run `201c6d05…`) stays **inconclusive**: step 1 (attribution with boundary bounds) passed; step 2 never exercised genuine model paging during answers (the model paged only while loading). The owner decided on 8 Oct to **defer** further paging-pressure work and the optional monitor-start change, and to go on to the practical-mode evaluation with the scoped monitor off. Validation under real model paging remains undone (D9, 5.8).

### 8.2 The first practical-mode hardware evaluation (C2): complete

**The owner's direction (8 Oct):** a practical Windows llama.cpp optimizer whose wait targets **10–15 minutes**. That is a design target, not permission to weaken evidence or to discard a passing result because it finishes slightly late. Unbounded, the acceptance thresholds and the monitor default stay unchanged. The earlier proposal of 45- and 60-minute budgets with a separate 8-block comparison is **withdrawn**.

**What ran** (owner-authorised, reviewed; one isolated run each, 15-minute soft target with the existing extension, scratch stores, the default per-answer monitor, the existing fresh whole-setup final check only):

| | Job 123: Llama 3.2 3B | Job 124: Granite 3.1 MoE |
| --- | --- | --- |
| Start | job 101's: 23 layers, context 4096, the rest engine default | job 102's: 33 layers, context 4096, the rest engine default |
| Time | 7.6 min (tuner 440 s) | 7.7 min (tuner 447 s) |
| First strong candidate | 27 layers, +5.5% screening, at 106 s (28–29 layers did not keep the free-memory margin; no margin relaxed) | 0 layers, +126.5%, at 64 s |
| Final check | 4 blocks, 238 s: unsure, +4.1% (−1.9% to +10.1%) | 4 blocks, 204 s: better, writing +221%, first word −68%, score +144% |
| Outcome | not confirmed; nothing saved | 0 layers + 10 threads, saved to scratch only |
| Isolation | 199 real files unchanged (both checks) | 199 real files unchanged (both checks) |
| Monitoring | 165 s, 38% of tuner time | 230 s, 51% |

**What it shows:**

- The search order works: both runs measured their strongest early placement within two minutes.
- A large gain is found and confirmed well inside 10 minutes.
- Llama ended "unsure" with about half the target left. Its check was sized from a 1.98% screening width (the load-to-load term was the 1.2% fallback, because the scratch store starts empty) while its blocks varied by 5.1 points. That is a confirmation-planning and calibration limit, not evidence that 15 minutes is too short, and not proof that more blocks would have passed.
- **Neither run measured the long question.** Batch costs remain predictions, and there is no long-question gain or non-regression evidence. Job 101/102 comparisons are from an earlier code version under different conditions; no retention figure (and no "95%") is claimed.

Reports: `claude/practical-eval-prep-2026-10-08.md` (preparation), the reviewer's `reviewer-practical-evaluation-2026-10-09.md`, and the job 123/124 artifacts in `_jobs\out\`.

### 8.3 What follows: confirmation planning (the C2 follow-up)

**Built (9 Oct, `claude/confirmation-planning-2026-10-09.md`):** the final check's block count is planned before it starts (section 4.4): more blocks for small or uncertain gains, only when the whole check finishes by the soft target; strong gains unchanged; the test and its acceptance rule unchanged; the plan recorded with measurements and predictions kept apart.

**Simulation (250 tunes per world, real `BudgetedTune`):** large gains unchanged; confirmation of checked small gains up (28% → 38% when the noise is underestimated as in job 123, 84% → 92% with drift, 85% → 93% with unequal side costs) for about 1–1.5 more minutes; no run past the target because of the change (checked per seed: 0 of 250 pairs per world newly past the target).

**The evidence also says this is not the main limit** for small gains on this laptop:

1. Block noise and block cost dominate: with job 123's readings, no fixed count inside 15 minutes confirms a 4% gain reliably (about 74% at the best affordable count). Monitoring is 38–51% of tuner time; the scoped monitor (D9) is the larger lever once validated.
2. Isolated runs always plan with the fallback load-to-load spread.
3. About a third of simulated +4% gains never reach a check (selection).
4. A predeclared two-look check (4, then the planned count; the existing Unbounded procedure) was as good or faster in most worlds; not adopted (D16).
5. The alternating-order test's false-accept rate rises a little above 5% under persistent drift (about 6–7% in simulation), in both modes (D17).

**Hardware follow-up completed (9 Oct):** job 125 ran once (run `d59cba6e-7bb8-4eab-8f0f-411070de4520`), finishing in 14.6 minutes including the wrapper, 14.4 minutes for tune, without an extension. Llama 23 to 27 layers passed an 11-block short-question check: balanced score +3.37% (90% interval +0.57% to +6.18%). Confirmation took 655.5 seconds against 667.8 predicted. The old selection count on THIS run was 10, so this is not proof that the additional block caused the passing verdict. Saved only in scratch; both isolation checks found all 199 real data files unchanged. Long-question behavior remains unanswered. Reporting defects and next C3 instructions are in `reviewer-confirmation-planning-2026-10-09.md`. No rerun is needed for the reporting fixes. One run cannot establish a confirmation rate.

**What job 125 established:** it confirmed a +3.37% balanced short-question gain for 27 versus 23 layers under the existing acceptance procedure. It did **not** show that the new confirmation planning raises success rates or caused this pass (the old formula would also have chosen 10 blocks; one run cannot show a rate). Its details:

- the plan was recorded on hardware;
- the forecast was 2% above the measured check time (667.8 s predicted, 655.5 s measured), with no overrun;
- an 11-block check fit inside 15 minutes;
- the realised block spread (5.13 points) was about 6% above the planning spread (4.86 points);
- the 80% planned power was not attainable within the 16-block cap (70.7% at 16; 54% at the chosen 11), so the plan was limited by the soft target;
- the long question was never measured.

**Reporting corrections (done, 9 Oct, `claude/confirmation-reporting-and-c3-2026-10-09.md`):**

- a capped plan no longer claims the planned power: `blocks_for_power` is null when unattainable, and the limit is named;
- the 87% / 71% / 46% four-block figures for job 123 are presented as estimates under three planning assumptions;
- a deferred reading whose exact setup was confirmed is no longer "unanswered".

Selected counts are unchanged.

**Supporting evidence, with denominators and intervals:**

- **Calibration estimates:** the 80th percentile for fresh final checks is 1.74, with a 90% interval of 0.97–2.42 from 10 rows; the allowance is not recalibrated.
- **A drift Monte Carlo:** 5.0% false "better" with no drift, about 6–7% at persistence 0.3–0.6.
- **Paired per-seed simulation:** 0 of 250 pairs newly past the target in each of 9 worlds.

D16/D17 stay experimental.

**Still pending:** the production default (D3). Practical mode stays internal; Unbounded stays the default.

---

## 9. Everything that remains

### 9.1 Cross-cutting work (not tied to one numbered step)

| # | Item | Status | Notes |
| --- | --- | --- | --- |
| C1 | **Commit the work** | Owner's action | Up to date at `329aad8` (16 Oct): C3-7 `df01ea3`, C4-1 `47b8d15`, C4-2 `f9d79ca`, C4-3 `b7d1e23`, R1-R5 repairs `329aad8`. Work after that is uncommitted until the owner commits it. Reviewer/audit files and job artifacts remain ignored; only the owner commits and pushes |
| C2 | **Practical-mode evaluation** (8.2) | **Done** (jobs 123 and 124, 9 Oct) | Follow-up job 125 confirmed a **+3.37% balanced short-question configuration gain**, 27 versus 23 layers, under the existing acceptance procedure. It also completed the planner timing/recording evaluation. It did not establish improved planner success rates, long-question benefit or 95% gain retention |
| C3 | **Unify tune's paths and the cards** (7.6, items 1?2) | **Shared declarations, runtime bindings and both dispatch policies implemented; independent C3-7 audit pending** (9?12 Oct) | Canonical declarations, optional card links, validated applicability and executable bindings remain in place. C3-6 moved Unbounded's option-path loop into `UnboundedPolicy`. C3-7 moves Practical's actual `_explore` control loop into `PracticalPolicy`, with captured `Step.base`, temporary exhaustion, sizing, weak repeats and one-time batch reopening preserved. Planner/evidence/acceptance owners remain unchanged. Error traces now record terminal stages and appear in reports already written by existing failure owners. Format-3 references preserve numeric simulated timing, full workload contents and per-run model/context requests; Practical round order is checked. Exact real-clock exclusions and the controlled-clock deadline-test replay are documented. Gates: 231 Practical scenarios and 328 simulated plus 107 test-driven Unbounded traces against the preserved C3-6 baseline. This completes the agreed shared-dispatch extraction, not a generic rule engine, new ordering policy or activation of more Practical paths. C4 remains separate |
| C4 | **Learning wired into `tune`** (7.6, items 3-4) | Initial consumers implemented; C4-3 shadow by default after 15 Oct review fixes | Practical stores immutable observations, forecasts compatible measured costs and reports goal-labelled candidate ordering proposals within existing eligible Steps. Applying proposals is restricted to explicit development evaluation until activation criteria are studied. Latest failed repeats cannot revive old readings; clean contrary evidence from unconfirmed runs can suppress promotion. Three eligible independent recent runs remain required. Fresh measurements and confirmation remain authoritative. Hardware reuse initially supports one detected GPU on Windows. All 231 no-history Practical replays still match owner C4-2; type checks pass on 446 files. Ten three-arm simulations retain control, explicit application and default shadow results, including known losses only under explicit application. Future path admission and interaction design require development-time studies. No historical acceptance evidence, skipped checks or new paths |
| C5 | **Retune reuse rules** (7.5) | Designed | Skip re-screening only when every dependency and the start reading still match |
| C6 | **Performance-first long-question length rule** | Waiting on the owner since 3 Oct | Study: option 1 (457 pieces, about 17 s per answer) took 74.5 min, option 2 (2,013 pieces, about 65 s per answer) took 236.4 min on the 9B. Balanced uses at most about 20 s per answer |
| C7 | **NVIDIA machine plan refresh** | Plan from 1 Oct is stale | Add: the CUDA build, NVIDIA memory readings, an engine-switch list, tune and practical runs, the layer study |
| C8 | **Usage-weighted trade-offs** | Open (owner, 5 Oct) | Weights and limits set before testing |
| C9 | **Cause of paging with flash attention off** (Gemma) | Not investigated | |
| C10 | **2.3 s stop during loading** (job 120) | Not investigated | It works, but is slower than stops during a request |

### 9.2 Open decisions (owner, with the reviewer's advice)

| # | Decision | Background | Where |
| --- | --- | --- | --- |
| D1 | ~~What job 122 means~~ **Decided 8 Oct:** inconclusive; paging-pressure work and the monitor-start change deferred | 8.1 | `claude/job122-results-2026-10-08.md` |
| D2 | ~~The practical-tune evaluation design~~ **Done:** jobs 123 and 124 | 8.2 | `claude/practical-eval-prep-2026-10-08.md` |
| D3 | The production default (whether practical mode becomes a choice or the default, and its duration) | **Pending.** Owner's design direction: a 10–15 minute soft target; Unbounded stays the default until decided | `claude/bounded-tune-decision-2026-10-06.md`, 8.2 |
| D4 | A cheaper guard for settings whose mechanism can't change writing speed much (prompt batch) | The 2% short guard costs 7–16 blocks on noisy models | `claude/budgeted-search-implementation-2026-10-06.md` §7 |
| D5 | Leave-one-out confirmation when a final check fails only on a guard question | Keeps the found gain at the cost of one more check | Same, §7 |
| D6 | Guarding placement changes on the long question when it has been sized anyway | Placement moves the long question a lot (MoE +41% long) | Same, §7 |
| D7 | A single weak-component final when nothing else is found and time allows (Qwen's +2.1% 10 layers) | The same evidence standard, spending the budget on it | Same, §7 |
| D8 | Whether pacing should allow for future repeats and discarded blocks | Today it projects only what it has observed | Onboarding handoff §13 |
| D9 | Enabling the scoped paging monitor (even only in practical mode) | Correct on attribution; about 12 s cheaper per measurement; genuine model paging not validated; unknown answers can still stall comparisons | Job 121 and 122 reports |
| D10 | Re-measuring a block whose paging is unknown or not settled, instead of ending the comparison "unsure" | Proposed after job 121 | `claude/job121-results-2026-10-08.md` |
| D11 | Grouped hybrid placement | Disabled after slow single runs | `claude/hardware-checks-2026-10-06.md` |
| D12 | Shared load in final checks (both questions from one fresh load) | Saves about 3% on Llama; second-in-load order effect unproven; not enabled | `claude/job120-results-2026-10-07.md` |
| D13 | First-look floor (about 3%) in modes that prioritise shorter tuning | Owner: allowed only where shorter tuning is explicitly prioritised; Performance first has none | `claude/search-order-v2-2026-10-06.md` §3 |
| D14 | C6: the Performance-first long-question rule | | Progress report |
| D15 | Temperature source on Windows (first task of step 6) | Needs a helper source | Build plan |
| D16 | A predeclared two-look final check in practical mode (4 blocks, then the planned count; alpha split evenly, as Unbounded already does) | Simulated as good or faster than the fixed count in most worlds; changes Practical's acceptance procedure, so not adopted | `claude/confirmation-planning-2026-10-09.md` §4 |
| D17 | The alternating-order fair test under persistent load-to-load drift | Simulated false-accept rate about 6–7% instead of 5%; affects both modes; not changed | Same, §4 |

### 9.3 Step 6: scorecard additions

**Build:**

- **The "computer stays usable" timer:** a tiny fixed task timed every few seconds.
- **Readings:** temperature (processor and card), fan speed, battery drain per minute, and power where the machine reports it.
- **The guard rule** for each new measure.
- **The three remaining goals:** computer stays usable, quiet and cool, save battery.

**Depends on:** choosing a temperature and fan source on Windows (D15).

**Interactions:**

- Heat readings let "quiet and cool" judge settings.
- The usable timer gives the watcher its responsiveness signal.
- A reading the machine can't give shows as "unknown", never a guess.

**Done when:** each reading shows a value or "unknown" on the laptop, and each new goal can run a tune that judges by its measures.

### 9.4 Step 7: more model settings, tuned together (finish)

**Build:**

- the memory-residency path (5.8);
- prompt reuse (5.9);
- word drafting (5.10; later in the step);
- a real tune that exercises per-piece placement and the memory format;
- bringing those two into practical mode, following 7.4 (after 8.2);
- C3, the card unification, which is what "tuned together" needs.

**Done when** (from the plan): the combined search matches or beats one-at-a-time.

Proposed measure: on the same starts, the combined setup's verified gain is at least the best single change's verified gain, with no guard breach.

### 9.5 Step 8: operating-system cards and the undo journal

**Build the undo journal first:**

- before any operating-system change, write the old value and how to restore it;
- close the entry when the test ends;
- on the next start, undo any open entries.

**Then the cards** from 5.11 and 5.12, each with admin rights asked per change and approval where the card requires it.

**Interactions:** core choice pairs with threads; priority and power plan affect every speed reading; keep-in-memory and large pages pair with the memory-residency path.

**Done when:** a test cut off by a crash is undone on the next start.

### 9.6 Step 9: the watcher

**Build:**

1. **Low-cost watching:** under 2% of the model's speed. The scoped monitor's design (one reader per session, raw counts) is the likely base. Its cost was measured at about +1 s per measurement versus +12.8 s (job 121).
2. **Comparison against the saved setup,** not just against how things looked when watching began.
3. **Recommend-only fixes** with one-click apply.
4. **Asking for an in-depth test** when the machine is clearly struggling. The request is dropped if ignored, and asked again later only if things are still bad.
5. **The in-depth test:**
   1. save the model's progress (llama.cpp can save and restore a conversation slot; to be confirmed for the installed build);
   2. run a test sized by the effort mode and aimed at the goal mode, through the same acceptance rules as `tune`;
   3. restore the model;
   4. ask for approval.
6. **Kept changes saved** to the profile.

**Today:** an older `monitor` command exists. According to the 30 Sep audit, which has not been re-verified since, it watches a model InferenceOS starts and feeds with its own prompts, can run the layer search and asks approval, but kept changes are not saved and its cost was 4–8%. It is not the designed watcher.

**Done when:** watching costs under 2% on the laptop, a struggling scenario produces a request, and an approved in-depth test restores the model's progress and saves an approved change.

### 9.7 Step 10: Ollama and LM Studio connectors

**Build:** one connector per program that can:

- start and stop it;
- read its settings;
- apply and undo a setting;
- read per-request speed;
- discover the installed version's options.

| Program | How |
| --- | --- |
| Ollama | Start the service, load models through it; per-request settings are allowed for testing |
| LM Studio | Its command-line tool; exact controls to be checked against the installed version |

**Interactions:**

- Each card's "programs" field says how it is set in each program.
- A path unsupported by a program is simply inapplicable there.
- All results are tied to the program and version.

**Done when:** `tune` and `start` work through each connector on at least one model.

### 9.8 Step 11: the app

**Views:**

- **Home:** model, program, effort and goal; "Optimize before loading" or "Watch".
- **Live:** the scorecard against the saved setup.
- **Waiting for you:** proposed changes with Keep or Undo.
- **History:** every test, with undo on anything still in place.
- **Settings.**
- **Advanced:** the single-step commands.

The window vs browser decision is made at this step.

### 9.9 Step 12: Linux

Port the counters (paging from the kernel's counters), process control, the machine check and the operating-system cards. Re-learn every curve; assume nothing carries over.

### 9.10 Later, not yet placed

- **Image models** with sample images, plus a quality check for image answers.
- **Machine speed tests** (drive, memory and core speed) in the machine check. A memory-speed test already exists for diagnosis.
- **The NVIDIA machine** (C7).
- **Calculation code and builds** (5.13): a general capability so that new hardware needs only testing.
- **Context budgeting:** only after the separate context-injection project is finished, and only as a distinct, newly authorised effort.

## 10. The target design: how it should all fit together

This section separates what is **decided** (marked "decided") from the implementer's **proposal** for meeting it (marked "proposal"). The proposal needs the reviewer's scrutiny and the owner's go before any of it is built.

### 10.1 The parts and their jobs

**Connectors** (decided, step 10):

- one per program;
- start, stop, apply, undo, read speed, discover options.

**Facts:**

- the machine: cores, memory, card and its free memory, power source and plan, driver; later, speed tests;
- the model: its blocks and their kinds, experts, sizes, how many layers fit;
- the engine: build, back end, the options it really has, the switch list for this hardware.

**Diagnosis:**

- **What it does:** from one or two cheap timed runs, it says what is limiting right now. Examples: processor-bound vs card-bound writing, memory-speed ceiling, prompt reading, paging, other programs, heat.
- **What it produces:** which paths are worth entering, which need a cheap probe, and which are clearly inapplicable.
- "Nothing worth fixing" is a valid output.

**Cards** (decided format; proposal for the extra fields):

- every optimization is one card;
- **proposal:** to make coordination general, each card would also declare its interaction facts:
  - the resources it uses or frees: card memory, processor share, prompt reading, system memory;
  - the measures it moves;
  - the settings it depends on;
  - the settings it invalidates when kept;
  - its search method (each path keeps its own specialised search);
  - a cost model, learned from measurements.

  The `AFFECTS` table and the planner's ordering rules would then be **computed from the cards**, not hand-written.

**Coordinator** (proposal, generalising today's planner):

- orders the applicable cards by the ordering principle (7.4) and by expected gain per minute;
- enforces evidence scoping and targeted rechecks;
- keeps earlier complete setups and subsets as candidates;
- plans verification time from measured costs;
- stops when the effort mode's time is used or gains stop coming.

**Measurement service** (built):

- fresh loads, warm-ups and run counts;
- disturbance repeats;
- phase timings;
- the paging monitor (per-answer today; the scoped monitor if enabled);
- heat notes.

**Acceptance** (built, decided):

- the fair comparison, guards, answer quality, applied-settings verification;
- one confirmation of the whole combination;
- trade-offs offered, never kept on their own.

**Safety layer:**

- engine stop/start safety (built);
- the undo journal (step 8);
- the context-size check on every tried setup (built);
- admin rights per change; approval for kept changes and for cards that touch other programs (decided).

**Profiles and learning** (partly built):

- saved setups per model, context, power source and program (built);
- **to add (C4):** values tried and their outcomes, measured costs, near-match starting guesses, the retune reuse rules (7.5).

**The watcher** (step 9): watches at under 2% cost, compares against the profile, recommends, and asks for in-depth tests that go through the same coordinator and acceptance.

**The app** (step 11): a screen over all of the above.

### 10.2 One tune, in the target design

1. **Connector** starts nothing yet. **Facts** are read; past results are checked (with the 7.5 reuse rules).
2. **Measure the start;** check that it is still safe for memory.
3. **Diagnosis** names the limiting cause, or "nothing worth fixing", which ends the tune.
4. **Coordinator** picks the applicable cards. It orders them resource-shaping first, then placement, then work allocation, then calculation path, then system, then workload. Within each tier, it orders by expected gain per minute and checks that verification still fits the effort mode's time.
5. **Each card runs its own search.** Every reading is scoped to its exact setup. A kept upstream choice triggers targeted rechecks of its declared dependants.
6. **The best combination is confirmed once,** with the strictest quality requirement. Trade-offs are offered separately.
7. **Approval,** then an atomic save to the profile. Learning records values tried and costs.
8. **`start`** uses the profile. **The watcher** watches it.

### 10.3 What stays specialised

Integration must not force one search on every path:

- The layer search keeps its own logic: range, load-only tries for the top, extremes first, intermediate counts.
- The memory format keeps its known-answer policy.
- Engine switches keep proof-of-effect.
- Operating-system cards keep approval and the undo journal.

The coordinator only decides **when** each runs, **in which setup**, and **what must be rechecked**.

---

## 11. Risks, unknowns and limits of the evidence

1. **The practical mode has run on hardware three times** (jobs 123, 124 and 125, short question only). Its retention numbers still come from simulated worlds whose curves between real readings are invented.
2. **Genuine model paging during answers has never been produced.** The paging monitor is proven on attribution of independently caused paging, not on a model that really pages while answering (job 122).
3. **Verification cost is the bottleneck on this laptop.**
   - A long-question check at the 4-block minimum takes about 25 min.
   - A 2% "not worse" short guard needs 7–16 blocks on noisy models.
   - In the three 15-minute hardware runs only short-question changes (placement, threads) were checked; batch was left out on a predicted cost and never measured.
   - This is a property of the acceptance rules (5% false keep, 2% margin), which the owner has kept.
4. **Small gains are hard to confirm in 15 minutes on this laptop.** Job 123's blocks varied by about 5 points; a 4% gain then needs about 11 blocks (about 10 minutes of checking). A quick "no verified change" is often the honest answer.
5. **Interactions measured directly are few.** Threads × batch rests on reasoning and simulation.
6. **The cards and tune's paths are separate** (7.6). Claims of "full integration" would be false today.
7. **Hardware breadth:** one AMD laptop. NVIDIA, Linux, Ollama and LM Studio are untested. Curves must be re-learned on each.
8. **Current baseline (16 Oct):** owner commit `329aad8` (R1-R5 repairs) after `b7d1e23` (C4-3), `f9d79ca` (C4-2), `47b8d15` (C4-1) and `df01ea3` (C3-7). Source snapshots and raw before/after evidence are retained locally under ignored `_jobs/out/`; these artifacts are not a remote backup.
9. **The 30 Sep audit** (`claude/current-state-audit.md`) describes the code before steps 1–5 and the later work. Re-verify before relying on it.

---

## 12. Working rules (for the implementer and the reviewer)

**The owner's rules:**

- Plain, simple wording; short sentences; no unexplained jargon. Answer the question asked.
- Ask before building. Nothing proceeds until what is to be done is clear.
- Never commit or push unless the owner asks. Job scripts, the runner and agent notes are git-ignored.
- Keep Unbounded unchanged. Keep the practical mode internal. Choose no default duration. Never claim "95% retention" generally.
- Keep the monitor disabled by default (`tune_experimental_measurement_monitor = False`). Thresholds, acceptance policy and Unbounded behaviour change only by the owner's decision.
- Never queue laptop jobs without the owner's approval. Prepared jobs wait in `_jobs\later\`. Don't rerun a job without a reason.
- Preserve each file's line endings. Some are CRLF, for example `backends/resident_server.py` and `tests/experiment_helpers.py`.

**Implementer process:**

- Reproduce a defect on the old code before fixing it.
- Run the focused tests, the full suite (21 known Windows-only failures are expected), the type check and the Unbounded simulation check (0 of 160 differ).
- Deliver to the laptop with mtime guards and byte-for-byte checks.
- Write a dated fix report under `claude/`, and update `claude/progress-report.md`.

**Reviewer:**

- Reviews only from what the owner pastes.
- Is asked at critical points: before tune's first real keeps on new rules, after a large build, and before hardware jobs.
- Is not asked to confirm fixes of their own findings unless needed.

**Hardware rules:**

- Plugged in only.
- Compared tests run under the same thermal conditions.
- Tests that ran on battery are redone.

---

## 13. Glossary and document index

### 13.1 Plain terms and their code names

| Plain term | Meaning | Code name |
| --- | --- | --- |
| Layers on the card | How many model layers the graphics card runs | `gpu_offload_layers`, `-ngl` |
| Layer top | The most layers that load without running out of card memory | `find_layer_top` |
| Threads | Processor threads while writing | `threads`, `-t` |
| Reading threads | Processor threads while reading the question | `threads_batch`, `-tb` |
| Prompt batch | Prompt tokens processed at once | `ubatch_size`, `-ub` |
| Conversation memory format | How keys and values are stored | KV cache type, `--cache-type-k/-v` |
| Per-piece placement | Which single pieces go on the card | `placement_overrides`, `--override-tensor` |
| Writing speed | Words per second while the answer appears | `decode_tokens_per_sec` |
| First word | Time until the answer starts | `ttft_s` |
| Paging | Pages read from disk to resolve memory faults | `hard_page_faults_per_sec`, `\Memory\Pages Input/sec` |
| Measurement | Fresh load + warm-up + N measured answers | `measure_steady` |
| Block | One start-vs-candidate pair of measurements | `compare_fairly` |
| Look | A planned decision point after a fixed number of blocks | `min_blocks` / `max_blocks` |
| Light check | Unbounded's 4-block screening check; only chooses | `tune_path_check = light` |
| Final check | One fresh check of the whole combination; the only acceptance | `_check_whole_setup` |
| Guard | A measure that must not get worse | `guard_workload_id`, paging guard |
| Smallest worthwhile gain | The effort mode's minimum gain to keep | 5% / 1.5% / 15% |
| Unbounded | The full search with no time limit (default `tune`) | `Tuner` |
| Practical mode | Time-limited search (internal) | `BudgetedTune`, `tune_experimental_budget_s` |
| Planner | Decides what to measure next in practical mode | `tune/planner.py` |
| Scoped paging monitor | One raw-count reader per measurement (experimental) | `MeasurementMonitor`, `tune_experimental_measurement_monitor` |
| Bounds (paging) | Least and most pages an answer can have, when some pages can't be placed | `pages_lower` / `pages_upper` |
| Memory-safety correction | Lowering an unsafe saved layer count to what loads | safety correction in `pipeline.py` / `budgeted.py` |
| Job | A PowerShell script run by the laptop's runner | `_jobs\NNN_*.ps1` |

### 13.2 Project document index (the most useful ones)

**Read first:**

| Document | What it holds |
| --- | --- |
| `claude/project-scope-and-roadmap-2026-10-08.md` | This document |
| `claude/project-core-idea.md` | The owner's purpose, in brief |
| `InferenceOS Description` | The owner's full operating instructions (sections 15–16 are the decided design) |
| `claude/progress-report.md` | Live status, step tracker, dated log |
| `claude/build-plan.md` | The plan and decisions table, mirroring the living doc |

**Design and decisions:**

- `claude/optimization-paths.md` and `claude/optimization-candidates.md`: the path list and the diagnosis-first view;
- `claude/bounded-tune-decision-2026-10-06.md`: why a time-limited mode; the verification floor;
- `claude/search-order-v2-2026-10-06.md`: the ordering policy, retune reuse, verification breakdown;
- `claude/budgeted-search-implementation-2026-10-06.md`: the planner and practical mode as built; open conflicts;
- `claude/reviewer-onboarding-handoff-2026-10-07.md`: the most detailed description of measurement, the fair comparison, Unbounded and practical mode.

**Results:**

- `claude/model-tunes-2026-10-05.md`: the six-model tunes;
- `claude/hardware-checks-2026-10-06.md`;
- `claude/job120-results-2026-10-07.md`, `claude/job121-results-2026-10-08.md`, `claude/job122-results-2026-10-08.md`;
- `claude/layer-study-laptop-results.md`;
- `claude/practical-eval-prep-2026-10-08.md` and the reviewer's `reviewer-practical-evaluation-2026-10-09.md` (jobs 123 and 124);
- `claude/confirmation-planning-2026-10-09.md` (the follow-up).

**Paging monitor series** (newest last):

- `claude/monitor-correction-2026-10-07.md`;
- `claude/monitor-correction-2-2026-10-08.md`;
- `claude/monitor-correction-3-2026-10-08.md`;
- `claude/boundary-bounds-2026-10-08.md`;
- `claude/validation-fixes-2026-10-08.md`;
- `claude/probe-hardening-2026-10-08.md`.

**Reviews and fix reports:**

- `claude/review-*.md`, `claude/review*-fixes-*.md`, `claude/reviewer-followup-*.md`, `claude/reviewer-handoff-*.md`.

**Other:**

- `claude/nvidia-machine-plan.md` (needs refresh, C7);
- `claude/current-state-audit.md` (30 Sep; outdated in places).

Job 128 v7 (18 Oct): independent review corrections replace the detached-symbol list
with a character-property policy and reject sign words across punctuation boundaries;
the wrapper pins frozen governing rules before dispatch. See the same maintained
`reviewer-implementation-handoff-current.md` for decisions and verification. Not run.

Job 128 v8 (18 Oct): the full intended-use audit fixes split punctuation/sign context,
unsupported Unicode spelling crashes and formatted-unit regressions; strengthens native
and chat work/identity/cleanup/power evidence; prevents mixed-configuration conclusions;
requires complete effective-SWA logs; reserves output names atomically and permits
`done` only at final publication. Results carry frozen-rule and model/engine provenance.
Launch failure/containment bookkeeping and explicit Win32 handle signatures are fixed
without changing tune scheduling or acceptance. The same maintained handoff records
each choice, rationale, evidence and remaining limits. Software checks pass (2897 broad tests; Practical 231/231 and
Unbounded 328 scenarios unchanged); job 128 remains prepared in `later`, with no hardware run or production-default change.
