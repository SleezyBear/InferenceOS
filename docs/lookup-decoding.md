# Lookup decoding: development capability and study

Lookup decoding proposes upcoming tokens by finding a matching sequence in the current
prompt/history. The main model verifies the proposal. This implementation exposes only
llama.cpp b11139 `ngram-simple` and an explicit `none` control. It does not load a helper
model, download anything or read/write persistent lookup-cache files.

It is **not an automatic tuning path**. No saved setup, production default, Practical
admission or history identity changes. None of the new engine flags is emitted during
an ordinary tune or start. The optional backend parameters and generated-ID request are
used by the development evaluation only.
The legacy process-per-request CLI adapter explicitly refuses these native-server-only
options instead of silently running without them.

## What is implemented

- Validated, bounded lookup/draft lengths and algorithm-specific command flags.
- Optional native generated token IDs collected across streaming chunks. Empty terminal
  arrays do not erase earlier IDs; malformed or incomplete IDs refuse the reading.
- Native effective generation settings and drafted/accepted counts. b11139 omits both
  draft counters when no work is drafted; that does not establish a lookup gain.
- Explicit type flags append to b11139's inert `none` default. Only the registered
  active algorithm is accepted; combinations with another draft method are refused.
- Matched fresh-load requests, retained checkpoints, strict work/quality/power/cleanup
  gates, model/setup/input pins and before/after data-folder maps.
- An offline auditor that recomputes the full population, outputs, statistics and
  per-load implementation evidence rather than trusting a stored winner label.
- A simulated engine and adversarial software tests. Simulated speedups are not hardware
  evidence. No lookup hardware result exists at delivery of this software batch.

## Prospective job 130

The wrapper remains in `_jobs/later/130_lookup_decoding_study.ps1`, prepared for independent
review and separate owner hardware authorization. It has not been queued or run.
The frozen governing artifact is `130_rules_lookup-study-1-r5.json` in that folder. The
existing single implementation handoff records all decisions and software evidence;
this page is capability/protocol documentation, not another handoff.

The proposed population is Llama 3.2 3B Q4_K_M and Gemma 3 1B Q4_K_M, using their verified
saved AC setups at context 4096, read only. Each baseline's complete configuration
digest is frozen too; a later retune requires revised reviewed rules. Four workloads cover copying prose,
copying code, extracting rows and composing prose with less copying. For each model:

- Five blocks per workload, explicit `none` versus `ngram-simple`, alternating order.
- A separate fresh load for each arm, exactly one request per load, no priming or
  prompt reuse. Forty loads per model, eighty total, ten-minute inter-model rest.
- Seed 42, temperature 0, natural EOS with a maximum of 128 generated tokens. Complete
  token IDs, text and ending must agree between the paired arms. Shorter work in one arm
  cannot count as a speedup. Output identity preserves the reference; it does not prove
  that the reference is factually correct.
- Lookup n-gram size 12 and draft m-gram size 16. No draft-model `n_max`, cache file,
  synthetic acceptance, helper weights, alternate context, parameter sweep or retries.
- Same trace logging and process-image capture in both arms. Input hashing/inspection
  happens outside the request measurement. Instrumentation still affects execution
  conditions, so conclusions apply to this instrumented study.

The primary endpoint is paired relative reduction in **whole request wall time**. Draft
acceptance and decode throughput alone cannot establish an improvement. TTFT and process
peak RSS are secondary observations, with a clear first-word regression labelled as a
trade-off. Five fixed blocks provide one final look. For primary latency gains, the
one-sided family alpha is .05 across eight registered model/workload comparisons
(.00625 each). Secondary TTFT intervals are descriptive; there is no joint error
guarantee across latency, TTFT and memory. No early success look,
pooling across models/workloads, dropped negative case or post-result threshold change.
Block-noise independence and representativeness remain assumptions; a narrow interval
does not establish performance on ordinary prompts, another machine or another engine.
The native accepted counter is verified draft work, not the number of emitted draft
tokens: b11139 updates it before stopping a verification batch at a natural EOS.
TTFT here is time to the first nonempty streamed content, as in the existing backend.
The endpoint includes prompt reading and answer generation, but excludes engine/model
startup, tokenizer inspection and input hashing. It is a resident-request comparison,
not a demonstration that a one-shot launch or the full product becomes faster.
Each measured request is the first after a fresh load, so lazy initialization may be
included. There is no warmed steady-state arm or throughput/concurrency claim.
Here "whole request" means the bounded native request with its 128-token ceiling;
a limit ending can stop before the entire requested document is copied. Matching that
window does not certify later tokens or completion of the user's larger task.

Every candidate needs observed implementation/size initialization in its own complete
load logs, matching observed engine-library backing-file hashes, effective native
settings and complete request work. Accepted draft activity is required for an attributed
lookup gain. A legitimate zero-opportunity case stays a reported no-opportunity result.
Ready process/log and implementation evidence is gated before the measured request.
The log reader may wait up to two seconds for the existing readiness text before its
single snapshot; this adds no inference probe. Incomplete evidence refuses measurement.

The following prevent later dispatch: power failure, changed frozen inputs/model,
unverified work, output mismatch, request/start error, interruption or unresolved process/
credential cleanup. Every attempted row remains visible. Ctrl+C is checkpointed and
re-raised. Only final publication after input/data/release checks may report `done`.
An offline audit must bind the result/run/model/rules and demonstrate all expected
observations before performance findings can be interpreted. An unsuccessful first
model stops the wrapper before the second.
After release, the runner also inspects the stored ready/before-stop observations
before another load. A late capture problem stops dispatch while retaining the actual
completed reading and clean-release evidence; it is not relabelled a process leak.

Engine identity is enforced during dispatch as well as offline. Before backend
construction, the executable actually selected by the configurable toolchain directory
must resolve to the frozen executable path and match its hash. Each load checks the
selection and input generations again, and the factory uses that validated path rather
than resolving a second independent executable. Even an identical copy at an unregistered
path is refused. The ready and released images use the auditor's same required-library
and frozen-hash checker: all five required executable/implementation/Vulkan components
must be observed, and every observed frozen engine component must match. Contradictory
ready evidence prevents generation; contradictory release evidence prevents another
load. Windows system libraries are observed but are not frozen toolchain components.
These backing-file observations do not attest resident instructions or eliminate races.

Both arms force `--offline`. The Windows system/user `llama.cpp/config.ini` paths are
checked before engine construction and each load, with their environment/path receipt
retained. Existing or unreadable config files refuse the study without editing them.
Startup evidence that a config file was loaded also refuses measurement. This prevents
an ambient preset from silently adding a helper model or other experiment settings;
the ordinary product's environment handling is unchanged.

Even a complete positive study does not activate a tune path. Admission requires a later
reviewed design using this evidence and a fresh confirmation of the complete selected
configuration. If it does not establish a useful improvement, retain the baseline.
This study holds the existing saved layer/thread/batch/cache choices fixed. It does not
test joint retuning, interaction with prompt reuse, or ordinary cached chat traffic.
Lookup adds CPU and verification work, so a gain on one saved setup cannot simply be
added to gains from other paths. Those combinations need their own quality and complete-
configuration checks before admission; job 129's narrow Gemma result supplies no general
permission to combine reuse and lookup.

## Software-only reproduction

Use a fresh output filename; result and audit writers refuse to overwrite evidence:

```powershell
.venv\Scripts\python.exe -B scripts/evaluate_lookup_decoding.py --simulate --out _jobs/out/lookup_SIMULATED_new.json
.venv\Scripts\python.exe -B -m pytest tests/test_lookup_decoding.py tests/test_lookup_study.py tests/test_lookup_job.py
```

The real wrapper and its frozen digest are intentionally not turned into an ordinary
product command. Review the frozen rules and authorize the hardware study separately.
After such a run, its offline audit can be repeated without any engine:

```powershell
.venv\Scripts\python.exe -B scripts/audit_lookup_decoding.py RESULT.json --rules _jobs/later/130_rules_lookup-study-1-r5.json --rules-sha256 FROZEN_SHA256 --out NEW_AUDIT.json
```

## Sources and limits

The implementation was checked against the installed help fixture and primary b11139
sources: [speculative implementations](https://github.com/ggml-org/llama.cpp/blob/b11139/common/speculative.cpp),
[native response counters](https://github.com/ggml-org/llama.cpp/blob/b11139/tools/server/server-common.cpp),
[stream/slot handling](https://github.com/ggml-org/llama.cpp/blob/b11139/tools/server/server-context.cpp),
and [request schema](https://github.com/ggml-org/llama.cpp/blob/b11139/tools/server/server-schema.cpp).
Source URLs and downloaded hashes are retained with the implementation review artifacts.
Source inspection is not proof that a resident process executes those exact instructions.

Frozen input hashes and per-load file-generation checks are bounded observations, not
OS file locks. Mapped-library observations identify backing files, not resident code.
Retained equal data maps do not exclude transient writes. The previous unexplained
access violation remains a separate unresolved limitation; this work does not diagnose it.
The native stream can omit individual IDs while buffering incomplete UTF-8. Such a
reading is explicitly unusable if the collected IDs do not equal the final written-token
count; the study neither invents IDs nor repeats the request to repair that evidence.
Exact greedy preservation is deliberately strict: hardware/backend variation may make
the study inconclusive even when a user might accept the different wording.
