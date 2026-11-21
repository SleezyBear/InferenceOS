# Prepared 9 Oct 2026: second isolated practical-mode evaluation, authorized by owner.
# Runs ONE practical-mode tune after the completed Llama evaluation:
#   model      Granite 3.1 MoE (models\study\granite-3.1-3b-a800m-instruct-Q4_K_M.gguf, SHA-256 checked)
#   start      job 102's starting configuration, reconstructed from its report:
#              33 graphics card layers, context 4096, every other setting the engine
#              default (no saved setup is read: the scratch store starts empty)
#   modes      effort performance_first, goal balanced (as job 102)
#   target     15 minutes soft target (900 s); existing extension policy:
#              max(20% of the target, 120 s) = 180 s. These are planning allowances,
#              not hard limits: work starts or continues while its PROJECTED finish is
#              within 18 minutes (a final check at least three quarters done: within
#              21), checked at measurement boundaries; a completed passing check is
#              kept even when it finishes late
#   monitor    the default per-answer monitor (the scoped monitor stays off)
#   rules      every setting except the evaluation's own (scratch folders, modes, target,
#              fixed start) at its code default: workloads, goal weights, quality checks,
#              guards, acceptance thresholds (checked before the run; any difference refuses)
#   checks     the existing fresh whole-setup final check only (no extra comparison,
#              no Unbounded rerun)
# Isolation: every store the tune reads or writes (saved setups, load-to-load spread,
# machine check, memory-speed cache), its reports and logs live in a fresh scratch folder
# under _jobs\out. The owner's data folder is only read: it is fingerprinted (size, time,
# SHA-256 of every file) before and after, by this wrapper AND by the Python runner;
# any difference is an isolation failure (exit 6), also after a failure or a stop.
# Expected: 15-minute soft target, with the same extension policy as job 123.
# No duration or gain is promised. This is a contrasting CPU-placement case after
# job 123 completed without a confirmed gain. The historical starting configuration
# and model SHA-256 come from job 102. Owner authorized follow-up testing on 9 Oct.
# Exit: 0 the evaluation completed (any outcome, including "no change" -- the report says
# why), 2 the tune failed, 3 runner error, 6 isolation breach, 7 refused before starting,
# 8 reporting error (missing, malformed or contradictory evidence: the runner's exit code,
# the summary's status and outcome, the completion record and the report and explanation
# files must all agree before this wrapper reports completion).
# ASCII only: Windows PowerShell 5.1 reads a file without a BOM as ANSI.
# The parameters exist for the wrapper's own regression test
# (tests/test_job_scripts.py); the runner starts the job with none.
param(
  [string]$Root = 'C:\Users\madha\Projects\InferenceOS',
  [string]$Python = '.\.venv\Scripts\python.exe',
  [string]$RealData = 'data',
  [switch]$SkipPowerWait
)
Set-Location $Root
$plugged = [bool]$SkipPowerWait
for ($i = 0; (-not $plugged) -and $i -lt 360; $i++) {
  try { $b = Get-CimInstance Win32_Battery -ErrorAction Stop } catch { $b = $null }
  if ($null -eq $b -or $b.BatteryStatus -ne 1) { $plugged = $true; break }
  if ($i -eq 0) { "on battery: waiting to be plugged in $(Get-Date -Format s)" }
  Start-Sleep -Seconds 10
}
if (-not $plugged) {
  "still on battery after 60 minutes: nothing run $(Get-Date -Format s)"
  exit 4
}
$sw = [Diagnostics.Stopwatch]::StartNew()
$runId = [guid]::NewGuid().ToString()
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $runId.Substring(0, 8)
New-Item -ItemType Directory -Force -Path '_jobs\out' | Out-Null
$scratch = "_jobs\out\124_scratch_$stamp"
$report = "_jobs\out\124_practical_granite_moe_$stamp.json"
$summary = "_jobs\out\124_practical_granite_moe_${stamp}_summary.json"
$explain = "_jobs\out\124_practical_granite_moe_${stamp}_explain"

function Get-Fingerprint([string]$Folder) {
  # Every file in the owner's data folder: size, last write time and SHA-256 (read only).
  $map = @{}
  if (Test-Path -LiteralPath $Folder) {
    $base = (Resolve-Path -LiteralPath $Folder).Path
    foreach ($f in Get-ChildItem -LiteralPath $Folder -Recurse -File) {
      $rel = $f.FullName.Substring($base.Length).TrimStart('\', '/')
      $hash = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash
      $map[$rel] = "$($f.Length)|$($f.LastWriteTimeUtc.Ticks)|$hash"
    }
  }
  return $map
}

function Compare-Fingerprint($Before, $After) {
  $problems = @()
  foreach ($k in $Before.Keys) {
    if (-not $After.ContainsKey($k)) { $problems += "${k}: removed" }
    elseif ($Before[$k] -ne $After[$k]) { $problems += "${k}: changed" }
  }
  foreach ($k in $After.Keys) {
    if (-not $Before.ContainsKey($k)) { $problems += "${k}: created" }
  }
  return $problems
}

"== practical-mode evaluation: Granite 3.1 MoE, 15 min target, job 102 start $(Get-Date -Format s)"
"run id $runId; scratch $scratch"
$before = Get-Fingerprint $RealData
"owner's data folder fingerprinted: $($before.Count) file(s)"
$code = $null
try {
  & $Python -u scripts\run_practical_eval.py `
    --model models\study\granite-3.1-3b-a800m-instruct-Q4_K_M.gguf --context 4096 --budget-s 900 `
    --start-config 'gpu_offload_layers=33' --effort performance_first --goal balanced `
    --scratch $scratch --out $report --summary $summary --explain $explain --run-id $runId `
    --expect-model-sha256 48e0edcd578fd4462f26127f04c651d0e650741110185297741089aea01a82b3 `
    --historical _jobs\out\102_granite31moe_report.json `
    --historical-label 'job 102 (Unbounded, 5 Oct 2026, earlier code version)' 2>&1
  $code = $global:LASTEXITCODE
} catch {
  "could not start the runner: $($_.Exception.Message)"
  $code = $null
}
$after = Get-Fingerprint $RealData
$problems = Compare-Fingerprint $before $after
$minutes = [Math]::Round($sw.Elapsed.TotalMinutes, 1)
if ($problems.Count -gt 0) {
  "ISOLATION BREACH: the owner's data folder changed: $($problems -join '; ')"
  "finished after $minutes min"
  exit 6
}
"owner's data folder unchanged ($($after.Count) file(s))"
$result = $null
try { $result = Get-Content -Raw -LiteralPath $summary -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { $result = $null }
if ($null -eq $result -or $result.run_id -ne $runId) {
  "no summary for this run ($summary); runner $(if ($null -eq $code) { 'did not start' } else { "exit $code" })"
  "finished after $minutes min"
  exit 3
}
if (-not $result.isolation.real_data_unchanged -and $null -ne $result.isolation) {
  "ISOLATION BREACH (runner): $($result.isolation.problems -join '; ')"
  "finished after $minutes min"
  exit 6
}
"summary: $summary"
"report : $report"
"explain: $explain.md"
$status = [string]$result.status
$outcome = [string]$result.outcome
"runner status: $status; tune outcome: $outcome; runner exit $(if ($null -eq $code) { 'none' } else { $code })"
"finished after $minutes min"
if ($code -eq 7 -and $status -like 'refused*') { "REFUSED before starting (see the summary's preflight list)"; exit 7 }
if ($code -eq 6) { "ISOLATION BREACH (runner)"; exit 6 }
if ($code -eq 3 -and $status -eq 'error') { "RUNNER ERROR"; exit 3 }
if ($null -eq $code) { "RUNNER ERROR (the runner did not start)"; exit 3 }
if ($code -eq 8 -and $status -like 'reporting error*') { "REPORTING ERROR: $status"; exit 8 }
# Completion (or a tune failure) needs the evidence the runner recorded, checked here too.
$problems = @()
$complete = $false
if ($null -ne $result.completion) { $complete = ($result.completion.complete -eq $true) }
if (-not $complete) { $problems += 'the summary does not record complete evidence' }
if ($null -ne $result.completion -and $result.completion.problems) { $problems += @($result.completion.problems) }
if (-not (Test-Path -LiteralPath $report -PathType Leaf)) { $problems += "the report is missing ($report)" }
if ([string]::IsNullOrWhiteSpace($outcome)) { $problems += 'the summary has no outcome' }
if ($code -eq 0) {
  if ($status -ne 'done') { $problems += "runner exit 0 but the summary status is '$status'" }
  if ($outcome -like 'failed*') { $problems += "runner exit 0 but the outcome is '$outcome'" }
  foreach ($f in @("$explain.md", "$explain.json")) {
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { $problems += "the explanation is missing ($f)" }
  }
} elseif ($code -notin @(3, 6, 7, 8)) {
  if ($status -ne 'failed') { $problems += "runner exit $code but the summary status is '$status'" }
  if (-not ($outcome -like 'failed*')) { $problems += "runner exit $code but the outcome is '$outcome'" }
} else {
  $problems += "runner exit $code does not match the summary status '$status'"
}
if ($problems.Count -gt 0) {
  "REPORTING ERROR (evaluation not completed): $($problems -join '; ')"
  exit 8
}
if ($code -ne 0) { "TUNE FAILED: $outcome"; exit 2 }
"EVALUATION COMPLETED: $outcome"
exit 0
