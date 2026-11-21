# PREPARED, NOT QUEUED (boundary-aware paging bounds, after job 121): validates the
# measurement-scoped monitor's bounded evidence on hardware -- paging at answer edges
# and genuine model paging. Move this file into _jobs\ only when the owner approves.
# Nothing is tuned or saved; the monitor stays off by default whatever the result.
#   Step 1  attribution against independently caused paging (no model): bursts inside,
#           before, after and across the edge of timed pseudo-answers, plus quiet ones;
#           the truth of each answer's share comes from the generator's clock reads
#           around every chunk it touched; the evidence (exact / bounded / unknown) and
#           full-precision raw readings are saved for replay                  ~4 min
#   Step 2  genuine model paging: Llama 3.2 3B run on the processor only from an
#           uncached copy of the model file, so its first answer really reads the
#           weights from disk; scoped monitor only, warm-ups judged too, 2 measurements
#           (the first cold, the second warm); an independent raw counter sampler at a
#           different cadence re-judges every answer as a CONSISTENCY check (same
#           algorithm, other read moments: bounds must overlap; certain pages cannot
#           exceed what it counted); inconclusive if paging was not exercised ~2 min
# Expected total: about 6-8 minutes. It never runs on battery: if the laptop is not
# plugged in within 60 minutes it exits (status 4).
# "all steps passed" is printed only when every step started and exited 0 AND each
# wrote a NEW results file for THIS run (fresh name, this run's id, status "done").
# Step 2 runs only after step 1 passed; otherwise it is "not run" (exit 2 when step 1
# failed, 5 when it was only inconclusive). Inconclusive is never a success.
# ASCII only: Windows PowerShell 5.1 reads a file without a BOM as ANSI.
# The parameters exist for the wrapper's own regression test
# (tests/test_job_scripts.py); the runner starts the job with none.
param(
  [string]$Root = 'C:\Users\madha\Projects\InferenceOS',
  [string]$Python = '.\.venv\Scripts\python.exe',
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
$failed = @()
$runId = [guid]::NewGuid().ToString()
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $runId.Substring(0, 8)
New-Item -ItemType Directory -Force -Path '_jobs\out' | Out-Null

function Get-ExitText($code) {
  if ($null -eq $code) { return 'did not start' }
  return "exit $code"
}

function Test-Result([string]$Label, [string]$Path, $Code) {
  # Returns @(state, text): 'passed' only when the step exited 0 AND wrote a NEW results
  # file for THIS run with status "done"; 'inconclusive' when this run's file says
  # "inconclusive: ..." (nothing was wrong, but nothing was proven either); otherwise
  # 'failed' (a failure, an error, no start, no file, or another run's file).
  $result = $null
  try { $result = Get-Content -Raw -LiteralPath $Path -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { $result = $null }
  $mine = ($null -ne $result) -and ($result.run_id -eq $runId)
  if ($mine -and ("$($result.status)" -like 'inconclusive*')) { return @('inconclusive', "$Label $($result.status)") }
  if ($null -eq $Code) { return @('failed', "$Label (did not start)") }
  if ($Code -ne 0) {
    if ($mine) { return @('failed', "$Label (exit $Code): $($result.status)") }
    return @('failed', "$Label (exit $Code)")
  }
  if ($null -eq $result) { return @('failed', "$Label exited 0 but wrote no readable results file for this run ($Path)") }
  if (-not $mine) { return @('failed', "$Label exited 0 but the results file is not this run's (run id '$($result.run_id)')") }
  if ($result.status -ne 'done') { return @('failed', "$Label exited 0 but its results file says status '$($result.status)'") }
  return @('passed', "$Label passed")
}

"== step 1: attribution against independently caused paging $(Get-Date -Format s)"
$attrOut = "_jobs\out\122_attribution_$stamp.json"
if (Test-Path -LiteralPath $attrOut) { Remove-Item -LiteralPath $attrOut -Force -ErrorAction SilentlyContinue }
$global:LASTEXITCODE = $null
try {
  & $Python -u scripts\check_monitor_attribution.py --out $attrOut --run-id $runId 2>&1
} catch { "could not start: $($_.Exception.Message)" }
$step1 = Test-Result 'attribution validation' $attrOut $global:LASTEXITCODE
$costOut = "_jobs\out\122_model_paging_$stamp.json"

if ($step1[0] -ne 'passed') {
  # Reviewer, after the boundary-bounds patch: a step 1 that failed, was inconclusive,
  # could not start or left no valid results (an unresolved sampler cleanup among them)
  # stops the job. No model and no further sampler is started.
  $step2 = @('not run', "model paging probe not run (step 1: $($step1[1]))")
  "== step 2: not run (step 1 did not pass) $(Get-Date -Format s)"
} else {
  "== step 2: genuine model paging, with a second sampler as a consistency check $(Get-Date -Format s)"
  if (Test-Path -LiteralPath $costOut) { Remove-Item -LiteralPath $costOut -Force -ErrorAction SilentlyContinue }
  $global:LASTEXITCODE = $null
  try {
    & $Python -u scripts\check_verify_cost.py --model models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf --layers 0 --rounds 2 --short-runs 3 --interrupt-rounds 0 --skip-part-b --skip-part-a0 --monitors measurement --finalize-warmups --cold-model-copy --consistency-sampler --out $costOut --run-id $runId 2>&1
  } catch { "could not start: $($_.Exception.Message)" }
  $step2 = Test-Result 'model paging probe' $costOut $global:LASTEXITCODE
}

"step 1: $($step1[0]) - $($step1[1])"
"step 2: $($step2[0]) - $($step2[1])"
"results: $attrOut ; $costOut (run id $runId)"
"finished after $([int]$sw.Elapsed.TotalMinutes) min"
$states = @($step1[0], $step2[0])
if ($states -contains 'failed' -or ($step1[0] -eq 'passed' -and $step2[0] -eq 'not run')) {
  "FAILED: " + ((@($step1, $step2) | Where-Object { $_[0] -ne 'passed' } | ForEach-Object { $_[1] }) -join "; ")
  exit 2
}
if ($states -contains 'inconclusive' -or $states -contains 'not run') {
  "INCONCLUSIVE (not a success): " + ((@($step1, $step2) | Where-Object { $_[0] -ne 'passed' } | ForEach-Object { $_[1] }) -join "; ")
  exit 5
}
"all steps passed"
exit 0
