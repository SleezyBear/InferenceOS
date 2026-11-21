# PREPARED, NOT QUEUED (reviewer follow-up): one short hardware check. Move this file
# into _jobs\ only when the owner approves. Nothing is tuned or saved.
#   Step 1  Windows SetupStore lock regression (real msvcrt.locking branch)   ~1 min
#   Step 2  file_sha256 wall time for the probe model (3 runs)                 ~1 min
#   Step 3  probe: Part C interruption (loading / request dispatched-pre-first-chunk
#           / decode x 3 rounds), Part A0 raw Get-Counter lines with source
#           timestamps, Part A monitor conditions, Part B shared load          ~30 min
# Expected total: about 35 minutes. It never runs on battery: if the laptop is not
# plugged in within 60 minutes it exits (status 4).
# "all steps passed" is printed only when every step started and exited 0 AND the
# probe wrote a NEW results file for THIS run (fresh file name, run id inside it, and
# status "done": every requested part and the cleanup passed).
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
# Every output of this run gets a name no earlier run used, and the probe also writes
# this run's id into its results, so an older file can never be taken for this run's.
$runId = [guid]::NewGuid().ToString()
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $runId.Substring(0, 8)
New-Item -ItemType Directory -Force -Path '_jobs\out' | Out-Null

function Get-ExitText($code) {
  if ($null -eq $code) { return 'did not start' }
  return "exit $code"
}

"== step 1: Windows SetupStore lock regression $(Get-Date -Format s)"
# A command that fails to start must not inherit the previous command's exit code.
$global:LASTEXITCODE = $null
try {
  & $Python -m pytest -q -s -p no:cacheprovider tests\test_budgeted_tune.py -k "concurrent_writers or lock_wait or after_release or non_contention or windows_lock" --junitxml="_jobs\out\120_store_lock_$stamp.xml" 2>&1
} catch { "could not start: $($_.Exception.Message)" }
$code = $global:LASTEXITCODE
if ($code -ne 0) { $failed += "store lock regression ($(Get-ExitText $code))" }

"== step 2: model hash wall time $(Get-Date -Format s)"
$global:LASTEXITCODE = $null
try {
  & $Python -u scripts\time_model_hash.py --model models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf --repeats 3 --out "_jobs\out\120_hash_time_$stamp.json" 2>&1
} catch { "could not start: $($_.Exception.Message)" }
$code = $global:LASTEXITCODE
if ($code -ne 0) { $failed += "hash timing ($(Get-ExitText $code))" }

"== step 3: probe $(Get-Date -Format s)"
$probeOut = "_jobs\out\120_verify_cost_llama_$stamp.json"
$fresh = $true
if (Test-Path -LiteralPath $probeOut) {
  try { Remove-Item -LiteralPath $probeOut -Force -ErrorAction Stop } catch { $fresh = $false }
}
if (-not $fresh) {
  $failed += "probe not run: an old results file at $probeOut could not be removed"
} else {
  $global:LASTEXITCODE = $null
  try {
    & $Python -u scripts\check_verify_cost.py --model models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf --layers 29 --long-passages 18 --rounds 3 --interrupt-rounds 3 --out $probeOut --run-id $runId 2>&1
  } catch { "could not start: $($_.Exception.Message)" }
  $probe = $global:LASTEXITCODE
  if ($probe -ne 0) {
    $failed += "probe ($(Get-ExitText $probe); 2 = a requested part or the cleanup failed, see its status; 3 = error)"
  } else {
    # The results file must exist, be this run's (run id) and say "done".
    $result = $null
    try { $result = Get-Content -Raw -LiteralPath $probeOut -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { $result = $null }
    if ($null -eq $result) {
      $failed += "probe exited 0 but wrote no readable results file for this run ($probeOut)"
    } elseif ($result.run_id -ne $runId) {
      $failed += "probe exited 0 but the results file is not this run's (run id '$($result.run_id)', expected '$runId')"
    } elseif ($result.status -ne 'done') {
      $failed += "probe exited 0 but its results file says status '$($result.status)'"
    }
  }
}
"results: $probeOut (run id $runId)"

"finished after $([int]$sw.Elapsed.TotalMinutes) min"
if ($failed.Count -gt 0) { "FAILED: " + ($failed -join "; "); exit 2 }
"all steps passed"
exit 0
