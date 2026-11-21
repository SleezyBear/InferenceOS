# PREPARED, NOT QUEUED (monitor correction): validates the measurement-scoped
# monitor's attribution on hardware and measures its overhead again. Move this file
# into _jobs\ only when the owner approves. Nothing is tuned or saved; the monitor
# stays off by default whatever the result.
#   Step 1  attribution against independently caused paging (no model): bursts of
#           page reads from an uncached scratch file, placed inside, before, after
#           and across the edge of timed pseudo-answers, plus quiet ones      ~4 min
#   Step 2  overhead of the corrected scoped monitor vs the per-answer monitor vs
#           no monitor (Llama 3.2 3B, short question, 3 rounds; Part C and Part B
#           disabled, Part A0 raw counters kept)                              ~4 min
# Expected total: about 8-10 minutes. It never runs on battery: if the laptop is not
# plugged in within 60 minutes it exits (status 4).
# "all steps passed" is printed only when every step started and exited 0 AND each
# wrote a NEW results file for THIS run (fresh name, this run's id, status "done").
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
  # Returns a failure text, or $null when the step passed for THIS run.
  if ($Code -ne 0) { return "$Label ($(Get-ExitText $Code))" }
  $result = $null
  try { $result = Get-Content -Raw -LiteralPath $Path -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { $result = $null }
  if ($null -eq $result) { return "$Label exited 0 but wrote no readable results file for this run ($Path)" }
  if ($result.run_id -ne $runId) { return "$Label exited 0 but the results file is not this run's (run id '$($result.run_id)')" }
  if ($result.status -ne 'done') { return "$Label exited 0 but its results file says status '$($result.status)'" }
  return $null
}

"== step 1: attribution against independently caused paging $(Get-Date -Format s)"
$attrOut = "_jobs\out\121_attribution_$stamp.json"
if (Test-Path -LiteralPath $attrOut) { Remove-Item -LiteralPath $attrOut -Force -ErrorAction SilentlyContinue }
$global:LASTEXITCODE = $null
try {
  & $Python -u scripts\check_monitor_attribution.py --out $attrOut --run-id $runId 2>&1
} catch { "could not start: $($_.Exception.Message)" }
$problem = Test-Result 'attribution validation' $attrOut $global:LASTEXITCODE
if ($null -ne $problem) { $failed += $problem }

"== step 2: monitor overhead with the corrected scoped monitor $(Get-Date -Format s)"
$costOut = "_jobs\out\121_overhead_$stamp.json"
if (Test-Path -LiteralPath $costOut) { Remove-Item -LiteralPath $costOut -Force -ErrorAction SilentlyContinue }
$global:LASTEXITCODE = $null
try {
  & $Python -u scripts\check_verify_cost.py --model models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf --layers 29 --rounds 3 --interrupt-rounds 0 --skip-part-b --out $costOut --run-id $runId 2>&1
} catch { "could not start: $($_.Exception.Message)" }
$problem = Test-Result 'overhead probe' $costOut $global:LASTEXITCODE
if ($null -ne $problem) { $failed += $problem }

"results: $attrOut ; $costOut (run id $runId)"
"finished after $([int]$sw.Elapsed.TotalMinutes) min"
if ($failed.Count -gt 0) { "FAILED: " + ($failed -join "; "); exit 2 }
"all steps passed"
exit 0
