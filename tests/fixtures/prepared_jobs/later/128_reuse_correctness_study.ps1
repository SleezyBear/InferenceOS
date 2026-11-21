# PREPARED, NOT QUEUED (17 Oct 2026): reuse-correctness study (job 128), after job 127
# saw Gemma 3 1B answer one question wrongly with partial prompt reuse. Pre-registration:
# reviewer-job128-prereg-2026-10-17.md. Move this file into _jobs\ only when the owner
# approves. Nothing is tuned or saved; saved setups and tune history are only read; every
# data-folder file is fingerprinted before and after; results live in _jobs\out.
# For each model: its saved setup for context 4096 on mains power (read only), fresh
# engine loads only, 9 known-answer prompt pairs (the exact job 127 pair plus 8 more) x
# 3 blocks x (fresh arm, reuse arm), then a chat exposure check (each pair, 2 blocks, as
# 3 fresh loads: the ordinary client, a matched reference primed the same way but asked
# with reuse off, and an unprimed reference; client and matched alternate in order).
#   Gemma 3 1B (sliding-window attention): the engine default AND --swa-full.
#   Llama 3.2 3B (no sliding window): the control, engine default only.
# 10 minutes of rest between the models. Expected total: about 1 hour 15 minutes (an
# estimate from job 127's per-trial times, taking a chat load as long as a trial; chat
# loads are not measured yet). Amended pre-registration:
# reviewer-job128-prereg-r4-2026-10-17.md, amended in reviewer-implementation-handoff-current.md (reuse-correctness-8).
# Mains power: checked before every model here (fails closed), and before and after every
# trial inside the evaluation, which stops at once if power is lost (exit 9).
# The job STOPS (no further models) on: an unconfirmed engine cleanup (exit 5), a
# changed data folder (6), a pre-registered validity gate (7), lost mains power (9 or
# here), an error (3) or any unknown exit. It continues after 0, 2 (incomplete) and 4
# (refused, e.g. no saved setup). "all steps passed" only when every model exited 0 and
# wrote a NEW results file for THIS run (run id inside) with status "done" and the data
# folder unchanged. A wrong answer with reuse is a RESULT, not a failure. ASCII only
# (Windows PowerShell 5.1).
# Test-only parameters (used by tests\test_job_scripts.py, labelled in the output):
# -SkipPowerWait passes --no-power-check and is printed as TEST RUN; -PowerStateScript
# replaces the power query with a script that prints ac, battery or unknown.
param(
  [string]$Root = 'C:\Users\madha\Projects\InferenceOS',
  [string]$Python = '.\.venv\Scripts\python.exe',
  [ValidateSet(3)][int]$Blocks = 3,
  [ValidateRange(0,86400)][int]$RestSeconds = 600,
  [ValidateRange(0,86400)][int]$PowerWaitSeconds = 3600,
  [string]$PowerStateScript = '',
  [switch]$SkipPowerWait
)
Set-Location $Root

# Frozen governing rules: fail before power queries or any model dispatch.
$rulesFile = Join-Path $Root '_jobs\later\128_rules_reuse-correctness-8.txt'
$rulesSha256 = '4a80891cb9f607da6d341f27d0da090148292a199d389d4101b6038b8e293734'
try {
  $actualRulesSha256 = (Get-FileHash -LiteralPath $rulesFile -Algorithm SHA256 -ErrorAction Stop).Hash
  if ($actualRulesSha256 -ne $rulesSha256) { throw 'governing rules SHA-256 mismatch' }
} catch {
  Write-Host "REFUSED: cannot verify frozen job 128 rules: $($_.Exception.Message)"
  exit 4
}
Write-Host "governing rules: reuse-correctness-8 SHA-256 $rulesSha256"

function Get-PowerState {
  if ($PowerStateScript -ne '') {
    try { return [string](& $PowerStateScript) } catch { return 'unknown' }
  }
  try {
    Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
    $line = [string][System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus
    if ($line -eq 'Online') { return 'ac' }
    if ($line -eq 'Offline') { return 'battery' }
    return 'unknown'
  } catch { return 'unknown' }
}

function Wait-ForMains {
  if ($SkipPowerWait) { return $true }
  $waited = 0
  while ($true) {
    $state = Get-PowerState
    if ($state -eq 'ac') { return $true }
    if ($waited -ge $PowerWaitSeconds) {
      # Write-Host, not output: a function's output would become part of its return value.
      Write-Host "not on mains power ($state) after $waited s: stopping $(Get-Date -Format s)"
      return $false
    }
    if ($waited -eq 0) { Write-Host "power state '$state': waiting for mains power $(Get-Date -Format s)" }
    Start-Sleep -Seconds 10
    $waited += 10
  }
}

if ($SkipPowerWait) { "TEST RUN: power checks disabled (-SkipPowerWait); results are labelled and not evidence" }
if (-not (Wait-ForMains)) { exit 4 }
$sw = [Diagnostics.Stopwatch]::StartNew()
$failed = @()
$stopped = $false
$runId = [guid]::NewGuid().ToString()
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $runId.Substring(0, 8)
New-Item -ItemType Directory -Force -Path '_jobs\out' | Out-Null

function Get-ExitText($code) {
  if ($null -eq $code) { return 'did not start' }
  return "exit $code"
}

$models = @(
  @{ Name = 'gemma1b'; File = 'models\study\google_gemma-3-1b-it-Q4_K_M.gguf'; Args = @('--swa-full-variant') },
  @{ Name = 'llama3b'; File = 'models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf'; Args = @() }
)
for ($m = 0; $m -lt $models.Count; $m++) {
  $model = $models[$m]
  if ($m -gt 0) {
    "resting $RestSeconds s before the next model $(Get-Date -Format s)"
    Start-Sleep -Seconds $RestSeconds
  }
  if (-not (Wait-ForMains)) {
    $failed += "$($model.Name): not started, not on mains power"
    $stopped = $true
    break
  }
  $out = "_jobs\out\128_correctness_$($model.Name)_$stamp.json"
  "== $($model.Name): reuse correctness study $(Get-Date -Format s)"
  $extra = @('--study', 'correctness', '--chat-exposure') + $model.Args
  if ($SkipPowerWait) { $extra += @('--no-power-check') }
  $global:LASTEXITCODE = $null
  try {
    & $Python -u scripts\evaluate_prompt_reuse.py --model $model.File --context 4096 --blocks $Blocks --out $out --run-id $runId --preregistration $rulesFile --preregistration-sha256 $rulesSha256 @extra 2>&1
  } catch { "could not start: $($_.Exception.Message)" }
  $code = $global:LASTEXITCODE
  "results: $out (run id $runId)"
  if ($code -eq 0) {
    $result = $null
    try { $result = Get-Content -Raw -LiteralPath $out -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { $result = $null }
    if ($null -eq $result) {
      $failed += "$($model.Name): exited 0 but wrote no readable results file ($out)"
      $stopped = $true; break
    } elseif ($result.run_id -ne $runId) {
      $failed += "$($model.Name): results file is not this run's (run id '$($result.run_id)')"
      $stopped = $true; break
    } elseif ($result.status -ne 'done') {
      $failed += "$($model.Name): results file says status '$($result.status)'"
      $stopped = $true; break
    } elseif ($result.study.version -ne 'reuse-correctness-8' -or $result.preregistration.sha256 -ne $rulesSha256) {
      $failed += "$($model.Name): result study version or governing rules digest does not match"
      $stopped = $true; break
    } elseif (-not $SkipPowerWait -and $result.simulated -ne $false) {
      $failed += "$($model.Name): result is not explicitly a real run"
      $stopped = $true; break
    } elseif ($result.data_folder_unchanged -ne $true) {
      $failed += "$($model.Name): the data folder changed"
      $stopped = $true
      break
    }
  } elseif ($code -eq 2 -or $code -eq 4) {
    $failed += "$($model.Name) ($(Get-ExitText $code); 2 = incomplete, 4 = refused)"
  } else {
    $failed += "$($model.Name) ($(Get-ExitText $code); 3 = error, 5 = cleanup not confirmed, 6 = data folder changed, 7 = validity gate, 9 = mains power lost): STOPPED, no further models"
    $stopped = $true
    break
  }
}

"finished after $([int]$sw.Elapsed.TotalMinutes) min"
if ($stopped) { "STOPPED early: the first stopping report is kept as written" }
if ($failed.Count -gt 0) { "FAILED: " + ($failed -join "; "); exit 2 }
if ($SkipPowerWait) { "TEST RUN finished (power checks disabled): not evidence"; exit 0 }
"all steps passed"
exit 0
