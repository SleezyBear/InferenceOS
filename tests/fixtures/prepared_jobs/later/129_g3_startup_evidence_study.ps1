# PREPARED FOR REVIEW ONLY. Not queued or authorized to run.
# Focused G3 follow-up: Gemma's exact job127 regression, both variants, 3 blocks,
# 12 fresh engine loads. No chat, tuning, saves, E7 or default changes.
# Frozen first-answer judge v8. Identical trace logging in both variants; no speed claim.
# Estimate 10-20 minutes including hashing, not a deadline. Rules/pins checked before
# dispatch and after completion. Each trial checks AC and cleanup in the Python runner.
# A completed run can leave G3 unanswered. The audit is required for interpretation.
# PowerStateScript and Python are injectable for executable wrapper tests only.
param(
  [string]$Root = 'C:\Users\madha\Projects\InferenceOS',
  [string]$Python = '.\.venv\Scripts\python.exe',
  [ValidateRange(0,86400)][int]$PowerWaitSeconds = 3600,
  [string]$PowerStateScript = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Root
$rulesFile = Join-Path $Root '_jobs\later\129_rules_g3-evidence-2.json'
$rulesSha = '17feb7d76046ba4036f99606ef2bc20db9299e2f31ea320749a7ba8d534ba94f'
function Confirm-Inputs {
  try {
    if ((Get-FileHash -LiteralPath $rulesFile -Algorithm SHA256).Hash -ne $rulesSha) { throw 'rules digest differs' }
    $rules = Get-Content -Raw -LiteralPath $rulesFile | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($rules.model_file) -or $rules.model_sha256 -notmatch '^[0-9a-f]{64}$') {
      throw 'frozen model identity is missing or malformed'
    }
    $modelFile = Join-Path $Root $rules.model_file
    if ((Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash -ne $rules.model_sha256) {
      throw 'model differs from the frozen content identity'
    }
    foreach ($group in @($rules.python_files, $rules.engine_files)) {
      foreach ($item in $group.PSObject.Properties) {
        $path = Join-Path $Root $item.Name
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.Value) { throw "input differs: $($item.Name)" }
      }
    }
    return $true
  } catch {
    Write-Host "REFUSED: cannot verify frozen G3 inputs: $($_.Exception.Message)"
    return $false
  }
}
function Get-PowerState {
  try {
    if ($PowerStateScript -ne '') { return [string](& $PowerStateScript) }
    Add-Type -AssemblyName System.Windows.Forms
    $state = [string][System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus
    if ($state -eq 'Online') { return 'ac' }
    if ($state -eq 'Offline') { return 'battery' }
  } catch { return 'unknown' }
  return 'unknown'
}
if (-not (Confirm-Inputs)) { exit 4 }
$waited = 0
while ((Get-PowerState) -ne 'ac') {
  if ($waited -ge $PowerWaitSeconds) { Write-Host 'REFUSED: mains power not verified'; exit 4 }
  Write-Host "waiting for mains power ($waited s)"
  Start-Sleep -Seconds 10
  $waited += 10
}
# Power may have taken an hour: validate the software and engine again before dispatch.
if (-not (Confirm-Inputs)) { exit 4 }
$runId = [guid]::NewGuid().ToString()
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $runId.Substring(0,8)
$out = "_jobs\out\129_g3_gemma1b_$stamp.json"
$audit = "_jobs\out\129_g3_gemma1b_$stamp.audit.json"
Write-Host "G3 protocol g3-evidence-2; rules SHA-256 $rulesSha; run $runId"
$global:LASTEXITCODE = $null
try {
  & $Python -u scripts\evaluate_prompt_reuse.py --study correctness --model models\study\google_gemma-3-1b-it-Q4_K_M.gguf --context 4096 --blocks 3 --cases correctness/regression-pump-b-unit-7 --swa-full-variant --capture-load-evidence --preregistration $rulesFile --preregistration-sha256 $rulesSha --run-id $runId --out $out
} catch { Write-Host "evaluation launch failed: $($_.Exception.Message)" }
$code = $global:LASTEXITCODE
Write-Host "record: $out (exit $code)"
if (-not (Confirm-Inputs)) { exit 2 }
try {
  $result = Get-Content -Raw -LiteralPath $out | ConvertFrom-Json
  if ($result.run_id -ne $runId -or $result.preregistration.sha256 -ne $rulesSha -or
      $result.study.version -ne 'reuse-correctness-8' -or $result.simulated -ne $false) {
    throw 'record identity/version/rules are not this real run'
  }
} catch { Write-Host "STOPPED: result identity cannot be verified: $($_.Exception.Message)"; exit 2 }
# Offline audit is useful even after incomplete work. It never starts a model.
$global:LASTEXITCODE = $null
try {
  & $Python -B scripts\audit_prompt_reuse_evidence.py $out --rules $rulesFile --rules-sha256 $rulesSha --out $audit
} catch { Write-Host "audit launch failed: $($_.Exception.Message)" }
$auditCode = $global:LASTEXITCODE
if ($code -ne 0 -or $result.status -ne 'done' -or $auditCode -ne 0) {
  Write-Host 'INCOMPLETE: keep both records; G3 may remain unanswered. No further model is dispatched.'
  exit 2
}
try {
  $checked = Get-Content -Raw -LiteralPath $audit | ConvertFrom-Json
  $actualRecordSha = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($checked.source_sha256 -ne $actualRecordSha -or $checked.rules_sha256 -ne $rulesSha -or
      $checked.protocol -ne 'g3-evidence-2' -or $checked.observations_complete -ne $true) { throw 'audit does not bind this record' }
} catch { Write-Host "INCOMPLETE: audit invalid: $($_.Exception.Message)"; exit 2 }
Write-Host "Operational observations complete; G3 interpretable: $($checked.g3_interpretable). Independent review required. No mitigation/default change follows."
exit 0
