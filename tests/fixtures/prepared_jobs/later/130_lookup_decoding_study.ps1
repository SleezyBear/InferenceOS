# PREPARED FOR INDEPENDENT REVIEW ONLY. Not queued or authorized to run.
# Model-free ngram-simple versus explicit none, 4 workloads x 5 blocks x 2 arms
# on Llama3B and Gemma1B: 80 fresh loads, no prompt reuse, tuning, saves or defaults.
# Seed42, natural EOS / max128 tokens; IDs, text and ending must match; whole latency primary.
# All rules/model/software/engine pins checked before dispatch and after each run.
# Estimate 20-50 minutes plus a 10-minute inter-model rest; not a deadline.
# Python/PowerStateScript/RestSeconds overrides are for executable tests only.
param(
  [string]$Root = 'C:\Users\madha\Projects\InferenceOS',
  [string]$Python = '.\.venv\Scripts\python.exe',
  [ValidateRange(0,86400)][int]$PowerWaitSeconds = 3600,
  [ValidateRange(0,3600)][int]$RestSeconds = 600,
  [string]$PowerStateScript = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Root
$rulesFile = Join-Path $Root '_jobs\later\130_rules_lookup-study-1-r5.json'
$rulesSha = '273b217c6243d500adbd1c40b2a1d7cc474163a74b471ddeb26f99ca37601f2e'
function Confirm-Inputs {
  try {
    if ((Get-FileHash -LiteralPath $rulesFile -Algorithm SHA256).Hash -ne $rulesSha) { throw 'rules digest differs' }
    $script:rules = Get-Content -Raw -LiteralPath $rulesFile | ConvertFrom-Json
    if ($rules.protocol -ne 'lookup-study-1' -or $rules.version -ne 'lookup-decoding-1' -or
        $rules.model_order.Count -ne 2) { throw 'protocol/model population differs' }
    foreach ($name in $rules.model_order) {
      $model = $rules.models.$name
      if ($model.sha256 -notmatch '^[0-9a-f]{64}$' -or $model.configuration_sha256 -notmatch '^[0-9a-f]{64}$' -or
          $model.label -notmatch '^[a-z0-9]{1,20}$') {
        throw 'model identity/label malformed'
      }
      if ((Get-FileHash -LiteralPath (Join-Path $Root $name) -Algorithm SHA256).Hash -ne $model.sha256) {
        throw "model differs: $name"
      }
    }
    foreach ($group in @($rules.python_files, $rules.engine_files)) {
      foreach ($item in $group.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $Root $item.Name) -Algorithm SHA256).Hash -ne $item.Value) {
          throw "input differs: $($item.Name)"
        }
      }
    }
    return $true
  } catch { Write-Host "REFUSED: frozen lookup inputs: $($_.Exception.Message)"; return $false }
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
function Wait-Mains {
  $waited = 0
  while ((Get-PowerState) -ne 'ac') {
    if ($waited -ge $PowerWaitSeconds) { Write-Host 'REFUSED: mains power not verified'; return $false }
    Write-Host "waiting for mains power ($waited s)"
    Start-Sleep -Seconds 10
    $waited += 10
  }
  return $true
}
if ($RestSeconds -ne 600 -and $PowerStateScript -eq '') { Write-Host 'REFUSED: test-only rest override'; exit 4 }
if (-not (Confirm-Inputs) -or -not (Wait-Mains)) { exit 4 }
$runId = [guid]::NewGuid().ToString()
$stamp = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $runId.Substring(0,8)
$index = 0
foreach ($name in $rules.model_order) {
  if ($index -gt 0) {
    Write-Host "inter-model rest: $RestSeconds seconds"
    Start-Sleep -Seconds $RestSeconds
  }
  if (-not (Wait-Mains) -or -not (Confirm-Inputs)) { exit 4 }
  $label = $rules.models.$name.label
  $out = "_jobs\out\130_lookup_${label}_$stamp.json"
  $audit = "_jobs\out\130_lookup_${label}_$stamp.audit.json"
  Write-Host "lookup protocol lookup-study-1; rules $rulesSha; run $runId; model $label"
  $global:LASTEXITCODE = $null
  try {
    & $Python -u scripts\evaluate_lookup_decoding.py --model $name --preregistration $rulesFile --preregistration-sha256 $rulesSha --run-id $runId --out $out
  } catch { Write-Host "evaluation launch failed: $($_.Exception.Message)" }
  $code = $global:LASTEXITCODE
  Write-Host "record: $out (exit $code)"
  if (-not (Confirm-Inputs)) { exit 2 }
  try {
    $record = Get-Content -Raw -LiteralPath $out | ConvertFrom-Json
    if ($record.run_id -ne $runId -or $record.preregistration.sha256 -ne $rulesSha -or
        $record.version -ne 'lookup-decoding-1' -or $record.protocol -ne 'lookup-study-1' -or
        $record.simulated -ne $false) { throw 'result identity/version/rules are not this real run' }
    $expectedModel = [IO.Path]::GetFullPath((Join-Path $Root $name))
    if ([string]::IsNullOrWhiteSpace($record.model) -or
        [IO.Path]::GetFullPath($record.model) -ne $expectedModel -or
        $record.model_file_identity.sha256 -ne $rules.models.$name.sha256 -or
        $record.configuration_sha256 -ne $rules.models.$name.configuration_sha256) {
      throw 'record does not belong to this model'
    }
  } catch { Write-Host "STOPPED: record identity invalid: $($_.Exception.Message)"; exit 2 }
  $global:LASTEXITCODE = $null
  try {
    & $Python -B scripts\audit_lookup_decoding.py $out --rules $rulesFile --rules-sha256 $rulesSha --out $audit
  } catch { Write-Host "audit launch failed: $($_.Exception.Message)" }
  $auditCode = $global:LASTEXITCODE
  if ($code -ne 0 -or $record.status -ne 'done' -or $auditCode -ne 0) {
    Write-Host 'STOPPED: incomplete/error/quality/validity/cleanup evidence; no next model. Keep both records.'
    exit 2
  }
  try {
    $checked = Get-Content -Raw -LiteralPath $audit | ConvertFrom-Json
    $sourceSha = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($checked.source_sha256 -ne $sourceSha -or $checked.rules_sha256 -ne $rulesSha -or
        $checked.protocol -ne 'lookup-study-1' -or $checked.observations_complete -ne $true -or
        $checked.performance_interpretable -ne $true -or $checked.automatic_admission -ne $false) {
      throw 'audit does not bind this complete observation set'
    }
  } catch { Write-Host "STOPPED: audit invalid: $($_.Exception.Message)"; exit 2 }
  $index += 1
}
Write-Host 'Registered observations complete. Independent results review required; no tune/default change.'
exit 0
