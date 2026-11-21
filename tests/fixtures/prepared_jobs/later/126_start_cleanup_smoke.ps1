# PREPARED ONLY. Owner-run interactive smoke test; not an unattended queue job.
# Run from a visible terminal, for example:
# powershell -NoProfile -File _jobs\later\126_start_cleanup_smoke.ps1
# This loads one model using its existing approved setup. When told it is ready,
# press Ctrl+C ONCE. The Python harness checks the owner tag and removal of that
# exact key file, without reading its contents. It writes fresh artifacts under out/.
# No tuning, configuration save, requests, or stale-key sweep stress test is added.
# Normal backend startup still performs its existing bounded stale-key sweep.
# An absent/mismatched saved setup is a refusal, not permission to tune one.
# ASCII only for Windows PowerShell 5.1.
param(
    [string]$Root = 'C:\Users\madha\Projects\InferenceOS',
    [string]$Model = 'models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    [int]$Context = 4096,
    [int]$Port = 18080
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Root
& '.\.venv\Scripts\python.exe' -B '_jobs\later\126_start_cleanup_smoke.py' --model $Model --context $Context --port $Port
exit $LASTEXITCODE
