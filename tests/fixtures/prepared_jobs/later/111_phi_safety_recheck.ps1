Set-Location C:\Users\madha\Projects\InferenceOS
# Review 6: correctness evidence for Phi-4 mini's memory-safety correction (32 -> 30
# layers). Measures both in turns on the short question and runs the answer check; prints
# whether the 30-layer answers passed and whether any run failed. Nothing is saved.
for ($i = 0; $i -lt 360; $i++) {
  $b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
  if ($null -eq $b -or $b.BatteryStatus -ne 1) { break }
  if ($i -eq 0) { "on battery: waiting to be plugged in $(Get-Date -Format s)" }
  Start-Sleep -Seconds 10
}
.\.venv\Scripts\python.exe -u scripts\check_tune_result.py --model models\study\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf --context 4096 --start gpu_offload_layers=32 --final gpu_offload_layers=30 --workloads short-short --turns 4 --out _jobs\out\111_phi_safety.json 2>&1
"=== done $(Get-Date -Format s) ==="
