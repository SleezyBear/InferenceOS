Set-Location C:\Users\madha\Projects\InferenceOS
# Short real-engine check of the new settings (no tuning, nothing saved): does the engine
# accept each placement / cache setup, answer (incl. two known-answer questions), and show
# it in its own log (per-piece "overridden" lines, buffer sizes, cache formats)? Also the
# hardware probe the hybrid grouped placement needs before tune may accept it.
# (6 Oct 2026, updated after review 7)
for ($i = 0; $i -lt 360; $i++) {
  $b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
  if ($null -eq $b -or $b.BatteryStatus -ne 1) { break }
  if ($i -eq 0) { "on battery: waiting to be plugged in $(Get-Date -Format s)" }
  Start-Sleep -Seconds 10
}
"=== Granite 3.1 MoE (experts) $(Get-Date -Format s) ==="
.\.venv\Scripts\python.exe -u scripts\check_new_paths.py --model models\study\granite-3.1-3b-a800m-instruct-Q4_K_M.gguf --layers 0 --out _jobs\out\110_granite31moe 2>&1
"=== Llama 3.2 3B (output step) $(Get-Date -Format s) ==="
.\.venv\Scripts\python.exe -u scripts\check_new_paths.py --model models\study\Llama-3.2-3B-Instruct-Q4_K_M.gguf --layers 23 --out _jobs\out\110_llama3b 2>&1
"=== Granite 4.0 H Tiny (hybrid) $(Get-Date -Format s) ==="
.\.venv\Scripts\python.exe -u scripts\check_new_paths.py --model models\study\granite-4.0-h-tiny-Q4_K_M.gguf --layers 13 --out _jobs\out\110_granite4h 2>&1
"=== done $(Get-Date -Format s) ==="
