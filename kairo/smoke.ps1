# What an attempt of this project can do where Kairo runs it: its integrity
# level (a confined attempt runs at Low), the KAIRO_* directories, whether the
# snapshot and the Kairo directories can be written, the Rust toolchain and the
# network. Writes smoke.txt to the run's output directory.
$ErrorActionPreference = 'Continue'
$out = $env:KAIRO_OUTPUT_DIR
$lines = [System.Collections.Generic.List[string]]::new()
function Note([string]$s) { $lines.Add($s); Write-Host $s }

Note "cwd: $(Get-Location)"
Note "integrity: $((whoami /groups | Select-String 'Mandatory Label') -replace '\s+', ' ')"
# directories and ids only: KAIRO_ATTEMPT_TOKEN is a credential
Get-ChildItem env: | Where-Object { $_.Name -like 'KAIRO_*' -and $_.Name -match '(_DIR|_ID|SCRATCH)$' } |
    Sort-Object Name | ForEach-Object { Note "$($_.Name)=$($_.Value)" }
Note "TMP=$env:TMP"
Note "SETSUNA_SMOKE=$env:SETSUNA_SMOKE"
Note "env count: $((Get-ChildItem env:).Count)"
foreach ($dir in @((Get-Location).Path, $env:KAIRO_OUTPUT_DIR, $env:KAIRO_SCRATCH, $env:KAIRO_CACHE_DIR, $env:LOCALAPPDATA)) {
    if (-not $dir) { continue }
    $probe = Join-Path $dir ".smoke-$PID"
    try {
        Set-Content -LiteralPath $probe -Value x -ErrorAction Stop
        Remove-Item -LiteralPath $probe -ErrorAction SilentlyContinue
        Note "write ok: $dir"
    } catch {
        Note "write refused: $dir ($($_.Exception.Message))"
    }
}
foreach ($tool in 'cargo', 'rustc') {
    try { Note "${tool}: $(& $tool --version 2>&1)" } catch { Note "${tool}: $($_.Exception.Message)" }
}
try {
    $r = Invoke-WebRequest -Method Head -Uri 'https://huggingface.co/esehe/Cymbidium-v1.1-xsmall/resolve/main/config.json' -UseBasicParsing -TimeoutSec 20
    Note "network: $($r.StatusCode)"
} catch {
    Note "network: $($_.Exception.Message)"
}
if ($out) { $lines | Set-Content -LiteralPath (Join-Path $out 'smoke.txt') -Encoding utf8 }
exit 0
