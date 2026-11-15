# Fetch a public artifact pinned by crates/tsf/models.json (a model, or the
# reading lexicon or the analyzer) into the run's output directory, its files
# beside one another as the pin lays them out under its entry, with
# fetched.txt naming the entry. Declared `regenerable` in kairo.toml: temporal
# data, fetched again at the same revision when needed.
[CmdletBinding()]
param(
    # A model the catalog pins (default: its own)
    [string]$Model,
    [ValidateSet('lexicon', 'analyzer')]
    [string]$Component
)
$ErrorActionPreference = 'Stop'
$out, $scratch = $env:KAIRO_OUTPUT_DIR, $env:KAIRO_SCRATCH
if (-not $out -or -not $scratch) { throw 'KAIRO_OUTPUT_DIR and KAIRO_SCRATCH are not set: run this as a Kairo task (kairo.toml)' }

$fetch = @{ CacheDir = (Join-Path $scratch 'download') }
if ($Model) { $fetch.Model = $Model }
if ($Component) { $fetch.Component = $Component }
# the manifest's (or the component entry's) path, the script's last line
$entry = (& (Join-Path $PSScriptRoot '../crates/tsf/fetch-model.ps1') @fetch | Select-Object -Last 1).Trim()
if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) { throw "fetch-model.ps1 gave no file: $entry" }
$dir = Split-Path -Parent $entry
Get-ChildItem -LiteralPath $dir -Force | Copy-Item -Destination $out -Recurse
"entry $(Split-Path -Leaf $entry)" | Set-Content -LiteralPath (Join-Path $out 'fetched.txt') -Encoding utf8
Write-Host "fetched $entry -> $out"
