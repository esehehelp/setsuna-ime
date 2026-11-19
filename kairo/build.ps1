# Build what the runs of this project execute into the run's output directory
# (bin/: the engine and the bench) with build.txt saying from what.
# Declared `regenerable` in kairo.toml: its output is temporal data, made
# again from the same snapshot when a reader needs it and it was evicted.
# cargo's target and home stay in the project's cache on this node
# (KAIRO_CACHE_DIR), so a later build there is incremental.
[CmdletBinding()]
param(
    [string[]]$Packages = @('setsuna-engine', 'setsuna-bench')
)
$ErrorActionPreference = 'Stop'
$out, $cache = $env:KAIRO_OUTPUT_DIR, $env:KAIRO_CACHE_DIR
if (-not $out -or -not $cache) { throw 'KAIRO_OUTPUT_DIR and KAIRO_CACHE_DIR are not set: run this as a Kairo task (kairo.toml)' }

$env:CARGO_TARGET_DIR = Join-Path $cache 'target'
$env:CARGO_HOME = Join-Path $cache 'cargo-home'
$argv = @('build', '--release', '--locked')
foreach ($p in $Packages) { $argv += @('-p', $p) }
& cargo @argv
if ($LASTEXITCODE -ne 0) { throw "cargo build exited $LASTEXITCODE" }

$bin = New-Item -ItemType Directory -Force -Path (Join-Path $out 'bin')
foreach ($p in $Packages) {
    Copy-Item -LiteralPath (Join-Path $env:CARGO_TARGET_DIR "release/$p.exe") -Destination $bin
}

$info = @(
    "source $env:KAIRO_SOURCE_ID",
    "packages $($Packages -join ' ')",
    (& rustc -vV | Select-Object -First 1)
)
Get-ChildItem -LiteralPath $bin -File | Sort-Object Name | ForEach-Object {
    $info += "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  bin/$($_.Name)"
}
$info | Set-Content -LiteralPath (Join-Path $out 'build.txt') -Encoding utf8
$info | Write-Host
