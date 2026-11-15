# Type the bench's sets into the IME session (setsuna-bench harness) with the
# settings of -Config, and score the run: run.jsonl, score.txt and what it ran
# with (the config as given, the build's build.txt) in the run's output
# directory, which Kairo keeps on the durable store. The binaries come from
# the input BUILD (kairo/build.ps1); the config names the other inputs as
# ${KAIRO_INPUT_<NAME>}.
[CmdletBinding()]
param(
    [string]$Config = 'kairo/bench/harness.toml',
    [string]$Sets = 'crates/bench/sets/v1',
    [string[]]$Only = @()
)
$ErrorActionPreference = 'Stop'
$out, $build = $env:KAIRO_OUTPUT_DIR, $env:KAIRO_INPUT_BUILD
if (-not $out -or -not $build) { throw 'KAIRO_OUTPUT_DIR and KAIRO_INPUT_BUILD are not set: run this as a Kairo task (kairo.toml)' }

$bench = Join-Path $build 'bin/setsuna-bench.exe'
# a plain copy: the snapshot's files are read-only, and the output directory
# is deleted once Kairo has its durable copy
Get-Content -LiteralPath $Config -Raw | Set-Content -LiteralPath (Join-Path $out 'config.toml') -NoNewline -Encoding utf8
Copy-Item -LiteralPath (Join-Path $build 'build.txt') -Destination (Join-Path $out 'build.txt')
$run = Join-Path $out 'run.jsonl'
# no reranker: the IME no longer uses one
$argv = @('harness', '--sets', $Sets, '--config', $Config, '--out', $run, '--no-rerank')
if ($Only) { $argv += @('--only', ($Only -join ',')) }
& $bench @argv
if ($LASTEXITCODE -ne 0) { throw "setsuna-bench harness exited $LASTEXITCODE" }
& $bench score --sets $Sets "setsuna=$run" | Tee-Object -FilePath (Join-Path $out 'score.txt')
if ($LASTEXITCODE -ne 0) { throw "setsuna-bench score exited $LASTEXITCODE" }
