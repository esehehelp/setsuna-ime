# Build the Windows release: target\dist\setsuna-ime-<version>-windows-x64.zip,
# setup.ps1 with that zip's version and SHA-256 filled in, and SHA256SUMS.
# Nothing is installed.
#
# Usage:
#   pwsh -File crates\tsf\package.ps1
#   pwsh -File target\dist\setup.ps1 -Archive target\dist\setsuna-ime-<version>-windows-x64.zip   # try it
#   gh release create v<version> target\dist\*                                                    # publish
#
# The zip holds the TSF DLL, the engine, reranker (with llama.cpp's DLLs) and
# rewriter daemons, ONNX Runtime (crates\engine\fetch-onnxruntime.ps1), the
# settings GUI, the mixed input model (mixed\), the licenses and install.ps1,
# uninstall.ps1, fetch-model.ps1 and models.json: install.ps1 downloads the
# model pinned there. Its scripts get a UTF-8 BOM, which Windows PowerShell
# 5.1 needs to read non-ASCII text, and install.cmd runs install.ps1 on a
# double-click.

#Requires -Version 5.1
[CmdletBinding()]
param(
    # Package the binaries already in target\release.
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

$Repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Release = Join-Path $Repo 'target\release'
$Dist = Join-Path $Repo 'target\dist'

$cargoToml = Get-Content -LiteralPath (Join-Path $Repo 'Cargo.toml') -Raw
if ($cargoToml -notmatch '(?ms)^\[workspace\.package\].*?^version = "(\d+\.\d+\.\d+[^"]*)"') {
    throw 'no workspace version in Cargo.toml'
}
$Version = $Matches[1]
$Name = "setsuna-ime-$Version-windows-x64"

if (-not $SkipBuild) {
    cargo build --release -p setsuna-tsf -p setsuna-engine -p setsuna-settings -p setsuna-rewrite -p setsuna-rerank
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
}
& (Join-Path $Repo 'crates\engine\fetch-onnxruntime.ps1') -Dest $Release

$Out = Join-Path $Dist $Name
$Zip = Join-Path $Dist "$Name.zip"
$Setup = Join-Path $Dist 'setup.ps1'
foreach ($p in @($Out, $Zip, $Setup, (Join-Path $Dist 'SHA256SUMS'))) {
    if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Recurse -Force }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# binaries
$binaries = @('setsuna_tsf.dll', 'setsuna-engine.exe', 'onnxruntime.dll', 'setsuna-settings.exe',
              'setsuna-rewrite.exe', 'setsuna-rerank.exe') +
            @(Get-ChildItem -LiteralPath $Release -File -Filter '*.dll' |
              Where-Object { $_.Name -match '^(llama|ggml)' } | ForEach-Object { $_.Name })
foreach ($b in $binaries) {
    $src = Join-Path $Release $b
    if (-not (Test-Path -LiteralPath $src)) { throw "missing build output: $src" }
    Copy-Item -LiteralPath $src -Destination $Out
}
if (-not ($binaries -match '^ggml-cpu-')) { throw "no ggml CPU backends in $Release" }

# install.ps1 installs the Visual C++ runtime when older than its floor,
# which must not be below the toolset (linker version) of any binary.
$installText = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'install.ps1') -Raw
if ($installText -notmatch '\$k\.Major -eq 14 -and \$k\.Minor -ge (\d+)') { throw 'no Visual C++ runtime floor in install.ps1' }
$floor = [int]$Matches[1]
foreach ($b in $binaries) {
    $bytes = [IO.File]::ReadAllBytes((Join-Path $Out $b))
    $optional = [BitConverter]::ToInt32($bytes, 0x3C) + 24
    $major, $minor = $bytes[$optional + 2], $bytes[$optional + 3]
    if ($major -gt 14 -or ($major -eq 14 -and $minor -gt $floor)) {
        throw "$b is linked by toolset $major.$minor, above install.ps1's Visual C++ runtime floor 14.$floor"
    }
}

# scripts, with a BOM
$utf8Bom = New-Object Text.UTF8Encoding $true
foreach ($s in @('install.ps1', 'uninstall.ps1', 'fetch-model.ps1')) {
    $text = [IO.File]::ReadAllText((Join-Path $PSScriptRoot $s), (New-Object Text.UTF8Encoding $false))
    [IO.File]::WriteAllText((Join-Path $Out $s), $text, $utf8Bom)
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'models.json') -Destination $Out
# PSModulePath emptied: Windows PowerShell started from pwsh would load
# pwsh's modules (no Get-FileHash).
$cmd = @(
    '@echo off',
    'rem Setsuna IME installer: double-click, or install.cmd -ModelPrecision fp32 and the other options of install.ps1.',
    'setlocal',
    'set "PSModulePath="',
    'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*',
    'set rc=%errorlevel%',
    'if not "%rc%"=="0" echo Install failed with exit code %rc%. Log: %LOCALAPPDATA%\setsuna-ime\install.log',
    'pause',
    'exit /b %rc%'
)
[IO.File]::WriteAllText((Join-Path $Out 'install.cmd'), (($cmd -join "`r`n") + "`r`n"), (New-Object Text.ASCIIEncoding))
[IO.File]::WriteAllText((Join-Path $Out 'VERSION'), "$Version`r`n", (New-Object Text.ASCIIEncoding))

# mixed input model (its README carries the sources' attribution)
$mixed = Join-Path $Out 'mixed'
New-Item -ItemType Directory -Force -Path $mixed | Out-Null
foreach ($f in @('kana.tsv', 'latin.tsv', 'params.txt', 'README.md')) {
    Copy-Item -LiteralPath (Join-Path $Repo "crates\ime-core\data\mixed\$f") -Destination $mixed
}

# licenses
$licenses = Join-Path $Out 'licenses'
New-Item -ItemType Directory -Force -Path (Join-Path $licenses 'onnxruntime'), (Join-Path $licenses 'llama.cpp') | Out-Null
Copy-Item -LiteralPath (Join-Path $Repo 'LICENSE'), (Join-Path $Repo 'THIRD_PARTY_NOTICES.md') -Destination $licenses
# ONNX Runtime's, from the zip fetch-onnxruntime.ps1 checked
$ortDll = (Get-Item -LiteralPath (Join-Path $Release 'onnxruntime.dll')).VersionInfo.ProductVersion
# "1.28.0.<date>.<build>.<commit>"
$ortDll = if ($ortDll -match '^\d+\.\d+\.\d+') { $Matches[0] }
$ortZip = Get-ChildItem -LiteralPath (Join-Path $Repo 'local\onnxruntime') -Filter 'onnxruntime-win-x64-*.zip' |
    Where-Object { $ortDll -and $_.BaseName -eq "onnxruntime-win-x64-$ortDll" } | Select-Object -First 1
if (-not $ortZip) { throw "no ONNX Runtime zip in local\onnxruntime for onnxruntime.dll $ortDll" }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($ortZip.FullName)
try {
    foreach ($f in @('LICENSE', 'ThirdPartyNotices.txt')) {
        $entry = $archive.Entries | Where-Object { $_.FullName -eq "$($ortZip.BaseName)/$f" }
        if (-not $entry) { throw "$($ortZip.Name) has no $f" }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $licenses "onnxruntime\$f"), $true)
    }
} finally {
    $archive.Dispose()
}
# llama.cpp (MIT), built into the reranker from the llama-cpp-sys-2 crate
$lock = Get-Content -LiteralPath (Join-Path $Repo 'Cargo.lock') -Raw
if ($lock -notmatch '(?m)^name = "llama-cpp-sys-2"\r?\nversion = "([^"]+)"') { throw 'no llama-cpp-sys-2 in Cargo.lock' }
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$llamaLicense = Get-ChildItem -Path (Join-Path $cargoHome "registry\src\*\llama-cpp-sys-2-$($Matches[1])\llama.cpp\LICENSE") |
    Select-Object -First 1
if (-not $llamaLicense) { throw "no llama.cpp LICENSE of llama-cpp-sys-2 $($Matches[1]) under $cargoHome" }
Copy-Item -LiteralPath $llamaLicense.FullName -Destination (Join-Path $licenses 'llama.cpp\LICENSE')

[IO.Compression.ZipFile]::CreateFromDirectory($Out, $Zip, [IO.Compression.CompressionLevel]::Optimal, $true)
Remove-Item -LiteralPath $Out -Recurse -Force

$sha = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
$text = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'setup.ps1'))
$text = $text.Replace("'@VERSION@'", "'$Version'").Replace("'@ZIP_SHA256@'", "'$sha'")
if ($text.Contains('@VERSION@') -or $text.Contains('@ZIP_SHA256@')) { throw 'setup.ps1 placeholders not filled' }
[IO.File]::WriteAllText($Setup, $text, (New-Object Text.UTF8Encoding $false))
$sums = foreach ($f in @($Zip, $Setup)) {
    "$((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLowerInvariant())  $(Split-Path -Leaf $f)"
}
[IO.File]::WriteAllText((Join-Path $Dist 'SHA256SUMS'), (($sums -join "`n") + "`n"), (New-Object Text.ASCIIEncoding))
Get-ChildItem -LiteralPath $Dist | Format-Table Name, Length
