# Fetch Microsoft's official ONNX Runtime build and put onnxruntime.dll next
# to the engine (the engine loads it from there; see src/runtime.rs).
#
# Usage:
#   pwsh -File crates\engine\fetch-onnxruntime.ps1                 # 1.28.0 -> target\release
#   pwsh -File crates\engine\fetch-onnxruntime.ps1 -Dest target\debug
#   pwsh -File crates\engine\fetch-onnxruntime.ps1 -Version 1.30.0 # another pinned build
#
# The pinned version matters for results, not only speed: int8 kernels differ
# between releases. With Aster-v1.2-small int8 over the research general dev set,
# official 1.28.0 gave the same candidates, scores and alignments as the
# ort crate's bundled 1.28 (1603/1603) at ~20% less engine time, while 1.25.1
# and 1.30.0 changed the top-1 of 4-8% of rows (gold accuracy within noise).
# Change the default only together with a re-benchmark.
#
# The download (GitHub release zip) is cached under local\onnxruntime and
# checked against the SHA-256 below. Windows x64 only; elsewhere put
# libonnxruntime.so / .dylib next to the engine or set SETSUNA_ORT_DYLIB.

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Version = '1.28.0',
    [string]$Dest = (Join-Path $PSScriptRoot '..\..\target\release')
)

$ErrorActionPreference = 'Stop'

# SHA-256 of onnxruntime-win-x64-<version>.zip from
# https://github.com/microsoft/onnxruntime/releases
$Known = @{
    '1.25.1' = '33f2e8a63774811f99a5fc224cac32f4eed8c27643d46c6cc685319fa8f18019'
    '1.28.0' = 'abef733dacbe2f571547a7150b479b5cb9cc0df22f96c24983a42cadb1b4f8bc'
    '1.30.0' = 'c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949'
}
if (-not $Known.ContainsKey($Version)) {
    throw "no pinned SHA-256 for ONNX Runtime $Version (known: $($Known.Keys -join ', '))"
}

$Name = "onnxruntime-win-x64-$Version"
$Cache = Join-Path $PSScriptRoot "..\..\local\onnxruntime"
New-Item -ItemType Directory -Force -Path $Cache | Out-Null
$Zip = Join-Path $Cache "$Name.zip"

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not (Test-Path -LiteralPath $Zip) -or (Get-Sha256 $Zip) -ne $Known[$Version]) {
    $Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$Name.zip"
    Write-Host "downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
    $Got = Get-Sha256 $Zip
    if ($Got -ne $Known[$Version]) {
        Remove-Item -LiteralPath $Zip -Force
        throw "SHA-256 mismatch for $Name.zip: got $Got, expected $($Known[$Version])"
    }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
try {
    $entry = $archive.Entries | Where-Object { $_.FullName -eq "$Name/lib/onnxruntime.dll" }
    if (-not $entry) { throw "$Name.zip has no lib/onnxruntime.dll" }
    $Out = Join-Path (Resolve-Path -LiteralPath $Dest) 'onnxruntime.dll'
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Out, $true)
} finally {
    $archive.Dispose()
}
Write-Host "ONNX Runtime $Version -> $Out"
