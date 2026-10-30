# Setsuna IME for Windows from a release: downloads the release zip this
# script was made for (crates\tsf\package.ps1 fills in its version and
# SHA-256), checks it, unpacks it into a temporary directory and runs its
# install.ps1 (UAC for regsvr32; the model is downloaded and pinned there).
#
# Usage (Windows PowerShell 5.1 or pwsh):
#   irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1 | iex
#   & ([scriptblock]::Create((irm https://github.com/esehehelp/setsuna-ime/releases/latest/download/setup.ps1))) -RerankManifest <LM scorer manifest>
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Archive setsuna-ime-<version>-windows-x64.zip
#
# Options are install.ps1's (-ModelPrecision, -RerankManifest,
# -NameDictionary, -Force, -NoElevate), and -Archive <zip> installs a local
# release zip instead of the download. Under iex this runs in the caller's
# session: it throws on failure instead of exiting. ASCII only, as iex reads
# a download without knowing its encoding.

param(
    [ValidateSet('int8', 'fp32')]
    [string]$ModelPrecision = 'int8',
    [string]$RerankManifest,
    [string]$NameDictionary,
    [string]$Archive,
    [switch]$Force,
    [switch]$NoElevate
)

function Install-SetsunaIme {
    param(
        [string]$ModelPrecision,
        [string]$RerankManifest,
        [string]$NameDictionary,
        [string]$Archive,
        [switch]$Force,
        [switch]$NoElevate
    )
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'

    $Version = '@VERSION@'
    $ZipSha256 = '@ZIP_SHA256@'
    $Releases = 'https://github.com/esehehelp/setsuna-ime/releases/download'

    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        throw 'Setsuna IME needs x64 Windows'
    }
    $work = Join-Path ([IO.Path]::GetTempPath()) "setsuna-ime-setup-$([Guid]::NewGuid().ToString('N').Substring(0, 8))"
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    try {
        if ($Archive) {
            $zip = (Resolve-Path -LiteralPath $Archive).Path
        } else {
            if ($Version.StartsWith('@') -or $ZipSha256 -notmatch '^[a-f0-9]{64}$') {
                throw 'this setup.ps1 is not from a release: use -Archive <setsuna-ime-*-windows-x64.zip>'
            }
            $name = "setsuna-ime-$Version-windows-x64.zip"
            $zip = Join-Path $work $name
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            Write-Host "downloading $Releases/v$Version/$name"
            Invoke-WebRequest -Uri "$Releases/v$Version/$name" -OutFile $zip -UseBasicParsing
            $got = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($got -ne $ZipSha256) {
                throw "SHA-256 mismatch for ${name}: got $got, expected $ZipSha256"
            }
        }

        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($zip, $work)
        $installer = @(Get-ChildItem -LiteralPath $work -Directory |
            ForEach-Object { Join-Path $_.FullName 'install.ps1' } |
            Where-Object { Test-Path -LiteralPath $_ })
        if ($installer.Count -ne 1) { throw "not a Setsuna release zip: $zip" }

        $argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installer[0], '-ModelPrecision', $ModelPrecision)
        if ($RerankManifest) { $argv += @('-RerankManifest', (Resolve-Path -LiteralPath $RerankManifest).Path) }
        if ($NameDictionary) { $argv += @('-NameDictionary', (Resolve-Path -LiteralPath $NameDictionary).Path) }
        if ($Force) { $argv += '-Force' }
        if ($NoElevate) { $argv += '-NoElevate' }
        & (Get-Process -Id $PID).Path @argv
        if ($LASTEXITCODE -ne 0) {
            throw "install.ps1 failed with exit code $LASTEXITCODE (log: $env:LOCALAPPDATA\setsuna-ime\install.log)"
        }
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Install-SetsunaIme -ModelPrecision $ModelPrecision -RerankManifest $RerankManifest -NameDictionary $NameDictionary `
    -Archive $Archive -Force:$Force -NoElevate:$NoElevate
