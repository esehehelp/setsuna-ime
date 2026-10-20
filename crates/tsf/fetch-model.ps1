# Fetch a public conversion artifact pinned by models.json. No login required.
# Writes only to the download cache; returns the local manifest path on stdout.
# Verified cache files work offline. Downloads are staged before replacement.
#
# Usage:
#   pwsh -File crates/tsf/fetch-model.ps1
#   pwsh -File crates/tsf/fetch-model.ps1 -Precision fp32 -CacheDir local/models/downloads

#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('int8', 'fp32')]
    [string]$Precision = 'int8',
    [string]$CacheDir = (Join-Path $env:LOCALAPPDATA 'setsuna-ime\cache\models'),
    [string]$CatalogPath = (Join-Path $PSScriptRoot 'models.json')
)
$ErrorActionPreference = 'Stop'

function Assert-ArtifactPath {
    param([string]$Path)
    # Forward-slash relative paths only. Also reject Windows ADS, device names,
    # trailing dots/spaces and URL syntax; every segment is URL-escaped below.
    if (-not $Path -or [IO.Path]::IsPathRooted($Path)) {
        throw "invalid artifact path: $Path"
    }
    foreach ($part in $Path.Split('/')) {
        if (-not $part -or $part -match '[<>:"\\|?*\x00-\x1f]' -or
            $part -match '[. ]$' -or $part -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)') {
            throw "invalid artifact path: $Path"
        }
    }
}

function Test-VerifiedFile {
    param([string]$Path, $Spec)
    (Test-Path -LiteralPath $Path -PathType Leaf) -and
        ((Get-Item -LiteralPath $Path).Length -eq $Spec.size) -and
        ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Spec.sha256)
}

$catalog = Get-Content -LiteralPath $CatalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($catalog.schema -ne 'setsuna-model-download-v1' -or
    $catalog.repository -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$' -or
    $catalog.revision -notmatch '^[a-f0-9]{40}$') {
    throw 'invalid model catalog (expected a repository and pinned commit)'
}
Assert-ArtifactPath $catalog.repository
$variant = $catalog.variants.$Precision
if (-not $variant) { throw "model catalog has no $Precision variant" }
Assert-ArtifactPath $variant.manifest
$specs = @{}
foreach ($file in $variant.files) {
    Assert-ArtifactPath $file.path
    if ($file.sha256 -notmatch '^[a-f0-9]{64}$' -or $null -eq $file.size -or
        $file.size -le 0 -or $specs.ContainsKey($file.path)) {
        throw "invalid or duplicate file specification: $($file.path)"
    }
    $specs[$file.path] = $file
}
if (-not $specs.ContainsKey($variant.manifest)) { throw 'manifest is not pinned in the catalog' }

$destination = Join-Path $CacheDir "$($catalog.repository)/$($catalog.revision)"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$destination = (Resolve-Path -LiteralPath $destination).Path
$baseUrl = "https://huggingface.co/$($catalog.repository)/resolve/$($catalog.revision)"
# Windows PowerShell 5.1 may otherwise negotiate an obsolete TLS version.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Get-ArtifactFile {
    param([string]$RelativePath)
    Assert-ArtifactPath $RelativePath
    if (-not $specs.ContainsKey($RelativePath)) {
        throw "manifest references an unpinned file: $RelativePath"
    }
    $spec = $specs[$RelativePath]
    $path = Join-Path $destination $RelativePath
    if (Test-VerifiedFile $path $spec) {
        Write-Host "model cache verified: $RelativePath"
        return $path
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    $temporary = "$path.$([Guid]::NewGuid().ToString('N')).part"
    $escaped = ($RelativePath.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
    Write-Host "downloading model: $RelativePath"
    try {
        # Never attach HF credentials: only public, pinned artifacts are used.
        Invoke-WebRequest -Uri "$baseUrl/$escaped" -OutFile $temporary -UseBasicParsing -TimeoutSec 120 | Out-Null
        if (-not (Test-VerifiedFile $temporary $spec)) {
            throw "size or SHA-256 mismatch: $RelativePath"
        }
        Move-Item -LiteralPath $temporary -Destination $path -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
    return $path
}

$manifestPath = Get-ArtifactFile $variant.manifest
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.schema -ne 'neo-ime-ctc-nat-export-v1' -or $manifest.proposal.precision -ne $Precision) {
    throw 'downloaded manifest has an unsupported schema or precision'
}
# All names remain relative to the manifest, including nested artifact paths.
$manifestDir = Split-Path -Parent $variant.manifest
$artifacts = @($manifest.proposal.path, $manifest.vocab)
if ($manifest.proposal.external_data) { $artifacts += @($manifest.proposal.external_data) }
$relativeFiles = @($artifacts | ForEach-Object {
    Assert-ArtifactPath $_
    if ($manifestDir) { "$($manifestDir.Replace('\', '/'))/$_" } else { $_ }
})
# Check the whole set before downloading any payload.
foreach ($relative in $relativeFiles) {
    if (-not $specs.ContainsKey($relative)) { throw "manifest references an unpinned file: $relative" }
}
foreach ($relative in $relativeFiles) { Get-ArtifactFile $relative | Out-Null }
Write-Host "model ready: $($catalog.repository) @ $($catalog.revision) ($Precision)"
return $manifestPath
