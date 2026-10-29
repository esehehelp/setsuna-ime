# Offline regression tests; no Pester, credentials, registry writes or real HTTP.
# Run: pwsh -NoProfile -File tests/model-download.ps1
# Generated fixtures/caches are kept under local/model-download-tests-<guid>.
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$fetch = Join-Path $repo 'crates/tsf/fetch-model.ps1'
$root = Join-Path $repo "local/model-download-tests-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $root | Out-Null
$state = @{ requests = @(); offline = $false; badDownload = $false; interrupt = $false; payloads = @{} }
$script:checks = 0
$revision = '0123456789abcdef0123456789abcdef01234567'
$prefix = "https://huggingface.co/example/converter/resolve/$revision/"

function Assert {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "FAIL: $Message" }
    $script:checks++
    Write-Host "PASS: $Message"
}
function Assert-Throws {
    param([scriptblock]$Action, [string]$Message)
    $caught = $false
    try { & $Action | Out-Null } catch {
        $caught = $true
        if ($_.Exception.Message -notlike "*$Message*") { throw }
    }
    Assert $caught "rejects $Message"
}
function Get-Hash {
    param([byte[]]$Bytes)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
function Save-Catalog {
    $script:catalog | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $script:catalogPath -Encoding UTF8
}
function New-Fixture {
    param([string]$Name, [string]$ProposalPath = 'graphs/model.onnx', [string]$Schema = 'neo-ime-ctc-nat-export-v1')
    $script:cache = Join-Path $root "$Name/cache"
    $dir = Join-Path $root $Name
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    $script:catalogPath = Join-Path $dir 'catalog.json'
    $manifest = @{
        schema = $Schema
        proposal = @{ path = $ProposalPath; precision = 'int8'; external_data = @('graphs/weights.bin') }
        vocab = 'vocab.tsv'
        rescore = @{ pairs = 'pairs.bin'; words = 'words.tsv' }
    } | ConvertTo-Json -Depth 5
    $state.payloads = @{
        'bundle/model.manifest.json' = [Text.Encoding]::UTF8.GetBytes($manifest)
        'bundle/graphs/model.onnx' = [Text.Encoding]::UTF8.GetBytes('tiny-onnx')
        'bundle/graphs/weights.bin' = [Text.Encoding]::UTF8.GetBytes('tiny-weights')
        'bundle/vocab.tsv' = [Text.Encoding]::UTF8.GetBytes('tiny-vocab')
        'bundle/pairs.bin' = [Text.Encoding]::UTF8.GetBytes('tiny-pairs')
        'bundle/words.tsv' = [Text.Encoding]::UTF8.GetBytes('tiny-words')
    }
    $files = @($state.payloads.Keys | Sort-Object | ForEach-Object {
        @{ path = $_; size = $state.payloads[$_].Length; sha256 = Get-Hash $state.payloads[$_] }
    })
    $script:catalog = @{
        schema = 'setsuna-model-download-v1'; repository = 'example/converter'; revision = $revision
        variants = @{ int8 = @{ manifest = 'bundle/model.manifest.json'; files = $files } }
    }
    Save-Catalog
    $state.requests = @()
    $state.offline = $false
    $state.badDownload = $false
    $state.interrupt = $false
}

# The mutable state object is shared with this dynamically scoped HTTP stub.
# This intercepts all requests made by the helper, including nested functions.
function Invoke-WebRequest {
    param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing, [int]$TimeoutSec,
          [int]$ConnectionTimeoutSeconds, [int]$OperationTimeoutSeconds)
    $state.requests += $Uri
    if ($state.offline) { throw 'network disabled by test' }
    if (-not $Uri.StartsWith($prefix)) { throw "unexpected URL: $Uri" }
    $relative = [Uri]::UnescapeDataString($Uri.Substring($prefix.Length))
    if (-not $state.payloads.ContainsKey($relative)) { throw "unexpected artifact: $relative" }
    if ($relative.EndsWith('.onnx') -and $state.interrupt) {
        [IO.File]::WriteAllBytes($OutFile, [byte[]]@(1, 2, 3))
        throw 'interrupted download'
    }
    $bytes = $state.payloads[$relative]
    if ($relative.EndsWith('.onnx') -and $state.badDownload) {
        $bytes = [Text.Encoding]::UTF8.GetBytes('bad-onnx!') # same length, different hash
    }
    [IO.File]::WriteAllBytes($OutFile, $bytes)
}

New-Fixture 'fresh'
$path = @(& $fetch -CacheDir $cache -CatalogPath $catalogPath)
Assert ($path.Count -eq 1 -and (Test-Path -LiteralPath $path[0])) 'returns exactly one manifest path'
Assert ($state.requests.Count -eq 6) 'fetches manifest, ONNX, external data, vocabulary and rescore tables'
Assert ($path[0] -eq (Join-Path $cache "example/converter/$revision/bundle/model.manifest.json")) 'uses revision-scoped cache'
foreach ($relative in $state.payloads.Keys) {
    $placed = Join-Path $cache "example/converter/$revision/$relative"
    Assert ((Get-FileHash -LiteralPath $placed -Algorithm SHA256).Hash -eq (Get-Hash $state.payloads[$relative])) "verified $relative"
}
$state.offline = $true
$state.requests = @()
$again = & $fetch -CacheDir $cache -CatalogPath $catalogPath
Assert ($again -eq $path[0] -and $state.requests.Count -eq 0) 'verified cache works fully offline'

$onnx = Join-Path (Split-Path -Parent $path[0]) 'graphs/model.onnx'
[IO.File]::WriteAllText($onnx, 'bad-onnx!')
$state.offline = $false
$repaired = & $fetch -CacheDir $cache -CatalogPath $catalogPath
Assert ($state.requests.Count -eq 1 -and $state.requests[0].EndsWith('/graphs/model.onnx')) 'only corrupt cached file is downloaded again'
Assert ((Get-FileHash -LiteralPath $onnx -Algorithm SHA256).Hash -eq (Get-Hash $state.payloads['bundle/graphs/model.onnx'])) 'corrupt cache is repaired'

New-Fixture 'interrupted'
$state.interrupt = $true
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'interrupted download'
Assert (@(Get-ChildItem -LiteralPath $cache -Recurse -Filter '*.part').Count -eq 0) 'interrupted temporary file is cleaned up'
Assert (@(Get-ChildItem -LiteralPath $cache -Recurse -Filter '*.onnx').Count -eq 0) 'partial artifact is never published'
$state.interrupt = $false
$recovered = & $fetch -CacheDir $cache -CatalogPath $catalogPath
Assert (Test-Path -LiteralPath $recovered) 'retry recovers after interrupted download'

New-Fixture 'bad-hash'
$state.badDownload = $true
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'SHA-256 mismatch'
Assert (@(Get-ChildItem -LiteralPath $cache -Recurse -Filter '*.part').Count -eq 0) 'hash failure cleans temporary file'
Assert (@(Get-ChildItem -LiteralPath $cache -Recurse -Filter '*.onnx').Count -eq 0) 'hash failure does not publish bad artifact'

New-Fixture 'bad-size'
$catalog.variants.int8.files | Where-Object { $_.path.EndsWith('.onnx') } | ForEach-Object { $_.size++ }
Save-Catalog
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'size or SHA-256 mismatch'

New-Fixture 'unpinned' 'graphs/not-pinned.onnx'
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'unpinned file'
Assert ($state.requests.Count -eq 1) 'validates every reference before fetching payloads'

foreach ($unsafe in @('../outside.onnx', 'C:/outside.onnx', '/outside.onnx', 'graphs\outside.onnx',
                       'graphs/file:stream', 'graphs/CON.onnx', 'graphs/file.', 'graphs//file.onnx')) {
    New-Fixture ([Guid]::NewGuid().ToString('N')) $unsafe
    Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'invalid artifact path'
    Assert ($state.requests.Count -eq 1) "unsafe manifest path rejected before payloads: $unsafe"
}
New-Fixture 'bad-catalog-path'
$catalog.variants.int8.files[0].path = '../outside'
Save-Catalog
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'invalid artifact path'
Assert ($state.requests.Count -eq 0) 'catalog path traversal rejected before HTTP'

New-Fixture 'floating-revision'
$catalog.revision = 'main'
Save-Catalog
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'invalid model catalog'
Assert ($state.requests.Count -eq 0) 'floating revision rejected before HTTP'

New-Fixture 'unsupported-schema' 'graphs/model.onnx' 'wrong-schema'
Assert-Throws { & $fetch -CacheDir $cache -CatalogPath $catalogPath } 'unsupported schema or precision'
Assert ($state.requests.Count -eq 1) 'schema checked before payloads'

# Installer parameter sets preserve local/offline installs without ambiguity.
$installer = Join-Path $repo 'crates/tsf/install.ps1'
$sets = (Get-Command $installer).ParameterSets
Assert (@($sets | Where-Object { $_.Name -eq 'Download' -and $_.IsDefault }).Count -eq 1) 'installer defaults to download'
$local = $sets | Where-Object { $_.Name -eq 'Local' }
Assert (@($local.Parameters | Where-Object { $_.Name -eq 'Manifest' -and $_.IsMandatory }).Count -eq 1) 'local manifest remains supported'
Assert (@($local.Parameters | Where-Object { $_.Name -eq 'ModelPrecision' }).Count -eq 0) 'local manifest and download precision are mutually exclusive'

# A real child installer must propagate network failure as code 30, without
# publishing models, writing config or reaching COM/registry operations.
$source = Join-Path $root 'installer-source'
New-Item -ItemType Directory -Path $source | Out-Null
foreach ($name in @('setsuna_tsf.dll', 'setsuna-engine.exe', 'onnxruntime.dll')) {
    [IO.File]::WriteAllText((Join-Path $source $name), 'stub')
}
$wrapper = Join-Path $root 'installer-failure.ps1'
@'
param($Installer, $Source, $Sandbox)
$env:LOCALAPPDATA = Join-Path $Sandbox 'Local'
$env:APPDATA = Join-Path $Sandbox 'Roaming'
function Invoke-WebRequest { throw 'offline fixture' }
& $Installer -SourceDir $Source -NoElevate
exit $LASTEXITCODE
'@ | Set-Content -LiteralPath $wrapper -Encoding UTF8
$shell = (Get-Process -Id $PID).Path
& $shell -NoProfile -File $wrapper -Installer $installer -Source $source -Sandbox (Join-Path $root 'installer-failure')
Assert ($LASTEXITCODE -eq 30) 'installer propagates download failure as exit 30'
Assert (-not (Test-Path -LiteralPath (Join-Path $root 'installer-failure/Roaming/setsuna-ime/config.toml'))) 'download failure leaves config untouched'
Assert (-not (Test-Path -LiteralPath (Join-Path $root 'installer-failure/Local/setsuna-ime/models'))) 'download failure never starts installation'

Write-Host "$script:checks checks passed. Test artifacts: $root"
