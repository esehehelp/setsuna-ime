# Setsuna TSF IME installer.
#
# Usage:
#   pwsh -File crates\tsf\install.ps1                           # download pinned int8 model + install
#   pwsh -File crates\tsf\install.ps1 -ModelPrecision fp32      # download pinned fp32 model instead
#   pwsh -File crates\tsf\install.ps1 -Manifest <m>             # local model, no download (UAC for regsvr32)
#   pwsh -File crates\tsf\install.ps1 -Manifest <m> -NoElevate  # files-only (CI)
#   pwsh -File crates\tsf\install.ps1 -Manifest <m> -Force      # overwrite even if same
#   pwsh -File crates\tsf\install.ps1 -Manifest <m> -RerankManifest <r>  # also the reranker
#   pwsh -File crates\tsf\install.ps1 -Manifest <m> -NameDictionary <tsv>  # also mixed input's names
#
# Without -Manifest, fetch-model.ps1 downloads the public artifact pinned in
# models.json, verifies its size/SHA-256, and reuses verified cache files offline.
# <m> is the model's `*.manifest.json` written by export. The manifest and
# every file it lists are copied, keeping their relative paths.
#
# In a release zip (crates\tsf\package.ps1) the binaries sit next to this
# script and the mixed input model in mixed\; it runs on Windows PowerShell
# 5.1 as well as pwsh. The Visual C++ runtime the binaries link (14.44 or
# later) is installed from Microsoft when missing.
#
# Exit codes:
#   0  success / already up to date
#   10 unsupported architecture (need x64)
#   20 regsvr32 failed
#   30 model file(s) not found, download failed, or integrity check failed
#   40 source DLL, engine.exe or onnxruntime.dll not found
#   50 file copy failed
#   60 Visual C++ runtime missing and could not be installed
#
# Layout produced:
#   %LOCALAPPDATA%\setsuna-ime\setsuna_tsf.dll
#   %LOCALAPPDATA%\setsuna-ime\setsuna_tsf_<sha256>.dll  (fallback when the stable DLL is locked)
#   %LOCALAPPDATA%\setsuna-ime\setsuna-engine.exe
#   %LOCALAPPDATA%\setsuna-ime\setsuna-engine_<sha256>.exe  (fallback when running daemons lock it;
#                                                     config engine_exe points at it)
#   %LOCALAPPDATA%\setsuna-ime\onnxruntime.dll  (official ONNX Runtime the engine loads;
#                                           a content-addressed fallback when locked,
#                                           config ort_dylib points at it)
#   %LOCALAPPDATA%\setsuna-ime\setsuna-settings.exe  (settings GUI, when built)
#   %LOCALAPPDATA%\setsuna-ime\setsuna-rewrite.exe  (rewriter, when built; a content-addressed
#                                            fallback when locked, config [rewriter] exe follows)
#   %LOCALAPPDATA%\setsuna-ime\uninstall.ps1
#   %LOCALAPPDATA%\setsuna-ime\models\onnx\<manifest> + the files it lists
#   %LOCALAPPDATA%\setsuna-ime\rerank\setsuna-rerank.exe + llama.cpp's DLLs (with -RerankManifest;
#                                  rerank_<sha256>\ when a running reranker locks them)
#   %LOCALAPPDATA%\setsuna-ime\models\rerank\<rerank manifest> + its weights
#   %LOCALAPPDATA%\setsuna-ime\models\mixed\kana.tsv, latin.tsv, params.txt  (crates\ime-core\data\mixed,
#                                  or -MixedModel <dir>)
#   %LOCALAPPDATA%\setsuna-ime\models\names\<names dictionary>  (with -NameDictionary)
#   %APPDATA%\setsuna-ime\config.toml   (user-editable runtime config)
#   HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\setsuna-ime
#
# Idempotent: re-running converges to the desired state. Per-user file layout
# under %LOCALAPPDATA%; regsvr32 still requires admin because registration.rs
# writes HKEY_CLASSES_ROOT (admin-mapped); registering under
# HKCU\Software\Classes instead would allow a no-elevation install.

#Requires -Version 5.1
[CmdletBinding(DefaultParameterSetName = 'Download')]
param(
    # Default: the build output, or this script's own directory in a release zip.
    [string]$SourceDir,
    [Parameter(Mandatory, ParameterSetName = 'Local')]
    [ValidateNotNullOrEmpty()]
    [string]$Manifest,
    [Parameter(ParameterSetName = 'Download')]
    [ValidateSet('int8', 'fp32')]
    [string]$ModelPrecision = 'int8',
    # LM scorer manifest (neo-ime-lm-scorer-export-v1): install the reranker too.
    [string]$RerankManifest,
    # Mixed input model directory (kana.tsv, latin.tsv, params.txt): English
    # words typed in romaji stay English ([frontend] mixed_model). Default:
    # the one of this repository, or mixed\ of a release zip; "" for none.
    [string]$MixedModel,
    # Latin names dictionary for mixed input ([frontend] name_dictionary_file).
    [string]$NameDictionary,
    [switch]$Force,
    [switch]$NoElevate
)

$ErrorActionPreference = 'Stop'

# Defaults that name this script's directory are set here: Windows PowerShell
# 5.1 leaves $PSScriptRoot empty in an advanced script's parameter defaults.
$Packaged = Test-Path -LiteralPath (Join-Path $PSScriptRoot 'setsuna_tsf.dll')
if (-not $PSBoundParameters.ContainsKey('SourceDir')) {
    $SourceDir = if ($Packaged) { $PSScriptRoot } else { Join-Path $PSScriptRoot '..\..\target\release' }
}
if (-not $PSBoundParameters.ContainsKey('MixedModel')) {
    $MixedModel = if ($Packaged) { Join-Path $PSScriptRoot 'mixed' } else { Join-Path $PSScriptRoot '..\ime-core\data\mixed' }
}

# ---- constants ----
$InstallDir = Join-Path $env:LOCALAPPDATA 'setsuna-ime'
$ConfigDir  = Join-Path $env:APPDATA      'setsuna-ime'
$LogFile    = Join-Path $InstallDir 'install.log'
$DllName    = 'setsuna_tsf.dll'
$EngineExe  = 'setsuna-engine.exe'
$OrtDll     = 'onnxruntime.dll'
$SettingsExe = 'setsuna-settings.exe'
$RewriteExe = 'setsuna-rewrite.exe'
$RerankExe  = 'setsuna-rerank.exe'
$UninstReg  = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\setsuna-ime'
# The PowerShell running this (pwsh or Windows PowerShell): the elevated
# install runs on it too.
$PsExe      = (Get-Process -Id $PID).Path
# Microsoft's link to the latest supported redistributable
$VcRuntimeUrl = 'https://aka.ms/vc14/vc_redist.x64.exe'
# GitHub and Microsoft's downloads need TLS 1.2, not the default of older
# Windows PowerShell 5.1 installs.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# ---- helpers ----
function Write-Log {
    param([string]$Level, [string]$Msg)
    $line = "$(Get-Date -Format o) [$Level] $Msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction Stop } catch { }
}

function Resolve-Path-Safe {
    param([string]$p)
    try { (Resolve-Path -LiteralPath $p -ErrorAction Stop).Path } catch { $p }
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pri = [Security.Principal.WindowsPrincipal] $id
    $pri.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# config.toml is UTF-8 without a BOM whichever PowerShell writes it
# (Windows PowerShell's -Encoding utf8 adds one).
function Write-Utf8Lines {
    param([string]$Path, [string[]]$Lines)
    [IO.File]::WriteAllLines($Path, $Lines, (New-Object Text.UTF8Encoding $false))
}

# The Visual C++ 2015-2022 runtime (VCRUNTIME140, MSVCP140, VCOMP140) the
# DLL, the daemons, ONNX Runtime and llama.cpp link, at least the 14.44
# toolset they are built with: a binary may crash on an older MSVCP140 (a
# 14.40 build's std::mutex does). Raise it with the build's toolset.
function Test-VcRuntime {
    try {
        $k = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction Stop
        ($k.Installed -eq 1) -and (($k.Major -gt 14) -or ($k.Major -eq 14 -and $k.Minor -ge 44))
    } catch {
        $false
    }
}

# Microsoft's redistributable, checked to carry Microsoft's signature, run
# quietly (elevated).
function Install-VcRuntime {
    $exe = Join-Path ([IO.Path]::GetTempPath()) "setsuna-vc_redist-$PID.exe"
    try {
        Write-Log INFO "downloading the Visual C++ runtime: $VcRuntimeUrl"
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $VcRuntimeUrl -OutFile $exe -UseBasicParsing
        $sig = Get-AuthenticodeSignature -LiteralPath $exe
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
            throw "vc_redist.x64.exe is not signed by Microsoft ($($sig.Status))"
        }
        $p = Start-Process -FilePath $exe -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
        # 3010: installed, a restart pending; 1638: a newer one is installed
        if ($p.ExitCode -notin 0, 3010, 1638) {
            throw "vc_redist.x64.exe exited with $($p.ExitCode)"
        }
        Write-Log INFO "Visual C++ runtime installed (exit code $($p.ExitCode))"
    } finally {
        Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
    }
}

# The release's version (VERSION of a release zip), else the workspace's.
function Get-SetsunaVersion {
    $file = Join-Path $SourceDir 'VERSION'
    if (Test-Path -LiteralPath $file) {
        return (Get-Content -LiteralPath $file -TotalCount 1).Trim()
    }
    $cargo = Join-Path $PSScriptRoot '..\..\Cargo.toml'
    if ((Test-Path -LiteralPath $cargo) -and
        ((Get-Content -LiteralPath $cargo -Raw) -match '(?ms)^\[workspace\.package\].*?^version = "([^"]+)"')) {
        return $Matches[1]
    }
    'unknown'
}

# `<stem>_<sha256[:12]><ext>` of the destination name: where a locked binary
# (a DLL loaded by input hosts, an engine daemon they started) is installed
# instead of overwriting it.
function Get-ContentAddressedName {
    param([string]$Source, [string]$Destination)
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.Substring(0, 12).ToLowerInvariant()
    $leaf = Split-Path -Leaf $Destination
    "$([IO.Path]::GetFileNameWithoutExtension($leaf))_$hash$([IO.Path]::GetExtension($leaf))"
}

# Whether $Destination exists with the content of $Source. By hash: a rebuilt
# binary often keeps its size.
function Test-SameContent {
    param([string]$Source, [string]$Destination)
    (Test-Path -LiteralPath $Destination) -and
        ((Get-Item -LiteralPath $Source).Length -eq (Get-Item -LiteralPath $Destination).Length) -and
        ((Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash -eq
         (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash)
}

function Copy-InstallFile {
    param(
        [string]$Source,
        [string]$Destination,
        [switch]$AllowLockedFallback
    )

    # Reinstalling from the installed model: nothing to copy.
    if ((Test-Path -LiteralPath $Destination) -and
        ((Resolve-Path -LiteralPath $Source).Path -eq (Resolve-Path -LiteralPath $Destination).Path)) {
        Write-Log INFO "skip (already in place) $Destination"
        return $Destination
    }

    if (-not $Force -and (Test-SameContent -Source $Source -Destination $Destination)) {
        Write-Log INFO "skip (unchanged) $Destination"
        return $Destination
    }

    try {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
        Write-Log INFO "copied $Source -> $Destination"
        return $Destination
    } catch {
        if (-not $AllowLockedFallback) {
            throw
        }

        $fallback = Join-Path $InstallDir (Get-ContentAddressedName -Source $Source -Destination $Destination)
        if (-not (Test-SameContent -Source $Source -Destination $fallback)) {
            Copy-Item -LiteralPath $Source -Destination $fallback -Force
            Write-Log WARN "$Destination was locked; copied $Source -> $fallback"
        } else {
            Write-Log WARN "$Destination was locked; using existing $fallback"
        }
        return $fallback
    }
}

function Set-OrAppendConfigValue {
    param(
        [string[]]$Lines,
        [string]$Section,
        [string]$Key,
        [string]$Value
    )

    $out = New-Object System.Collections.Generic.List[string]
    $inSection = $false
    $sectionSeen = $false
    $keyWritten = $false

    foreach ($line in $Lines) {
        if ($line -match '^\s*\[(.+)\]\s*$') {
            if ($inSection -and -not $keyWritten) {
                $out.Add("$Key = $Value")
                $keyWritten = $true
            }
            $inSection = ($Matches[1] -eq $Section)
            if ($inSection) { $sectionSeen = $true }
            $out.Add($line)
            continue
        }

        if ($inSection -and $line -match "^\s*$([regex]::Escape($Key))\s*=") {
            if (-not $keyWritten) {
                $out.Add("$Key = $Value")
                $keyWritten = $true
            }
            continue
        }

        $out.Add($line)
    }

    if ($inSection -and -not $keyWritten) {
        $out.Add("$Key = $Value")
        $keyWritten = $true
    }
    if (-not $sectionSeen) {
        $out.Add("")
        $out.Add("[$Section]")
        $out.Add("$Key = $Value")
    }
    return $out.ToArray()
}

function Remove-ConfigValue {
    param(
        [string[]]$Lines,
        [string]$Section,
        [string]$Key
    )

    $out = New-Object System.Collections.Generic.List[string]
    $inSection = $false
    foreach ($line in $Lines) {
        if ($line -match '^\s*\[(.+)\]\s*$') {
            $inSection = ($Matches[1] -eq $Section)
        } elseif ($inSection -and $line -match "^\s*$([regex]::Escape($Key))\s*=") {
            continue
        }
        $out.Add($line)
    }
    return $out.ToArray()
}

function Update-ExistingConfig {
    param(
        [string]$ConfigPath,
        [string]$ManifestPath,
        [string]$EnginePath,
        [string]$OrtPath,
        [string]$RerankManifestPath,
        [string]$RerankExePath,
        [string]$RewriteExePath,
        [string]$MixedModelPath,
        [string]$NameDictionaryPath
    )

    $lines = @(Get-Content -LiteralPath $ConfigPath -Encoding UTF8 -ErrorAction Stop)
    $qManifestPath = '"' + ($ManifestPath.Replace('\','/').Replace('"','\"')) + '"'
    $qEnginePath = '"' + ($EnginePath.Replace('\','/').Replace('"','\"')) + '"'
    $qOrtPath = '"' + ($OrtPath.Replace('\','/').Replace('"','\"')) + '"'

    # `model_dir` would compete with the explicit manifest.
    $lines = Remove-ConfigValue -Lines $lines -Section 'engine' -Key 'model_dir'
    $lines = Set-OrAppendConfigValue -Lines $lines -Section 'engine' -Key 'model_manifest' -Value $qManifestPath
    $lines = Set-OrAppendConfigValue -Lines $lines -Section 'engine' -Key 'engine_exe' -Value $qEnginePath
    $lines = Set-OrAppendConfigValue -Lines $lines -Section 'engine' -Key 'ort_dylib' -Value $qOrtPath
    # Decode knobs are left alone: the manifest's decode_defaults carry the
    # recipe the artifact was verified with; a knob kept in config.toml is a
    # deliberate user override.
    if ($RerankManifestPath) {
        $qRerankManifest = '"' + ($RerankManifestPath.Replace('\','/').Replace('"','\"')) + '"'
        $qRerankExe = '"' + ($RerankExePath.Replace('\','/').Replace('"','\"')) + '"'
        $lines = Set-OrAppendConfigValue -Lines $lines -Section 'rerank' -Key 'manifest' -Value $qRerankManifest
        $lines = Set-OrAppendConfigValue -Lines $lines -Section 'rerank' -Key 'exe' -Value $qRerankExe
    }
    # mixed_model = "" is mixed input turned off: kept
    if ($MixedModelPath -and -not ($lines -match '^\s*mixed_model\s*=\s*""\s*$')) {
        $q = '"' + ($MixedModelPath.Replace('\','/').Replace('"','\"')) + '"'
        $lines = Set-OrAppendConfigValue -Lines $lines -Section 'frontend' -Key 'mixed_model' -Value $q
    }
    if ($NameDictionaryPath) {
        $q = '"' + ($NameDictionaryPath.Replace('\','/').Replace('"','\"')) + '"'
        $lines = Set-OrAppendConfigValue -Lines $lines -Section 'frontend' -Key 'name_dictionary_file' -Value $q
    }
    # The rewriter is found next to the DLL; a locked one's fallback is named.
    if ($RewriteExePath -and ($RewriteExePath -ne (Join-Path $InstallDir $RewriteExe))) {
        $qRewriteExe = '"' + ($RewriteExePath.Replace('\','/').Replace('"','\"')) + '"'
        $lines = Set-OrAppendConfigValue -Lines $lines -Section 'rewriter' -Key 'exe' -Value $qRewriteExe
    } elseif ($RewriteExePath) {
        $lines = Remove-ConfigValue -Lines $lines -Section 'rewriter' -Key 'exe'
    }
    Write-Utf8Lines -Path $ConfigPath -Lines $lines
}

# ---- pre-flight ----
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Log INFO "install starting (Source=$SourceDir, Manifest=$Manifest, Force=$Force, NoElevate=$NoElevate)"

if ([Environment]::Is64BitOperatingSystem -ne $true) {
    Write-Log ERROR 'x64 required'; exit 10
}

$SourceDir = Resolve-Path-Safe $SourceDir
$DllSrc    = Join-Path $SourceDir $DllName
$EngineSrc = Join-Path $SourceDir $EngineExe

if (-not (Test-Path -LiteralPath $DllSrc))    { Write-Log ERROR "source DLL not found: $DllSrc";    exit 40 }
if (-not (Test-Path -LiteralPath $EngineSrc)) { Write-Log ERROR "engine.exe not found: $EngineSrc"; exit 40 }
$OrtSrc = Join-Path $SourceDir $OrtDll
if (-not (Test-Path -LiteralPath $OrtSrc)) {
    Write-Log ERROR "onnxruntime.dll not found: $OrtSrc (run crates\engine\fetch-onnxruntime.ps1)"; exit 40
}

# Resolve the public artifact before elevation. The elevated invocation receives
# this local manifest, so it neither downloads again nor changes cache ownership.
if ($PSCmdlet.ParameterSetName -eq 'Download') {
    try {
        $Manifest = & (Join-Path $PSScriptRoot 'fetch-model.ps1') -Precision $ModelPrecision
    } catch {
        Write-Log ERROR "model download failed: $_ (retry, or use -Manifest for a local artifact)"; exit 30
    }
}
$Manifest = Resolve-Path-Safe $Manifest

# The manifest names every artifact file relative to itself. Each is copied
# to the same relative path under the install dir, so the installed manifest
# resolves unchanged; a path leaving the artifact directory is refused.
if (-not (Test-Path -LiteralPath $Manifest)) {
    Write-Log ERROR "manifest not found: $Manifest"; exit 30
}
$ManifestItem = Get-Item -LiteralPath $Manifest
$ManifestJson = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$ArtifactFiles = @($ManifestJson.proposal.path, $ManifestJson.vocab)
if ($ManifestJson.proposal.external_data) { $ArtifactFiles += @($ManifestJson.proposal.external_data) }
if ($ManifestJson.input_vocab) { $ArtifactFiles += @($ManifestJson.input_vocab) }
foreach ($rel in $ArtifactFiles) {
    if ([IO.Path]::IsPathRooted($rel) -or $rel -match '(^|[\\/])\.\.([\\/]|$)') {
        Write-Log ERROR "manifest path $rel leaves the artifact directory"; exit 30
    }
    $src = Join-Path $ManifestItem.Directory.FullName $rel
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Log ERROR "artifact file not found: $src"; exit 30
    }
}
Write-Log INFO "using manifest: $($ManifestItem.FullName)"

# The reranker: its exe and llama.cpp's DLLs from the build, and the LM
# scorer artifact its manifest lists.
$RerankItem = $null
if ($RerankManifest) {
    $RerankManifest = Resolve-Path-Safe $RerankManifest
    if (-not (Test-Path -LiteralPath $RerankManifest)) {
        Write-Log ERROR "rerank manifest not found: $RerankManifest"; exit 30
    }
    $RerankItem = Get-Item -LiteralPath $RerankManifest
    $RerankJson = Get-Content -LiteralPath $RerankManifest -Raw | ConvertFrom-Json
    $RerankWeights = $RerankJson.weights.path
    if ([IO.Path]::IsPathRooted($RerankWeights) -or $RerankWeights -match '(^|[\\/])\.\.([\\/]|$)') {
        Write-Log ERROR "rerank manifest path $RerankWeights leaves the artifact directory"; exit 30
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RerankItem.Directory.FullName $RerankWeights))) {
        Write-Log ERROR "rerank weights not found: $RerankWeights"; exit 30
    }
    $RerankSrc = Join-Path $SourceDir $RerankExe
    if (-not (Test-Path -LiteralPath $RerankSrc)) {
        Write-Log ERROR "$RerankExe not found: $RerankSrc (cargo build --release -p setsuna-rerank)"; exit 40
    }
    Write-Log INFO "using rerank manifest: $($RerankItem.FullName)"
}

# Mixed input: the model's three files, and the names dictionary.
$MixedFiles = @('kana.tsv', 'latin.tsv', 'params.txt')
if ($MixedModel) {
    $MixedModel = Resolve-Path-Safe $MixedModel
    foreach ($f in $MixedFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $MixedModel $f))) {
            Write-Log ERROR "mixed model file not found: $(Join-Path $MixedModel $f)"; exit 30
        }
    }
    Write-Log INFO "using mixed model: $MixedModel"
}
if ($NameDictionary) {
    $NameDictionary = Resolve-Path-Safe $NameDictionary
    if (-not (Test-Path -LiteralPath $NameDictionary)) {
        Write-Log ERROR "names dictionary not found: $NameDictionary"; exit 30
    }
    Write-Log INFO "using names dictionary: $NameDictionary"
}

# ---- elevate when regsvr32 needed ----
if (-not (Test-IsAdmin) -and -not $NoElevate) {
    Write-Log INFO 'elevating for regsvr32'
    $argv = @(
        '-NoProfile','-ExecutionPolicy','Bypass','-File',$PSCommandPath,
        '-SourceDir',("`"$SourceDir`""),'-Manifest',("`"$Manifest`"")
    )
    if ($RerankManifest) { $argv += @('-RerankManifest', ("`"$RerankManifest`"")) }
    # always: "" (none) would be the default again
    $argv += @('-MixedModel', ("`"$MixedModel`""))
    if ($NameDictionary) { $argv += @('-NameDictionary', ("`"$NameDictionary`"")) }
    if ($Force) { $argv += '-Force' }
    try {
        $p = Start-Process $PsExe -ArgumentList $argv -Verb RunAs -Wait -PassThru
        Write-Log INFO "elevated install exit code: $($p.ExitCode)"
        exit $p.ExitCode
    } catch {
        Write-Log ERROR "elevation declined: $_"; exit 1
    }
}

# ---- Visual C++ runtime ----
# Without it regsvr32 cannot load the DLL, nor input hosts afterwards.
if (Test-VcRuntime) {
    Write-Log INFO 'Visual C++ runtime 14.44+ present'
} elseif ($NoElevate) {
    Write-Log WARN "Visual C++ runtime 14.44+ missing: install $VcRuntimeUrl"
} else {
    try {
        Install-VcRuntime
    } catch {
        Write-Log ERROR "Visual C++ runtime missing and not installed: $_ (install $VcRuntimeUrl and re-run)"; exit 60
    }
    if (-not (Test-VcRuntime)) {
        Write-Log ERROR "Visual C++ runtime still missing after its installer (install $VcRuntimeUrl and re-run)"; exit 60
    }
}

# ---- copy files ----
$ModelDstDir = Join-Path $InstallDir 'models\onnx'
New-Item -ItemType Directory -Force -Path $ModelDstDir | Out-Null

$InstalledManifest = Join-Path $ModelDstDir $ManifestItem.Name
$copies = @(
    @{ src = $ManifestItem.FullName;  dst = $InstalledManifest },
    @{ src = (Join-Path $PSScriptRoot 'uninstall.ps1'); dst = (Join-Path $InstallDir 'uninstall.ps1') }
)
foreach ($rel in $ArtifactFiles) {
    $dst = Join-Path $ModelDstDir $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    $copies += @{ src = (Join-Path $ManifestItem.Directory.FullName $rel); dst = $dst }
}

$DllInstalled = Join-Path $InstallDir $DllName
try {
    $DllInstalled = Copy-InstallFile -Source $DllSrc -Destination $DllInstalled -AllowLockedFallback
} catch {
    Write-Log ERROR "copy failed $DllSrc -> $DllInstalled`: $_"; exit 50
}

# Engine daemons started by running input hosts lock the exe; the new one goes
# next to it under a content-addressed name and config engine_exe follows.
$EngineInstalled = Join-Path $InstallDir $EngineExe
try {
    $EngineInstalled = Copy-InstallFile -Source $EngineSrc -Destination $EngineInstalled -AllowLockedFallback
} catch {
    Write-Log ERROR "copy failed $EngineSrc -> $EngineInstalled`: $_"; exit 50
}

# The ONNX Runtime the engine loads (by absolute path, config ort_dylib).
# Running daemons lock it; a new build then goes next to it under a
# content-addressed name.
$OrtInstalled = Join-Path $InstallDir $OrtDll
try {
    $OrtInstalled = Copy-InstallFile -Source $OrtSrc -Destination $OrtInstalled -AllowLockedFallback
} catch {
    Write-Log ERROR "copy failed $OrtSrc -> $OrtInstalled`: $_"; exit 50
}

# The settings GUI is optional: the IME works without it.
$SettingsSrc = Join-Path $SourceDir $SettingsExe
if (Test-Path -LiteralPath $SettingsSrc) {
    try {
        Copy-InstallFile -Source $SettingsSrc -Destination (Join-Path $InstallDir $SettingsExe) | Out-Null
    } catch {
        Write-Log WARN "copy failed $SettingsSrc`: $_ (close setsuna-settings and re-run)"
    }
} else {
    Write-Log WARN "settings GUI not found: $SettingsSrc"
}

# The rewriter is optional too. Running ones (one per app that used rewrite
# mode) lock it: a new build then goes next to it under a content-addressed
# name and config [rewriter] exe follows.
$RewriteInstalled = $null
$RewriteSrc = Join-Path $SourceDir $RewriteExe
if (Test-Path -LiteralPath $RewriteSrc) {
    try {
        $RewriteInstalled = Copy-InstallFile -Source $RewriteSrc -Destination (Join-Path $InstallDir $RewriteExe) -AllowLockedFallback
    } catch {
        Write-Log WARN "copy failed $RewriteSrc`: $_"
    }
} else {
    Write-Log WARN "rewriter not found: $RewriteSrc"
}

foreach ($c in $copies) {
    try {
        Copy-InstallFile -Source $c.src -Destination $c.dst | Out-Null
    } catch {
        Write-Log ERROR "copy failed $($c.src) -> $($c.dst): $_"; exit 50
    }
}

# The reranker and llama.cpp's DLLs go together into rerank\ (it loads them
# from its own directory); when a running reranker locks them, the whole set
# goes into rerank_<sha256 of the exe>\ and config exe follows.
$RerankInstalled = $null
$RerankManifestInstalled = $null
if ($RerankItem) {
    $RerankFiles = @(Get-Item -LiteralPath $RerankSrc) +
                   @(Get-ChildItem -LiteralPath $SourceDir -File -Filter '*.dll' |
                     Where-Object { $_.Name -match '^(llama|ggml)' })
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $RerankSrc).Hash.Substring(0, 12).ToLowerInvariant()
    $placed = $false
    foreach ($dir in @((Join-Path $InstallDir 'rerank'), (Join-Path $InstallDir "rerank_$hash"))) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        try {
            foreach ($file in $RerankFiles) {
                Copy-InstallFile -Source $file.FullName -Destination (Join-Path $dir $file.Name) | Out-Null
            }
            $RerankInstalled = Join-Path $dir $RerankExe
            $placed = $true
            break
        } catch {
            Write-Log WARN "rerank files in $dir are locked: $_"
        }
    }
    if (-not $placed) { Write-Log ERROR 'could not place the reranker'; exit 50 }
    $RerankModelDir = Join-Path $InstallDir 'models\rerank'
    New-Item -ItemType Directory -Force -Path $RerankModelDir | Out-Null
    $RerankManifestInstalled = Join-Path $RerankModelDir $RerankItem.Name
    try {
        Copy-InstallFile -Source $RerankItem.FullName -Destination $RerankManifestInstalled | Out-Null
        $dst = Join-Path $RerankModelDir $RerankWeights
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
        Copy-InstallFile -Source (Join-Path $RerankItem.Directory.FullName $RerankWeights) -Destination $dst | Out-Null
    } catch {
        Write-Log ERROR "copy failed (rerank model): $_"; exit 50
    }
}

# Mixed input: read when the IME activates, so a file in use is not locked.
$MixedInstalled = $null
$NamesInstalled = $null
try {
    if ($MixedModel) {
        $MixedInstalled = Join-Path $InstallDir 'models\mixed'
        New-Item -ItemType Directory -Force -Path $MixedInstalled | Out-Null
        # its README carries the sources' attribution
        foreach ($f in $MixedFiles + @('README.md')) {
            if (Test-Path -LiteralPath (Join-Path $MixedModel $f)) {
                Copy-InstallFile -Source (Join-Path $MixedModel $f) -Destination (Join-Path $MixedInstalled $f) | Out-Null
            }
        }
    }
    if ($NameDictionary) {
        $NamesDir = Join-Path $InstallDir 'models\names'
        New-Item -ItemType Directory -Force -Path $NamesDir | Out-Null
        $NamesInstalled = Join-Path $NamesDir (Split-Path -Leaf $NameDictionary)
        Copy-InstallFile -Source $NameDictionary -Destination $NamesInstalled | Out-Null
    }
} catch {
    Write-Log ERROR "copy failed (mixed input): $_"; exit 50
}

# Files unpacked from a downloaded zip carry its mark of the web; the
# settings GUI started from the language bar would get SmartScreen's prompt.
Get-ChildItem -LiteralPath $InstallDir -Recurse -File | Unblock-File -ErrorAction SilentlyContinue

# ---- config.toml ----
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$ConfigPath = Join-Path $ConfigDir 'config.toml'
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    $ModelConfigDir = (Join-Path $InstallDir 'models\onnx').Replace('\','/').Replace('"','\"')
    $ModelConfigFile = (Join-Path $ModelConfigDir $ManifestItem.Name).Replace('\','/').Replace('"','\"')
    $EngineConfigExe = $EngineInstalled.Replace('\','/').Replace('"','\"')
    $OrtConfigDll = $OrtInstalled.Replace('\','/').Replace('"','\"')
    $DefaultConfig = @"
# Setsuna TSF IME configuration. Edit and save; activations after the save
# pick up new values (Win+Space twice).
# Missing fields fall back to built-in defaults.

[engine]
# Model artifact manifest. Point this at another manifest to switch models.
# Decode settings (beam width, top_k, symbol split, ...) come from the
# manifest's decode_defaults; set a knob here only to override it.
model_manifest = "$ModelConfigFile"
engine_exe = "$EngineConfigExe"
# Official ONNX Runtime the engine loads.
ort_dylib = "$OrtConfigDll"

[ui.candidate_window]
font_face = "Meiryo UI"
font_pt = 16
page_size = 10
highlight_color = "#D77800"
highlight_text_color = "#FFFFFF"
bg_color = "#FFFFFF"
border_color = "#808080"
footer_color = "#707070"

[frontend]
live_min_dispatch_chars = 3
force_convert_on_space = true

[frontend.symbols]
comma = "、"
period = "。"
slash = "／"
question = "？"
left_bracket = "「"
right_bracket = "」"
yen = "￥"
shift_8 = "（"
shift_9 = "）"
"@
    Write-Utf8Lines -Path $ConfigPath -Lines ($DefaultConfig -split "`r?`n")
    if ($RerankInstalled -or $RewriteInstalled -or $MixedInstalled -or $NamesInstalled) {
        Update-ExistingConfig -ConfigPath $ConfigPath -ManifestPath $InstalledManifest -EnginePath $EngineInstalled -OrtPath $OrtInstalled -RerankManifestPath $RerankManifestInstalled -RerankExePath $RerankInstalled -RewriteExePath $RewriteInstalled -MixedModelPath $MixedInstalled -NameDictionaryPath $NamesInstalled
    }
    Write-Log INFO "wrote default config: $ConfigPath"
} else {
    $StableDll = Join-Path $InstallDir $DllName
    if ($NoElevate -and ($DllInstalled -ne $StableDll)) {
        Write-Log WARN "NoElevate with fallback DLL: config preserved because registered DLL was not updated"
    } else {
        try {
            Update-ExistingConfig -ConfigPath $ConfigPath -ManifestPath $InstalledManifest -EnginePath $EngineInstalled -OrtPath $OrtInstalled -RerankManifestPath $RerankManifestInstalled -RerankExePath $RerankInstalled -RewriteExePath $RewriteInstalled -MixedModelPath $MixedInstalled -NameDictionaryPath $NamesInstalled
            Write-Log INFO "updated engine config: $ConfigPath"
        } catch {
            Write-Log WARN "config exists but engine update failed (preserved): $ConfigPath : $_"
        }
    }
}

# ---- regsvr32 ----
if (-not $NoElevate) {
    Write-Log INFO "regsvr32 /s $DllInstalled"
    $p = Start-Process regsvr32.exe -ArgumentList '/s', "`"$DllInstalled`"" -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) {
        Write-Log ERROR "regsvr32 failed with exit code $($p.ExitCode)"; exit 20
    }
    Write-Log INFO 'regsvr32 OK'
} else {
    Write-Log WARN 'NoElevate: skipped regsvr32 (DLL not activated)'
}

# ---- Apps & Features entry (HKCU) ----
New-Item -Path $UninstReg -Force | Out-Null
Set-ItemProperty $UninstReg DisplayName     'Setsuna Japanese IME'
Set-ItemProperty $UninstReg DisplayVersion  (Get-SetsunaVersion)
Set-ItemProperty $UninstReg Publisher       'Setsuna'
Set-ItemProperty $UninstReg InstallLocation $InstallDir
Set-ItemProperty $UninstReg InstallDate     (Get-Date -Format 'yyyyMMdd')
# Windows PowerShell: always there, whichever PowerShell installed this.
$WinPs = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
Set-ItemProperty $UninstReg UninstallString "`"$WinPs`" -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $InstallDir 'uninstall.ps1')`""
Set-ItemProperty $UninstReg NoModify        1 -Type DWord
Set-ItemProperty $UninstReg NoRepair        1 -Type DWord
Write-Log INFO "registered HKCU Uninstall entry: $UninstReg"

Write-Log INFO "install complete at $InstallDir"
Write-Host ""
Write-Host "Setsuna IME installed."
Write-Host "Switch input source via Win+Space -> 'Setsuna Japanese IME'."
Write-Host "Uninstall: 'Apps & Features' > 'Setsuna Japanese IME' OR powershell -ExecutionPolicy Bypass -File $(Join-Path $InstallDir 'uninstall.ps1')"
exit 0
