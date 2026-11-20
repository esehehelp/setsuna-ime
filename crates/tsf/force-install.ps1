# Setsuna TSF IME force installer for local development.
#
# This is intentionally stronger than install.ps1:
#   - self-elevates,
#   - stops TSF/input hosts that keep the DLL loaded,
#   - unregisters the currently registered COM server when present,
#   - installs the current release build with the given model manifest,
#   - points %APPDATA%\setsuna-ime\config.toml at the manifest (decode settings
#     come from the manifest's decode_defaults),
#   - restarts ctfmon and runs a backend smoke test.
#
# Usage:
#   pwsh -File crates\tsf\force-install.ps1 -Manifest <m>
#   pwsh -File crates\tsf\force-install.ps1 -Manifest <m> -SkipBuild
#   pwsh -File crates\tsf\force-install.ps1 -Manifest <m> -PinActiveProfile
#   pwsh -File crates\tsf\force-install.ps1 -Manifest <m> -RerankManifest <r>   # with the reranker

#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Manifest,
    [string]$RerankManifest,
    [switch]$SkipBuild,
    [switch]$KeepHostsRunning,
    [switch]$NoSmoke,
    [switch]$PinActiveProfile
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
# where install.ps1 puts the code and the models; its log stays the user's
$InstallDir = Join-Path $env:ProgramFiles 'setsuna-ime'
$UserDir = Join-Path $env:LOCALAPPDATA 'setsuna-ime'
$ConfigPath = Join-Path $env:APPDATA 'setsuna-ime\config.toml'
$DllName = 'setsuna_tsf.dll'
$TextServiceClsid = '{C3F4DFA2-6E28-4AA0-9DAA-407B575EF52D}'
$ProfileGuid = '{ACEB29C0-964E-4FF1-8BDC-B507AAB78873}'
$KeyboardTipCategory = '{34745C63-B2F0-4784-8B67-5E12C8701A31}'
# the machine's registration, which the elevated regsvr32 /u is run on:
# HKEY_CLASSES_ROOT puts the user's own HKCU\Software\Classes over it
$ClsidReg = 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{C3F4DFA2-6E28-4AA0-9DAA-407B575EF52D}\InProcServer32'
$LogFile = Join-Path $UserDir 'force-install.log'

function Write-Log {
    param([string]$Level, [string]$Msg)
    New-Item -ItemType Directory -Force -Path $UserDir | Out-Null
    $line = "$(Get-Date -Format o) [$Level] $Msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction Stop } catch { }
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pri = [Security.Principal.WindowsPrincipal] $id
    $pri.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# One argument for Start-Process -ArgumentList, which joins them with spaces
# as they are: quoted, and a trailing backslash doubled so that it does not
# escape the closing quote (a path holds no quote).
function Format-Argument([string]$Value) {
    '"' + ($Value -replace '(\\+)$', '$1$1') + '"'
}

function Restart-AsAdmin {
    $argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Format-Argument $PSCommandPath))
    $argv += @('-Manifest', (Format-Argument $Manifest))
    if ($RerankManifest) { $argv += @('-RerankManifest', (Format-Argument $RerankManifest)) }
    # built already, as the user (Invoke-Build)
    $argv += '-SkipBuild'
    if ($KeepHostsRunning) { $argv += '-KeepHostsRunning' }
    if ($NoSmoke) { $argv += '-NoSmoke' }
    if ($PinActiveProfile) { $argv += '-PinActiveProfile' }
    # on the PowerShell running this: pwsh may not be installed
    $p = Start-Process (Get-Process -Id $PID).Path -ArgumentList $argv -Verb RunAs -Wait -PassThru
    exit $p.ExitCode
}

# The release build of what install.ps1 installs: the IME DLL, the shim and
# setsuna-tsf-check, the daemons and the settings GUI.
function Invoke-Build {
    $packages = @('-p', 'setsuna-tsf', '-p', 'setsuna-tsf-shim', '-p', 'setsuna-engine', '-p', 'setsuna-settings',
                  '-p', 'setsuna-rewrite')
    if ($RerankManifest) {
        $packages += @('-p', 'setsuna-rerank')
        # bindgen (llama.cpp's bindings) needs libclang
        if (-not $env:LIBCLANG_PATH) {
            $clang = Get-Command clang -ErrorAction SilentlyContinue
            if ($clang) { $env:LIBCLANG_PATH = Split-Path -Parent $clang.Source }
        }
    }
    Write-Log INFO "cargo build $($packages -join ' ') --release"
    Push-Location $RepoRoot
    try {
        cargo build @packages --release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}

function Stop-InputHosts {
    param([string]$Reason = 'reload')
    foreach ($name in @('setsuna-engine', 'setsuna-rerank', 'setsuna-rewrite', 'ctfmon', 'TextInputHost')) {
        $procs = @(Get-Process -Name $name -ErrorAction SilentlyContinue)
        foreach ($p in $procs) {
            try {
                Write-Log INFO "stopping $($p.ProcessName) pid=$($p.Id) ($Reason)"
                Stop-Process -Id $p.Id -Force -ErrorAction Stop
            } catch {
                Write-Log WARN "could not stop $($p.ProcessName) pid=$($p.Id): $_"
            }
        }
    }
    Start-Sleep -Milliseconds 800
}

function Get-RegisteredDllPath {
    try {
        $registered = (Get-ItemProperty -LiteralPath $ClsidReg -ErrorAction Stop).'(default)'
        if ($registered -and (Test-Path -LiteralPath $registered)) {
            return $registered
        }
    } catch { }
    return Join-Path $InstallDir $DllName
}

function Unregister-CurrentDll {
    $registered = Get-RegisteredDllPath
    if (-not (Test-Path -LiteralPath $registered)) {
        Write-Log WARN "registered DLL not found, skipping unregister: $registered"
        return
    }
    Write-Log INFO "regsvr32 /u /s $registered"
    $p = Start-Process regsvr32.exe -ArgumentList '/u', '/s', "`"$registered`"" -Wait -PassThru -NoNewWindow
    Write-Log INFO "regsvr32 /u exit code: $($p.ExitCode)"
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

function Quote-TomlPath {
    param([string]$Path)
    '"' + ($Path.Replace('\', '/').Replace('"', '\"')) + '"'
}

# The manifest installed, as config.toml's model. engine_exe stays as
# install.ps1 wrote it: the engine it installed under a content-addressed name
# when a running one held setsuna-engine.exe (set here to the plain name, the
# IME ran the old engine).
function Pin-BestConfig {
    $manifestDst = Join-Path (Join-Path $InstallDir 'models\onnx') (Split-Path -Leaf $Manifest)

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ConfigPath) | Out-Null
    if (Test-Path -LiteralPath $ConfigPath) {
        $lines = @(Get-Content -LiteralPath $ConfigPath -Encoding UTF8)
    } else {
        $lines = @('[engine]')
    }

    $lines = Remove-ConfigValue -Lines $lines -Section 'engine' -Key 'model_dir'
    $lines = Set-OrAppendConfigValue -Lines $lines -Section 'engine' -Key 'model_manifest' -Value (Quote-TomlPath $manifestDst)

    # UTF-8 without a BOM whichever PowerShell writes it (Windows PowerShell's -Encoding utf8 adds one)
    [IO.File]::WriteAllLines($ConfigPath, [string[]]$lines, (New-Object Text.UTF8Encoding $false))
    Write-Log INFO "pinned best config: $ConfigPath"
}

function Pin-ActiveProfile {
    $assemblyKey = "HKCU:\Software\Microsoft\CTF\Assemblies\0x00000411\$KeyboardTipCategory"
    New-Item -Path $assemblyKey -Force | Out-Null
    Set-ItemProperty -Path $assemblyKey -Name 'Default' -Value $TextServiceClsid
    Set-ItemProperty -Path $assemblyKey -Name 'Profile' -Value $ProfileGuid
    Set-ItemProperty -Path $assemblyKey -Name 'KeyboardLayout' -Value 0 -Type DWord

    $tipKey = "HKCU:\Software\Microsoft\CTF\TIP\$TextServiceClsid\LanguageProfile\0x00000411\$ProfileGuid"
    New-Item -Path $tipKey -Force | Out-Null
    Set-ItemProperty -Path $tipKey -Name 'Enable' -Value 1 -Type DWord

    Write-Log INFO "pinned active CTF profile: clsid=$TextServiceClsid profile=$ProfileGuid"
}

function Clear-SetsunaProfilePin {
    $assemblyKey = "HKCU:\Software\Microsoft\CTF\Assemblies\0x00000411\$KeyboardTipCategory"
    if (-not (Test-Path -LiteralPath $assemblyKey)) {
        return
    }
    $values = Get-ItemProperty -LiteralPath $assemblyKey
    if ($values.Default -ne $TextServiceClsid -and $values.Profile -ne $ProfileGuid) {
        return
    }

    $backup = Join-Path $UserDir 'ctf-assembly-keyboard-backup.txt'
    @(
        "time=$(Get-Date -Format o)"
        "key=$assemblyKey"
        "Default=$($values.Default)"
        "Profile=$($values.Profile)"
        "KeyboardLayout=$($values.KeyboardLayout)"
    ) | Set-Content -LiteralPath $backup -Encoding utf8

    foreach ($name in @('Default', 'Profile', 'KeyboardLayout')) {
        try {
            Remove-ItemProperty -LiteralPath $assemblyKey -Name $name -ErrorAction Stop
        } catch { }
    }
    Write-Log INFO "cleared Setsuna CTF keyboard profile pin (backup: $backup)"
}

function Invoke-Smoke {
    $engineExe = Join-Path $InstallDir 'setsuna-engine.exe'
    $manifestDst = Join-Path (Join-Path $InstallDir 'models\onnx') (Split-Path -Leaf $Manifest)

    # Decode settings come from the manifest, as they do for the IME.
    $args = @('convert', '--manifest', $manifestDst, '--reading', 'こんにちは')

    Write-Log INFO 'running backend smoke convert'
    $output = & $engineExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "backend smoke failed with exit code $LASTEXITCODE"
    }
    Write-Log INFO "smoke output: $($output | Select-Object -First 1)"
}

function Report-LoadedTsfModules {
    $registered = Get-RegisteredDllPath
    $hits = @()
    foreach ($p in Get-Process) {
        try {
            foreach ($m in $p.Modules) {
                if ($m.ModuleName -like 'setsuna_tsf*') {
                    $hits += [pscustomobject]@{
                        Process = $p.ProcessName
                        Id = $p.Id
                        Module = $m.ModuleName
                        File = $m.FileName
                        Registered = ($m.FileName -eq $registered)
                    }
                }
            }
        } catch { }
    }

    foreach ($hit in $hits) {
        $state = if ($hit.Registered) { 'registered' } else { 'stale' }
        Write-Log WARN "loaded TSF DLL ($state): $($hit.Process) pid=$($hit.Id) $($hit.File)"
    }
    if ($hits | Where-Object { -not $_.Registered }) {
        Write-Log WARN 'Close the stale processes above, or sign out/in, before testing Setsuna in those apps.'
    }
}

if (-not (Test-IsAdmin)) {
    # the build as the user, before the elevation: cargo runs the crates'
    # build scripts, which an elevated build ran as an administrator
    if (-not $SkipBuild) { Invoke-Build }
    Restart-AsAdmin
}

Write-Log INFO "force install starting (RepoRoot=$RepoRoot, Manifest=$Manifest, RerankManifest=$RerankManifest, SkipBuild=$SkipBuild, KeepHostsRunning=$KeepHostsRunning, PinActiveProfile=$PinActiveProfile)"

if (-not (Test-Path -LiteralPath $Manifest)) {
    Write-Log ERROR "manifest not found: $Manifest"
    exit 30
}

# started elevated (an administrator's shell): built here
if (-not $SkipBuild) {
    Invoke-Build
}

if (-not $KeepHostsRunning) {
    Stop-InputHosts -Reason 'before unregister'
}

Unregister-CurrentDll

$install = Join-Path $PSScriptRoot 'install.ps1'
Write-Log INFO "running install.ps1 with manifest $Manifest"
$installArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $install, '-Force', '-Manifest', $Manifest)
if ($RerankManifest) { $installArgs += @('-RerankManifest', $RerankManifest) }
& (Get-Process -Id $PID).Path @installArgs
if ($LASTEXITCODE -ne 0) {
    Write-Log ERROR "install.ps1 failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Pin-BestConfig
if ($PinActiveProfile) {
    Pin-ActiveProfile
} else {
    Clear-SetsunaProfilePin
    Write-Log INFO 'skipped direct CTF active-profile pin; use -PinActiveProfile to force current Japanese keyboard profile'
}

if (-not $NoSmoke) {
    Invoke-Smoke
}

if (-not $KeepHostsRunning) {
    Stop-InputHosts -Reason 'after registration'
    Start-Sleep -Milliseconds 1200
    Write-Log INFO 'starting ctfmon'
    Start-Process ctfmon.exe | Out-Null
}

Report-LoadedTsfModules

Write-Log INFO 'force install complete'
Write-Host ''
Write-Host 'Force install complete. Switch away from and back to Setsuna with Win+Space.'
Write-Host 'If MS-IME is still broken, sign out and sign in once so TSF reloads all input processors.'
