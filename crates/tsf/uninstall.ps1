# Setsuna TSF IME uninstaller.
#
# Usage:
#   pwsh -File uninstall.ps1                # interactive; asks before deleting config
#   pwsh -File uninstall.ps1 -Yes           # non-interactive (CI / Apps & Features)
#   pwsh -File uninstall.ps1 -KeepConfig    # preserve user config dir
#   pwsh -File uninstall.ps1 -NoElevate     # skip regsvr32 /u (CI)
#   pwsh -File uninstall.ps1 -InstallDir D:\setsuna-ime   # install.ps1's -InstallDir
#
# Idempotent: each step is wrapped in Test-Path; running a second time on a
# clean machine exits 0.

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$KeepConfig,
    [switch]$NoElevate,
    # where install.ps1 put the code and the models (its -InstallDir); by
    # default the directory of this copy when it is the installed one (beside
    # installed.toml), else %ProgramFiles%\setsuna-ime
    [string]$InstallDir
)

# Don't abort on first failure — partial uninstall must keep converging.
$ErrorActionPreference = 'Continue'

if (-not $InstallDir) {
    $InstallDir = if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'installed.toml')) {
        $PSScriptRoot
    } else {
        Join-Path $env:ProgramFiles 'setsuna-ime'
    }
}
# a full path with no separator at its end (the quoted -InstallDir passed
# on would end in \")
$InstallDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InstallDir)
if ($InstallDir.Length -gt 3) { $InstallDir = $InstallDir.TrimEnd('\', '/') }
# installs before the code moved to Program Files, and the install log and cache
$UserDir    = Join-Path $env:LOCALAPPDATA 'setsuna-ime'
$ConfigDir  = Join-Path $env:APPDATA      'setsuna-ime'
# what regsvr32 /u is run on when the registry names nothing there: the shim
# registered as the IME, else the IME DLL itself (an install before the shim)
$DllPath    = Join-Path $InstallDir 'setsuna_tsf_shim.dll'
foreach ($p in (Join-Path $InstallDir 'setsuna_tsf.dll'), (Join-Path $UserDir 'setsuna_tsf_shim.dll'), (Join-Path $UserDir 'setsuna_tsf.dll')) {
    if (-not (Test-Path -LiteralPath $DllPath)) { $DllPath = $p }
}
$ClsidReg   = 'Registry::HKEY_CLASSES_ROOT\CLSID\{C3F4DFA2-6E28-4AA0-9DAA-407B575EF52D}\InProcServer32'
$UninstReg  = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\setsuna-ime'
$LogFile    = Join-Path $env:LOCALAPPDATA 'setsuna-ime\uninstall.log'

function Write-Log {
    param([string]$Level, [string]$Msg)
    $line = "$(Get-Date -Format o) [$Level] $Msg"
    Write-Host $line
    try { Add-Content -Path $LogFile -Value $line -ErrorAction Stop } catch { }
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pri = [Security.Principal.WindowsPrincipal] $id
    $pri.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# What install.ps1 granted AppContainers (ALL APPLICATION PACKAGES, ALL
# RESTRICTED APPLICATION PACKAGES) on the install dir and config.toml.
function Revoke-AppContainerRead {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    try {
        & icacls $Path /remove:g '*S-1-15-2-1' '*S-1-15-2-2' 2>&1 | Out-Null
        Write-Log INFO "AppContainers may no longer read $Path"
    } catch {
        Write-Log WARN "icacls could not take AppContainers' read of $Path back: $_"
    }
}

function Get-RegisteredDllPath {
    try {
        $registered = (Get-ItemProperty -LiteralPath $ClsidReg -ErrorAction Stop).'(default)'
        if ($registered -and (Test-Path -LiteralPath $registered)) {
            return $registered
        }
    } catch { }
    return $DllPath
}

Write-Log INFO "uninstall starting (Yes=$Yes, KeepConfig=$KeepConfig, NoElevate=$NoElevate, InstallDir=$InstallDir)"

$RegisteredDllPath = Get-RegisteredDllPath

# 1. Elevate for regsvr32 /u and Program Files (something still there + caller
# not admin + elevation allowed).
if (((Test-Path -LiteralPath $RegisteredDllPath) -or (Test-Path -LiteralPath $InstallDir)) -and
    -not (Test-IsAdmin) -and -not $NoElevate) {
    Write-Log INFO 'elevating for regsvr32 /u'
    $argv = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$PSCommandPath)
    if ($Yes)        { $argv += '-Yes' }
    if ($KeepConfig) { $argv += '-KeepConfig' }
    # the directory found here, whoever the elevated one runs as
    $argv += @('-InstallDir', ("`"$InstallDir`""))
    try {
        # on the PowerShell running this: pwsh may not be installed
        $p = Start-Process (Get-Process -Id $PID).Path -ArgumentList $argv -Verb RunAs -Wait -PassThru
        Write-Log INFO "elevated uninstall exit code: $($p.ExitCode)"
        exit $p.ExitCode
    } catch {
        Write-Log ERROR "elevation declined: $_"
        exit 1
    }
}

# 2. Unregister DLL.
if ((Test-Path -LiteralPath $RegisteredDllPath) -and -not $NoElevate) {
    Write-Log INFO "regsvr32 /u /s $RegisteredDllPath"
    $p = Start-Process regsvr32.exe -ArgumentList '/u','/s', "`"$RegisteredDllPath`"" -Wait -PassThru -NoNewWindow
    Write-Log INFO "regsvr32 exit: $($p.ExitCode)"
}

# 3. Stop any leftover daemon process: the stable engine and rewriter and
# their content-addressed fallbacks (`setsuna-engine_<hash>.exe`), and the
# reranker, only those running from the install dir.
Get-Process -Name 'setsuna-engine*', 'setsuna-rewrite*', 'setsuna-rerank' -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and ($_.Path.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase) -or
                                 $_.Path.StartsWith($UserDir, [StringComparison]::OrdinalIgnoreCase)) } |
    ForEach-Object {
    Write-Log INFO "stopping daemon pid=$($_.Id)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

# 4. Remove install dir (ctfmon may briefly hold the DLL — retry).
if (Test-Path -LiteralPath $InstallDir) {
    $removed = $false
    for ($i = 0; $i -lt 10; $i++) {
        try {
            Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction Stop
            $removed = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if ($removed) {
        Write-Log INFO "removed install dir: $InstallDir"
    } else {
        Write-Log WARN "could not remove install dir after retries: $InstallDir"
        Revoke-AppContainerRead -Path $InstallDir
    }
}

# 4b. What installs before the move to Program Files left in the user's
# directory (the install log stays until the end).
if ((Test-Path -LiteralPath $UserDir) -and ($UserDir -ne $InstallDir)) {
    Get-ChildItem -LiteralPath $UserDir -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '^(install|uninstall)\.log$' } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    Revoke-AppContainerRead -Path $UserDir
}

# 5. Remove HKCU Uninstall entry.
if (Test-Path -LiteralPath $UninstReg) {
    Remove-Item -LiteralPath $UninstReg -Recurse -Force -ErrorAction SilentlyContinue
    Write-Log INFO "removed registry entry: $UninstReg"
}

# 6. Optionally remove user config (keep by default).
if ($KeepConfig) {
    Write-Log INFO "keeping user config: $ConfigDir"
} elseif (Test-Path -LiteralPath $ConfigDir) {
    $doDelete = $Yes
    if (-not $doDelete) {
        try {
            $r = Read-Host "Delete user config at $ConfigDir? [y/N]"
            $doDelete = $r -match '^[Yy]'
        } catch {
            # Non-interactive shell: skip deletion.
            Write-Log WARN 'non-interactive prompt failed; keeping config'
        }
    }
    if ($doDelete) {
        Remove-Item -LiteralPath $ConfigDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Log INFO "removed config dir: $ConfigDir"
    } else {
        Write-Log INFO "config preserved (user opt-out)"
    }
}
# a kept config.toml is closed to AppContainers again
Revoke-AppContainerRead -Path (Join-Path $ConfigDir 'config.toml')

Write-Log INFO 'uninstall complete'
Write-Host ''
Write-Host 'Setsuna IME uninstalled.'
exit 0
