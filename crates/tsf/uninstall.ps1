# Setsuna TSF IME uninstaller.
#
# Usage:
#   pwsh -File uninstall.ps1                # interactive; asks before deleting config
#   pwsh -File uninstall.ps1 -Yes           # non-interactive (CI / Apps & Features)
#   pwsh -File uninstall.ps1 -KeepConfig    # preserve user config dir
#   pwsh -File uninstall.ps1 -NoElevate     # skip regsvr32 /u (CI)
#
# Idempotent: each step is wrapped in Test-Path; running a second time on a
# clean machine exits 0.

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$KeepConfig,
    [switch]$NoElevate
)

# Don't abort on first failure — partial uninstall must keep converging.
$ErrorActionPreference = 'Continue'

$InstallDir = Join-Path $env:LOCALAPPDATA 'setsuna-ime'
$ConfigDir  = Join-Path $env:APPDATA      'setsuna-ime'
$DllName    = 'setsuna_tsf.dll'
$DllPath    = Join-Path $InstallDir $DllName
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

function Get-RegisteredDllPath {
    try {
        $registered = (Get-ItemProperty -LiteralPath $ClsidReg -ErrorAction Stop).'(default)'
        if ($registered -and (Test-Path -LiteralPath $registered)) {
            return $registered
        }
    } catch { }
    return $DllPath
}

Write-Log INFO "uninstall starting (Yes=$Yes, KeepConfig=$KeepConfig, NoElevate=$NoElevate)"

$RegisteredDllPath = Get-RegisteredDllPath

# 1. Elevate for regsvr32 /u (DLL still present + caller not admin + elevation allowed).
if ((Test-Path -LiteralPath $RegisteredDllPath) -and -not (Test-IsAdmin) -and -not $NoElevate) {
    Write-Log INFO 'elevating for regsvr32 /u'
    $argv = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$PSCommandPath)
    if ($Yes)        { $argv += '-Yes' }
    if ($KeepConfig) { $argv += '-KeepConfig' }
    try {
        $p = Start-Process pwsh -ArgumentList $argv -Verb RunAs -Wait -PassThru
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
    Where-Object { $_.Path -and $_.Path.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase) } |
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
    }
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

Write-Log INFO 'uninstall complete'
Write-Host ''
Write-Host 'Setsuna IME uninstalled.'
exit 0
