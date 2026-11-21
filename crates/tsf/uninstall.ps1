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
# regsvr32 /u is run on the DLL the machine's registration names, and on
# nothing else: it runs elevated and loads what it is given, and -InstallDir
# is the caller's to give (a DLL there need not be Setsuna's). Every install,
# before the shim too, registered for the machine.
# the machine's registration: HKEY_CLASSES_ROOT puts the user's own
# HKCU\Software\Classes, which the user may write, over it
$ClsidReg   = 'Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Classes\CLSID\{C3F4DFA2-6E28-4AA0-9DAA-407B575EF52D}\InProcServer32'
# Apps & Features' entry: the machine's (install.ps1's), and the user's of
# the installs before
$UninstRegs = @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\setsuna-ime',
                'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\setsuna-ime')
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

# One argument for Start-Process -ArgumentList, which joins them with spaces
# as they are: quoted, and a trailing backslash doubled so that it does not
# escape the closing quote (a path holds no quote).
function Format-Argument([string]$Value) {
    '"' + ($Value -replace '(\\+)$', '$1$1') + '"'
}

function Test-ReparsePoint([System.IO.FileSystemInfo]$Item) {
    ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
}

# $Path and what it holds gone, as this one deletes them elevated: a junction
# or a link inside goes itself, never what it points at (Windows PowerShell's
# Remove-Item -Recurse went through a junction and deleted where it led).
# What cannot go (a DLL a running app still has loaded) is passed over and
# the rest goes; the paths left are written out, each before its directory.
function Remove-Tree([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    try {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    } catch {
        return $Path
    }
    if ($item.PSIsContainer -and -not (Test-ReparsePoint $item)) {
        $children = @()
        try {
            $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop)
        } catch { }
        foreach ($child in $children) {
            Remove-Tree $child.FullName
        }
    }
    try {
        if ($item.PSIsContainer) {
            # a directory, or the link of one (removing it leaves its target)
            [IO.Directory]::Delete($item.FullName, $false)
        } else {
            $item.Attributes = $item.Attributes -band -bnot [IO.FileAttributes]::ReadOnly
            [IO.File]::Delete($item.FullName)
        }
    } catch {
        $item.FullName
    }
}

# $Path deleted when Windows next starts (MOVEFILE_DELAY_UNTIL_REBOOT, an
# administrator's): only under Program Files, where no user can put a link
# for the start to delete through.
function Remove-AtRestart([string]$Path) {
    $programFiles = [IO.Path]::Combine([Environment]::GetFolderPath('ProgramFiles'), 'x').TrimEnd('x')
    if (-not $Path.StartsWith($programFiles, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    if (-not ('Setsuna.Uninstall.Native' -as [type])) {
        Add-Type -Namespace Setsuna.Uninstall -Name Native -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool MoveFileEx(string existing, string replacement, int flags);
'@
    }
    return [Setsuna.Uninstall.Native]::MoveFileEx($Path, $null, 4)
}

# Whether $Dir is an install of install.ps1's to delete: a directory of its
# own (no drive's root, no link) that holds what an install leaves there.
# -InstallDir is the caller's to give (Apps & Features' entry of an install
# before was the user's to change), and this runs elevated.
function Test-InstallDir([string]$Dir) {
    try {
        $item = Get-Item -LiteralPath $Dir -Force -ErrorAction Stop
    } catch {
        return $false
    }
    if (-not $item.PSIsContainer -or (Test-ReparsePoint $item) -or -not $item.Parent) { return $false }
    $marks = @('installed.toml', 'core.txt', 'uninstall.ps1') |
        Where-Object { Test-Path -LiteralPath (Join-Path $Dir $_) -PathType Leaf }
    $dlls = @(Get-ChildItem -LiteralPath $Dir -Filter 'setsuna_tsf*.dll' -File -Force -ErrorAction SilentlyContinue)
    return (@($marks).Count -gt 0 -and $dlls.Count -gt 0)
}

# The DLL the machine's registration names, when it is there; else none.
function Get-RegisteredDllPath {
    try {
        $registered = (Get-ItemProperty -LiteralPath $ClsidReg -ErrorAction Stop).'(default)'
        if ($registered -and (Test-Path -LiteralPath $registered -PathType Leaf)) {
            return $registered
        }
    } catch { }
    return $null
}

Write-Log INFO "uninstall starting (Yes=$Yes, KeepConfig=$KeepConfig, NoElevate=$NoElevate, InstallDir=$InstallDir)"

$RegisteredDllPath = Get-RegisteredDllPath

# 1. Elevate for regsvr32 /u and Program Files (something still there + caller
# not admin + elevation allowed).
if (($RegisteredDllPath -or (Test-Path -LiteralPath $InstallDir)) -and
    -not (Test-IsAdmin) -and -not $NoElevate) {
    Write-Log INFO 'elevating for regsvr32 /u'
    $argv = @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Format-Argument $PSCommandPath))
    if ($Yes)        { $argv += '-Yes' }
    if ($KeepConfig) { $argv += '-KeepConfig' }
    # the directory found here, whoever the elevated one runs as
    $argv += @('-InstallDir', (Format-Argument $InstallDir))
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
if ($RegisteredDllPath -and -not $NoElevate) {
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

# 4. Remove install dir (ctfmon may briefly hold the DLL — retry), when it is
# one of install.ps1's.
if ((Test-Path -LiteralPath $InstallDir) -and -not (Test-InstallDir $InstallDir)) {
    Write-Log WARN "not an install of Setsuna's (no installed.toml, core.txt or uninstall.ps1 with its DLL, or a link or a drive's root): left as it is: $InstallDir"
} elseif (Test-Path -LiteralPath $InstallDir) {
    $left = @()
    for ($i = 0; $i -lt 10; $i++) {
        $left = @(Remove-Tree $InstallDir)
        if ($left.Count -eq 0) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($left.Count -eq 0) {
        Write-Log INFO "removed install dir: $InstallDir"
    } else {
        # the DLLs apps still have loaded (until they exit): gone at the next restart
        $later = @($left | Where-Object { Remove-AtRestart $_ })
        Write-Log WARN "$($left.Count) in use in $InstallDir, $($later.Count) of them to be deleted at the next restart: $($left -join ', ')"
        Revoke-AppContainerRead -Path $InstallDir
    }
}

# 4b. What installs before the move to Program Files left in the user's
# directory (the install log stays until the end); not when the directory is
# a link (its listing would be where the link leads, deleted elevated).
$userDirItem = Get-Item -LiteralPath $UserDir -Force -ErrorAction SilentlyContinue
if ($userDirItem -and (Test-ReparsePoint $userDirItem)) {
    Write-Log WARN "$UserDir is a link: what it holds is left as it is"
} elseif ((Test-Path -LiteralPath $UserDir) -and ($UserDir -ne $InstallDir)) {
    Get-ChildItem -LiteralPath $UserDir -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '^(install|uninstall)\.log$' } |
        ForEach-Object {
            $null = @(Remove-Tree $_.FullName)
        }
    Revoke-AppContainerRead -Path $UserDir
}

# 5. Remove the Uninstall entries (the machine's needs an administrator).
foreach ($reg in $UninstRegs) {
    if (Test-Path -LiteralPath $reg) {
        Remove-Item -LiteralPath $reg -Recurse -Force -ErrorAction SilentlyContinue
        Write-Log INFO "removed registry entry: $reg"
    }
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
        $left = @(Remove-Tree $ConfigDir)
        if ($left.Count -gt 0) { Write-Log WARN "could not remove all of $ConfigDir`: $($left -join ', ')" }
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
