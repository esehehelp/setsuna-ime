# Build the Android app: the Rust half and the engine for arm64, then the APK.
#
# Usage:
#   pwsh -File crates\android\build.ps1             # debug APK
#   pwsh -File crates\android\build.ps1 -Release    # release APK (signed with the debug key)
#   pwsh -File crates\android\build.ps1 -Install    # and install it on the phone adb sees
#
# Needs: the Android SDK (ANDROID_HOME, else %LOCALAPPDATA%\Android\Sdk) with
# platform 35 and an NDK under ndk\ (ANDROID_NDK_HOME picks another), a JDK
# 17 or later (JAVA_HOME), `rustup target add aarch64-linux-android` and
# `cargo install cargo-ndk`.
#
# The APK's native libraries (app\build\rust\jniLibs\arm64-v8a):
#   libsetsuna_android.so  the Rust half (crates/android), loaded by the service
#   libsetsuna_engine.so   the engine binary (crates/engine), run as the daemon:
#                          the system extracts a library executable, another file not
#   libonnxruntime.so      from Microsoft's onnxruntime-android (Gradle)
#
# The models are not in the APK: push-models.ps1 copies them onto the phone.

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Release,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$App = Join-Path $PSScriptRoot 'app'

if (-not $env:ANDROID_HOME) { $env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
if (-not (Test-Path -LiteralPath $env:ANDROID_HOME)) { throw "Android SDK not found: $env:ANDROID_HOME (set ANDROID_HOME)" }
if (-not $env:ANDROID_NDK_HOME) {
    $ndk = Get-ChildItem -LiteralPath (Join-Path $env:ANDROID_HOME 'ndk') -Directory -ErrorAction SilentlyContinue |
        Sort-Object { [version]$_.Name } | Select-Object -Last 1
    if (-not $ndk) { throw "no NDK in $env:ANDROID_HOME\ndk (sdkmanager 'ndk;27.2.12479018')" }
    $env:ANDROID_NDK_HOME = $ndk.FullName
}

# the libraries as small as they go: the APK carries them
$env:CARGO_PROFILE_RELEASE_STRIP = 'symbols'
Push-Location $Repo
try {
    & cargo ndk -t arm64-v8a -P 26 build --release -p setsuna-android -p setsuna-engine
    if ($LASTEXITCODE -ne 0) { throw "cargo ndk failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
    Remove-Item Env:\CARGO_PROFILE_RELEASE_STRIP
}

$Out = Join-Path $Repo 'target\aarch64-linux-android\release'
$Jni = Join-Path $App 'build\rust\jniLibs\arm64-v8a'
New-Item -ItemType Directory -Force -Path $Jni | Out-Null
Copy-Item -LiteralPath (Join-Path $Out 'libsetsuna_android.so') -Destination $Jni -Force
Copy-Item -LiteralPath (Join-Path $Out 'setsuna-engine') -Destination (Join-Path $Jni 'libsetsuna_engine.so') -Force

$Task = if ($Release) { 'assembleRelease' } else { 'assembleDebug' }
Push-Location $App
try {
    & .\gradlew.bat --console=plain $Task
    if ($LASTEXITCODE -ne 0) { throw "gradle $Task failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

$Variant = if ($Release) { 'release' } else { 'debug' }
$Apk = Get-ChildItem -LiteralPath (Join-Path $App "build\outputs\apk\$Variant") -Filter '*.apk' |
    Sort-Object LastWriteTime | Select-Object -Last 1
Write-Host "APK: $($Apk.FullName) ($([math]::Round($Apk.Length / 1MB, 1)) MB)"

if ($Install) {
    $Adb = Join-Path $env:ANDROID_HOME 'platform-tools\adb.exe'
    & $Adb install -r $Apk.FullName
    if ($LASTEXITCODE -ne 0) { throw "adb install failed ($LASTEXITCODE)" }
}
