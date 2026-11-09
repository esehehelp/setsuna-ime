# Copy the models onto the phone adb sees, into the app's files directory,
# where it looks for them (files/models):
#   onnx/<manifest> and the files it lists
#   lexicon/<lexicon>
#   analyzer/<its directory's name>/analyzer.manifest.json and its files
#   names/latin-names.tsv (mixed input's names: HTCondor, GitHub kept as
#     typed, not read as romaji)
#
# Usage:
#   pwsh -File crates\android\push-models.ps1                  # those this PC's Windows install uses
#   pwsh -File crates\android\push-models.ps1 -Manifest <m> [-Lexicon <f>] [-Analyzer <a>] [-NameDictionary <tsv>]
#   pwsh -File crates\android\push-models.ps1 -NoDictionaries  # the model only
#
# Without -Manifest, the model, the lexicon, the analyzer and the names are those
# %APPDATA%\setsuna-ime\config.toml names. Each file goes to /data/local/tmp
# and from there into the app's directory by `run-as`, which only a
# debuggable build (build.ps1 without -Release) lets in; a file already there
# with the same size is not sent again, and what an earlier push left in the
# directories pushed into (another model) is removed. The app's private directory, not
# /sdcard/Android/data: a file adb puts there is the shell's, which the app
# cannot read.

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Manifest,
    [string]$Lexicon,
    [string]$Analyzer,
    # not $Names: PowerShell's names are case-blind, and $names below lists the model's files
    [string]$NameDictionary,
    [switch]$NoDictionaries
)

$ErrorActionPreference = 'Stop'
$Package = 'io.github.esehehelp.setsuna'
$Stage = '/data/local/tmp/setsuna-models'
if (-not $env:ANDROID_HOME) { $env:ANDROID_HOME = Join-Path $env:LOCALAPPDATA 'Android\Sdk' }
$Adb = Join-Path $env:ANDROID_HOME 'platform-tools\adb.exe'

# config.toml's value of `key` in `[section]`, or ''
function Get-ConfigValue {
    param([string]$Section, [string]$Key)
    $path = Join-Path $env:APPDATA 'setsuna-ime\config.toml'
    if (-not (Test-Path -LiteralPath $path)) { return '' }
    $in = $false
    foreach ($line in Get-Content -LiteralPath $path -Encoding UTF8) {
        if ($line -match '^\s*\[([^\]]+)\]') { $in = $Matches[1].Trim() -eq $Section; continue }
        if ($in -and $line -match "^\s*$Key\s*=\s*""([^""]*)""") { return $Matches[1] }
    }
    return ''
}

if (-not $Manifest) {
    $Manifest = Get-ConfigValue 'engine' 'model_manifest'
    if (-not $NoDictionaries) {
        if (-not $Lexicon) { $Lexicon = Get-ConfigValue 'lexicon' 'file' }
        if (-not $Analyzer) { $Analyzer = Get-ConfigValue 'engine' 'analyzer_manifest' }
        if (-not $NameDictionary) { $NameDictionary = Get-ConfigValue 'frontend' 'name_dictionary_file' }
    }
}
if (-not $Manifest -or -not (Test-Path -LiteralPath $Manifest)) { throw "model manifest not found: '$Manifest' (pass -Manifest)" }

# [local path, path under files/models]
$Files = New-Object System.Collections.Generic.List[object]
function Add-Relative {
    param([string]$Dir, [string[]]$Names, [string]$Under)
    foreach ($n in $Names) {
        if (-not $n) { continue }
        if ([IO.Path]::IsPathRooted($n) -or $n -match '(^|[\\/])\.\.([\\/]|$)') { throw "$n leaves its directory" }
        if ($n -match "['\s]") { throw "a name with a quote or a space: $n" }
        $src = Join-Path $Dir $n
        if (-not (Test-Path -LiteralPath $src)) { throw "not found: $src" }
        $Files.Add(@($src, "$Under/$($n -replace '\\', '/')"))
    }
}

$m = Get-Item -LiteralPath $Manifest
$j = Get-Content -LiteralPath $m.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
$names = @($m.Name, $j.proposal.path, $j.vocab, $j.input_vocab, $j.rescore.pairs, $j.rescore.words)
if ($j.proposal.external_data) { $names += @($j.proposal.external_data) }
if ($j.crf) { $names += $j.crf.path }
Add-Relative $m.Directory.FullName $names 'onnx'

if ($Lexicon) {
    $l = Get-Item -LiteralPath $Lexicon
    Add-Relative $l.Directory.FullName @($l.Name) 'lexicon'
}
if ($Analyzer) {
    $a = Get-Item -LiteralPath $Analyzer
    $aj = Get-Content -LiteralPath $a.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $names = @($a.Name, $aj.dictionary.path)
    if ($aj.alternates) { $names += @($aj.alternates) }
    Add-Relative $a.Directory.FullName $names "analyzer/$($a.Directory.Name)"
}

if ($NameDictionary) {
    # under the name the app looks for (crates/android/src/host.rs)
    if (-not (Test-Path -LiteralPath $NameDictionary)) { throw "not found: $NameDictionary" }
    $Files.Add(@((Get-Item -LiteralPath $NameDictionary).FullName, 'names/latin-names.tsv'))
}

$state = (& $Adb get-state 2>&1) -join ''
if ($state -ne 'device') { throw "no phone: adb get-state said '$state'" }
$probe = (& $Adb shell "run-as $Package id 2>&1") -join ''
if ($probe -notmatch '^uid=') { throw "run-as $Package failed: $probe (install a debug build: build.ps1 -Install)" }

& $Adb shell "mkdir -p $Stage" | Out-Null
foreach ($f in $Files) {
    $src, $rel = $f
    $size = (Get-Item -LiteralPath $src).Length
    $there = (& $Adb shell "run-as $Package stat -c %s files/models/$rel 2>/dev/null") -join ''
    if ($there -eq "$size") { Write-Host "same  $rel"; continue }
    Write-Host "push  $rel ($([math]::Round($size / 1MB, 1)) MB)"
    $staged = "$Stage/$($rel -replace '/', '_')"
    & $Adb push $src $staged | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "adb push $src failed ($LASTEXITCODE)" }
    $dir = $rel.Substring(0, $rel.LastIndexOf('/'))
    # the app reads the staged copy (world-readable, in a directory others may pass)
    & $Adb shell "chmod 644 $staged && run-as $Package sh -c 'mkdir -p files/models/$dir && cp $staged files/models/$rel.new && mv files/models/$rel.new files/models/$rel'"
    $there = (& $Adb shell "run-as $Package stat -c %s files/models/$rel 2>/dev/null") -join ''
    & $Adb shell "rm -f $staged" | Out-Null
    if ($there -ne "$size") { throw "copy of $rel into the app failed" }
}
& $Adb shell "rmdir $Stage 2>/dev/null" | Out-Null

# What an earlier push left in the directories pushed into goes: the app
# takes the only model, lexicon and analyzer there (another model's
# manifest beside this one's, and none starts), and the one before is
# kept for nothing (the model before, once the default changed).
$pushed = @{}
foreach ($f in $Files) { $pushed[$f[1]] = $true }
foreach ($top in @($Files | ForEach-Object { $_[1].Split('/')[0] } | Sort-Object -Unique)) {
    if ($top -eq 'names') { continue }
    # analyzer/<name>/...: the other analyzers' directories; elsewhere the files
    $depth = if ($top -eq 'analyzer') { 2 } else { 1 }
    $kept = @($pushed.Keys | Where-Object { $_.StartsWith("$top/") } | ForEach-Object {
        ($_.Split('/')[0..($depth - 1)]) -join '/'
    } | Sort-Object -Unique)
    $there = & $Adb shell "run-as $Package sh -c 'cd files/models/$top 2>/dev/null && ls -1'"
    foreach ($name in @($there | ForEach-Object { "$_".Trim() } | Where-Object { $_ })) {
        if ($name -match "['\s/]") { continue }
        if ($kept -contains "$top/$name") { continue }
        Write-Host "drop  $top/$name"
        & $Adb shell "run-as $Package rm -rf files/models/$top/$name" | Out-Null
    }
}
Write-Host "models in the app's files/models"
