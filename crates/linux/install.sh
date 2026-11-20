#!/usr/bin/env bash
# Setsuna IME installer for Linux (fcitx5). Run as root:
#
#   curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash
#   curl -fsSL .../install.sh | sudo bash -s -- --precision fp32
#
# Downloads the release archive this script was made for (crates/linux/package.sh
# fills in its SHA-256 below) and the conversion model, the reading lexicon and
# the analyzer pinned in it, checks them, installs them, then adds Setsuna to
# the input methods of the user who ran sudo (or pkexec) and restarts their
# fcitx5 so it loads the addon. A lexicon or analyzer that cannot be had is
# left out with a warning (the IME converts without it; run again to retry).
#
# Options:
#   --precision int8|fp32   the model (default int8)
#   --model <name>          another model the archive pins (Bletilla-v0.1-small)
#                           instead of its own
#   --archive <file>        a local release archive instead of the download
#   --lexicon <file>        a local reading lexicon (a compiled .lexicon) instead
#                           of the download: the dictionary's words come into
#                           the candidates
#   --analyzer <manifest>   a local analyzer (its manifest, the dictionary and
#                           alternates it names) instead of the download:
#                           segment editing cuts a conversion by its words
#   --no-dictionaries       download neither the lexicon nor the analyzer
#   --no-corrector          do not download the reading corrector (models.json's
#                           `corrector`: a reading with a slip converted as it
#                           reads it)
#   --rerank-manifest <m>   also the reranker's LM (an LM scorer manifest, the
#                           weights and tokenizer it names): candidates are reranked
#   --name-dictionary <tsv> also mixed input's Latin names dictionary
#   --no-register           install only; leave fcitx5 as it is
#   --root <dir>            install under <dir> (staging, tests): no root
#                           needed, implies --no-register
# A reranker model, names dictionary, lexicon, analyzer or reading corrector
# installed before stays when this install does not bring one
# (--no-dictionaries, --no-corrector, a failed download, the option not given
# again).
#
# Layout:
#   <fcitx5 addon dir>/setsuna.so
#   /usr/share/fcitx5/{addon,inputmethod,themes}/, /usr/share/icons/hicolor/scalable/apps/setsuna*.svg
#   /usr/lib/setsuna-ime/  the engine, reranker and rewriter daemons,
#                          llama.cpp's libraries, mixed/, models/onnx/, licenses/,
#                          uninstall.sh, installed-files
#   /usr/lib/setsuna-ime/models/rerank/<manifest> + its weights (and tokenizer)  (with --rerank-manifest)
#   /usr/lib/setsuna-ime/names/latin-names.tsv  (with --name-dictionary)
#   /usr/lib/setsuna-ime/lexicon/words.lexicon  (downloaded, or --lexicon)
#   /usr/lib/setsuna-ime/analyzer/<manifest> + its dictionary and alternates  (downloaded, or --analyzer)
#   /usr/lib/setsuna-ime/corrector/<manifest> + its weights and vocabulary  (downloaded)
# These are the addon's defaults where config.toml sets nothing.
#   /usr/bin/setsuna-settings  (when WebKitGTK 4.1 is there)
#   /var/cache/setsuna-ime/models  (downloads while installing: hard-linked into
#                                   place, then removed)
# The pinned files are kept once, where they are installed: an installed one
# that is the pinned file stays and is not downloaded again (offline too).
# /usr/lib/setsuna-ime is compiled into the addon, so no config.toml is needed.
# Uninstall: sudo /usr/lib/setsuna-ime/uninstall.sh
#
# Exit codes:
#   0  success
#   1  usage, or not run as root
#   10 unsupported architecture, or fcitx5 not installed
#   30 download failed or integrity check failed
#   50 install failed
set -euo pipefail

# Filled in by crates/linux/package.sh.
VERSION=@VERSION@
ARCHIVE_SHA256_X86_64=@ARCHIVE_SHA256_X86_64@
RELEASES=https://github.com/esehehelp/setsuna-ime/releases/download

precision=int8
model=
archive=
register=1
root=
rerank_manifest=
name_dictionary=
lexicon=
analyzer=
dictionaries=1
corrector=1

log() { echo "setsuna-install: $*" >&2; }
fail() {
    local code=$1
    shift
    log "error: $*"
    exit "$code"
}

while (($#)); do
    case $1 in
        --precision) precision=${2:?}; shift 2 ;;
        --model) model=${2:?}; shift 2 ;;
        --archive) archive=${2:?}; shift 2 ;;
        --rerank-manifest) rerank_manifest=${2:?}; shift 2 ;;
        --name-dictionary) name_dictionary=${2:?}; shift 2 ;;
        --lexicon) lexicon=${2:?}; shift 2 ;;
        --analyzer) analyzer=${2:?}; shift 2 ;;
        --no-dictionaries) dictionaries=0; shift ;;
        --no-corrector) corrector=0; shift ;;
        --no-register) register=0; shift ;;
        --root) root=${2:?}; register=0; shift 2 ;;
        -h | --help) sed -n '1d; /^#/!q; s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) fail 1 "unknown argument: $1" ;;
    esac
done
[[ $precision == int8 || $precision == fp32 ]] || fail 1 "--precision must be int8 or fp32"
[[ -z $model || $model =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail 1 "invalid --model: $model"
if [[ -n $root ]]; then
    mkdir -p "$root"
    root=$(cd "$root" && pwd)
elif [[ $EUID -ne 0 ]]; then
    fail 1 "run as root: curl -fsSL <install.sh URL> | sudo bash"
fi

# ---- pre-flight ----
case $(uname -m) in
    x86_64) archive_sha=$ARCHIVE_SHA256_X86_64 ;;
    *) fail 10 "no release build for $(uname -m) (x86_64 only)" ;;
esac
for tool in curl tar sha256sum; do
    command -v "$tool" >/dev/null || fail 1 "$tool not found"
done
# fcitx5's addon directory: where its own classic UI addon is
addon_dir=
for d in /usr/lib/fcitx5 /usr/lib64/fcitx5 /usr/lib/*/fcitx5; do
    if [[ -f $d/libclassicui.so ]]; then
        addon_dir=$d
        break
    fi
done
[[ -n $addon_dir ]] || fail 10 "fcitx5 not found: install fcitx5 first"

# The reranker's LM: its manifest, the weights file it names and the LM's own
# tokenizer when it names one, relative to it (copied to the same relative
# paths, so the manifest resolves).
rerank_files=()
if [[ -n $rerank_manifest ]]; then
    [[ -f $rerank_manifest ]] || fail 30 "rerank manifest not found: $rerank_manifest"
    for key in weights tokenizer; do
        if command -v jq >/dev/null; then
            f=$(jq -r ".$key.path // empty" "$rerank_manifest") ||
                fail 30 "rerank manifest is not JSON: $rerank_manifest"
        else
            # the "path" inside the key's object (neither object nests another)
            f=$(tr -d '\n' <"$rerank_manifest" |
                sed -n 's/.*"'"$key"'"[[:space:]]*:[[:space:]]*{[^}]*"path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        fi
        if [[ -z $f ]]; then
            [[ $key == tokenizer ]] && continue
            fail 30 "cannot read the weights path: $rerank_manifest"
        fi
        if [[ $f == /* || $f =~ (^|/)\.\.(/|$) ]]; then
            fail 30 "rerank manifest $key path leaves the artifact directory: $f"
        fi
        [[ -f $(dirname "$rerank_manifest")/$f ]] ||
            fail 30 "rerank $key not found: $(dirname "$rerank_manifest")/$f"
        rerank_files+=("$f")
    done
fi
if [[ -n $name_dictionary && ! -f $name_dictionary ]]; then
    fail 30 "names dictionary not found: $name_dictionary"
fi
if [[ -n $lexicon ]]; then
    [[ -f $lexicon ]] || fail 30 "lexicon not found: $lexicon"
    [[ $(head -c 8 -- "$lexicon") == LEXICON1 ]] || fail 30 "not a compiled lexicon: $lexicon"
fi
# The analyzer: its manifest and the dictionary and alternates it names,
# relative to it (copied to the same relative paths, so the manifest resolves).
analyzer_files=()
if [[ -n $analyzer ]]; then
    [[ -f $analyzer ]] || fail 30 "analyzer manifest not found: $analyzer"
    if command -v jq >/dev/null; then
        files=$(jq -r '.dictionary.path // empty, .alternates // empty' "$analyzer") ||
            fail 30 "analyzer manifest is not JSON: $analyzer"
    else
        files=$(tr -d '\n' <"$analyzer" |
            sed -n 's/.*"dictionary"[[:space:]]*:[[:space:]]*{[^}]*"path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        files+=$'\n'$(tr -d '\n' <"$analyzer" |
            sed -n 's/.*"alternates"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    fi
    while IFS= read -r f; do
        [[ -n $f ]] || continue
        if [[ $f == /* || $f =~ (^|/)\.\.(/|$) ]]; then
            fail 30 "analyzer manifest path leaves its directory: $f"
        fi
        [[ -f $(dirname "$analyzer")/$f ]] || fail 30 "analyzer file not found: $(dirname "$analyzer")/$f"
        analyzer_files+=("$f")
    done <<<"$files"
    ((${#analyzer_files[@]})) || fail 30 "the analyzer manifest names no dictionary: $analyzer"
fi

work=$(mktemp -d)
kwinrc_orig=
kwinrc_backup=
kwinrc_user=
kwinrc_uid=
kwinrc_bus=
kwinrc_im=
cleanup() {
    if [[ -n $kwinrc_backup && -f $kwinrc_backup ]]; then
        if [[ -n $kwinrc_im ]]; then
            runuser -u "$kwinrc_user" -- env XDG_RUNTIME_DIR="/run/user/$kwinrc_uid" \
                DBUS_SESSION_BUS_ADDRESS="unix:path=$kwinrc_bus" \
                kwriteconfig6 --file kwinrc --group Wayland --key InputMethod --notify "$kwinrc_im" \
                >/dev/null 2>&1 || true
        fi
        runuser -u "$kwinrc_user" -- mv -f "$kwinrc_backup" "$kwinrc_orig" 2>/dev/null || true
    fi
    rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

file_size() { wc -c <"$1" | tr -d ' '; }
file_sha256() { sha256sum -- "$1" | cut -d' ' -f1; }
# try_fetch <url> <path> <size or empty> <sha256>: download, check and move
# into place; 1 (and nothing in place) when that fails
try_fetch() {
    local tmp
    mkdir -p "$(dirname "$2")" && tmp=$(mktemp "$2.XXXXXXXX.part") || return 1
    if ! curl -fsSL --retry 2 --connect-timeout 30 --max-time 3600 -o "$tmp" "$1"; then
        rm -f "$tmp"
        log "download failed: $1"
        return 1
    fi
    if [[ -n $3 && $(file_size "$tmp") != "$3" ]] || [[ $(file_sha256 "$tmp") != "$4" ]]; then
        rm -f "$tmp"
        log "size or SHA-256 mismatch: $1"
        return 1
    fi
    mv -f "$tmp" "$2"
}
# fetch_to: try_fetch or stop
fetch_to() {
    try_fetch "$@" || fail 30 "could not get $1"
}
pin_path_re='^([A-Za-z0-9][A-Za-z0-9._-]*/)*[A-Za-z0-9][A-Za-z0-9._-]*$'
repository_re='^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$'

# ---- the release archive ----
if [[ -z $archive ]]; then
    [[ $VERSION != @* && $archive_sha =~ ^[a-f0-9]{64}$ ]] ||
        fail 1 "this install.sh is not from a release: use --archive <setsuna-ime-*.tar.gz>"
    name=setsuna-ime-$VERSION-linux-$(uname -m).tar.gz
    log "downloading $name"
    archive=$work/$name
    fetch_to "$RELEASES/v$VERSION/$name" "$archive" "" "$archive_sha"
fi
[[ -f $archive ]] || fail 1 "archive not found: $archive"
tar -xzf "$archive" -C "$work" || fail 30 "could not unpack $archive"
src=$(find "$work" -mindepth 1 -maxdepth 1 -type d -name 'setsuna-ime-*' | head -1)
[[ -n $src && -f $src/models.tsv && -f $src/addon/setsuna.so ]] || fail 30 "not a Setsuna release archive: $archive"
[[ $VERSION != @* ]] || VERSION=$(basename "$src" | sed 's/^setsuna-ime-//; s/-linux-.*//')

# ---- the pinned files: the model, the reading lexicon and the analyzer ----
# Each is installed once: a file installed before that is the pinned one stays
# as it is, the others are downloaded into the cache, checked, hard-linked into
# place (copied where the cache is on another file system) and the cache goes
# after the install. A later install with the same pins downloads nothing.
module=/usr/lib/setsuna-ime
installed=()
cache_root=$root/var/cache/setsuna-ime/models
pinned_src=() # the file to install ("": the installed one is it)
pinned_dst=() # where, absolute
verified() {
    [[ -f $1 && $(file_size "$1") == "$2" && $(file_sha256 "$1") == "$3" ]]
}
# obtain <what> <repository> <revision> <path> <size> <sha256> <destination>:
# queue a pinned file; 1 when it cannot be had
obtain() {
    local what=$1 repo=$2 rev=$3 path=$4 size=$5 sha=$6 dst=$7 c
    c=$cache_root/$repo/$rev/$path
    if verified "$root$dst" "$size" "$sha"; then
        log "$what installed already: $path"
        c=
    elif verified "$c" "$size" "$sha"; then
        log "$what cache verified: $path"
    else
        log "downloading $what: $path"
        try_fetch "https://huggingface.co/$repo/resolve/$rev/$path" "$c" "$size" "$sha" || return 1
    fi
    pinned_src+=("$c")
    pinned_dst+=("$dst")
}

# the model (models.tsv's repository, revision and <precision> lines; with
# --model another one's "model <name> <repository> <revision>" line and its
# <name>:<precision> lines)
[[ $model != $(awk -F'\t' '$1 == "name" { print $2 }' "$src/models.tsv") ]] || model=
if [[ -z $model ]]; then
    repository=$(awk -F'\t' '$1 == "repository" { print $2 }' "$src/models.tsv")
    revision=$(awk -F'\t' '$1 == "revision" { print $2 }' "$src/models.tsv")
    key=$precision
else
    repository=$(awk -F'\t' -v m="$model" '$1 == "model" && $2 == m { print $3 }' "$src/models.tsv")
    revision=$(awk -F'\t' -v m="$model" '$1 == "model" && $2 == m { print $4 }' "$src/models.tsv")
    [[ -n $repository ]] || fail 1 "the archive pins no model $model"
    key=$model:$precision
fi
[[ $repository =~ $repository_re && $revision =~ ^[a-f0-9]{40}$ ]] ||
    fail 30 "invalid model pins in the archive"
manifest=
model_files=() # the model's files, under models/onnx
while IFS=$'\t' read -r p path size sha; do
    [[ $p == "$key" ]] || continue
    [[ $path =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $size =~ ^[1-9][0-9]*$ && $sha =~ ^[a-f0-9]{64}$ ]] ||
        fail 30 "invalid model pin: $path"
    obtain model "$repository" "$revision" "$path" "$size" "$sha" "$module/models/onnx/$path" ||
        fail 30 "could not get the model"
    model_files+=("$path")
    [[ $path == *manifest.json ]] && manifest=$path
done <"$src/models.tsv"
[[ -n $manifest ]] || fail 30 "the archive pins no $precision model"

# component_dst <name> <entry> <path>: where a component's file goes. The
# lexicon is words.lexicon; the analyzer's and the reading corrector's files
# keep their paths relative to its manifest, as the manifest names them.
component_dst() {
    local entry_dir
    case $1 in
        lexicon)
            [[ $3 == "$2" ]] || return 1
            echo "$module/lexicon/words.lexicon"
            ;;
        analyzer | corrector)
            entry_dir=$(dirname "$2")
            if [[ $entry_dir == . ]]; then
                echo "$module/$1/$3"
            else
                [[ $3 == "$entry_dir"/* ]] || return 1
                echo "$module/$1/${3#"$entry_dir"/}"
            fi
            ;;
        *) return 1 ;;
    esac
}
# fetch_component <name>: queue the files models.tsv pins for it (component
# lines); 1, with nothing queued, when it cannot be had. Sets comp_entry.
fetch_component() {
    local name=$1 repo= rev= entry= p path size sha dst found=0 n=${#pinned_src[@]}
    IFS=$'\t' read -r repo rev entry < <(awk -F'\t' -v OFS='\t' -v n="$name" \
        '$1 == "component" && $2 == n { print $3, $4, $5; exit }' "$src/models.tsv") || true
    if [[ -z $entry ]]; then
        log "the archive pins no $name"
        return 1
    fi
    if ! [[ $repo =~ $repository_re && $rev =~ ^[a-f0-9]{40}$ && $entry =~ $pin_path_re ]]; then
        log "invalid $name pins in the archive"
        return 1
    fi
    comp_entry=$entry
    while IFS=$'\t' read -r p path size sha; do
        [[ $p == "$name" ]] || continue
        if ! [[ $path =~ $pin_path_re && $size =~ ^[1-9][0-9]*$ && $sha =~ ^[a-f0-9]{64}$ ]] ||
            ! dst=$(component_dst "$name" "$entry" "$path"); then
            log "invalid $name pin: $path"
            pinned_src=("${pinned_src[@]:0:n}")
            pinned_dst=("${pinned_dst[@]:0:n}")
            return 1
        fi
        if ! obtain "$name" "$repo" "$rev" "$path" "$size" "$sha" "$dst"; then
            pinned_src=("${pinned_src[@]:0:n}")
            pinned_dst=("${pinned_dst[@]:0:n}")
            return 1
        fi
        [[ $path == "$entry" ]] && found=1
    done <"$src/models.tsv"
    if ((!found)); then
        log "the archive pins no file $entry for $name"
        pinned_src=("${pinned_src[@]:0:n}")
        pinned_dst=("${pinned_dst[@]:0:n}")
        return 1
    fi
}
# What this install brings (else what was installed before stays)
have_lexicon=0
have_analyzer=0
analyzer_name=
have_corrector=0
corrector_name=
[[ -n $lexicon ]] && have_lexicon=1
[[ -n $analyzer ]] && have_analyzer=1 && analyzer_name=$(basename "$analyzer")
if ((dictionaries && !have_lexicon)); then
    if fetch_component lexicon; then
        have_lexicon=1
    else
        log "warning: no reading lexicon: installing without it (run again to retry)"
    fi
fi
if ((dictionaries && !have_analyzer)); then
    if fetch_component analyzer; then
        have_analyzer=1
        analyzer_name=$(basename "$comp_entry")
    else
        log "warning: no analyzer: installing without it (run again to retry)"
    fi
fi
if ((corrector)); then
    if fetch_component corrector; then
        have_corrector=1
        corrector_name=$(basename "$comp_entry")
    else
        log "warning: no reading corrector: installing without it (run again to retry)"
    fi
fi

# ---- install ----
# put <source> <absolute destination> <mode>: copied beside the destination
# and renamed over it, so running daemons keep the files they have open
put() {
    local dst=$root$2 tmp
    mkdir -p "$(dirname "$dst")"
    tmp=$(mktemp "$dst.XXXXXXXX")
    if ! { cp -- "$1" "$tmp" && chmod "$3" "$tmp" && mv -f "$tmp" "$dst"; }; then
        rm -f "$tmp"
        fail 50 "could not install $2"
    fi
    installed+=("$2")
}
# put_tree <source dir> <absolute destination dir>
put_tree() {
    local f rel mode
    while IFS= read -r -d '' f; do
        rel=${f#"$1"/}
        mode=644
        [[ -x $f ]] && mode=755
        put "$f" "$2/$rel" "$mode"
    done < <(find "$1" -type f -print0 | sort -z)
}
# place <source or ""> <absolute destination>: a pinned file, hard-linked from
# the cache beside the destination (copied where it cannot be linked) and
# renamed over it; "" keeps the installed one, which is the pinned file
place() {
    local dst=$root$2 tmp
    if [[ -n $1 ]]; then
        mkdir -p "$(dirname "$dst")"
        tmp=$dst.setsuna-new.$$
        rm -f -- "$tmp"
        if ! { { ln -- "$1" "$tmp" 2>/dev/null || cp -- "$1" "$tmp"; } &&
            chmod 644 "$tmp" && mv -f "$tmp" "$dst"; }; then
            rm -f -- "$tmp"
            fail 50 "could not install $2"
        fi
    fi
    installed+=("$2")
}

put "$src/addon/setsuna.so" "$addon_dir/setsuna.so" 755
put_tree "$src/share" /usr/share
put_tree "$src/module" "$module"
# Another model's manifest would compete with this one (the engine reads the
# only one in models/onnx), and its files would be kept for nothing (the
# model before, on a change of the default).
for m in "$root$module"/models/onnx/*; do
    [[ -f $m ]] || continue
    keep=0
    for f in "${model_files[@]}"; do
        [[ $(basename "$m") != "$f" ]] || keep=1
    done
    ((keep)) || rm -f -- "$m"
done
# One analyzer, one reading corrector: the addon gives the engine the only
# manifest there.
if ((have_analyzer)); then
    for m in "$root$module"/analyzer/*manifest.json; do
        if [[ -e $m && $(basename "$m") != "$analyzer_name" ]]; then
            rm -f -- "$m"
        fi
    done
fi
if ((have_corrector)); then
    for m in "$root$module"/corrector/*manifest.json; do
        if [[ -e $m && $(basename "$m") != "$corrector_name" ]]; then
            rm -f -- "$m"
        fi
    done
fi
# the model, lexicon and analyzer from the pins (a lexicon renamed over the
# old one, which a running fcitx5 keeps mapped)
for i in "${!pinned_dst[@]}"; do
    place "${pinned_src[i]}" "${pinned_dst[i]}"
done
if [[ -f $src/bin/setsuna-settings ]]; then
    if ldd "$src/bin/setsuna-settings" 2>/dev/null | grep -q 'not found'; then
        log "the settings GUI needs WebKitGTK 4.1: not installed"
        # nor its menu entry
        rm -f -- "$root/usr/share/applications/setsuna-settings.desktop"
        mapfile -t installed < <(printf '%s\n' "${installed[@]}" | grep -vxF /usr/share/applications/setsuna-settings.desktop)
    else
        put "$src/bin/setsuna-settings" /usr/bin/setsuna-settings 755
    fi
fi
if [[ -n $rerank_manifest ]]; then
    # one LM: the reranker reads the only manifest there. The one given is
    # kept aside first: it may be the installed one the old ones' removal
    # takes (an install again with its own --rerank-manifest)
    cp -- "$rerank_manifest" "$work/rerank.manifest.json"
    rm -f -- "$root$module"/models/rerank/*manifest.json
    put "$work/rerank.manifest.json" "$module/models/rerank/$(basename "$rerank_manifest")" 644
    for f in "${rerank_files[@]}"; do
        put "$(dirname "$rerank_manifest")/$f" "$module/models/rerank/$f" 644
    done
fi
if [[ -n $name_dictionary ]]; then
    put "$name_dictionary" "$module/names/latin-names.tsv" 644
fi
if [[ -n $lexicon ]]; then
    # a local one (--lexicon), renamed over the old one
    put "$lexicon" "$module/lexicon/words.lexicon" 644
fi
if [[ -n $analyzer ]]; then
    # a local one (--analyzer)
    put "$analyzer" "$module/analyzer/$(basename "$analyzer")" 644
    for f in "${analyzer_files[@]}"; do
        put "$(dirname "$analyzer")/$f" "$module/analyzer/$f" 644
    done
fi
# Files of the previous install that this one no longer has (another model);
# its reranker model, names dictionary, lexicon and analyzer stay unless
# replaced.
if [[ -f $root$module/installed-files ]]; then
    while IFS= read -r f; do
        [[ $f == /* && ! -d $root$f ]] || continue
        printf '%s\n' "${installed[@]}" | grep -qxF -- "$f" && continue
        if [[ ($f == "$module"/models/rerank/* && -z $rerank_manifest) ||
            ($f == "$module"/names/* && -z $name_dictionary) ||
            ($f == "$module"/lexicon/* && $have_lexicon == 0) ||
            ($f == "$module"/analyzer/* && $have_analyzer == 0) ||
            ($f == "$module"/corrector/* && $have_corrector == 0) ]] && [[ -f $root$f ]]; then
            installed+=("$f")
            continue
        fi
        rm -f -- "$root$f"
    done <"$root$module/installed-files"
fi
printf '%s\n' "${installed[@]}" "$module/installed-files" >"$root$module/installed-files"
# The download cache: what this install needed is installed (one copy each),
# and what earlier installs left there (other revisions, other models) no
# install uses.
rm -rf -- "$cache_root"
rmdir -- "$root/var/cache/setsuna-ime" 2>/dev/null || true
if [[ -z $root ]] && command -v gtk-update-icon-cache >/dev/null && [[ -f /usr/share/icons/hicolor/index.theme ]]; then
    gtk-update-icon-cache -q -t /usr/share/icons/hicolor || true
fi
log "installed Setsuna $VERSION ($precision model) into ${root:-/}"

# Extract every double-quoted string from a busctl reply line (handling spaces
# inside quotes and \" / \\ escapes) into the `busctl_strings` array.
parse_busctl_strings() {
    local s=$1 i=0 len=${#1} in_q=0 cur="" c
    busctl_strings=()
    while ((i < len)); do
        c=${s:i:1}
        if ((!in_q)); then
            if [[ $c == '"' ]]; then
                in_q=1
                cur=""
            fi
        else
            if [[ $c == '\' && $((i + 1)) -lt $len ]]; then
                i=$((i + 1))
                c=${s:i:1}
                case $c in
                    n) cur+=$'\n' ;;
                    t) cur+=$'\t' ;;
                    r) cur+=$'\r' ;;
                    *) cur+=$c ;;
                esac
            elif [[ $c == '"' ]]; then
                busctl_strings+=("$cur")
                in_q=0
            else
                cur+=$c
            fi
        fi
        i=$((i + 1))
    done
    ((!in_q))
}

# ---- add Setsuna to the user's fcitx5 ----
register_user() {
    local user=$1 uid home bus pid env_file
    uid=$(id -u "$user")
    home=$(getent passwd "$user" | cut -d: -f6)
    bus=/run/user/$uid/bus
    as_user() {
        runuser -u "$user" -- env XDG_RUNTIME_DIR="/run/user/$uid" \
            DBUS_SESSION_BUS_ADDRESS="unix:path=$bus" "$@"
    }
    fcitx() {
        as_user busctl --user call org.fcitx.Fcitx5 /controller org.fcitx.Fcitx.Controller1 "$@"
    }

    pid=$(pgrep -u "$uid" -x fcitx5 | head -1 || true)
    if [[ -z $pid ]]; then
        # Not running: the profile is safe to edit; fcitx5 reads it when it starts.
        # read as the user, as kwinrc is below
        local profile=$home/.config/fcitx5/profile n next
        if as_user test -f "$profile"; then
            if as_user grep -qx 'Name=setsuna' "$profile"; then
                log "Setsuna is already in $user's fcitx5 input methods"
            else
                n=$(as_user sed -n 's/^\[Groups\/0\/Items\/\([0-9][0-9]*\)\]$/\1/p' "$profile" | sort -n | tail -1)
                if [[ -n $n ]]; then
                    next=$((10#$n + 1))
                else
                    next=0
                fi
                printf '\n[Groups/0/Items/%s]\n# Name\nName=setsuna\n# Layout\nLayout=\n' "$next" |
                    as_user tee -a "$profile" >/dev/null
                log "added Setsuna to $user's fcitx5 input methods"
            fi
        else
            log "fcitx5 is not running: add Setsuna in fcitx5-configtool once it is"
        fi
        return
    fi
    [[ -S $bus ]] || { log "no session bus for $user: add Setsuna in fcitx5-configtool"; return; }

    # Restart fcitx5 so it loads the new addon. KDE Plasma (Wayland) starts it
    # through fcitx5-wayland-launcher and starts that again only when its
    # InputMethod setting changes; elsewhere fcitx5 comes back by D-Bus
    # activation, or from the environment it ran with.
    local kwin_im=
    if pgrep -u "$uid" -x kwin_wayland >/dev/null && command -v kreadconfig6 >/dev/null &&
        command -v kwriteconfig6 >/dev/null; then
        kwin_im=$(as_user kreadconfig6 --file kwinrc --group Wayland --key InputMethod || true)
        [[ $kwin_im == *fcitx5* ]] || kwin_im=
    fi
    env_file=$work/fcitx5.environ
    cp "/proc/$pid/environ" "$env_file" 2>/dev/null || : >"$env_file"
    log "restarting fcitx5 for $user"
    fcitx Exit >/dev/null 2>&1 || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    if [[ -n $kwin_im ]]; then
        # kwinrc is the user's: it is read and written as the user only (as
        # root, a link the user put in its place, or in the temporary
        # file's, would be followed to a file of root's)
        local kwinrc=$home/.config/kwinrc orig_line=
        if as_user test -f "$kwinrc"; then
            orig_line=$(as_user awk '/^\[/{g=$0; next} g=="[Wayland]" && /^InputMethod(\[[^]]*\])?=/{print; exit}' "$kwinrc" || true)
            kwinrc_backup=$(as_user mktemp "$kwinrc.setsuna-install.XXXXXX")
            as_user cp -p "$kwinrc" "$kwinrc_backup"
            kwinrc_orig=$kwinrc
            kwinrc_user=$user
            kwinrc_uid=$uid
            kwinrc_bus=$bus
            kwinrc_im=$kwin_im
        fi
        as_user kwriteconfig6 --file kwinrc --group Wayland --key InputMethod --notify "" || true
        sleep 1
        as_user kwriteconfig6 --file kwinrc --group Wayland --key InputMethod --notify "$kwin_im" || true
        # Restore the key's original flags (e.g. InputMethod[$e]=...) without
        # overwriting unrelated keys KWin may have written meanwhile.
        if [[ -n $orig_line && $orig_line == InputMethod\[* ]]; then
            local fix='
                /^\[/ { in_w = ($0 == "[Wayland]") }
                in_w && !done && /^InputMethod(\[[^]]*\])?=/ { print ENVIRON["SETSUNA_LINE"]; done = 1; next }
                { print }
            '
            # shellcheck disable=SC2016 # expanded by the user's shell
            as_user env SETSUNA_LINE="$orig_line" SETSUNA_AWK="$fix" sh -c '
                rc=$1
                [ -f "$rc" ] || exit 0
                tmp=$(mktemp "$rc.setsuna-fix.XXXXXX") || exit 0
                if awk "$SETSUNA_AWK" "$rc" >"$tmp"; then
                    chmod --reference="$rc" "$tmp" 2>/dev/null || true
                    mv -f "$tmp" "$rc" || rm -f -- "$tmp"
                else
                    rm -f -- "$tmp"
                fi
            ' sh "$kwinrc" || true
        fi
        if [[ -n $kwinrc_backup ]]; then
            as_user rm -f -- "$kwinrc_backup"
            kwinrc_backup=
        fi
    elif [[ ! -f /usr/share/dbus-1/services/org.fcitx.Fcitx5.service ]]; then
        local -a vars
        mapfile -d '' vars <"$env_file"
        as_user env -i "${vars[@]}" fcitx5 -d >/dev/null 2>&1 || true
    fi
    for _ in $(seq 1 60); do
        fcitx AvailableInputMethods 2>/dev/null | grep -q '"setsuna"' && break
        sleep 0.5
    done
    if ! fcitx AvailableInputMethods 2>/dev/null | grep -q '"setsuna"'; then
        log "fcitx5 did not come back with Setsuna: log in again, then add Setsuna in fcitx5-configtool"
        return
    fi

    # Append Setsuna to the current input method group.
    local group info layout count i
    local -a busctl_strings entries
    if ! info=$(fcitx CurrentInputMethodGroup) || ! parse_busctl_strings "$info" || ((${#busctl_strings[@]} < 1)); then
        log "could not read $user's current fcitx5 group: add Setsuna in fcitx5-configtool"
        return
    fi
    group=${busctl_strings[0]}
    if ! info=$(fcitx InputMethodGroupInfo s "$group") || ! parse_busctl_strings "$info" ||
        ((${#busctl_strings[@]} < 1 || (${#busctl_strings[@]} - 1) % 2 != 0)); then
        log "could not read $user's fcitx5 group $group: add Setsuna in fcitx5-configtool"
        return
    fi
    layout=${busctl_strings[0]}
    entries=("${busctl_strings[@]:1}")
    count=$((${#entries[@]} / 2))
    for ((i = 0; i < count; i++)); do
        if [[ ${entries[2 * i]} == setsuna ]]; then
            log "Setsuna is already in $user's input methods"
            return
        fi
    done
    fcitx SetInputMethodGroupInfo 'ssa(ss)' "$group" "$layout" "$((count + 1))" "${entries[@]}" setsuna "" &&
        log "added Setsuna to $user's input methods (group $group)"
}

if ((register)); then
    user=${SUDO_USER:-}
    if [[ -z $user && -n ${PKEXEC_UID:-} ]]; then
        user=$(getent passwd "$PKEXEC_UID" | cut -d: -f1)
    fi
    if [[ -z $user || $user == root ]]; then
        log "run through sudo from your user to have Setsuna added to your fcitx5; or add it in fcitx5-configtool"
    elif ! command -v runuser >/dev/null || ! command -v busctl >/dev/null; then
        log "runuser or busctl not found: add Setsuna in fcitx5-configtool"
    else
        register_user "$user"
    fi
fi

cat >&2 <<EOF

Setsuna IME $VERSION installed. Switch to it with fcitx5's trigger key (Ctrl+Space by default).
Uninstall: sudo $module/uninstall.sh
EOF
