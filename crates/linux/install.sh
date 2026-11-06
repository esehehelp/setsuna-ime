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
#   --archive <file>        a local release archive instead of the download
#   --lexicon <file>        a local reading lexicon (a compiled .lexicon) instead
#                           of the download: the dictionary's words come into
#                           the candidates
#   --analyzer <manifest>   a local analyzer (its manifest, the dictionary and
#                           alternates it names) instead of the download:
#                           segment editing cuts a conversion by its words
#   --no-dictionaries       download neither the lexicon nor the analyzer
#   --rerank-manifest <m>   also the reranker's LM (an LM scorer manifest, the
#                           weights and tokenizer it names): candidates are reranked
#   --name-dictionary <tsv> also mixed input's Latin names dictionary
#   --no-register           install only; leave fcitx5 as it is
#   --root <dir>            install under <dir> (staging, tests): no root
#                           needed, implies --no-register
# A reranker model, names dictionary, lexicon or analyzer installed before
# stays when this install does not bring one (--no-dictionaries, a failed
# download, the option not given again).
#
# Layout:
#   <fcitx5 addon dir>/setsuna.so
#   /usr/share/fcitx5/{addon,inputmethod,themes}/, /usr/share/icons/hicolor/scalable/apps/setsuna*.svg
#   /usr/lib/setsuna-ime/  the engine, reranker and rewriter daemons, libonnxruntime.so,
#                          llama.cpp's libraries, mixed/, models/onnx/, licenses/,
#                          uninstall.sh, installed-files
#   /usr/lib/setsuna-ime/models/rerank/<manifest> + its weights (and tokenizer)  (with --rerank-manifest)
#   /usr/lib/setsuna-ime/names/latin-names.tsv  (with --name-dictionary)
#   /usr/lib/setsuna-ime/lexicon/words.lexicon  (downloaded, or --lexicon)
#   /usr/lib/setsuna-ime/analyzer/<manifest> + its dictionary and alternates  (downloaded, or --analyzer)
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
archive=
register=1
root=
rerank_manifest=
name_dictionary=
lexicon=
analyzer=
dictionaries=1

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
        --archive) archive=${2:?}; shift 2 ;;
        --rerank-manifest) rerank_manifest=${2:?}; shift 2 ;;
        --name-dictionary) name_dictionary=${2:?}; shift 2 ;;
        --lexicon) lexicon=${2:?}; shift 2 ;;
        --analyzer) analyzer=${2:?}; shift 2 ;;
        --no-dictionaries) dictionaries=0; shift ;;
        --no-register) register=0; shift ;;
        --root) root=${2:?}; register=0; shift 2 ;;
        -h | --help) sed -n '1d; /^#/!q; s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) fail 1 "unknown argument: $1" ;;
    esac
done
[[ $precision == int8 || $precision == fp32 ]] || fail 1 "--precision must be int8 or fp32"
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
trap 'rm -rf "$work"' EXIT

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

# the model (models.tsv's repository, revision and <precision> lines)
repository=$(awk -F'\t' '$1 == "repository" { print $2 }' "$src/models.tsv")
revision=$(awk -F'\t' '$1 == "revision" { print $2 }' "$src/models.tsv")
[[ $repository =~ $repository_re && $revision =~ ^[a-f0-9]{40}$ ]] ||
    fail 30 "invalid model pins in the archive"
manifest=
while IFS=$'\t' read -r p path size sha; do
    [[ $p == "$precision" ]] || continue
    [[ $path =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $size =~ ^[1-9][0-9]*$ && $sha =~ ^[a-f0-9]{64}$ ]] ||
        fail 30 "invalid model pin: $path"
    obtain model "$repository" "$revision" "$path" "$size" "$sha" "$module/models/onnx/$path" ||
        fail 30 "could not get the model"
    [[ $path == *manifest.json ]] && manifest=$path
done <"$src/models.tsv"
[[ -n $manifest ]] || fail 30 "the archive pins no $precision model"

# component_dst <name> <entry> <path>: where a component's file goes. The
# lexicon is words.lexicon; the analyzer's files keep their paths relative to
# its manifest, as the manifest names them.
component_dst() {
    local entry_dir
    case $1 in
        lexicon)
            [[ $3 == "$2" ]] || return 1
            echo "$module/lexicon/words.lexicon"
            ;;
        analyzer)
            entry_dir=$(dirname "$2")
            if [[ $entry_dir == . ]]; then
                echo "$module/analyzer/$3"
            else
                [[ $3 == "$entry_dir"/* ]] || return 1
                echo "$module/analyzer/${3#"$entry_dir"/}"
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
# only one in models/onnx).
for m in "$root$module"/models/onnx/*manifest.json; do
    if [[ -e $m && $(basename "$m") != "$manifest" ]]; then
        rm -f -- "$m"
    fi
done
# One analyzer: the addon gives the engine the only manifest there.
if ((have_analyzer)); then
    for m in "$root$module"/analyzer/*manifest.json; do
        if [[ -e $m && $(basename "$m") != "$analyzer_name" ]]; then
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
    # one LM: the reranker reads the only manifest there
    rm -f -- "$root$module"/models/rerank/*manifest.json
    put "$rerank_manifest" "$module/models/rerank/$(basename "$rerank_manifest")" 644
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
            ($f == "$module"/analyzer/* && $have_analyzer == 0) ]] && [[ -f $root$f ]]; then
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
        local profile=$home/.config/fcitx5/profile n
        if [[ -f $profile ]] && ! grep -qx 'Name=setsuna' "$profile"; then
            n=$(grep -o '^\[Groups/0/Items/[0-9]*\]' "$profile" | tr -dc '0-9\n' | sort -n | tail -1)
            printf '\n[Groups/0/Items/%s]\n# Name\nName=setsuna\n# Layout\nLayout=\n' "$((${n:--1} + 1))" |
                as_user tee -a "$profile" >/dev/null
            log "added Setsuna to $user's fcitx5 input methods"
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
        local kwinrc=$home/.config/kwinrc
        as_user cp -p "$kwinrc" "$kwinrc.setsuna-install"
        as_user kwriteconfig6 --file kwinrc --group Wayland --key InputMethod --notify ""
        sleep 1
        as_user kwriteconfig6 --file kwinrc --group Wayland --key InputMethod --notify "$kwin_im"
        # the file exactly as it was (kwriteconfig6 rewrites the key's flags)
        as_user mv -f "$kwinrc.setsuna-install" "$kwinrc"
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
    local -a tok entries
    group=$(fcitx CurrentInputMethodGroup | cut -d'"' -f2)
    info=$(fcitx InputMethodGroupInfo s "$group")
    # sa(ss) "<layout>" <count> "<name>" "<layout>" ...
    read -ra tok <<<"$info"
    tok=("${tok[@]//\"/}")
    layout=${tok[1]}
    count=${tok[2]}
    entries=("${tok[@]:3}")
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
