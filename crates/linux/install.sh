#!/usr/bin/env bash
# Setsuna IME installer for Linux (fcitx5). Run as root:
#
#   curl -fsSL https://github.com/esehehelp/setsuna-ime/releases/latest/download/install.sh | sudo bash
#   curl -fsSL .../install.sh | sudo bash -s -- --precision fp32
#
# Downloads the release archive this script was made for (crates/linux/package.sh
# fills in its SHA-256 below) and the conversion model pinned in it, checks
# both, installs them, then adds Setsuna to the input methods of the user who
# ran sudo (or pkexec) and restarts their fcitx5 so it loads the addon.
#
# Options:
#   --precision int8|fp32   the model (default int8)
#   --archive <file>        a local release archive instead of the download
#   --no-register           install only; leave fcitx5 as it is
#   --root <dir>            install under <dir> (staging, tests): no root
#                           needed, implies --no-register
#
# Layout:
#   <fcitx5 addon dir>/setsuna.so
#   /usr/share/fcitx5/{addon,inputmethod,themes}/, /usr/share/icons/hicolor/scalable/apps/setsuna*.svg
#   /usr/lib/setsuna-ime/  the engine and rewriter daemons, libonnxruntime.so,
#                          mixed/, models/onnx/, licenses/, uninstall.sh, installed-files
#   /usr/bin/setsuna-settings  (when WebKitGTK 4.1 is there)
#   /var/cache/setsuna-ime/models  (verified model downloads, reused offline)
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

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

file_size() { wc -c <"$1" | tr -d ' '; }
file_sha256() { sha256sum -- "$1" | cut -d' ' -f1; }
# fetch_to <url> <path> <size or empty> <sha256>: download, check and move into place
fetch_to() {
    local tmp
    mkdir -p "$(dirname "$2")"
    tmp=$(mktemp "$2.XXXXXXXX.part")
    if ! curl -fsSL --retry 2 --connect-timeout 30 --max-time 900 -o "$tmp" "$1"; then
        rm -f "$tmp"
        fail 30 "download failed: $1"
    fi
    if [[ -n $3 && $(file_size "$tmp") != "$3" ]] || [[ $(file_sha256 "$tmp") != "$4" ]]; then
        rm -f "$tmp"
        fail 30 "size or SHA-256 mismatch: $1"
    fi
    mv -f "$tmp" "$2"
}

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

# ---- the model (pinned by the archive's models.tsv) ----
repository=$(awk -F'\t' '$1 == "repository" { print $2 }' "$src/models.tsv")
revision=$(awk -F'\t' '$1 == "revision" { print $2 }' "$src/models.tsv")
[[ $repository =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ && $revision =~ ^[a-f0-9]{40}$ ]] ||
    fail 30 "invalid model pins in the archive"
cache=$root/var/cache/setsuna-ime/models/$repository/$revision
model_files=()
manifest=
while IFS=$'\t' read -r p path size sha; do
    [[ $p == "$precision" ]] || continue
    [[ $path =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && $size =~ ^[1-9][0-9]*$ && $sha =~ ^[a-f0-9]{64}$ ]] ||
        fail 30 "invalid model pin: $path"
    if [[ ! -f $cache/$path || $(file_size "$cache/$path") != "$size" || $(file_sha256 "$cache/$path") != "$sha" ]]; then
        log "downloading model: $path"
        fetch_to "https://huggingface.co/$repository/resolve/$revision/$path" "$cache/$path" "$size" "$sha"
    else
        log "model cache verified: $path"
    fi
    model_files+=("$path")
    [[ $path == *manifest.json ]] && manifest=$path
done <"$src/models.tsv"
[[ -n $manifest ]] || fail 30 "the archive pins no $precision model"

# ---- install ----
module=/usr/lib/setsuna-ime
installed=()
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
for path in "${model_files[@]}"; do
    put "$cache/$path" "$module/models/onnx/$path" 644
done
if [[ -f $src/bin/setsuna-settings ]]; then
    if ldd "$src/bin/setsuna-settings" 2>/dev/null | grep -q 'not found'; then
        log "the settings GUI needs WebKitGTK 4.1: not installed"
    else
        put "$src/bin/setsuna-settings" /usr/bin/setsuna-settings 755
    fi
fi
# Files of the previous install that this one no longer has (another model).
if [[ -f $root$module/installed-files ]]; then
    while IFS= read -r f; do
        [[ $f == /* && ! -d $root$f ]] || continue
        printf '%s\n' "${installed[@]}" | grep -qxF -- "$f" || rm -f -- "$root$f"
    done <"$root$module/installed-files"
fi
printf '%s\n' "${installed[@]}" "$module/installed-files" >"$root$module/installed-files"
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
