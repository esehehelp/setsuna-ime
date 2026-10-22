#!/usr/bin/env bash
# Fetch a public conversion artifact pinned by crates/tsf/models.json. No login
# required. Writes only to the download cache; prints the local manifest path
# on stdout (progress goes to stderr). Verified cache files work offline.
# Downloads are staged before replacement.
#
# Usage:
#   crates/linux/fetch-model.sh
#   crates/linux/fetch-model.sh --precision fp32 --cache-dir local/models/downloads
#
# --base-url replaces https://huggingface.co (a mirror, or file:// in
# tests/model-download.sh); the pinned sizes and SHA-256 are checked all the same.
# Needs curl, jq and sha256sum.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
precision=int8
cache_dir=${XDG_CACHE_HOME:-$HOME/.cache}/setsuna-ime/models
catalog=$here/../tsf/models.json
base_url=https://huggingface.co

die() {
    echo "fetch-model: $*" >&2
    exit 1
}

while (($#)); do
    case $1 in
        --precision) precision=${2:?}; shift 2 ;;
        --cache-dir) cache_dir=${2:?}; shift 2 ;;
        --catalog) catalog=${2:?}; shift 2 ;;
        --base-url) base_url=${2:?}; shift 2 ;;
        -h | --help) sed -n '1d; /^#/!q; s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ $precision == int8 || $precision == fp32 ]] || die "precision must be int8 or fp32"
for tool in curl jq sha256sum; do
    command -v "$tool" >/dev/null || die "$tool is required"
done

# Forward-slash relative paths only: no empty, . or .. segment, no control
# characters, backslashes or URL syntax; every segment is URL-escaped below.
check_path() {
    local p=$1 part bad='[[:cntrl:]\?#%:"<>|*]'
    [[ -n $p && $p != /* && $p != */ && $p != *//* ]] && ! [[ $p =~ $bad ]] ||
        die "invalid artifact path: $p"
    local -a parts
    IFS=/ read -ra parts <<<"$p"
    for part in "${parts[@]}"; do
        [[ $part != . && $part != .. ]] || die "invalid artifact path: $p"
    done
}

file_size() { wc -c <"$1" | tr -d ' '; }
file_sha256() { sha256sum -- "$1" | cut -d' ' -f1; }

# verified <path> <relative path>: the file has the pinned size and SHA-256
verified() {
    [[ -f $1 && $(file_size "$1") == "${spec_size[$2]}" && $(file_sha256 "$1") == "${spec_sha[$2]}" ]]
}

[[ -f $catalog ]] || die "model catalog not found: $catalog"
jq -e . "$catalog" >/dev/null 2>&1 || die "model catalog is not JSON: $catalog"
schema=$(jq -r '.schema // ""' "$catalog")
repository=$(jq -r '.repository // ""' "$catalog")
revision=$(jq -r '.revision // ""' "$catalog")
if [[ $schema != setsuna-model-download-v1 ||
    ! $repository =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ||
    ! $revision =~ ^[a-f0-9]{40}$ ]]; then
    die "invalid model catalog (expected a repository and pinned commit)"
fi
check_path "$repository"
jq -e --arg p "$precision" '.variants[$p] | type == "object"' "$catalog" >/dev/null ||
    die "model catalog has no $precision variant"
manifest_rel=$(jq -r --arg p "$precision" '.variants[$p].manifest // ""' "$catalog")
check_path "$manifest_rel"

specs=$(jq -r --arg p "$precision" \
    '.variants[$p].files[] | [.path, (.size | tostring), (.sha256 // "")] | @tsv' "$catalog") ||
    die "invalid file specifications in the catalog"
declare -A spec_size spec_sha
while IFS=$'\t' read -r path size sha; do
    check_path "$path"
    if [[ ! $sha =~ ^[a-f0-9]{64}$ || ! $size =~ ^[1-9][0-9]*$ || -n ${spec_size[$path]+x} ]]; then
        die "invalid or duplicate file specification: $path"
    fi
    spec_size[$path]=$size
    spec_sha[$path]=$sha
done <<<"$specs"
[[ -n ${spec_size[$manifest_rel]+x} ]] || die "manifest is not pinned in the catalog"

dest=$cache_dir/$repository/$revision
mkdir -p "$dest"

# fetch <relative path>: the verified cached file, downloaded when missing or
# changed. The new file is staged next to it and moved over it only once checked.
fetch() {
    local rel=$1
    check_path "$rel"
    [[ -n ${spec_size[$rel]+x} ]] || die "manifest references an unpinned file: $rel"
    local path=$dest/$rel
    if verified "$path" "$rel"; then
        echo "model cache verified: $rel" >&2
        return
    fi
    mkdir -p "$(dirname "$path")"
    local escaped tmp
    escaped=$(jq -rn --arg p "$rel" '$p | split("/") | map(@uri) | join("/")')
    tmp=$(mktemp "$path.XXXXXXXX.part")
    echo "downloading model: $rel" >&2
    # Never attach HF credentials: only public, pinned artifacts are used.
    if ! curl -fsSL --retry 2 --connect-timeout 30 --max-time 600 -o "$tmp" \
        "$base_url/$repository/resolve/$revision/$escaped"; then
        rm -f "$tmp"
        die "download failed: $rel"
    fi
    if ! verified "$tmp" "$rel"; then
        rm -f "$tmp"
        die "size or SHA-256 mismatch: $rel"
    fi
    mv -f "$tmp" "$path"
}

fetch "$manifest_rel"
manifest_path=$dest/$manifest_rel
if ! jq -e --arg p "$precision" \
    '.schema == "neo-ime-ctc-nat-export-v1" and .proposal.precision == $p' \
    "$manifest_path" >/dev/null 2>&1; then
    die "downloaded manifest has an unsupported schema or precision"
fi
# All names remain relative to the manifest, including nested artifact paths.
manifest_dir=$(dirname "$manifest_rel")
list=$(jq -r '[.proposal.path, .vocab] + ([.proposal.external_data // empty] | flatten) + [.input_vocab // empty]
    | .[] | if type == "string" then . else error("not a string") end' "$manifest_path") ||
    die "manifest lists an invalid artifact path"
artifacts=()
while IFS= read -r rel; do
    check_path "$rel"
    [[ $manifest_dir == . ]] || rel=$manifest_dir/$rel
    artifacts+=("$rel")
done <<<"$list"
# Check the whole set before downloading any payload.
for rel in "${artifacts[@]}"; do
    [[ -n ${spec_size[$rel]+x} ]] || die "manifest references an unpinned file: $rel"
done
for rel in "${artifacts[@]}"; do
    fetch "$rel"
done
echo "model ready: $repository @ $revision ($precision)" >&2
echo "$manifest_path"
