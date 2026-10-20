#!/usr/bin/env bash
# Fetch Microsoft's official ONNX Runtime build and put libonnxruntime.so next
# to the engine (the engine loads it from there; see src/runtime.rs). The Linux
# counterpart of fetch-onnxruntime.ps1, whose notes on the pinned version
# apply: change the default only together with a re-benchmark.
#
# Usage:
#   crates/engine/fetch-onnxruntime.sh                     # 1.28.0 -> target/release
#   crates/engine/fetch-onnxruntime.sh --dest target/debug
#
# The download (GitHub release tgz) is cached under local/onnxruntime and
# checked against the SHA-256 below. Linux x64 and aarch64.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
version=1.28.0
dest=$here/../../target/release

die() {
    echo "fetch-onnxruntime: $*" >&2
    exit 1
}

while (($#)); do
    case $1 in
        --version) version=${2:?}; shift 2 ;;
        --dest) dest=${2:?}; shift 2 ;;
        -h | --help) sed -n '1d; /^#/!q; s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

case $(uname -m) in
    x86_64) arch=x64 ;;
    aarch64 | arm64) arch=aarch64 ;;
    *) die "no official ONNX Runtime build for $(uname -m)" ;;
esac

# SHA-256 of onnxruntime-linux-<arch>-<version>.tgz from
# https://github.com/microsoft/onnxruntime/releases
declare -A known=(
    [1.28.0-x64]=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407
    [1.28.0-aarch64]=e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb
)
want=${known[$version-$arch]:-}
[[ -n $want ]] || die "no pinned SHA-256 for ONNX Runtime $version ($arch)"

name=onnxruntime-linux-$arch-$version
cache=$here/../../local/onnxruntime
mkdir -p "$cache"
tgz=$cache/$name.tgz
sha() { sha256sum -- "$1" | cut -d' ' -f1; }

if [[ ! -f $tgz || $(sha "$tgz") != "$want" ]]; then
    url=https://github.com/microsoft/onnxruntime/releases/download/v$version/$name.tgz
    echo "downloading $url" >&2
    tmp=$(mktemp "$tgz.XXXXXXXX.part")
    if ! curl -fsSL --retry 2 --connect-timeout 30 --max-time 600 -o "$tmp" "$url"; then
        rm -f "$tmp"
        die "download failed: $url"
    fi
    got=$(sha "$tmp")
    if [[ $got != "$want" ]]; then
        rm -f "$tmp"
        die "SHA-256 mismatch for $name.tgz: got $got, expected $want"
    fi
    mv -f "$tmp" "$tgz"
fi

# Only the library itself, under the name the engine loads. It is staged in
# the destination and renamed over the old one, so a running engine keeps the
# file it has mapped.
mkdir -p "$dest"
staging=$(mktemp -d "$dest/.onnxruntime.XXXXXXXX")
trap 'rm -rf "$staging"' EXIT
tar -xzf "$tgz" -C "$staging" "$name/lib/libonnxruntime.so.$version"
chmod 755 "$staging/$name/lib/libonnxruntime.so.$version"
mv -f "$staging/$name/lib/libonnxruntime.so.$version" "$dest/libonnxruntime.so"
echo "ONNX Runtime $version -> $(cd "$dest" && pwd)/libonnxruntime.so" >&2
