#!/usr/bin/env bash
# Build the Linux release: target/dist/setsuna-ime-<version>-linux-<arch>.tar.gz,
# install.sh with that archive's SHA-256 filled in, and SHA256SUMS. Run as a
# normal user; nothing is installed.
#
# Usage:
#   crates/linux/package.sh
#   sudo bash target/dist/install.sh --archive target/dist/setsuna-ime-*.tar.gz   # try it
#   gh release create v<version> target/dist/*                                     # publish
#
# The archive holds the fcitx5 addon (its module dir compiled in as
# /usr/lib/setsuna-ime), the engine, reranker (with llama.cpp) and rewriter
# daemons, ONNX Runtime, the mixed input model, the settings GUI and
# models.tsv, the conversion model pins of crates/tsf/models.json (install.sh
# downloads the model). Needs what the build needs (cargo, cmake, fcitx5 and
# WebKitGTK 4.1 development files, and for llama.cpp a C/C++ compiler with
# OpenMP), curl and jq.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
build_dir=$repo/target/fcitx5-release
dist=$repo/target/dist

die() {
    echo "package: $*" >&2
    exit 1
}

case $(uname -m) in
    x86_64) arch=x86_64 ;;
    *) die "release builds are x86_64 only" ;;
esac
version=$(sed -n '/^\[workspace.package\]/,/^\[/{s/^version = "\(.*\)"/\1/p}' "$repo/Cargo.toml")
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] || die "no workspace version in Cargo.toml"
name=setsuna-ime-$version-linux-$arch

"$repo/crates/engine/fetch-onnxruntime.sh" --dest "$repo/target/release"
cmake -S "$here/fcitx5" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DSETSUNA_MODULE_DIR=/usr/lib/setsuna-ime \
    -DSETSUNA_BUILD_SETTINGS=ON \
    -DSETSUNA_BUILD_RERANK=ON \
    -DSETSUNA_ORT_LIBRARY="$repo/target/release/libonnxruntime.so" \
    -DSETSUNA_MODEL_MANIFEST=
cmake --build "$build_dir"

stage=$build_dir/stage
rm -rf "$stage" "$dist"
DESTDIR=$stage cmake --install "$build_dir" >/dev/null
out=$dist/$name
mkdir -p "$out/addon" "$out/module/licenses/onnxruntime" "$out/module/licenses/llama.cpp" "$out/bin"
mv "$(find "$stage" -name setsuna.so -path '*/fcitx5/*')" "$out/addon/"
mv "$stage/usr/share" "$out/share"
mv "$stage/usr/lib/setsuna-ime/"* "$out/module/"
mv "$stage/usr/bin/setsuna-settings" "$out/bin/"
strip "$out/addon/setsuna.so" "$out/module/setsuna-engine" "$out/module/setsuna-rewrite" \
    "$out/module/setsuna-rerank" "$out/bin/setsuna-settings"
install -m755 "$here/uninstall.sh" "$out/module/uninstall.sh"
install -m644 "$repo/LICENSE" "$repo/THIRD_PARTY_NOTICES.md" "$out/module/licenses/"
ort=$(basename "$(ls "$repo"/local/onnxruntime/onnxruntime-linux-x64-*.tgz | tail -1)" .tgz)
tar -xzf "$repo/local/onnxruntime/$ort.tgz" -C "$out/module/licenses/onnxruntime" --strip-components=1 \
    "$ort/LICENSE" "$ort/ThirdPartyNotices.txt"
# llama.cpp (MIT), built into the reranker from the llama-cpp-sys-2 crate
sys=$(sed -n '/^name = "llama-cpp-sys-2"$/{n;s/^version = "\(.*\)"$/\1/p}' "$repo/Cargo.lock")
install -m644 "$(ls "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/llama-cpp-sys-2-"$sys"/llama.cpp/LICENSE | head -1)" \
    "$out/module/licenses/llama.cpp/LICENSE"

# models.tsv: "repository" / "revision" lines, then <precision> <path> <size> <sha256>
catalog=$repo/crates/tsf/models.json
{
    jq -r '"repository\t\(.repository)", "revision\t\(.revision)"' "$catalog"
    jq -r '.variants | to_entries[] | .key as $p | .value.files[] | [$p, .path, .size, .sha256] | @tsv' "$catalog"
} >"$out/models.tsv"

tar -C "$dist" --owner=0 --group=0 --numeric-owner --sort=name -czf "$dist/$name.tar.gz" "$name"
rm -rf "$out"
sha=$(sha256sum "$dist/$name.tar.gz" | cut -d' ' -f1)
sed -e "s/^VERSION=@VERSION@$/VERSION=$version/" \
    -e "s/^ARCHIVE_SHA256_X86_64=@ARCHIVE_SHA256_X86_64@$/ARCHIVE_SHA256_X86_64=$sha/" \
    "$here/install.sh" >"$dist/install.sh"
grep -q '@VERSION@\|@ARCHIVE_SHA256' "$dist/install.sh" && die "install.sh placeholders not filled"
(cd "$dist" && sha256sum "$name.tar.gz" install.sh >SHA256SUMS)
ls -l "$dist" >&2
