#!/usr/bin/env bash
# Offline regression tests of crates/linux/fetch-model.sh: a file:// mirror
# stands in for Hugging Face; no credentials or real HTTP.
# Run: bash tests/model-download.sh
# Generated fixtures/caches are kept under local/model-download-tests-<random>.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fetch=$repo/crates/linux/fetch-model.sh
mkdir -p "$repo/local"
root=$(mktemp -d "$repo/local/model-download-tests-XXXXXXXX")
revision=0123456789abcdef0123456789abcdef01234567
mirror=$root/mirror
served=$mirror/example/converter/resolve/$revision
checks=0

pass() {
    checks=$((checks + 1))
    echo "PASS: $1"
}
assert() {
    local message=$1
    shift
    "$@" || { echo "FAIL: $message" >&2; exit 1; }
    pass "$message"
}
# assert_fails <message> <expected error text> <fetch-model.sh args...>
assert_fails() {
    local message=$1 expected=$2 err
    shift 2
    if err=$(run "$@" 2>&1 >/dev/null); then
        echo "FAIL: $message (succeeded)" >&2
        exit 1
    fi
    [[ $err == *"$expected"* ]] || { echo "FAIL: $message: $err" >&2; exit 1; }
    pass "$message"
}
sha() { sha256sum -- "$1" | cut -d' ' -f1; }
size() { wc -c <"$1" | tr -d ' '; }

# new_fixture <proposal path> [schema]: the mirror's files and a catalog
# pinning them, as catalog.json
new_fixture() {
    local proposal=$1 schema=${2:-neo-ime-ctc-nat-export-v1}
    rm -rf "$mirror" "$root/catalog.json"
    mkdir -p "$served/$(dirname "$proposal")"
    printf 'onnx bytes' >"$served/$proposal"
    printf '00\t<pad>\n' >"$served/vocab.tsv"
    printf 'pairs' >"$served/pairs.bin"
    printf 'words' >"$served/words.tsv"
    printf '{}' >"$served/config.json"
    jq -n --arg s "$schema" --arg p "$proposal" \
        '{schema: $s, proposal: {path: $p, precision: "int8"}, vocab: "vocab.tsv",
          rescore: {pairs: "pairs.bin", words: "words.tsv"}}' \
        >"$served/model.int8.manifest.json"
    local files=()
    for f in model.int8.manifest.json "$proposal" vocab.tsv pairs.bin words.tsv config.json; do
        files+=("$(jq -n --arg p "$f" --argjson n "$(size "$served/$f")" --arg h "$(sha "$served/$f")" \
            '{path: $p, size: $n, sha256: $h}')")
    done
    printf '%s\n' "${files[@]}" | jq -s --arg r "$revision" \
        '{schema: "setsuna-model-download-v1", repository: "example/converter", revision: $r,
          variants: {int8: {manifest: "model.int8.manifest.json", files: .}}}' \
        >"$root/catalog.json"
}
run() {
    "$fetch" --catalog "$root/catalog.json" --cache-dir "$root/cache" --base-url "file://$mirror" "$@"
}
edit_catalog() {
    jq "$1" "$root/catalog.json" >"$root/catalog.tmp" && mv "$root/catalog.tmp" "$root/catalog.json"
}
cached=$root/cache/example/converter/$revision

# A fresh download, nested proposal path included.
new_fixture graphs/model.onnx
out=$(run 2>/dev/null)
assert "prints the manifest path" [ "$out" = "$cached/model.int8.manifest.json" ]
assert "downloads the nested proposal" cmp -s "$served/graphs/model.onnx" "$cached/graphs/model.onnx"
assert "downloads the vocab" cmp -s "$served/vocab.tsv" "$cached/vocab.tsv"
assert "downloads the rescore tables" cmp -s "$served/words.tsv" "$cached/words.tsv"
assert "downloads every pinned file (config.json, the Hub's download count)" cmp -s "$served/config.json" "$cached/config.json"
assert "leaves no staged file" [ -z "$(find "$root/cache" -name '*.part')" ]

# Offline: verified cache files are reused.
mv "$mirror" "$root/offline"
out=$(run 2>&1 >/dev/null)
assert "reuses the verified cache offline" grep -q 'model cache verified: graphs/model.onnx' <<<"$out"
mv "$root/offline" "$mirror"

# A damaged cache file is downloaded again.
printf 'damaged!!!' >"$cached/vocab.tsv"
run >/dev/null 2>&1
assert "replaces a damaged cache file" cmp -s "$served/vocab.tsv" "$cached/vocab.tsv"

# A download that does not match its pin is refused and leaves nothing.
rm -rf "$root/cache"
printf 'tampered!!' >"$served/graphs/model.onnx"
assert_fails "refuses a mismatching download" "size or SHA-256 mismatch: graphs/model.onnx"
assert "keeps no mismatching file" [ ! -e "$cached/graphs/model.onnx" ]
assert "keeps no staged file after a mismatch" [ -z "$(find "$root/cache" -name '*.part')" ]

# The whole set is checked before any payload is downloaded.
new_fixture graphs/model.onnx
rm -rf "$root/cache"
edit_catalog '.variants.int8.files |= map(select(.path != "vocab.tsv"))'
assert_fails "refuses an unpinned artifact" "unpinned file: vocab.tsv"
assert "downloads no payload before the check" [ ! -e "$cached/graphs/model.onnx" ]

# The manifest's schema and precision.
new_fixture model.onnx other-schema-v1
assert_fails "refuses another schema" "unsupported schema or precision"

# Catalog checks.
new_fixture model.onnx
edit_catalog '.revision = "main"'
assert_fails "refuses an unpinned revision" "invalid model catalog"
new_fixture model.onnx
edit_catalog '.variants.int8.files[1].path = "../model.onnx"'
assert_fails "refuses a path leaving the artifact" "invalid artifact path: ../model.onnx"
new_fixture model.onnx
edit_catalog '.variants.int8.files[1].sha256 = "abc"'
assert_fails "refuses a malformed SHA-256" "invalid or duplicate file specification"
new_fixture model.onnx
edit_catalog '.variants.int8.files += [.variants.int8.files[1]]'
assert_fails "refuses a duplicate file" "invalid or duplicate file specification"
new_fixture model.onnx
assert_fails "refuses a missing variant" "no fp32 variant" --precision fp32

echo "all $checks checks passed ($root)"
