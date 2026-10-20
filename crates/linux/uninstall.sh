#!/usr/bin/env bash
# Remove Setsuna IME as install.sh put it. Run as root:
#
#   sudo /usr/lib/setsuna-ime/uninstall.sh
#
# Removes the files listed in installed-files next to this script, the
# directories only Setsuna uses and the model download cache. fcitx5 drops
# Setsuna from its input methods the next time it starts.
set -euo pipefail

die() {
    echo "uninstall: $*" >&2
    exit 1
}

dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Only an installed copy (<root>/usr/lib/setsuna-ime) removes anything; <root>
# is empty for a system install and the staging dir of install.sh --root.
[[ $dir == */usr/lib/setsuna-ime ]] || die "not an installed copy: $dir"
root=${dir%/usr/lib/setsuna-ime}
[[ -f $dir/installed-files ]] || die "$dir/installed-files not found"
[[ $EUID -eq 0 || -n $root ]] || die "run as root: sudo $0"

while IFS= read -r f; do
    [[ $f == /* ]] && rm -f -- "$root$f"
done <"$dir/installed-files"
for d in /usr/share/fcitx5/themes/setsuna /usr/share/fcitx5/themes/setsuna-dark; do
    [[ -d $root$d ]] && rmdir --ignore-fail-on-non-empty -- "$root$d"
done
rm -rf -- "$root/var/cache/setsuna-ime" "$dir"
echo "Setsuna IME removed. Restart fcitx5 (or log in again) to drop it from the input methods." >&2
