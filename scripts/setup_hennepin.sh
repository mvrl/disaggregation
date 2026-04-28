#!/usr/bin/env bash
# Unpacks the Hennepin Box archives into the layout aggregation/ expects.
#
# Reads BOX_DIR and HENNEPIN_DATA_ROOT from <repo>/paths.env (if present) and
# from your shell. Real shell-exported env vars always win.
#
# Each archive roots at its own top-level directory inside the zip:
#   1m_302px.zip                 -> $HENNEPIN_DATA_ROOT/1m_302px/{imgs,masks,vals.pkl}
#   1m_302px_region_combined.zip -> $HENNEPIN_DATA_ROOT/1m_302px_region_combined/{...}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source paths.env if it exists (KEY=VALUE lines). Already-exported env vars win.
if [ -f "$REPO_ROOT/paths.env" ]; then
    while IFS='=' read -r k v; do
        # strip leading/trailing whitespace, skip comments and blanks
        k="${k%%#*}"; k="${k%"${k##*[![:space:]]}"}"; k="${k#"${k%%[![:space:]]*}"}"
        [ -z "$k" ] && continue
        v="${v%%#*}"; v="${v%"${v##*[![:space:]]}"}"; v="${v#"${v%%[![:space:]]*}"}"
        [ -z "$v" ] && continue
        # only set if not already in env
        if [ -z "${!k:-}" ]; then
            export "$k=$v"
        fi
    done < "$REPO_ROOT/paths.env"
fi

BOX_DIR="${BOX_DIR:-$HOME/Library/CloudStorage/Box-Box/datasets/Hennepin_dataset}"
DATA_ROOT="${HENNEPIN_DATA_ROOT:-$REPO_ROOT/data/hennepin}"

ZIP_A="$BOX_DIR/1m_302px.zip"
ZIP_B="$BOX_DIR/1m_302px_region_combined.zip"

file_size() {
    if stat -f%z "$1" >/dev/null 2>&1; then
        stat -f%z "$1"
    else
        stat -c%s "$1"
    fi
}

for z in "$ZIP_A" "$ZIP_B"; do
    if [ ! -e "$z" ]; then
        echo "ERROR: missing zip: $z" >&2
        echo "       Set BOX_DIR (in paths.env or your shell) to the directory" >&2
        echo "       containing the Hennepin archives." >&2
        exit 1
    fi
    sz=$(file_size "$z")
    if [ "$sz" -lt 1000000 ]; then
        echo "ERROR: $z is $sz bytes — looks like a Box cloud-only placeholder." >&2
        echo "       In Finder, right-click the file, choose 'Download Now'," >&2
        echo "       wait for it to fully sync, then re-run this script." >&2
        exit 1
    fi
done

mkdir -p "$DATA_ROOT"
unzip -n "$ZIP_A" -d "$DATA_ROOT"
unzip -n "$ZIP_B" -d "$DATA_ROOT"

for sub in 1m_302px 1m_302px_region_combined; do
    [ -d "$DATA_ROOT/$sub/imgs" ]     || { echo "ERROR: missing imgs in $sub" >&2; exit 1; }
    [ -d "$DATA_ROOT/$sub/masks" ]    || { echo "ERROR: missing masks in $sub" >&2; exit 1; }
    [ -f "$DATA_ROOT/$sub/vals.pkl" ] || { echo "ERROR: missing vals.pkl in $sub" >&2; exit 1; }
done

echo "OK — Hennepin aggregation data unpacked to: $DATA_ROOT"
echo "If HENNEPIN_DATA_ROOT is empty in paths.env, set it to:  $DATA_ROOT"
