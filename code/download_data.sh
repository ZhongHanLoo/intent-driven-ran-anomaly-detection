#!/usr/bin/env bash
# Fetch the NCSRD-DS-5GDDoS dataset, version 3 (Zenodo record 13900057,
# licence CC-BY-4.0) into data/ and verify MD5 checksums. Idempotent:
# files that are already present and verified are not re-downloaded.
# Checksums were recorded from the project's verified copy (2026-07-02)
# and match Zenodo's own file records.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
mkdir -p data

BASE="https://zenodo.org/records/13900057/files"
FILES=(
  "amari_ue_data_merged_with_attack_number.csv a50e22bd98ed3779de986469889bfd0f"
  "summary_report.xlsx 87a093cb2794f13fa72559bfb5b14331"
  "README.txt 5df490488f70bbb246e57c5cde3eb2d1"
)

md5_of() {  # macOS ships `md5`, Linux ships `md5sum`
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

for entry in "${FILES[@]}"; do
  f="${entry%% *}"; want="${entry##* }"; path="data/$f"
  if [[ -f "$path" && "$(md5_of "$path")" == "$want" ]]; then
    echo "OK (already present, verified): $f"
    continue
  fi
  echo "downloading $f ..."
  curl -L --fail -o "$path" "$BASE/$f?download=1"
  got="$(md5_of "$path")"
  if [[ "$got" != "$want" ]]; then
    echo "CHECKSUM MISMATCH for $f: got $got, want $want" >&2
    echo "(delete the file and re-run; if it persists, check the Zenodo record)" >&2
    exit 1
  fi
  echo "OK (downloaded + verified): $f"
done
echo "dataset ready in data/ (main CSV ~242 MB: 686,026 rows x 58 columns)"
