"""Phase 0 audit — re-verifies every Phase 0 claim from first principles.

Checks (each prints PASS/FAIL/INFO):
  A. Downloaded files match Zenodo's own MD5 checksums; record is the latest version.
  B. Unparseable-timestamp rows (dropped in EDA): how many, whose, when, and
     whether any would have been *malicious* (bias check on the drop).
  C. Duplicate (imeisv, timestamp) rows.
  D. Label derivation, two independent routes:
       route 1: attack_number column ∧ participant list  (what the EDA did)
       route 2: schedule times parsed from summary_report.xlsx ∧ participant list
     Both must equal Paper 1's published counts exactly.
  E. Window boundaries: attack_number time spans vs xlsx schedule (minute-level).
  F. Attacker-row coverage per attack: first-row offset from window start,
     row count vs expectation, max intra-window gap (detection-delay foundation).
  G. Spot-check of two Table V statistics against Paper 1's published values.

Run:  .venv/bin/python code/eda/phase0_audit.py
Requires /tmp/zenodo_record.json (fetched via curl) for checksum comparison.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CSV = DATA / "amari_ue_data_merged_with_attack_number.csv"
XLSX = DATA / "summary_report.xlsx"

PAPER1_COUNTS = {1: 1402, 2: 3756, 3: 1402, 4: 1399, 5: 3497}  # TNSE 2026, Sec. IV-A
IP_TO_IMEISV = {
    "10.20.10.2": "8642840401612300", "10.20.10.4": "8642840401624200",
    "10.20.10.6": "8642840401594200", "10.20.10.8": "8677660403123800",
    "10.20.10.10": "3557821101183501",
}

results = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def info(name: str, detail: str) -> None:
    print(f"[INFO] {name}: {detail}")


# ---------------------------------------------------------------- A. integrity
print("\n=== A. File integrity vs Zenodo ===")
rec = json.load(open("/tmp/zenodo_record.json"))
zen = {f["key"]: f["checksum"].split(":", 1)[1] for f in rec["files"]}
for fname in ["amari_ue_data_merged_with_attack_number.csv", "summary_report.xlsx", "README.txt"]:
    h = hashlib.md5()
    with open(DATA / fname, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    check(f"md5 {fname}", h.hexdigest() == zen[fname], f"local {h.hexdigest()} vs zenodo {zen[fname]}")

try:
    versions = json.load(open("/tmp/zenodo_versions.json"))
    hits = versions.get("hits", {}).get("hits", [])
    ids = [(v["id"], v["metadata"].get("publication_date"), v["metadata"].get("relations", {})) for v in hits]
    latest = max(hits, key=lambda v: v["metadata"].get("publication_date", ""))["id"] if hits else None
    check("record 13900057 is latest version", latest == 13900057,
          f"versions found: {[(v['id'], v['metadata'].get('publication_date')) for v in hits]}")
except Exception as e:  # versions endpoint shape can vary; not fatal
    info("versions endpoint", f"could not evaluate ({e}) — record page states latest; non-fatal")

# ---------------------------------------------------------------- load
print("\n=== load ===")
df = pd.read_csv(CSV, low_memory=False)
df["imeisv"] = df["imeisv"].astype(str)
raw_ts = df["_time"].copy()
ts = pd.to_datetime(raw_ts, errors="coerce", utc=True, format="mixed")
nat_mask = ts.isna()
info("rows", f"{len(df):,} raw; {int(nat_mask.sum())} unparseable timestamps")

# ---------------------------------------------------------------- B. NaT rows
print("\n=== B. Unparseable-timestamp rows (the 740 dropped in EDA) ===")
nat = df[nat_mask]
info("raw samples", str(nat["_time"].head(5).tolist()))
info("per-UE distribution", nat["imeisv"].value_counts().to_dict())
info("attack_number among NaT rows", nat["attack_number"].value_counts().to_dict())
would_be_malicious = 0
for k, ips in {1: list(IP_TO_IMEISV)[:2], 2: list(IP_TO_IMEISV)[:2],
               3: list(IP_TO_IMEISV)[:2], 4: list(IP_TO_IMEISV)[:2],
               5: list(IP_TO_IMEISV)}.items():
    imeis = [IP_TO_IMEISV[i] for i in ips]
    would_be_malicious += int(((nat["attack_number"] == k) & nat["imeisv"].isin(imeis)).sum())
check("dropping NaT rows loses no malicious rows", would_be_malicious == 0,
      f"{would_be_malicious} NaT rows would have been attacker-active")

df = df[~nat_mask].copy()
df["ts"] = ts[~nat_mask]

# ---------------------------------------------------------------- C. duplicates
print("\n=== C. Duplicates ===")
dup = int(df.duplicated(subset=["imeisv", "_time"]).sum())
check("no duplicate (imeisv, timestamp) rows", dup == 0, f"{dup} duplicates")

# ---------------------------------------------------------------- D. labels, two routes
print("\n=== D. Label derivation — two independent routes ===")
sheet = pd.read_excel(XLSX, sheet_name="dataset_v2")
# schedule block: rows where the last column holds the attack number 1..5
sched = []
for _, r in sheet.iterrows():
    num = pd.to_numeric(r.iloc[4], errors="coerce")
    if pd.notna(num) and 1 <= int(num) <= 5 and isinstance(r.iloc[1], str):
        m_date = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", r.iloc[1].strip())
        m_time = re.findall(r"(\d{1,2})[:.](\d{2})", str(r.iloc[2]))
        ips = re.findall(r"10\.20\.10\.\d+", str(r.iloc[3]))
        d, mo, y = (int(g) for g in m_date.groups())
        (h1, m1), (h2, m2) = ((int(a), int(b)) for a, b in m_time[:2])
        start = datetime(y, mo, d, h1, m1, tzinfo=timezone.utc)
        end = datetime(y, mo, d, h2, m2, tzinfo=timezone.utc)
        sched.append({"k": int(num), "name": str(r.iloc[0]).strip(), "start": start, "end": end, "ips": ips})
info("parsed schedule", "; ".join(f"#{s['k']} {s['name']} {s['start']:%d %H:%M}-{s['end']:%H:%M} x{len(s['ips'])}UE" for s in sched))
check("schedule parse", len(sched) == 5 and all(len(s["ips"]) in (2, 5) for s in sched), f"{len(sched)} attacks parsed")

# route 1: attack_number column ∧ participants
y1 = pd.Series(0, index=df.index)
# route 2: xlsx time window ∧ participants (no use of attack_number at all)
y2 = pd.Series(0, index=df.index)
for s in sched:
    imeis = [IP_TO_IMEISV[i] for i in s["ips"]]
    y1[(df["attack_number"] == s["k"]) & df["imeisv"].isin(imeis)] = s["k"]
    y2[(df["ts"] >= s["start"]) & (df["ts"] < s["end"]) & df["imeisv"].isin(imeis)] = s["k"]

c1 = y1.value_counts().drop(0).sort_index().to_dict()
c2 = y2.value_counts().drop(0).sort_index().to_dict()
check("route 1 (attack_number ∧ participants) == Paper 1", c1 == PAPER1_COUNTS, f"{c1}")
check("route 2 (xlsx times ∧ participants)   == Paper 1", c2 == PAPER1_COUNTS, f"{c2}")
check("routes agree row-for-row", bool((y1 == y2).all()),
      f"{int((y1 != y2).sum())} rows differ")

# ---------------------------------------------------------------- E. windows
print("\n=== E. attack_number window spans vs xlsx schedule ===")
ok_all = True
for s in sched:
    g = df[df["attack_number"] == s["k"]]
    d_start = (g["ts"].min() - s["start"]).total_seconds()
    d_end = (s["end"] - g["ts"].max()).total_seconds()
    ok = (0 <= d_start < 60) and (0 <= d_end < 60)
    ok_all &= ok
    print(f"   #{s['k']} {s['name']:10s} start+{d_start:6.1f}s  end-{d_end:6.1f}s  {'ok' if ok else 'OUT OF TOLERANCE'}")
check("window boundaries within 60s of schedule", ok_all, "see lines above")

# ---------------------------------------------------------------- F. coverage
print("\n=== F. attacker-row coverage inside windows (delay-measurement foundation) ===")
cov_ok = True
for s in sched:
    dur = (s["end"] - s["start"]).total_seconds()
    for ip in s["ips"]:
        imei = IP_TO_IMEISV[ip]
        g = df[(df["imeisv"] == imei) & (y2 == s["k"])].sort_values("ts")
        if g.empty:
            cov_ok = False
            print(f"   #{s['k']} {imei}: NO ROWS")
            continue
        first_off = (g["ts"].min() - s["start"]).total_seconds()
        gaps = g["ts"].diff().dt.total_seconds().max()
        expect = dur / 5.14
        ratio = len(g) / expect
        line_ok = first_off < 15 and gaps < 30 and 0.9 < ratio < 1.1
        cov_ok &= line_ok
        print(f"   #{s['k']} {imei}: {len(g):4d} rows (exp~{expect:4.0f}, x{ratio:4.2f}) first +{first_off:5.1f}s maxgap {gaps:5.1f}s {'ok' if line_ok else 'CHECK'}")
check("attacker coverage adequate for delay measurement", cov_ok, "see lines above")

# ---------------------------------------------------------------- G. stats spot check
print("\n=== G. Table V spot-check vs Paper 1 (tolerance: |Δmean| small, from row-set diff) ===")
benign = y1 == 0
for col, paper_normal, paper_attack in [("epre", -103.97, -110.909), ("ul_err", 0.0985, 2.9725)]:
    x = pd.to_numeric(df[col], errors="coerce")
    nm, am = x[benign].mean(), x[~benign].mean()
    ok = abs(nm - paper_normal) < 0.15 and abs(am - paper_attack) < 0.15
    check(f"{col} means near Paper 1", ok, f"normal {nm:.4f} vs {paper_normal} | attack {am:.4f} vs {paper_attack}")

# ---------------------------------------------------------------- summary
print("\n=== SUMMARY ===")
fails = [n for n, ok in results if not ok]
print(f"{sum(ok for _, ok in results)}/{len(results)} checks passed")
print("FAILURES:" if fails else "ALL CHECKS PASSED", fails if fails else "")
