"""Phase 0 EDA for NCSRD-DS-5GDDoS (Zenodo record 13900057).

Answers, from the raw record itself (not from papers):
  1. What does the attack schedule (summary_report.xlsx) actually contain?
  2. Shape/columns/dtypes of the merged per-UE file; which label column exists?
  3. Time span, per-UE sampling cadence.
  4. Class balance overall and per attack; per-UE benign/malicious split.
  5. Label audit: do labelled attack windows match the xlsx schedule?
  6. Which of Paper 1's 14 features exist here, and under what names?
  7. Normal-vs-attack summary stats for those features (Paper 1 Table V spirit).
  8. Figures: bitrate timelines with attack shading; per-class distributions.

Run:  .venv/bin/python code/eda/eda_overview.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

CSV = DATA / "amari_ue_data_merged_with_attack_number.csv"
XLSX = DATA / "summary_report.xlsx"

# The 14 KPM features used by Paper 1 (AI-on-RAN, TNSE 2026), per its Fig. 2.
PAPER1_FEATURES = [
    "epre", "pusch_snr", "p_ue", "ul_mcs", "cqi", "ul_bitrate", "dl_mcs",
    "dl_retx", "ul_tx", "dl_tx", "ul_retx", "dl_bitrate", "dl_err", "ul_err",
]


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------ 1. xlsx
hr("1. summary_report.xlsx — attack schedule as shipped")
xl = pd.ExcelFile(XLSX)
print("sheets:", xl.sheet_names)
sheets = {}
for name in xl.sheet_names:
    df = xl.parse(name)
    sheets[name] = df
    print(f"\n--- sheet '{name}' ({df.shape[0]} rows x {df.shape[1]} cols)")
    print(df.to_string(max_rows=40))

# ------------------------------------------------------------------ 2. CSV
hr("2. merged per-UE file — shape, columns, label column")
df = pd.read_csv(CSV, low_memory=False)
print("shape:", df.shape)
print("columns:", list(df.columns))
label_col = next(
    (c for c in ["attack_number", "attack", "label", "malicious"] if c in df.columns),
    None,
)
print("label column detected:", label_col)
if "attack" in df.columns and label_col != "attack":
    print("also present: 'attack' column, values:", df["attack"].value_counts(dropna=False).to_dict())

# ------------------------------------------------------------------ 3. time
hr("3. timestamps & sampling cadence")
ts_col = next(c for c in ["timestamp", "_time", "time"] if c in df.columns)
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True, format="mixed")
print("timestamp NaT after parse:", int(df[ts_col].isna().sum()))
print("span:", df[ts_col].min(), "->", df[ts_col].max(), f"({df[ts_col].max() - df[ts_col].min()})")
df = df.sort_values([ "imeisv", ts_col ]).reset_index(drop=True)
gaps = df.groupby("imeisv")[ts_col].diff().dt.total_seconds()
print("per-record time delta (s): median", np.nanmedian(gaps), "| p05", np.nanpercentile(gaps.dropna(), 5), "| p95", np.nanpercentile(gaps.dropna(), 95))
print("distinct days:", sorted(df[ts_col].dropna().dt.date.unique()))
n_nat = int(df[ts_col].isna().sum())
df = df.dropna(subset=[ts_col]).reset_index(drop=True)
print(f"dropped {n_nat} rows with unparseable timestamps ({n_nat / (len(df) + n_nat) * 100:.3f}%)")

# ------------------------------------------------------------------ 4. class balance
hr("4. class balance")
df["imeisv"] = df["imeisv"].astype(str)
print("rows per UE (imeisv):")
print(df["imeisv"].value_counts().to_string())
if label_col:
    vc = df[label_col].value_counts(dropna=False).sort_index()
    print(f"\n{label_col} value counts:\n{vc.to_string()}")
    benign_mask = (df[label_col] == 0) | df[label_col].isna()
    mal = int((~benign_mask).sum())
    print(f"\nmalicious rows: {mal:,} / {len(df):,} = {mal / len(df) * 100:.2f}%")
    print("\nper-UE malicious share (%):")
    print((df.groupby("imeisv")[label_col].apply(lambda s: ((s != 0) & s.notna()).mean() * 100)).round(2).to_string())

# ------------------------------------------------------------------ 5. label audit
hr("5. label audit — three label columns and schedule cross-check")
for lc in ["attack", "malicious", "attack_number"]:
    if lc in df.columns:
        print(f"\n'{lc}' value counts:\n{df[lc].value_counts(dropna=False).sort_index().to_string()}")
if "attack" in df.columns and "attack_number" in df.columns:
    print("\ncrosstab attack x attack_number:")
    print(pd.crosstab(df["attack"], df["attack_number"], dropna=False).to_string())
if "malicious" in df.columns and "attack" in df.columns:
    print("\ncrosstab malicious x attack:")
    print(pd.crosstab(df["malicious"], df["attack"], dropna=False).to_string())
print("\nper attack_number: rows, time window, UEs involved:")
if label_col:
    for k, g in df[~benign_mask].groupby(label_col):
        print(
            f"{label_col}={k}: {len(g):,} rows | {g[ts_col].min()} -> {g[ts_col].max()} | "
            f"UEs: {sorted(g['imeisv'].unique())}"
        )
    print("\nper 'attack'==1 (if different): rows per UE:")
    if "attack" in df.columns:
        a1 = df[df["attack"] == 1]
        print(a1.groupby("imeisv").size().to_string())
        print("attack==1 time windows per attack_number:")
        print(a1.groupby("attack_number")[ts_col].agg(["min", "max", "size"]).to_string())

# --------------------------------------------------------- 5b. derived labels
hr("5b. DERIVED per-row ground truth (attacker-active rows only)")
# From summary_report.xlsx: IP -> IMEISV for the 5 malicious-role devices, and
# which devices actually ran each attack (attacks 1-4: the two Raspberry Pis;
# attack 5, GTP-U: all five). The merged file's own 'attack'/'attack_number'
# columns flag the whole time window for ALL 9 UEs, so they alone over-label
# (innocent phones inherit attack=1 during windows).
ip_to_imeisv = {
    "10.20.10.2": "8642840401612300", "10.20.10.4": "8642840401624200",
    "10.20.10.6": "8642840401594200", "10.20.10.8": "8677660403123800",
    "10.20.10.10": "3557821101183501",
}
attackers_per_attack = {
    1: ["10.20.10.2", "10.20.10.4"], 2: ["10.20.10.2", "10.20.10.4"],
    3: ["10.20.10.2", "10.20.10.4"], 4: ["10.20.10.2", "10.20.10.4"],
    5: ["10.20.10.2", "10.20.10.4", "10.20.10.6", "10.20.10.8", "10.20.10.10"],
}
for ip, imei in ip_to_imeisv.items():
    modal = df.loc[df["imeisv"] == imei, "bearer_0_ip"].mode()
    got = modal.iloc[0] if len(modal) else "n/a"
    print(f"  {imei} expected {ip:12s} | bearer_0_ip mode: {got:12s} | {'OK' if got == ip else 'MISMATCH'}")
y = pd.Series(0, index=df.index)
for k, ips in attackers_per_attack.items():
    imeis = [ip_to_imeisv[i] for i in ips]
    y[(df["attack_number"] == k) & (df["imeisv"].isin(imeis))] = k
df["y_attack"] = y
print("\nderived y_attack counts (0 = benign, 1-5 = attacker-active rows):")
print(df["y_attack"].value_counts().sort_index().to_string())
print(
    f"derived malicious rows: {(y > 0).sum():,} ({(y > 0).mean() * 100:.2f}%)\n"
    "Paper 1 reported: 11,456 total (1.7%) = SYN 1,402 / ICMP 3,756 / UDP 1,402 / DNS 1,399 / GTP-U 3,497"
)
benign_mask = df["y_attack"] == 0  # stats & figures below use the DERIVED label

# ------------------------------------------------------------------ 6. feature mapping
hr("6. Paper 1's 14 features — presence in this file")
cols = set(df.columns)
mapping = {}
for f in PAPER1_FEATURES:
    candidates = [c for c in cols if c == f or c.endswith("_" + f) or (("cell" in c) and c.split("cell_x_")[-1] == f)]
    # prefer exact, then cell_X_ prefixed
    exact = [c for c in candidates if c == f]
    cellx = [c for c in candidates if c.lower().startswith("cell_x_")]
    chosen = exact[0] if exact else (cellx[0] if cellx else (candidates[0] if candidates else None))
    mapping[f] = chosen
    print(f"  {f:12s} -> {chosen}")
missing = [f for f, c in mapping.items() if c is None]
print("MISSING here:", missing if missing else "none")

# ------------------------------------------------------------------ 7. stats table
hr("7. normal vs attack stats (found features)")
found = {f: c for f, c in mapping.items() if c}
stats = []
for f, c in found.items():
    x = pd.to_numeric(df[c], errors="coerce")
    stats.append({
        "feature": f, "column": c,
        "normal_mean": x[benign_mask].mean(), "normal_std": x[benign_mask].std(),
        "attack_mean": x[~benign_mask].mean(), "attack_std": x[~benign_mask].std(),
        "nan_%": x.isna().mean() * 100,
    })
sdf = pd.DataFrame(stats).round(4)
print(sdf.to_string(index=False))
sdf.to_csv(FIGS.parent / "normal_vs_attack_stats.csv", index=False)

# ------------------------------------------------------------------ 8. figures
hr("8. figures")
# (a) per-UE uplink bitrate timeline with attack rows highlighted
ul_col = found.get("ul_bitrate", "ul_bitrate")
fig, axes = plt.subplots(len(df["imeisv"].unique()), 1, figsize=(14, 1.6 * df["imeisv"].nunique()), sharex=True)
for ax, (ue, g) in zip(np.atleast_1d(axes), df.groupby("imeisv")):
    g2 = g.iloc[:: max(1, len(g) // 4000)]  # downsample for plotting
    x = pd.to_numeric(g2[ul_col], errors="coerce")
    ax.plot(g2[ts_col], x, lw=0.3, color="tab:blue")
    bad = g2[g2["y_attack"] > 0]
    ax.scatter(bad[ts_col], pd.to_numeric(bad[ul_col], errors="coerce"), s=1, color="red")
    ax.set_ylabel(str(ue)[-6:], fontsize=7)
fig.suptitle("Uplink bitrate per UE (red = labelled attack rows)")
fig.savefig(FIGS / "timeline_ul_bitrate_per_ue.png", dpi=130, bbox_inches="tight")
print("saved", FIGS / "timeline_ul_bitrate_per_ue.png")

# (b) class-conditional distributions for two key features
for f in ["dl_bitrate", "ul_retx"]:
    c = found.get(f)
    if not c:
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, lab, col in [(benign_mask, "normal", "tab:blue"), (~benign_mask, "attack", "tab:red")]:
        vals = x[m].dropna()
        q99 = vals.quantile(0.99)
        ax.hist(vals.clip(upper=q99), bins=80, alpha=0.55, density=True, label=lab, color=col)
    ax.set_title(f"{f} — normal vs attack (clipped at p99)")
    ax.legend()
    fig.savefig(FIGS / f"dist_{f}.png", dpi=130, bbox_inches="tight")
    print("saved", FIGS / f"dist_{f}.png")

hr("DONE")
