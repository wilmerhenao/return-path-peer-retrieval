"""
gen_fig_regime_timeline.py
Generate the 19-transition regime timeline figure.

Uses the final feature-time-only 19-transition census from
results/acceptance_full19_feature_time/run_20260730T032944Z/.

Under the census W3→W4 no longer has TF-IDF leading; only H1→H2 (COVID crash)
remains as the exception (18/19 win). The "noise floor" annotation is removed.

Saves to paper/figures/fig_regime_timeline.pdf and .png
"""

import os
import sys
import json
import hashlib
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure project root is on path so plots.style is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import STYLES, apply_paper_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(BASE, "paper", "figures")
os.makedirs(FIGURES, exist_ok=True)

INPUT_JSON = os.path.join(
    BASE,
    "results",
    "acceptance_full19_feature_time",
    "run_20260730T032944Z",
    "full19_feature_time_census.json",
)
PROV_LOG = os.path.join(FIGURES, "provenance.log")

# ---------------------------------------------------------------------------
# Load census NDCG@5 from CSV  (universe == "extension_19")
# ---------------------------------------------------------------------------
_ndcg = {}
with open(INPUT_JSON) as f:
    payload = json.load(f)
for row in payload["per_transition"]:
    transition = row["transition"].replace("->", "→")
    _ndcg[(transition, "ReturnPath")] = float(row["return_path"]["mean_ndcg"])
    _ndcg[(transition, "TF-IDF")] = float(row["tfidf"]["mean_ndcg"])


def _get(transition, method, fallback=None):
    return _ndcg.get((transition, method), fallback)


# ---------------------------------------------------------------------------
# Transition table  (label, regime, lw, tfidf, minilm)
# Historical windows: MiniLM not available (None → kept as NaN, not plotted)
# ---------------------------------------------------------------------------
TRANSITIONS = [
    ("H1→H2",  "HIGH_VOL", _get("H1→H2",  "ReturnPath"), _get("H1→H2",  "TF-IDF"), _get("H1→H2",  "MiniLM")),
    ("H2→H3",  "HIGH_VOL", _get("H2→H3",  "ReturnPath"), _get("H2→H3",  "TF-IDF"), _get("H2→H3",  "MiniLM")),
    ("H3→H4",  "HIGH_VOL", _get("H3→H4",  "ReturnPath"), _get("H3→H4",  "TF-IDF"), _get("H3→H4",  "MiniLM")),
    ("H4→H5",  "MED_VOL",  _get("H4→H5",  "ReturnPath"), _get("H4→H5",  "TF-IDF"), _get("H4→H5",  "MiniLM")),
    ("H5→H6",  "LOW_VOL",  _get("H5→H6",  "ReturnPath"), _get("H5→H6",  "TF-IDF"), _get("H5→H6",  "MiniLM")),
    ("H6→H7",  "MED_VOL",  _get("H6→H7",  "ReturnPath"), _get("H6→H7",  "TF-IDF"), _get("H6→H7",  "MiniLM")),
    ("H7→H8",  "HIGH_VOL", _get("H7→H8",  "ReturnPath"), _get("H7→H8",  "TF-IDF"), _get("H7→H8",  "MiniLM")),
    ("H8→H9",  "HIGH_VOL", _get("H8→H9",  "ReturnPath"), _get("H8→H9",  "TF-IDF"), _get("H8→H9",  "MiniLM")),
    ("H9→H10", "HIGH_VOL", _get("H9→H10", "ReturnPath"), _get("H9→H10", "TF-IDF"), _get("H9→H10", "MiniLM")),
    ("H10→H11","MED_VOL",  _get("H10→H11","ReturnPath"), _get("H10→H11","TF-IDF"), _get("H10→H11","MiniLM")),
    ("H11→H12","LOW_VOL",  _get("H11→H12","ReturnPath"), _get("H11→H12","TF-IDF"), _get("H11→H12","MiniLM")),
    ("H12→H13","LOW_VOL",  _get("H12→H13","ReturnPath"), _get("H12→H13","TF-IDF"), _get("H12→H13","MiniLM")),
    ("H13→W1", "LOW_VOL",  _get("H13→W1", "ReturnPath"), _get("H13→W1", "TF-IDF"), _get("H13→W1", "MiniLM")),
    ("W1→W2",  "MED_VOL",  _get("W1→W2",  "ReturnPath"), _get("W1→W2",  "TF-IDF"), _get("W1→W2",  "MiniLM")),
    ("W2→W3",  "MED_VOL",  _get("W2→W3",  "ReturnPath"), _get("W2→W3",  "TF-IDF"), _get("W2→W3",  "MiniLM")),
    ("W3→W4",  "MED_VOL",  _get("W3→W4",  "ReturnPath"), _get("W3→W4",  "TF-IDF"), _get("W3→W4",  "MiniLM")),
    ("W4→W5",  "HIGH_VOL", _get("W4→W5",  "ReturnPath"), _get("W4→W5",  "TF-IDF"), _get("W4→W5",  "MiniLM")),
    ("W5→W6",  "LOW_VOL",  _get("W5→W6",  "ReturnPath"), _get("W5→W6",  "TF-IDF"), _get("W5→W6",  "MiniLM")),
    ("W6→W7",  "LOW_VOL",  _get("W6→W7",  "ReturnPath"), _get("W6→W7",  "TF-IDF"), _get("W6→W7",  "MiniLM")),
]

TRANS_LABELS = [t[0] for t in TRANSITIONS]
REGIMES      = [t[1] for t in TRANSITIONS]
lw_vals      = [t[2] for t in TRANSITIONS]
tfidf_vals   = [t[3] for t in TRANSITIONS]
minilm_vals  = [t[4] for t in TRANSITIONS]

N = len(TRANSITIONS)
xs = list(range(N))

# ---------------------------------------------------------------------------
# Apply shared style
# ---------------------------------------------------------------------------
apply_paper_style()

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))

# --- Regime shading (axvspan, full y-range) --------------------------------
for i, regime in enumerate(REGIMES):
    if regime == "HIGH_VOL":
        ax.axvspan(i - 0.5, i + 0.5, color="0.0", alpha=0.12, linewidth=0, zorder=0)
    elif regime == "LOW_VOL":
        ax.axvspan(i - 0.5, i + 0.5, color="0.0", alpha=0.07, linewidth=0, zorder=0)
    # MED_VOL: unshaded

# --- Vertical divider between index 12 (H13→W1) and 13 (W1→W2) -----------
ax.axvline(x=12.5, color="0.4", linestyle="--", linewidth=0.7, zorder=1)

# --- Era labels ---
ax.text(6.0, 0.46, "← Historical", ha="center", va="bottom",
        fontsize=6, color="0.35", style="italic")
ax.text(15.5, 0.46, "Modern →", ha="center", va="bottom",
        fontsize=6, color="0.35", style="italic")

# --- Line series ---
lw_color, lw_ls, lw_mk, _lw_lbl_unused = STYLES["LW"]
ax.plot(xs, lw_vals,
        color=lw_color, linestyle=lw_ls, marker=lw_mk,
        label="Return path", zorder=3)

tfidf_color, tfidf_ls, tfidf_mk, tfidf_lbl = STYLES["TF-IDF"]
ax.plot(xs, tfidf_vals,
        color=tfidf_color, linestyle=tfidf_ls, marker=tfidf_mk,
        label=tfidf_lbl, zorder=3)

# --- Annotation: TF-IDF > LW at index 0 (H1→H2, COVID crash) -------------
# Under census: only this exception remains (18/19 wins for Return path).
ax.annotate(
    "COVID exception:\nTF-IDF leads",
    xy=(0, tfidf_vals[0]),
    xytext=(2.3, 0.455),
    fontsize=5.5,
    color=tfidf_color,
    arrowprops=dict(arrowstyle="-|>", color=tfidf_color,
                    lw=0.7, shrinkA=0, shrinkB=2),
    ha="center",
    va="top",
    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
              edgecolor="none", alpha=0.85),
    zorder=5,
)
# NOTE: "noise floor" annotation (W3→W4) removed — census shows RP wins there.

# --- Axes decoration -------------------------------------------------------
ax.set_xticks(xs)
ax.set_xticklabels(TRANS_LABELS, rotation=45, ha="right", fontsize=6)
ax.set_ylim(0.15, 0.50)
ax.set_ylabel("WF NDCG@5")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.grid(axis="y", color="0.87", linewidth=0.35, zorder=0)
ax.set_axisbelow(True)

ax.legend(loc="upper right", ncol=1, handlelength=2.0)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_pdf = os.path.join(FIGURES, "fig_regime_timeline.pdf")
out_png = os.path.join(FIGURES, "fig_regime_timeline.png")

fig.subplots_adjust(left=0.14, bottom=0.22)
plt.tight_layout()
fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
fig.savefig(out_png, bbox_inches="tight", dpi=150, format="png")
plt.close(fig)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
input_sha256 = hashlib.sha256(open(INPUT_JSON, "rb").read()).hexdigest()
ts = datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
n_wins = sum(1 for lbl, reg, lw, tfidf, mini in TRANSITIONS if lw is not None and tfidf is not None and lw > tfidf)
prov_lines = [
    f"--- gen_fig_regime_timeline.py  {ts} ---",
    f"  INPUT  sha256={input_sha256}  results/acceptance_full19_feature_time/run_20260730T032944Z/full19_feature_time_census.json",
    "  OUTPUT paper/figures/fig_regime_timeline.pdf",
    "  OUTPUT paper/figures/fig_regime_timeline.png",
    f"  notes  feature-time-only census; frozen config-b TF-IDF; RP wins {n_wins}/19; only H1->H2 exception",
]
with open(PROV_LOG, "a") as pf:
    pf.write("\n".join(prov_lines) + "\n")
print(f"Provenance written to: {PROV_LOG}")

# ---------------------------------------------------------------------------
# Data summary table
# ---------------------------------------------------------------------------
print("\n--- Regime timeline data summary ---")
print(f"{'#':<3} {'Transition':<12} {'Regime':<10} {'LW':>6} {'TF-IDF':>8} {'MiniLM':>8}")
print("-" * 52)
for i, (lbl, reg, lw, tfidf, mini) in enumerate(TRANSITIONS):
    mini_str = f"{mini:>8.4f}" if mini is not None else "     N/A"
    print(f"{i:<3} {lbl:<12} {reg:<10} {lw:>6.4f} {tfidf:>8.4f} {mini_str}")
