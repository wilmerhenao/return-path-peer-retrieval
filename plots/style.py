"""
plots/style.py
Shared plotting style for ICAIF 2026 paper figures.

Okabe-Ito colorblind-safe palette.
Fixed color + linestyle + marker per method family.
"""

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Method style registry
# Tuples: (color, linestyle, marker, label)
# Two plain-string entries at the end are accent colors for pipeline diagrams.
# ---------------------------------------------------------------------------
STYLES = {
    "LW":         ("#000000", "-",   "o", "LW / Raw-$z$"),
    "TF-IDF":     ("#0072B2", "--",  "s", "TF-IDF (vintage)"),
    "MiniLM":     ("#E69F00", ":",   "D", "MiniLM (vintage)"),
    "Blend":      ("#009E73", "-.",  "^", "No-lookahead blend"),
    "LOTO":       ("#555555", ":",   "x", "LOTO benchmark (atemporal)"),
    "MLP":        ("#888888", "-",   "s", "MLP"),
    "SEC-BERT":   ("#888888", "--",  "D", "SEC-BERT"),
    "Random":     ("#AAAAAA", ":",   "",  "Random"),
    # Accent colors for pipeline / flow diagrams (plain strings, not tuples)
    "text_route":   "#0072B2",
    "return_route": "#000000",
}

# ---------------------------------------------------------------------------
# rcParams for paper-quality figures
# ---------------------------------------------------------------------------
_RCPARAMS = {
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":           8,
    "axes.titlesize":      8,
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "figure.dpi":          150,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "lines.linewidth":     1.1,
    "lines.markersize":    4.0,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "0.6",
    "legend.borderpad":    0.3,
    "legend.labelspacing": 0.2,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
}


def apply_paper_style() -> None:
    """Apply paper-quality rcParams to the current matplotlib session."""
    plt.rcParams.update(_RCPARAMS)


def get_style(method: str) -> tuple:
    """
    Return (color, linestyle, marker, label) for a known method key.

    For the two accent-color keys ("text_route", "return_route") a plain
    string is stored; this function raises KeyError for those — use
    STYLES[key] directly for pipeline diagram accents.

    Raises KeyError if the method is not registered.
    """
    entry = STYLES[method]
    if not isinstance(entry, tuple):
        raise KeyError(
            f"'{method}' is an accent-color key, not a method style tuple. "
            f"Use STYLES['{method}'] directly."
        )
    return entry
