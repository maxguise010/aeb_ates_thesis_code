"""One style for all thesis figures: font, sizes, a fixed colour-blind-safe colour per option, and reproducible PNG
and PDF output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH_CM = 16.0                       # thesis text width (no preamble on disk, see README)
WIDTH_IN = TEXT_WIDTH_CM / 2.54
DPI = 300
FONT = "Inter" if any(f.name == "Inter" for f in fm.fontManager.ttflist) else "DejaVu Sans"

# one colour per option, used in every figure (Okabe-Ito, colour-blind safe)
OPT_COLOR = {"A1D": "#0072B2", "A2D": "#56B4E9", "A4D": "#009E73", "BWN110": "#E69F00", "Bstd": "#CC79A7",
             "TTES5k": "#D55E00"}
OPT_LABEL = {"A1D": "Option A 1D", "A2D": "Option A 2D", "A4D": "Option A 4D", "BWN110": "B-WN 110 °C",
             "Bstd": "B standard", "TTES5k": "TTES 5,000 m³"}
OPT_SHORT = {"A1D": "A 1D", "A2D": "A 2D", "A4D": "A 4D", "BWN110": "B WN\n110 °C", "Bstd": "B std\n90 °C",
             "TTES5k": "TTES\n5,000 m³"}
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e3"
DARK, MID, LIGHT = "#4d4d4d", "#8f8f8f", "#c8c8c8"        # neutrals (references, non-option series)
LOWER, RAISE = "#bdbdbd", "#525252"                       # tornado: lowers / raises the NPV (no option colour)

FS, FS_SMALL = 8.0, 7.0
LW, LW_THIN = 1.4, 0.8


def apply():
    plt.rcParams.update({
        "font.family": FONT, "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS_SMALL + 0.5, "ytick.labelsize": FS_SMALL + 0.5, "legend.fontsize": FS_SMALL + 0.5,
        "axes.linewidth": 0.6, "lines.linewidth": LW, "axes.edgecolor": "#9a9a9a", "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.labelcolor": INK, "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.hashsalt": "report",
    })


def figure(height_cm, nrows=1, ncols=1, **kw):
    apply()
    return plt.subplots(nrows, ncols, figsize=(WIDTH_IN, height_cm / 2.54), layout="constrained", **kw)


def despine(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)


def spread_labels(ys, min_gap):
    """y positions for end-of-line labels so that they do not overlap (keeps the order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def save(fig, folder, stem):
    """PNG at 300 dpi with the exact text width (no tight bounding box) and a vector PDF without timestamps."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=DPI, metadata={"Software": None})
    fig.savefig(folder / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                                                   "Creator": None})
    plt.close(fig)
    return [folder / f"{stem}.png", folder / f"{stem}.pdf"]
