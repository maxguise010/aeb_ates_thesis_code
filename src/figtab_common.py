"""Shared helpers for the figure and table scripts: thesis figure layout, writing tables, and recording new numbers in
the locked-numbers files."""
import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

TABLES = C.RESULTS / "tables"
FIGDATA = C.RESULTS / "figures_data"
for _d in (TABLES, FIGDATA, C.FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

CHAPTER = "Figures and tables (repository scripts)"
SECTION = "## Figures and tables (repository scripts)"

# colours of the printed thesis figures
BLUE, ORANGE, GREEN, YELLOW, PINK, DKGREEN = "#2a78d6", "#eb6834", "#1baf7a", "#e8a200", "#e87ba4", "#0a7d0a"
GREY, LIGHTORANGE, INK, MUTED, GRID = "#8c8c88", "#f5c0a5", "#0b0b0b", "#52514e", "#e6e5e1"
NAVY, MIDBLUE, LIGHTBLUE = "#123e7a", "#3a7fd5", "#a3c6ee"
RUST, STEEL = "#c4622d", "#2f6fa6"
LABEL = {"TTES5k": "TTES\n5,000 m³", "A1D": "A 1D", "A2D": "A 2D", "A4D": "A 4D", "BWN110": "B WN\n110 °C",
         "Bstd": "B std\n90 °C"}


def style():
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                         "axes.edgecolor": "#b5b4b0", "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "legend.frameon": False})


def despine(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)


def save(fig, name):
    p = C.FIGURES / name
    fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", p.relative_to(C.REPO).as_posix())


def write_table(df, name, title, fmt=None, note=""):
    """CSV (full precision) and markdown (thesis precision: fmt = {column: format string or callable})."""
    df.to_csv(TABLES / f"{name}.csv", index=False)
    fmt = fmt or {}
    cols = list(df.columns)
    lines = [f"# {title}", ""] + ([note, ""] if note else [])
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v, f = r[c], fmt.get(c)
            if f is None or (isinstance(v, float) and v != v) or isinstance(v, str):
                cells.append("–" if (isinstance(v, float) and v != v) else str(v))
            else:
                cells.append(f(v) if callable(f) else format(v, f))
        lines.append("| " + " | ".join(cells) + " |")
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written results/tables/{name}.csv and .md")


def locked_map():
    return pd.read_csv(C.RES_FINAL / "chapter_number_map.csv")


def locked(key):
    """value of a locked number (chapter_number_map.csv) by its key."""
    m = locked_map()
    v = m.loc[m.key == key, "value"]
    if len(v) != 1:
        raise KeyError(f"locked number not found once: {key}")
    return float(v.iloc[0])


class Rows:
    """Collects new numbers of one figure/table script and writes them to the map and the locked file."""

    def __init__(self, group):
        self.group, self.rows = group, []

    def put(self, key, value, unit, source, decimals=2):
        fr = inspect.currentframe().f_back
        self.rows.append(dict(chapter=CHAPTER, key=f"{self.group}: {key}", value=round(float(value), decimals), unit=unit,
                              script=f"src/{Path(fr.f_code.co_filename).name}:{fr.f_lineno}", source=source,
                              author_estimate="", old_value=None, old_location=""))

    def write(self):
        mp = C.RES_FINAL / "chapter_number_map.csv"
        m = pd.read_csv(mp)
        m = m[~((m.chapter == CHAPTER) & m.key.astype(str).str.startswith(self.group + ": "))]
        m = pd.concat([m, pd.DataFrame(self.rows)], ignore_index=True)
        m.to_csv(mp, index=False)
        own = m[m.chapter == CHAPTER]
        lk = C.RES_FINAL / "final_numbers_locked.md"
        txt = lk.read_text(encoding="utf-8").split("\n" + SECTION)[0].rstrip() + "\n"
        out = ["", SECTION, "", "Numbers of the figure and table scripts written for the repository (`src/fig_*.py`, "
               "`src/tab_*.py`). They are computed from the locked numbers and the derived data above; nothing is typed in.", ""]
        out += [f"- {r.key}: **{r.value} {r.unit}** | `{r.script}` | {r.source}" for r in own.itertuples()]
        lk.write_text(txt + "\n".join(out) + "\n", encoding="utf-8")
        print(f"{len(self.rows)} numbers of '{self.group}' written to chapter_number_map.csv and final_numbers_locked.md")
