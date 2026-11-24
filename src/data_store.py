"""Packs the caches of the model runs into data/run_caches.jsonl, unpacks them to outputs/cache/, and reads the
long-form reference tables."""
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import functools

import pandas as pd

STAMP = ".pack_sha256"


# ---------------------------------------------------------------------------------------------- caches
def unpack(jsonl, cache_root):
    """Unpack data/run_caches.jsonl (one JSON line per cache file of a model run: kind, file name, content) into
    cache_root/<kind>/<file>, as the model scripts wrote them. A kind is skipped when its stamp shows the same content.
    Files are written one by one with os.replace, so parallel worker processes that import config at the same time never
    see a half-written file; files that the model scripts added to a folder are kept."""
    by_kind = {}
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            by_kind.setdefault(d["kind"], []).append(d)
    for kind, items in by_kind.items():
        sha = hashlib.sha256(json.dumps([[i["file"], i["content"]] for i in items], sort_keys=True).encode("utf-8")).hexdigest()
        target = Path(cache_root) / kind
        stamp = target / STAMP
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == sha:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for i in items:
            name, content = i["file"], i["content"]
            tmp = target / f".{name}.{os.getpid()}.tmp"
            text = content if isinstance(content, str) else json.dumps(content)
            tmp.write_text(text, encoding="utf-8", newline="")
            os.replace(tmp, target / name)
        tmp = target / f"{STAMP}.{os.getpid()}.tmp"
        tmp.write_text(sha, encoding="utf-8")
        os.replace(tmp, stamp)


def pack_caches(cache_root, out_file):
    """All cache folders cache_root/<kind>/ -> one JSON-lines file (kind, file, content); JSON files are stored as parsed
    objects, other files (CSV) as text. Used after new model runs."""
    lines = []
    for folder in sorted(p for p in Path(cache_root).iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir(), key=lambda p: p.name):
            if not f.is_file() or f.name.startswith("."):
                continue
            text = f.read_bytes().decode("utf-8").replace("\r\n", "\n")
            content = json.loads(text) if f.suffix == ".json" else text
            lines.append(json.dumps({"kind": folder.name, "file": f.name, "content": content}, ensure_ascii=False, separators=(",", ":")))
    Path(out_file).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


# ---------------------------------------------------------------------------------------------- small loaders
@functools.lru_cache(maxsize=None)
def _dispatch_weekly(path):
    return pd.read_csv(path, float_precision="round_trip")


def dispatch_weekly(path, run):
    """Weekly aggregate of the hourly ATES dispatch of one run (data/dispatch_weekly.csv; runs A1D, A2D, A4D, B-WN110,
    B-std WN, B-std NW): per ISO week the charge/discharge sums and maxima of the run and, per doublet, the sums, maxima,
    hours above the well limit and energy above the well-limit cap that well_limit.py uses. Full float precision."""
    w = _dispatch_weekly(str(path))
    return w[w.run == run].reset_index(drop=True)


def extra_table(path, name):
    """One table of data/extra_results.json (results of the extra analyses that need the raw data), as pd.read_csv
    reads the original CSV file."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.read_csv(io.StringIO("\n".join(d[name]["lines"]) + "\n"))


# ---------------------------------------------------------------------------------------------- long-format tables
# one row per line of the CSV file: file (identifying column), row (-1 = header, then 0, 1, ...) and line
# (the line rewritten with the csv module, so that only the cell text counts and not the quoting style).
LONG_COLUMNS = ["file", "row", "line"]


def _canon(record):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(record)
    return buf.getvalue()


def to_long(path, name=None):
    """A CSV file as long rows (file, row, line): row -1 = header, then one row per record; every record rewritten in
    canonical CSV form, so that only the cell text counts."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        records = list(csv.reader(fh))
    name = name or Path(path).name
    return pd.DataFrame([(name, i - 1, _canon(r)) for i, r in enumerate(records)], columns=LONG_COLUMNS)


def read_long(path):
    return pd.read_csv(path, dtype={"file": str, "row": int, "line": str}, keep_default_na=False)


def csv_text(long, name):
    """The CSV text of one table of a long file."""
    t = long[long.file == name]
    if not len(t):
        raise KeyError(f"{name} is not in the reference file")
    return "\n".join(t.sort_values("row").line) + "\n"


def frame(long, name):
    """One table of a long file as a DataFrame, parsed as pd.read_csv would parse the original file."""
    return pd.read_csv(io.StringIO(csv_text(long, name)))


def compare_long(new, ref):
    """Records of two long frames that differ (exact cell text): (file, row, reference line, new line)."""
    m = ref.merge(new, on=["file", "row"], how="outer", suffixes=("_ref", "_new"), indicator=True)
    bad = m[(m._merge != "both") | (m.line_ref != m.line_new)]
    return bad.drop(columns="_merge")


# ---------------------------------------------------------------------------------------------- command line
if __name__ == "__main__":
    if sys.argv[1:] == ["pack"]:
        import config as C
        n = pack_caches(C.CACHE, C.RUN_CACHES)
        print(f"{n} cache files -> {C.RUN_CACHES}")
    else:
        print(__doc__ + "\nUsage: python src/data_store.py pack   (outputs/cache/<kind>/ -> data/run_caches.jsonl)")
