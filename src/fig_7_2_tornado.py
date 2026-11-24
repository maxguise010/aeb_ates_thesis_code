"""Thesis Figure 7.2 (NPV tornado) with the final inputs, for every option; called by report_figures.py."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

import final_config as FC
import scenarios as SC
import tank_capex as TC
import figtab_common as F

SET = "new_R_B"                     # f_corr basis R, tank case B (headline)
ORDER = ["TTES5k", "A1D", "BWN110", "A2D", "Bstd", "A4D"]
NAME = {"TTES5k": "TTES 5,000 m³", "A1D": "Option A 1D", "BWN110": "B-WN 110 °C", "A2D": "Option A 2D",
        "Bstd": "B standard", "A4D": "Option A 4D"}
# rows of the printed figure: (label, low bar key, high bar key); one-sided rows have one key
ROWS = [("Full dispatch-only reference credited", "full S_ref credited (potential)", None),
        ("Relaxed schedule not credited", "relaxed schedule not credited (lower)", None),
        ("Only bridged schedule value credited (2 h restart)", "bridged, R = 2 h", None),
        ("ROSIM correction on whole increment", "ROSIM correction on the whole increment", None),
        ("Discount rate 6.0% / 2.65% (real)", "rate 6.0 %", "rate 2.65 %"),
        ("CAPEX high / low", "CAPEX high", "CAPEX low"),
        ("Savings −1% / +1% per year", "growth -1 %/yr", "growth +1 %/yr"),
        ("OPEX +50% / −50%", "OPEX +50 %", "OPEX -50 %"),
        ("Heat demand +20% / −20%", "demand +20 %", "demand -20 %"),
        ("Price spread ×0.75 / ×1.25", "spread x0.75", "spread x1.25")]


def bars(D, s, share, o):
    """change of the base NPV (M EUR) per bar; for A1D identical to scenarios.tornado."""
    cx, ox = SC.capex_opex(o, s)
    if o == "TTES5k":
        lo, hi = TC.CASE_B_TUPLE[0], TC.CASE_B_TUPLE[2]
    else:
        lo, hi = FC.ATES[o]["capex_meur"][0], FC.ATES[o]["capex_meur"][2]
    f, fpv = s["f"][o], s["fpv"]
    d = D.loc[("base", o)]
    t = SC.tiers(d, o, f, share)
    nb = SC.npv_generic(cx, ox, t["base"], fpv, fpv)
    if o == "TTES5k":   # OPEX follows the CAPEX of the case (1.4 %)
        b = {"CAPEX low": SC.npv_generic(lo, TC.opex_keur(lo), t["base"], fpv, fpv) - nb,
             "CAPEX high": SC.npv_generic(hi, TC.opex_keur(hi), t["base"], fpv, fpv) - nb}
    else:
        b = {"CAPEX low": cx - lo, "CAPEX high": cx - hi}
    for r, k in ((0.06, "rate 6.0 %"), (0.0265, "rate 2.65 %")):
        b[k] = SC.npv_generic(cx, ox, t["base"], SC.af(r), SC.af(r)) - nb
    for g, k in ((-0.01, "growth -1 %/yr"), (0.01, "growth +1 %/yr")):
        b[k] = SC.npv_generic(cx, ox, t["base"], SC.fpv_g(g), fpv) - nb
    b["OPEX -50 %"], b["OPEX +50 %"] = 0.5 * ox * fpv / 1000, -0.5 * ox * fpv / 1000
    for scn, k in (("demand_m20", "demand -20 %"), ("demand_p20", "demand +20 %"), ("spread_075", "spread x0.75"),
                   ("spread_125", "spread x1.25")):
        b[k] = SC.npv_generic(cx, ox, SC.tiers(D.loc[(scn, o)], o, f, share)["base"], fpv, fpv) - nb
    b["full S_ref credited (potential)"] = SC.npv_generic(cx, ox, t["potential"], fpv, fpv) - nb
    b["relaxed schedule not credited (lower)"] = SC.npv_generic(cx, ox, t["lower"], fpv, fpv) - nb
    b["bridged, R = 2 h"] = SC.npv_generic(cx, ox, t["bridged"], fpv, fpv) - nb
    b["ROSIM correction on the whole increment"] = SC.npv_generic(cx, ox, f * (d.S_flex + d.S_stor), fpv, fpv) - nb
    return nb, b


def lab(v):
    return f"{v:+.2f}".replace("-", "−")


def panel(ax, o, nb, b, big=False):
    fs = 8.5 if big else 6.5
    n = len(ROWS)
    span = max(abs(nb + v) for v in b.values()) or 1
    for i, (label, k1, k2) in enumerate(ROWS):
        y = n - 1 - i
        ks = [k for k in (k1, k2) if k is not None]
        for k in sorted(ks, key=lambda k: -abs(b[k])):      # longer bar first, the shorter one stays visible
            v = b[k]
            ax.barh(y, v, left=nb, height=0.55, color=F.RUST if v < 0 else F.STEEL)
        vals = [b[k] for k in ks if abs(b[k]) > 1e-9]
        if len(vals) == 2 and (vals[0] < 0) == (vals[1] < 0):   # both bars on one side: one label "low / high"
            v = max(vals, key=abs)
            ax.text(nb + v + (0.012 * span if v >= 0 else -0.012 * span), y, " / ".join(lab(nb + x) for x in vals),
                    va="center", ha="left" if v >= 0 else "right", fontsize=fs)
        else:
            for v in vals:
                ax.text(nb + v + (0.012 * span if v >= 0 else -0.012 * span), y, lab(nb + v), va="center",
                        ha="left" if v >= 0 else "right", fontsize=fs)
    ax.axvline(nb, color=F.INK, linewidth=1)
    ax.text(nb, n - 0.35, f"base case {lab(nb)}", ha="center", va="bottom", color="#6d6d6a", fontsize=fs)
    ax.set_ylim(-0.7, n + 0.25)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=fs + (1 if big else 0))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=F.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    lo = min(nb + v for v in b.values())
    hi = max(nb + v for v in b.values())
    pad = 0.18 * (hi - lo)
    ax.set_xlim(min(lo, nb) - pad, max(hi, nb) + pad)
    ax.set_xlabel(f"NPV of {NAME[o]} at mid CAPEX (M€)", fontsize=fs + (1 if big else 0))


def main():
    D = SC.decompose()
    sets, share = SC.inputs()
    s = sets[SET]
    out, res = [], {}
    for o in ORDER:
        nb, b = bars(D, s, share, o)
        res[o] = (nb, b)
        out.append(dict(option=o, npv_base=nb, **b))
    pd.DataFrame(out).to_csv(F.FIGDATA / "figure_7_2.csv", index=False)
    F.style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    panel(ax, "A1D", *res["A1D"], big=True)
    ax.legend([Patch(color=F.RUST), Patch(color=F.STEEL)], ["Lowers NPV", "Raises NPV"], loc="lower right")
    F.save(fig, "ch7_npv_tornado_1d.png")
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 12.5))
    for ax, o in zip(axs.flat, ORDER):
        panel(ax, o, *res[o])
        ax.set_title(NAME[o], fontsize=9, loc="left")
    fig.legend([Patch(color=F.RUST), Patch(color=F.STEEL)], ["Lowers NPV", "Raises NPV"], loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    F.save(fig, "ch7_npv_tornado_all_options.png")
    R = F.Rows("Figure 7.2")
    for r in out:
        R.put(f"base NPV, {NAME[r['option']]}", r["npv_base"], "M EUR", "scenarios.tiers / npv_generic, set new_R_B", 2)
        for _, k1, k2 in ROWS:
            for k in (k1, k2):
                if k:
                    R.put(f"NPV with {k}, {NAME[r['option']]}", r["npv_base"] + r[k], "M EUR", "base NPV + bar", 2)
    R.write()
    print(pd.DataFrame(out).set_index("option").T.round(2).to_string())


if __name__ == "__main__":
    main()
