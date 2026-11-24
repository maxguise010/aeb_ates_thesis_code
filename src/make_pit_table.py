#!/usr/bin/env python3
"""Pit check (lower charging temperature): checks five storage-effect numbers against the TTES runs and writes one
table. Run by run_all.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C  # repository paths (config.py)
HERE = C.RES_SENS / "pit_check"   # output folder of this script
HERE.mkdir(parents=True, exist_ok=True)
ORIG = C.TTES_LOW_TEMP_CACHE
RERUN = C.PIT_RERUN_CACHE
YEARS = ["2023", "2024", "2025"]
T_RET, RHO, CP = 58.0, 950.0, 4220.0
SCRIPT = "src/run_ttes_low_temp.py"

CLAIMS = [  # (label, schedule, buffer, V, T, top-up, value from the author's notes)
    ("5,000 m3 at 110 C, no top-up", "relaxed", "thesis", 5000, 110, False, 51.4),
    ("5,000 m3 at 95 C, no top-up", "relaxed", "thesis", 5000, 95, False, 27.7),
    ("5,000 m3 at 90 C, no top-up", "relaxed", "thesis", 5000, 90, False, 22.0),
    ("5,000 m3 at 110 C, 90 C top-up enforced", "relaxed", "thesis", 5000, 110, True, 39.5),
    ("10,000 m3 at 95 C, 90 C top-up enforced", "relaxed", "thesis", 10000, 95, True, 61.4),
]


def key(schedule, v, t, buffer):
    return (f"{schedule}__{v}__{t}" + ("__ss16" if v >= 10_000 else "")
            + ("__cal" if buffer == "calibrated" else "") + ".json")


def load(folder, schedule, v, t, buffer):
    p = folder / key(schedule, v, t, buffer)
    return json.loads(p.read_text()) if p.exists() else None, p


def metrics(o, z):
    """All table quantities of one run from its cache and the zero run of the same schedule/buffer."""
    s_py = [(o["saving_py"][y] - z["saving_py"][y]) / 1e3 for y in YEARS]
    tc_py = [o["topup_cost_py"][y] / 1e3 for y in YEARS]
    tm_py = [o["topup_mwh_py"][y] for y in YEARS]
    cap = o["volume_m3"] * RHO * CP * (o["t_charge_c"] - T_RET) / 3.6e9
    return dict(
        capacity_mwh=cap,
        charged_dispatch_mwh_yr=o["charged_dispatch_mwh"] / 3,
        discharged_dispatch_mwh_yr=o["discharged_dispatch_mwh"] / 3,
        stored_replay_mwh_yr=o["stored_mwh"] / 3,
        discharged_heat_mwh_yr=o["useful_mwh"] / 3,
        thermal_losses_mwh_yr=o["loss_mwh"] / 3,
        cycle_efficiency=o["useful_mwh"] / o["stored_mwh"],
        cycles_per_yr=o["charged_dispatch_mwh"] / 3 / cap,
        hours_below_grid_min_yr=o["lowT_hours_any"] / 3,
        discharge_hours_yr=o["discharge_hours"] / 3,
        topup_mwh_yr=float(np.mean(tm_py)),
        topup_cost_keur_yr=float(np.mean(tc_py)),
        S_no_topup=float(np.mean(s_py)),
        S_with_topup=float(np.mean([s - c for s, c in zip(s_py, tc_py)])),
        S_no_topup_py=s_py,
        S_with_topup_py=[s - c for s, c in zip(s_py, tc_py)],
    )


def runs():
    for buffer in ("thesis", "calibrated"):
        for schedule in ("relaxed", "constrained"):
            for v in (5000, 10000):
                for t in (110, 95, 90):
                    yield buffer, schedule, v, t


def main():
    rows, cmp_rows = [], []
    have_rerun = RERUN.exists() and any(RERUN.glob("*.json"))
    for buffer, schedule, v, t in runs():
        o, po = load(ORIG, schedule, v, t, buffer)
        z, _ = load(ORIG, schedule, 0, 110, buffer)
        m = metrics(o, z)
        mr = None
        if have_rerun:
            orr, _ = load(RERUN, schedule, v, t, buffer)
            zr, _ = load(RERUN, schedule, 0, 110, buffer)
            if orr and zr:
                mr = metrics(orr, zr)
        res_csv = ("ttes_low_temp_results_ss16_cal.csv" if buffer == "calibrated" else "ttes_low_temp_results_ss16.csv")
        for topup in (False, True):
            s = m["S_with_topup"] if topup else m["S_no_topup"]
            label = (f"{schedule} {buffer}-buffer {v:,} m3 {t} C " + ("with 90 C top-up" if topup else "no top-up"))
            row = {
                "run_label": label, "volume_m3": v, "charge_temperature_c": t,
                "topup_enforced": "yes" if topup else "no", "schedule": schedule,
                "buffer_setting": buffer,
                "discharged_heat_mwh_yr": round(m["discharged_heat_mwh_yr"], 1),
                "storage_effect_keur_yr": round(s, 2),
                "storage_effect_2023_2024_2025_keur": " / ".join(
                    f"{x:.1f}" for x in (m["S_with_topup_py"] if topup else m["S_no_topup_py"])),
                "thermal_losses_mwh_yr": round(m["thermal_losses_mwh_yr"], 1),
                "cycle_efficiency": round(m["cycle_efficiency"], 3),
                "hours_below_grid_min_per_yr": round(m["hours_below_grid_min_yr"], 1),
                "topup_energy_mwh_yr": round(m["topup_mwh_yr"], 1) if topup else "",
                "topup_cost_keur_yr": round(m["topup_cost_keur_yr"], 2) if topup else "",
                "capacity_mwh": round(m["capacity_mwh"], 1),
                "cycles_per_yr": round(m["cycles_per_yr"], 1),
                "storage_effect_rerun_current_model_keur_yr": (
                    round(mr["S_with_topup"] if topup else mr["S_no_topup"], 2) if mr else "not rerun"),
                "script": SCRIPT + " --substeps-10k 16" + (" --buffer calibrated" if buffer == "calibrated" else ""),
                "result_file": (f"results/ttes_low_temp/{res_csv}; "
                                + f"data/derived/ttes_low_temp_cache/{po.name}"),
            }
            rows.append(row)
        if mr:
            for k in ("S_no_topup", "S_with_topup", "discharged_heat_mwh_yr", "thermal_losses_mwh_yr",
                      "topup_mwh_yr", "hours_below_grid_min_yr", "cycle_efficiency"):
                cmp_rows.append(dict(buffer_setting=buffer, schedule=schedule, volume_m3=v, t_charge_c=t, quantity=k,
                                     original=m[k], rerun_current_model=mr[k], difference=mr[k] - m[k]))

    tab = pd.DataFrame(rows)
    tab.to_csv(HERE / "pit_runs_table.csv", index=False)

    # cross-check against the original result CSVs (reproduction from the files)
    chk = []
    for f, buf in (("ttes_low_temp_results_ss16.csv", "thesis"), ("ttes_low_temp_results_ss16_cal.csv", "calibrated")):
        d = pd.read_csv(C.RES_TTES / f)
        for _, r in d.iterrows():
            mine = tab[(tab.buffer_setting == buf) & (tab.schedule == r.schedule) & (tab.volume_m3 == r.volume_m3)
                       & (tab.charge_temperature_c == r.t_charge_c)]
            chk.append(abs(mine[mine.topup_enforced == "no"].storage_effect_keur_yr.iloc[0] - r.S_stor_keur_yr))
            chk.append(abs(mine[mine.topup_enforced == "yes"].storage_effect_keur_yr.iloc[0]
                           - r.S_stor_net_topup_keur_yr))
    # (the project also printed the columns of its like-for-like table; that table is not needed here)
    print(f"max |table - original result CSV| = {max(chk):.4f} k EUR/yr over {len(chk)} values")

    # claims
    vr = []
    for label, sch, buf, v, t, topup, claim in CLAIMS:
        r = tab[(tab.buffer_setting == buf) & (tab.schedule == sch) & (tab.volume_m3 == v)
                & (tab.charge_temperature_c == t) & (tab.topup_enforced == ("yes" if topup else "no"))].iloc[0]
        val = r.storage_effect_keur_yr
        rr = r.storage_effect_rerun_current_model_keur_yr
        vr.append(dict(claim=label, value_in_notes=claim, value_in_files=val,
                       rounded_1dp=round(val, 1), verdict="confirmed" if round(val, 1) == claim else "differs",
                       rerun_current_model=rr, source=r.result_file))
    vr = pd.DataFrame(vr)
    vr.to_csv(HERE / "verification_claims.csv", index=False)
    print(vr.to_string(index=False))

    # final TTES (scenario_runner TTES5k, relaxed) vs the 110 C low-temp run: same dispatch?
    fin = json.loads((C.SCENARIO_CACHE / "base__TTES5k__relaxed__fd6d736c.json").read_text())
    o, _ = load(ORIG, "relaxed", 5000, 110, "thesis")
    ft = [dict(year=y, final_TTES5k_saving_keur=fin["per_year"][y], lowtemp_5000_110_saving_keur=o["saving_py"][y] / 1e3,
               difference=o["saving_py"][y] / 1e3 - fin["per_year"][y]) for y in YEARS]
    pd.DataFrame(ft).to_csv(HERE / "final_ttes_check.csv", index=False)
    print(pd.DataFrame(ft).to_string(index=False))

    if cmp_rows:
        c = pd.DataFrame(cmp_rows)
        c.to_csv(HERE / "rerun_comparison.csv", index=False)
        print("rerun vs original, max |difference| per quantity:")
        print(c.groupby("quantity").difference.apply(lambda s: s.abs().max()).to_string())

    # markdown version of the table
    cols = ["run_label", "volume_m3", "charge_temperature_c", "topup_enforced", "schedule", "discharged_heat_mwh_yr",
            "storage_effect_keur_yr", "thermal_losses_mwh_yr", "cycle_efficiency", "hours_below_grid_min_per_yr",
            "topup_energy_mwh_yr", "storage_effect_rerun_current_model_keur_yr"]
    head = ["run", "V (m³)", "T charge (°C)", "top-up", "schedule", "discharged (MWh/yr)", "S_stor (k€/yr)",
            "losses (MWh/yr)", "cycle eff.", "h/yr below 90 °C", "top-up (MWh/yr)", "S_stor rerun (k€/yr)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in tab.iterrows():
        vals = []
        for cname in cols:
            x = r[cname]
            if cname == "run_label":
                x = f"{r.buffer_setting} buffer"
            elif cname in ("volume_m3",):
                x = f"{x:,}"
            elif isinstance(x, float):
                x = f"{x:,.3f}" if cname == "cycle_efficiency" else (f"{x:,.1f}" if "mwh" in cname or "hours" in cname
                                                                       else f"{x:,.2f}")
            vals.append(str(x))
        lines.append("| " + " | ".join(vals) + " |")
    (HERE / "pit_runs_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
