"""Table of the f_corr runs of the submitted thesis (old targets), including the discharged heat of the old LCOH. Run
by run_all.py."""
import os
import pandas as pd

import config as C  # repository paths (config.py)
D = str(C.FCORR_VALIDATION)   # evaluation CSVs of the thesis f_corr runs
OUT = str(C.RES_FCORR)


def load(name):
    return pd.read_csv(os.path.join(D, name))


rows, direct_rows = [], []

# Option A: fcorr_validation.py evaluations (single scale)
for cfg, eta_rosim in (('A-1D', 43.56), ('A-2D', 45.07), ('A-4D', 46.20)):
    ev = load(f"{cfg}_evaluations.csv")
    base = ev.loc[ev['ground_loss_scale'] == 1.0].iloc[0]
    ok = ev[~ev['flagged_artefact']]
    d = ok.iloc[(ok['eta_cycle_pct'] - eta_rosim).abs().argmin()]
    rows.append(dict(config=cfg, eta_rosim_pct=eta_rosim, base_eta_disp_pct=base['eta_disp_pct'],
                     base_eta_cycle_pct=base['eta_cycle_pct'], f_corr_alt=eta_rosim / base['eta_cycle_pct'],
                     s_stor_base_eur_yr=base['s_stor_eur_yr'], s_stor_direct_eur_yr=d['s_stor_eur_yr'],
                     q_discharged_base_mwh_yr=base['discharged_mwh_yr'], q_discharged_direct_mwh_yr=d['discharged_mwh_yr']))
    direct_rows.append(dict(config=cfg, target=f"η_cycle {eta_rosim}", scales=f"{d['ground_loss_scale']}",
                            eta_disp_pct=d['eta_disp_pct'], eta_cycle_pct=d['eta_cycle_pct'],
                            charged_mwh_yr=d['charged_mwh_yr'], discharged_mwh_yr=d['discharged_mwh_yr'],
                            useful_to_customers_mwh_yr=d['useful_discharge_mwh_yr'], s_disp_eur_yr=d['s_disp_eur_yr'],
                            s_stor_eur_yr=d['s_stor_eur_yr'], equiv_full_cycles_per_yr=d['equiv_full_cycles_per_yr'],
                            ground_loss_mwh_yr=d['ground_loss_mwh_yr'], e_end_mwh=d['e_end_mwh'],
                            balance_residual_mwh=d['balance_residual_mwh'], flagged=d['flagged_artefact'],
                            source=f"{cfg}_evaluations.csv"))

# Option B: fcorr_validation_followup.py evaluations (per-doublet targets)
for cfg, targets in (('B-WN110', {'wn': 44.2}), ('B-std', {'wn': 36.9, 'nw': 37.6})):
    ev = load(f"followup_{cfg}_evaluations.csv")
    base = ev.loc[(ev['scale_1'] == 1.0) & (ev['scale_2'] == 1.0)].iloc[0]
    ok = ev[~ev['flagged_artefact']].copy()
    ok['dist'] = sum((ok[f'{k}_eta_cycle_pct'] - v).abs() for k, v in targets.items())
    d = ok.loc[ok['dist'].idxmin()]
    eta_rosim = sum(targets.values()) / len(targets)
    rows.append(dict(config=cfg, eta_rosim_pct=eta_rosim, base_eta_disp_pct=base['eta_disp_pct'],
                     base_eta_cycle_pct=base['eta_cycle_pct'], f_corr_alt=eta_rosim / base['eta_cycle_pct'],
                     s_stor_base_eur_yr=base['s_stor_eur_yr'], s_stor_direct_eur_yr=d['s_stor_eur_yr'],
                     q_discharged_base_mwh_yr=base['discharged_mwh_yr'], q_discharged_direct_mwh_yr=d['discharged_mwh_yr']))
    direct_rows.append(dict(config=cfg, target=", ".join(f"{k.upper()} η_cycle {v}" for k, v in targets.items()),
                            scales=f"{d['scale_1']}/{d['scale_2']}", eta_disp_pct=d['eta_disp_pct'],
                            eta_cycle_pct=d['eta_cycle_pct'],
                            per_doublet_eta_cycle=", ".join(f"{k.upper()} {d[f'{k}_eta_cycle_pct']:.2f}" for k in targets),
                            charged_mwh_yr=d['charged_mwh_yr'], discharged_mwh_yr=d['discharged_mwh_yr'],
                            useful_to_customers_mwh_yr=d['useful_to_customers_mwh_yr'], s_disp_eur_yr=d['s_disp_eur_yr'],
                            s_stor_eur_yr=d['s_stor_eur_yr'], equiv_full_cycles_per_yr=d['equiv_full_cycles_per_yr'],
                            ground_loss_mwh_yr=d['ground_loss_mwh_yr'], e_end_mwh=d['e_end_mwh'],
                            balance_residual_mwh=d['balance_residual_mwh'], flagged=d['flagged_artefact'],
                            source=f"followup_{cfg}_evaluations.csv"))

sens = []
for cfg in ('A-1D-sens', 'A-4D-sens'):
    ev = load(f"followup_{cfg}_evaluations.csv")
    tgt = {'A-1D-sens': 1.05 * 43.56, 'A-4D-sens': 1.05 * 46.20}[cfg]
    ok = ev[~ev['flagged_artefact']]
    d = ok.iloc[(ok['eta_cycle_pct'] - tgt).abs().argmin()]
    sens.append(dict(config=cfg, target_eta_cycle_pct=tgt, eta_cycle_pct=d['eta_cycle_pct'],
                     s_stor_eur_yr=d['s_stor_eur_yr'], flagged=d['flagged_artefact']))

t, dr, se = pd.DataFrame(rows), pd.DataFrame(direct_rows), pd.DataFrame(sens)
t.to_csv(os.path.join(OUT, "followup_final_table.csv"), index=False)
dr.to_csv(os.path.join(OUT, "followup_direct_runs.csv"), index=False)
se.to_csv(os.path.join(OUT, "followup_sensitivity.csv"), index=False)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
print()
print(dr.drop(columns=['source']).round(2).to_string(index=False))
print()
print(se.round(2).to_string(index=False))
