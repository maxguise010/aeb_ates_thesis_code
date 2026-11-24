"""Tank cost cases D, B (low/high) and H and the OPEX rule, computed from final_config.py; prints the cases when run
on its own."""
import final_config as FC

V = FC.TANK_VOLUME_M3
EXP = 1.0 + FC.DEA_B                       # cost ~ V^(1 + b) = V^0.691


def dea_specific(v=V):
    return FC.DEA_A * v ** FC.DEA_B        # EUR/m3


def cbs_ratio(year):
    base = FC.CBS_INDEX.get(year, FC.CBS_INDEX[2018]) if year <= 2018 else FC.CBS_INDEX[year]
    return FC.CBS_INDEX[FC.CBS_TARGET] / base


def project_table(v=V):
    rows = {}
    for name, p in FC.PROJECTS.items():
        scale = (v / p["volume_m3"]) ** EXP
        scaled = p["invest_meur"] * scale
        esc = cbs_ratio(p["year"]) if p["escalate"] else float("nan")
        rows[name] = dict(volume_m3=p["volume_m3"], invest_meur=p["invest_meur"], year=p["year"], t_max_c=p["t_max_c"],
                          design=p["design"], scale_factor=scale, scaled_meur=scaled, cbs_ratio=esc,
                          scaled_escalated_meur=scaled * esc if p["escalate"] else float("nan"),
                          eur_per_m3_2026=(scaled * esc * 1e6 / v) if p["escalate"] else float("nan"))
        if "usable_m3" in p:   # sensitivity: scale from the usable volume instead of the gross volume
            su = p["invest_meur"] * (v / p["usable_m3"]) ** EXP
            rows[name]["scaled_from_usable_meur"] = su
            rows[name]["scaled_from_usable_escalated_meur"] = su * esc
    return rows


# tank cost cases: D = Danish benchmark (atmospheric, without connection); B = mean of Düsseldorf and
# Duisburg, scaled to 5,000 m3 and indexed with CBS (headline, author estimate); H = 8.0 M EUR
# (earlier author estimate, pressure vessel with connection). Case E (break-even) is in final_numbers.py.
def cases():
    pt = project_table()
    vals = [pt[n]["scaled_escalated_meur"] for n in FC.BASE_PROJECTS]
    d = dea_specific() * V / 1e6
    b = sum(vals) / len(vals)
    return {
        "D": dict(capex_meur=d, eur_per_m3=dea_specific(), label="Danish benchmark: atmospheric tank without connection, Denmark (DEA ch. 141)", estimate=False),
        "B": dict(capex_meur=b, eur_per_m3=b * 1e6 / V, low_meur=min(vals), high_meur=max(vals),
                  label="Thesis base: estimate derived from published German projects (Duesseldorf, Duisburg), scaled with the Danish exponent and escalated with the CBS index; not a published price for a 110 C buffer", estimate=True),
        "B_low": dict(capex_meur=min(vals), eur_per_m3=min(vals) * 1e6 / V, label="Case B low (Duesseldorf scaled and escalated)", estimate=True),
        "B_high": dict(capex_meur=max(vals), eur_per_m3=max(vals) * 1e6 / V, label="Case B high (Duisburg scaled and escalated)", estimate=True),
        "H": dict(capex_meur=FC.TANK_HIGH_MEUR, eur_per_m3=FC.TANK_HIGH_MEUR * 1e6 / V, label="High sensitivity: author estimate, pressure vessel with connection, no published source", estimate=True),
    }


# tank OPEX: 'scaled' = 1.4 % of its own CAPEX (headline); 'absolute' = fixed 21 k EUR/yr (sensitivity only).
def opex_keur(capex_meur, rule=FC.TANK_HEADLINE_OPEX):
    return FC.TANK_OLD["opex_keur"] if rule == "absolute" else FC.TANK_OPEX_SHARE * capex_meur * 1000.0


CASES = cases()
CASE_B_TUPLE = (CASES["B"]["low_meur"], CASES["B"]["capex_meur"], CASES["B"]["high_meur"])   # (low, mid, high)

if __name__ == "__main__":
    for k, c in CASES.items():
        print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in c.items()})
    for n, r in project_table().items():
        print(n, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()})
