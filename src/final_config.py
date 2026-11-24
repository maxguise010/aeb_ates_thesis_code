"""The locked economic inputs (discount rate, lifetime, tank cost cases, OPEX, year-7 ROSIM efficiencies); every
final-number script imports them from here."""
# ------------------------------------------------------------------ discounting (thesis Eq. 7.1, Section 7.1)
RATE = 0.05                    # [THESIS] real discount rate (WACC)
LIFETIME = 30                  # [THESIS] years, ATES and tank
RATES_SENS = (0.03, 0.05, 0.07)


# annuity factor = present value of 1 EUR per year over n years: (1 - (1 + r)^-n) / r = 15.3725 at 5 % and
# 30 years. The submitted thesis used the rounded 15.37. CRF = 1 / annuity factor = 0.06505.
def annuity_factor(r=RATE, n=LIFETIME):
    """Present value of 1 EUR/yr over n years, (1 - (1 + r)^-n) / r."""
    return (1.0 - (1.0 + r) ** -n) / r


def crf(r=RATE, n=LIFETIME):
    """Capital recovery factor r / (1 - (1 + r)^-n) = 1 / annuity_factor."""
    return 1.0 / annuity_factor(r, n)


# ------------------------------------------------------------------ tank (TTES 5,000 m3, 110 C)
TANK_VOLUME_M3 = 5000.0
TANK_OLD = dict(capex_meur=(1.0, 1.5, 2.0), opex_keur=21.0)   # [THESIS] 200/300/400 EUR/m3, OPEX 1.4 % of mid
# OPEX rule of the thesis (Table 7.4): 1.4 % of CAPEX per year. At the old 1.5 M EUR that was 21 k EUR/yr;
# in the final numbers the tank OPEX scales with the CAPEX of each case (case B 5.04 M EUR -> 70.6 k EUR/yr).
TANK_OPEX_SHARE = TANK_OLD["opex_keur"] / 1000.0 / TANK_OLD["capex_meur"][1]   # [THESIS] = 1.4 % of CAPEX per year

# Danish Energy Agency, Technology Data for Energy Storage, ch. 141 TTES (amended July 2026), p. 62:
# P = a V^b (EUR/m3), valid 900-12,000 m3; scope tank + equipment, civil works, consultancy (p. 61);
# excludes pumps, heat exchangers, pipes, connection; atmospheric TTES only (p. 61). Price year not stated in ch. 141.
# case D uses this Danish formula directly; the exponent 1 + b = 0.691 also scales the German projects to 5,000 m3.
DEA_A, DEA_B = 3055.0, -0.309                                  # [SRC]
DEA_RANGE_M3 = (900.0, 12000.0)                                # [SRC]

# CBS 85728ENG "New dwellings; input price indices building costs 2021=100", Total building costs. [SRC]
# The table starts in January 2018: there is no 2017 value, so the 2018 value is used for 2017 too (proxy).
CBS_INDEX = {2018: 90.9, "2026-09": 129.5}                     # 2026-09 is provisional (CBS asterisk)
CBS_TARGET = "2026-09"

# German district heating tanks with a published total investment. [SRC] figures as in the sources:
# NRW: EEB ENERKO, "Waermespeicher in NRW", Tab. 2 (printed p. 17) and list (printed p. 10);
# Duisburg also energy4climate.nrw; Flensburg foerde.news 05.06.2024; Halle energie-experten.org 21.09.2018.
PROJECTS = {
    "Duesseldorf": dict(volume_m3=36000.0, invest_meur=10.0, year=2017, t_max_c=98, design="single-zone, atmospheric",
                        escalate=True),
    "Duisburg":    dict(volume_m3=43000.0, invest_meur=20.0, year=2018, t_max_c=115,
                        design="two-zone, not a pressure vessel; usable volume 28,000 m3", usable_m3=28000.0,
                        escalate=True),
    "Flensburg":   dict(volume_m3=28584.0, invest_meur=9.0, year=2024, t_max_c=None,
                        design="not stated; 2024 article reports a plan (completion planned Nov 2024)", escalate=False),
    "Halle":       dict(volume_m3=50000.0, invest_meur=10.0, year=2018, t_max_c=None,
                        design="not stated (usable volume); power-to-heat scope of the cost not stated", escalate=False),
}
BASE_PROJECTS = ("Duesseldorf", "Duisburg")    # [EST] base case = mean of these two (highest temperatures)
TANK_HIGH_MEUR = 8.0                           # [EST] author's earlier estimate, pressure vessel with connection
TANK_HEADLINE_OPEX = "scaled"                  # "scaled" = 1.4 % of CAPEX, "absolute" = 21 k EUR/yr

# ------------------------------------------------------------------ ATES (thesis Tables 7.1-7.4) [THESIS]
ATES = {
    "A1D":    dict(capex_meur=(8.5, 10.5, 14.0),  opex_keur=175.0),
    "A2D":    dict(capex_meur=(13.5, 17.0, 22.5), opex_keur=255.0),
    "A4D":    dict(capex_meur=(23.5, 30.0, 40.0), opex_keur=495.0),
    "BWN110": dict(capex_meur=(8.5, 10.5, 14.0),  opex_keur=175.0),
    # thesis Table 7.4: 200 + 54 (HP electricity) + 124 (HP maintenance) = 378. An earlier code version used 359 (opex_code_keur).
    # B-std OPEX = 378 k EUR/yr according to thesis Table 7.4 (200 + 54 heat-pump electricity + 124 maintenance).
    "Bstd":   dict(capex_meur=(17.7, 22.2, 30.2), opex_keur=378.0, opex_code_keur=359.0),
}
BRIDGE_R_H = 2                                  # restart time for the bridged tier (thesis Section 7.4.4)
R_TH = 1.51                                     # [THESIS] Eq. 3.2 at 110 C (n 0.28, rho_s 2,650, c_s 800); thermal radius
                                                # Eq. 3.3 (tab_6_3_6_10_thermal_radius.py). NB: the ROSIM settings use rho_s 2,400
# ------------------------------------------------------------------ further thesis inputs of the report tables
# pump replacement (Appendix B.3): pump of a hot well 140 k EUR (2020) x CBS index 1.363, replaced every 7 years.
PUMP_COST_2020_KEUR = 140.0                     # [THESIS] App. B.3, mid value per doublet (TU Delft study prices)
CBS_CIVIL_2020_2025 = 1.363                     # [THESIS] App. B.1, CBS input price index civil engineering 2020 -> 2025
PUMP_INTERVAL_YR = 7                            # [THESIS] App. B.3, replacement interval (years)
DOUBLETS = {"A1D": 1, "A2D": 2, "A4D": 4, "BWN110": 1, "Bstd": 2}   # [THESIS] Table 3.2
RHO_G_THIEM = 1000.0 * 9.81                     # [THESIS] Eq. 3.1 with rho = 1,000 kg/m3 (reproduces 221/179/117 m3/h)
WATER_PROPS = {110.0: (950.95, 4232.0), 90.0: (965.31, 4205.0)}   # [SRC] IAPWS-IF97 saturated liquid: rho (kg/m3), cp (J/kg/K)

# ------------------------------------------------------------------ final ROSIM runs, year 7 (rosim_check.py)
# ROSIM reported (B-WN110 cleaned), checked against the ROSIM exports (data/rosim) to 0.02 by rosim_check.py.
# annual efficiencies of the six final ROSIM runs (years 6, 7, 8); year 7 is the basis of the f_corr calibration.
# B-WN110 year 7 is the 'cleaned' value: the glitch weeks 339 and 340 of the cold well are interpolated.
EFF_TABLE = {"A1D": (43.76, 45.16, 46.33), "A2D": (48.12, 49.62, 50.92), "A4D": (49.80, 51.43, 52.85),
             "BWN110": (47.62, 48.68, 50.03), "Bstd WN": (34.00, 35.57, 36.88), "Bstd NW": (35.14, 36.90, 38.37)}
EFF_THESIS_Y7 = {"A1D": 43.56, "A2D": 45.07, "A4D": 46.20, "BWN110": 44.16, "Bstd WN": 36.88, "Bstd NW": 37.56}
SPACING_M = {"A1D": 100, "A2D": 100, "A4D": 100, "BWN110": 125, "Bstd": 75}   # warm-cold, from ROSIM settings
# two efficiency bases: R = reported by ROSIM (headline, convention of the thesis); D = energy computed with
# the return temperature of the dispatch (58 C for A, B-WN110 and B-std): the lower case.
FCORR_BASIS_HEADLINE = "R"                      # R = ROSIM reported (thesis convention), D = dispatch reference
