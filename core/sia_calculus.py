# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:16:16 2026

@author: infor
"""
# core/sia_calculus.py
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import calendar


def compute_theta_e_avg_C_from_monthly_T_ext(T_ext: list[float] | None) -> float | None:
    """
    Compute annual mean outdoor temperature theta_e_avg [°C] from monthly averages.
    Uses day-weighted mean over the 12 months (CECB wording: annual mean local climate temp).
    """
    if not isinstance(T_ext, list) or len(T_ext) != 12:
        return None

    try:
        temps = [float(x) for x in T_ext]
    except Exception:
        return None

    days = [calendar.monthrange(2021, m)[1] for m in range(1, 13)]  # non-leap year
    num = sum(t * d for t, d in zip(temps, days))
    den = float(sum(days))
    return num / den if den > 0 else None


def compute_f_cor(theta_e_avg_C: float | None) -> float | None:
    """
    CECB Eq.57: f_cor = 1 + (9.4 - theta_e_avg) * 0.06
    """
    if theta_e_avg_C is None or not np.isfinite(theta_e_avg_C):
        return None
    return 1.0 + (9.4 - float(theta_e_avg_C)) * 0.06


def theta_e_avg_from_monthly_t_ext(t_ext_monthly: list[float]) -> Optional[float]:
    """
    Compute annual mean outdoor temperature θe,avg from 12 monthly mean values.
    Uses day-weighted average (non-leap year) for robustness.
    """
    if not isinstance(t_ext_monthly, list) or len(t_ext_monthly) != 12:
        return None

    vals = np.array(t_ext_monthly, dtype=float)
    if not np.isfinite(vals).all():
        return None

    # Non-leap year day weights (CECB uses annual mean; leap handling is negligible here)
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    return float(np.sum(vals * days) / np.sum(days))


def f_cor_from_theta_e_avg(theta_e_avg_c: float) -> float:
    """
    CECB temperature correction factor:
      f_cor = 1 + (9.4°C - θe,avg) * 0.06 [1/K]
    """
    return float(1.0 + (9.4 - float(theta_e_avg_c)) * 0.06)

# -----------------------------
# Reference values (SIA 380)
# Qh,li = Qh,li0 + ΔQh,li * (Ath / AE)
# Here we use AE = SRE (your current available reference area)
# Units: kWh/m²·a
# -----------------------------
_QH_LI_REF = {
    # category_code: (Qh_li0, dQh_li)
    "I":  (13.0, 15.0),  # residential multi-family
    "II": (16.0, 15.0),  # residential single-family
    "III": (13.0, 15.0), # administration
    "IV": (14.0, 15.0),  # schools
    "V":  (7.0, 14.0),   # commerce/retail
    "VI": (16.0, 15.0),  # restaurants
    "VII": (18.0, 15.0), # assembly
    "VIII": (18.0, 17.0),# hospitals
    "IX": (10.0, 14.0),  # industry
    "X":  (14.0, 14.0),  # storage/depots
    "XI": (16.0, 14.0),  # sport
    "XII": (15.0, 18.0), # indoor pools
}

# -----------------------------
# Reference values for ELECTRICITY demand (from MOPEC)
# Units: MJ/m²·a  -> we convert to kWh/m²·a via /3.6
# -----------------------------
_E_EL_REF_MJ_M2A = {
    "I":   100.0,  # habitat collectif
    "II":   80.0,  # habitat individuel
    "III":  80.0,  # administration
    "IV":   40.0,  # ecoles
    "V":   120.0,  # commerce
    "VI":  120.0,  # restauration
    "VII":  60.0,  # lieux de rassemblement
    "VIII": 100.0, # hopitaux
    "IX":   60.0,  # industrie
    "X":    20.0,  # depots
    "XI":   20.0,  # installations sportives
    "XII": 200.0,  # piscines couvertes
}

def compute_e_el_limit_kwh_m2(type_usage_code: str) -> Dict[str, Any]:
    """
    Returns the SIA-like reference electricity demand limit (kWh/m²·a),
    converted from MJ/m²·a using 1 kWh = 3.6 MJ.
    """
    code = (type_usage_code or "").strip()
    cat = _TYPE_TO_CAT.get(code, None)
    if cat is None:
        return {
            "ok": False,
            "error": f"Unknown or missing type_usage: {type_usage_code!r}",
            "type_usage": type_usage_code,
        }

    mj = float(_E_EL_REF_MJ_M2A[cat])
    kwh = mj / 3.6

    return {
        "ok": True,
        "type_usage": code,
        "category": cat,
        "E_el_ref_MJ_m2a": float(mj),
        "E_el_ref_kWh_m2a": float(kwh),
        "conversion": "kWh/m²·a = MJ/m²·a / 3.6",
    }


# Map your Phase-1 EN codes -> category (I..XII)
_TYPE_TO_CAT = {
    "residential_multi": "I",
    "residential_single": "II",
    "administration": "III",
    "school": "IV",
    "commerce": "V",
    "restaurant": "VI",
    "assembly": "VII",
    "hospital": "VIII",
    "industry": "IX",
    "storage": "X",
    "sport": "XI",
    "indoor_pool": "XII",
}

# Class thresholds from your screenshot (R in %)
# A: 0-50, B: 50-100, C: 100-150, D: 150-200, E: 200-250, F: 250-300, G: >300
def _class_from_r_pct(r_pct: float) -> str:
    r = float(r_pct)
    if r <= 50.0: return "A"
    if r <= 100.0: return "B"
    if r <= 150.0: return "C"
    if r <= 200.0: return "D"
    if r <= 250.0: return "E"
    if r <= 300.0: return "F"
    return "G"


def compute_qh_li_kwh_m2(type_usage_code: str, sre_m2: float, ath_m2: float) -> Dict[str, Any]:
    """
    Returns:
      - qh_li_kwh_m2
      - inputs + category + sources for traceability
    """
    code = (type_usage_code or "").strip()
    cat = _TYPE_TO_CAT.get(code, None)
    if cat is None:
        return {
            "ok": False,
            "error": f"Unknown or missing type_usage: {type_usage_code!r}",
            "type_usage": type_usage_code,
        }

    q0, dq = _QH_LI_REF[cat]

    sre = float(sre_m2 or 0.0)
    ath = float(ath_m2 or 0.0)

    if sre <= 0:
        return {
            "ok": False,
            "error": "SRE must be > 0 to compute Qh,li.",
            "type_usage": type_usage_code,
            "category": cat,
        }

    ratio = ath / sre  # Ath/AE with AE=SRE
    qh_li = float(q0 + dq * ratio)

    return {
        "ok": True,
        "type_usage": code,
        "category": cat,
        "qh_li0_kwh_m2": float(q0),
        "dqh_li_kwh_m2": float(dq),
        "ath_over_sre": float(ratio),
        "qh_li_kwh_m2": float(qh_li),
        "assumption": {"AE_equals": "SRE"},
    }


def compute_sia_results(project: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute SIA/CECB-like thermal envelope label inputs (baseline measured).
    Stores ONLY derived KPIs from existing measured thermal demand timeseries.
    """
    ouvrages = project.get("ouvrages", []) or []
    ts_by_bat = results.get("timeseries_by_batiment") or {}

    out = {
        "thermal": {
            "meta": {
                "method": "Qh,li = Qh,li0 + dQh,li*(Ath/SRE); R% = 100 * (Qh_measured_specific / Qh,li); class A..G from R%",
                "AE_assumption": "AE = SRE",
                "units": {
                    "Qh_measured_kWh_y": "kWh/a",
                    "Qh_measured_specific_kWh_m2": "kWh/m2/a",
                    "Qh_li_kWh_m2": "kWh/m2/a",
                    "R_pct": "%",
                },
            },
            "by_batiment": {},
            "global": {},
        }
    }

        # Build per-building SRE and Ath from project geometry inputs
    # Priority:
    #   1) project["categories_ouvrages"] (Phase 1 "Units" table source)
    #   2) project["ouvrages"] (legacy)
    by_bat_geom: Dict[str, Dict[str, float]] = {}

    batiments = project.get("batiments", []) or []
    units = project.get("categories_ouvrages", []) or []

    def _bat_id_from_unit_row(row: Dict[str, Any]) -> Optional[str]:
        # 1) explicit batiment_id
        if "batiment_id" in row and row.get("batiment_id") not in (None, ""):
            try:
                return str(int(row.get("batiment_id")))
            except Exception:
                return str(row.get("batiment_id"))

        # 2) explicit batiment name -> index in project["batiments"]
        bname = row.get("batiment")
        if isinstance(bname, str) and bname.strip():
            matches = []
            for i, b in enumerate(batiments):
                if str(b.get("nom", "")).strip() == bname.strip():
                    matches.append(i)
            if len(matches) == 1:
                return str(matches[0])
            if len(matches) > 1:
                return None  # ambiguous

        # 3) only one building -> safe fallback
        if len(batiments) == 1:
            return "0"

        return None

    if units:
        # categories_ouvrages keys (current Phase 1): sre, surface_enveloppe, type
        for row in units:
            if not isinstance(row, dict):
                continue

            bat_id = _bat_id_from_unit_row(row)
            if bat_id is None:
                # Can't allocate to a building reliably (multi-building + missing mapping)
                continue

            sre = float(row.get("sre") or 0.0)
            ath = float(row.get("surface_enveloppe") or 0.0)
            tcode = (row.get("type") or "").strip()

            d = by_bat_geom.setdefault(bat_id, {"sre_m2": 0.0, "ath_m2": 0.0, "type_usage": ""})
            d["sre_m2"] += sre
            d["ath_m2"] += ath
            if tcode:
                d["type_usage"] = tcode

    else:
        # Legacy: derive from ouvrages if categories_ouvrages not present
        for ouv in ouvrages:
            bat_id = str(ouv.get("batiment_id", ""))
            if bat_id == "":
                continue

            sre = float(ouv.get("sre_m2") or 0.0)
            ath = float(ouv.get("surface_enveloppe_m2") or 0.0)
            tcode = (ouv.get("type_usage") or "").strip()

            d = by_bat_geom.setdefault(bat_id, {"sre_m2": 0.0, "ath_m2": 0.0, "type_usage": ""})
            d["sre_m2"] += sre
            d["ath_m2"] += ath
            if tcode:
                d["type_usage"] = tcode


    # Compute from thermal_measured demand_th_kWh
    g_sum_kwh = 0.0
    g_sum_sre = 0.0

    for bat_id, geom in by_bat_geom.items():
        ts = ts_by_bat.get(bat_id) or {}
        th = (ts.get("thermal_measured") or {})
        demand = th.get("demand_th_kWh", None)

        if not isinstance(demand, list):
            continue

        q_meas = float(np.nansum(np.array(demand, dtype=float)))
        sre = float(geom.get("sre_m2") or 0.0)
        ath = float(geom.get("ath_m2") or 0.0)
        tcode = geom.get("type_usage") or ""

        if sre <= 0:
            rec = {
                "ok": False,
                "error": "SRE is 0 (cannot normalize).",
                "Qh_measured_kWh_y": q_meas,
                "sre_m2": sre,
                "ath_m2": ath,
                "type_usage": tcode,
            }
            out["thermal"]["by_batiment"][bat_id] = rec
            continue

        q_spec = q_meas / sre
        ref = compute_qh_li_kwh_m2(tcode, sre_m2=sre, ath_m2=ath)
        if not ref.get("ok"):
            rec = {
                "ok": False,
                "error": ref.get("error"),
                "Qh_measured_kWh_y": q_meas,
                "Qh_measured_specific_kWh_m2": float(q_spec),
                "sre_m2": sre,
                "ath_m2": ath,
                "type_usage": tcode,
            }
            out["thermal"]["by_batiment"][bat_id] = rec
            continue

        q_lim = float(ref["qh_li_kwh_m2"])
        r_pct = 100.0 * (q_spec / q_lim) if q_lim > 0 else np.nan
        cls = _class_from_r_pct(r_pct) if np.isfinite(r_pct) else None

        out["thermal"]["by_batiment"][bat_id] = {
            "ok": True,
            "type_usage": tcode,
            "category": ref["category"],
            "sre_m2": sre,
            "ath_m2": ath,
            "ath_over_sre": ref["ath_over_sre"],
            "Qh_measured_kWh_y": q_meas,
            "Qh_measured_specific_kWh_m2": float(q_spec),
            "Qh_li_kWh_m2": q_lim,
            "R_pct": float(r_pct),
            "class": cls,
            "ref": ref,
        }

        g_sum_kwh += q_meas
        g_sum_sre += sre

    if g_sum_sre > 0:
        out["thermal"]["global"] = {
            "Qh_measured_kWh_y": float(g_sum_kwh),
            "Qh_measured_specific_kWh_m2": float(g_sum_kwh / g_sum_sre),
        }
        # ---------------------------------
        # Electricity (baseline measured)
        # ---------------------------------
        out["electricity"] = {"by_batiment": {}}
        
        g_sum_kwh_el = 0.0
        g_sum_sre_el = 0.0
        
        for bat_id, geom in by_bat_geom.items():
            ts = ts_by_bat.get(bat_id) or {}
        
            # Best-effort: electricity measured series
            meas = (ts.get("measured") or {})
            load = meas.get("load_kWh", None)
        
            if not isinstance(load, list):
                # If missing, skip silently (no hardcoding)
                continue
        
            e_meas = float(np.nansum(np.array(load, dtype=float)))
            sre = float(geom.get("sre_m2") or 0.0)
            tcode = geom.get("type_usage") or ""
        
            if sre <= 0:
                out["electricity"]["by_batiment"][bat_id] = {
                    "ok": False,
                    "error": "SRE is 0 (cannot normalize).",
                    "E_el_measured_kWh_y": e_meas,
                    "sre_m2": sre,
                    "type_usage": tcode,
                }
                continue
        
            e_spec = e_meas / sre  # kWh/m²·a (since baseline is annual sum)
            ref = compute_e_el_limit_kwh_m2(tcode)
            if not ref.get("ok"):
                out["electricity"]["by_batiment"][bat_id] = {
                    "ok": False,
                    "error": ref.get("error"),
                    "E_el_measured_kWh_y": e_meas,
                    "E_el_measured_specific_kWh_m2a": float(e_spec),
                    "sre_m2": sre,
                    "type_usage": tcode,
                }
                continue
        
            e_lim = float(ref["E_el_ref_kWh_m2a"])
            r_pct = 100.0 * (e_spec / e_lim) if e_lim > 0 else np.nan
            cls = _class_from_r_pct(r_pct) if np.isfinite(r_pct) else None
        
            out["electricity"]["by_batiment"][bat_id] = {
                "ok": True,
                "type_usage": tcode,
                "category": ref["category"],
                "sre_m2": sre,
                "E_el_measured_kWh_y": e_meas,
                "E_el_measured_specific_kWh_m2a": float(e_spec),
                "E_el_limit_kWh_m2a": float(e_lim),
                "R_pct": float(r_pct),
                "class": cls,
                "ref": ref,
            }
        
            g_sum_kwh_el += e_meas
            g_sum_sre_el += sre
        
        if g_sum_sre_el > 0:
            out["electricity"]["global"] = {
                "E_el_measured_kWh_y": float(g_sum_kwh_el),
                "E_el_measured_specific_kWh_m2a": float(g_sum_kwh_el / g_sum_sre_el),
            }

    return out
