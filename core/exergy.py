# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:17:32 2026

@author: vujic
"""

# core/exergy.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Tuple
import math


def _c_to_k(t_c: float) -> float:
    return float(t_c) + 273.15


def _phi(T0_K: float, T_K: float) -> float:
    """
    Quality factor for heat: phi = 1 - T0/T
    Guard: if T <= 0 or T <= T0, phi may be <= 0 (then exergy of heat is 0 for MVP).
    """
    if T_K is None or not (T_K > 0):
        return 0.0
    if T0_K is None or not (T0_K > 0):
        return 0.0
    return 1.0 - (T0_K / T_K)


def _get_annual_mean_T0_C(project: Dict[str, Any]) -> float:
    """
    MVP: reference temperature T0 = annual mean outdoor temperature
    from project.meta.climate_data['T_ext'] (stored in Phase 1).
    """
    meta = project.get("meta") or {}
    clim = meta.get("climate_data") or {}
    t_ext = clim.get("T_ext")

    if not isinstance(t_ext, list) or len(t_ext) == 0:
        raise ValueError("Exergy MVP requires project.meta.climate_data['T_ext'] as a non-empty list.")

    vals = []
    for v in t_ext:
        try:
            fv = float(v)
            if math.isfinite(fv):
                vals.append(fv)
        except Exception:
            continue

    if not vals:
        raise ValueError("Exergy MVP: could not parse any numeric value from climate_data['T_ext'].")

    return float(sum(vals) / len(vals))


def _get_beta_fuel(tech_params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Try to read a fuel chemical exergy factor from techno params.
    If missing, fallback to 1.0 but return source info (traceability).
    """
    candidates = [
        "Fuel exergy factor",
        "Fuel exergy",
        "Chemical exergy factor",
        "Exergy factor",
    ]

    for k in candidates:
        item = (tech_params or {}).get(k)
        if isinstance(item, dict):
            v = item.get("Values", None)
        else:
            v = item

        if v is None or v == "":
            continue

        try:
            fv = float(v)
            if fv > 0:
                return fv, f"techno:{k}"
        except Exception:
            continue

    return 1.0, "default:1.0_missing_beta"


def _get_exergy_cfg_for_batiment(project: Dict[str, Any], bat_id: int) -> Dict[str, float]:
    """
    Read exergy heating temperature settings from Phase 1 producer config.
    We pick the first thermal producer (engine=boiler_oil) found for this building.
    """
    ouvrages = project.get("ouvrages") or []
    for ov in ouvrages:
        if int(ov.get("batiment_id", -1)) != int(bat_id):
            continue

        for prod in (ov.get("producteurs") or []):
            if prod.get("engine") in ("boiler_oil",):
                ex_cfg = prod.get("exergy_config") or {}
                if "room_temp_C" in ex_cfg and "supply_temp_C" in ex_cfg and "return_temp_C" in ex_cfg:
                    return {
                        "room_temp_C": float(ex_cfg["room_temp_C"]),
                        "supply_temp_C": float(ex_cfg["supply_temp_C"]),
                        "return_temp_C": float(ex_cfg["return_temp_C"]),
                    }

    raise ValueError(
        f"Missing exergy temperatures for building {bat_id}. "
        f"Configure them in Phase 1 inside the thermal producer settings."
    )


# --- ADD near other helpers in core/exergy.py (e.g. after _get_beta_fuel or before compute_exergy_results) ---

def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _get_boiler_max_temp_C(tech_params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Read boiler maximum temperature from techno params (Excel).
    No hardcoding of numeric values: must come from techno sheet.
    Returns (Tmax_C, source_key).
    """
    candidates = [
        "Max temperature",
        "Max supply temperature",
        "T max",
        "Tmax",
        "Température max",
        "Température max chaudière",
        "Boiler max temperature",
    ]

    for k in candidates:
        item = (tech_params or {}).get(k)
        if isinstance(item, dict):
            v = item.get("Values", None)
        else:
            v = item
        if v is None or v == "":
            continue
        try:
            fv = float(v)
            if fv > -50 and fv < 300:  # just sanity for °C
                return fv, f"techno:{k}"
        except Exception:
            continue

    raise ValueError(
        "Exergy heating requires a boiler maximum temperature from techno parameters. "
        "Please add a parameter like 'Max temperature' (°C) in the thermal technology Excel sheet."
    )


def compute_exergy_results(project: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heating-only EXERGY (defendable efficiencies).

    Per building (boiler MVP):
      - eta_ex_machine = eta_th * (1 - T_dist_K / T_max_K)
      - eta_ex_distribution = (1 - T_room_K / T_dist_K)
      - eta_ex_global = eta_ex_machine * eta_ex_distribution

    Notes:
      - No climate term enters eta_ex_global (comparability focus).
      - Climate-based context can be stored separately (optional) but not used in the product.
    """
    ex = {
        "meta": {
            "scope": "heating_only",
            "units": "-",
            "method": "eta_ex_machine=eta_th*(1-Tdist/Tmax), eta_ex_dist=(1-Troom/Tdist), eta_global=product",
            "temps_unit": "degC",
        },
        "by_batiment": {},
        "global": {},
    }

    thermal_by_bat = results.get("thermal_by_batiment") or {}

    # Aggregation strategy for global:
    # weighted by delivered useful heat (heat_out_th_kWh), so global reflects "system performance for delivered service".
    g = {
        "heat_out_th_kWh": 0.0,
        "eta_ex_machine_wavg": None,
        "eta_ex_distribution_wavg": None,
        "eta_ex_global_wavg": None,
    }

    sum_w = 0.0
    sum_eta_m = 0.0
    sum_eta_d = 0.0
    sum_eta_g = 0.0

    # Optional climate info (does NOT enter global exergy)
    try:
        T0_C = _get_annual_mean_T0_C(project)
        ex["meta"]["T0_C_optional"] = float(T0_C)
        ex["meta"]["T0_source_optional"] = "annual_mean_outdoor_temperature"
    except Exception:
        ex["meta"]["T0_C_optional"] = None
        ex["meta"]["T0_source_optional"] = None

    for bat_id, th in thermal_by_bat.items():
        totals = (th or {}).get("totals") or {}
        engine = (th or {}).get("engine")

        # MVP: only oil boiler for now
        if engine != "boiler_oil":
            continue

        # Delivered heat & fuel
        Q_kWh = float(totals.get("heat_out_th_kWh", 0.0) or 0.0)
        fuel_kWh = float(totals.get("fuel_in_kWh", 0.0) or 0.0)

        # Thermal efficiency from simulation (defendable & consistent)
        eta_th = (Q_kWh / fuel_kWh) if fuel_kWh > 0 else None

        # Exergy temps from Phase 1 producer config
        temps = _get_exergy_cfg_for_batiment(project, int(bat_id))
        T_room_C = float(temps["room_temp_C"])
        T_sup_C = float(temps["supply_temp_C"])
        T_ret_C = float(temps["return_temp_C"])
        T_dist_C = 0.5 * (T_sup_C + T_ret_C)

        # Techno params used from flow block (best traceability)
        tech_params = None
        fb_list = (results.get("flows") or {}).get("batiments") or {}
        for b in (fb_list.get(bat_id) or []):
            if b.get("type") == "thermal_boiler_oil":
                meta = b.get("meta") or {}
                tech_params = meta.get("techno_params_used") or {}
                break

        Tmax_C, Tmax_source = _get_boiler_max_temp_C(tech_params or {})

        # Convert to Kelvin for ratios
        T_room_K = _c_to_k(T_room_C)
        T_dist_K = _c_to_k(T_dist_C)
        T_max_K = _c_to_k(Tmax_C)

        # --- Efficiencies (clamped) ---
        # Machine: includes thermal efficiency + "temperature mismatch" between max capability and used distribution level
        if eta_th is None:
            eta_ex_machine = None
        else:
            eta_ex_machine = _clamp01(float(eta_th) * (1.0 - (T_dist_K / T_max_K)))

        # Distribution: mismatch between distribution level and room need (no losses estimated)
        eta_ex_dist = _clamp01((T_room_K / T_dist_K)) if (T_dist_K > 0) else None

        # Global product
        if eta_ex_machine is None or eta_ex_dist is None:
            eta_ex_global = None
        else:
            eta_ex_global = _clamp01(float(eta_ex_machine) * float(eta_ex_dist))

        # Store per building (no kWh_ex, only defendable ratios + inputs)
        ex["by_batiment"][str(bat_id)] = {
            "heating": {
                "heat_out_th_kWh": Q_kWh,
                "fuel_in_kWh": fuel_kWh,
                "eta_th": eta_th,
                "eta_ex_machine": eta_ex_machine,
                "eta_ex_distribution": eta_ex_dist,
                "eta_ex_global": eta_ex_global,
                "temps_C": {
                    "room": T_room_C,
                    "supply": T_sup_C,
                    "return": T_ret_C,
                    "dist": T_dist_C,
                    "tmax_boiler": float(Tmax_C),
                },
                "sources": {
                    "tmax_boiler": Tmax_source,
                    "eta_th": "simulation: heat_out_th_kWh / fuel_in_kWh",
                },
                "formulas": {
                    "eta_ex_machine": "eta_th*(1 - Tdist_K/Tmax_K)",
                    "eta_ex_distribution": "(Troom_K/Tdist_K)",
                    "eta_ex_global": "eta_ex_machine*eta_ex_distribution",
                },
                "details": {
                    "Troom_K": T_room_K,
                    "Tdist_K": T_dist_K,
                    "Tmax_K": T_max_K,
                    "term_machine_temp": (1.0 - (T_dist_K / T_max_K)) if T_max_K > 0 else None,
                    "term_dist_temp": (1.0 - (T_room_K / T_dist_K)) if T_dist_K > 0 else None,
                },
            }
        }

        # Weighted aggregation on delivered heat (Q_kWh)
        if Q_kWh > 0 and eta_ex_machine is not None and eta_ex_dist is not None and eta_ex_global is not None:
            sum_w += Q_kWh
            sum_eta_m += Q_kWh * float(eta_ex_machine)
            sum_eta_d += Q_kWh * float(eta_ex_dist)
            sum_eta_g += Q_kWh * float(eta_ex_global)

        g["heat_out_th_kWh"] += Q_kWh

    if sum_w > 0:
        g["eta_ex_machine_wavg"] = sum_eta_m / sum_w
        g["eta_ex_distribution_wavg"] = sum_eta_d / sum_w
        g["eta_ex_global_wavg"] = sum_eta_g / sum_w
    else:
        g["eta_ex_machine_wavg"] = None
        g["eta_ex_distribution_wavg"] = None
        g["eta_ex_global_wavg"] = None

    ex["global"] = g
    return ex

