# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 08:22:29 2026

@author: vujic
"""

# core/thermal_producer_module.py
# -*- coding: utf-8 -*-

from __future__ import annotations

"""Thermal producers – MVP implementation.

Contract (Core):
  - Inputs are always *useful thermal demand* in kWh_th per timestep.
  - For boilers (oil/gas/biomass) and electric heaters, "Efficiency" is a *thermal efficiency* (<= 1).
  - For heat pumps, "Efficiency" may be provided as COP (> 1).
  - If the techno sheet stores Efficiency in %, we convert to fraction (e.g., 90% -> 0.9, 330% -> 3.3).

This module returns:
  - JSON-friendly timeseries (lists)
  - a Sankey flow_block for thermal supply
  - aggregated totals for KPI/economics/CO2 modules.

No hardcoding: every numeric comes from producer config (installed power) and techno params.
"""



from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.nodes_ontology import node_heat_load, node_heat_loss, node_heat_prod, node_fuel_oil
from core.utils.timebase import infer_dt_hours_from_index


def _get_param_value_and_unit(tech_params: Dict[str, Any], key: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, unit) for a given techno parameter key."""
    if not tech_params:
        return None, None
    item = tech_params.get(key)
    if isinstance(item, dict):
        v = item.get("Values", None)
        u = item.get("Units", None)
    else:
        v = item
        u = None
    if v is None or v == "":
        return None, u
    try:
        return float(v), u
    except Exception:
        return None, u


def _get_param_numeric(tech_params: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    v, _u = _get_param_value_and_unit(tech_params, key)
    return v if v is not None else default


def _normalize_efficiency_from_units(tech_params: Dict[str, Any]) -> float:
    """Normalize the 'Efficiency' parameter.

    Rules:
      - If Units contains '%', divide by 100.
      - Else keep as-is (accept either fraction or COP), but must be > 0.
    """
    v, u = _get_param_value_and_unit(tech_params, "Efficiency")
    if v is None:
        raise ValueError("Thermal producer: missing techno parameter 'Efficiency'.")
    if v <= 0:
        raise ValueError(f"Thermal producer: invalid Efficiency (<=0): {v}")
    if isinstance(u, str) and ("%" in u):
        v = v / 100.0
    return float(v)


def compute_oil_boiler_from_demand_ts(
    demand_th_kwh_ts: pd.Series,
    installed_power_kw: float,
    tech_params: Dict[str, Any],
    *,
    batiment_id: Any,
    batiment_nom: str,
    label: str = "Oil boiler",
    producer_index: int = 1,
) -> Dict[str, Any]:
    """Compute thermal supply for an oil boiler from thermal demand time series."""
    if demand_th_kwh_ts is None or len(demand_th_kwh_ts) == 0:
        raise ValueError("Oil boiler: demand_th_kwh_ts is empty.")

    try:
        p_inst = float(installed_power_kw)
    except Exception:
        raise ValueError("Oil boiler: installed_power_kw must be numeric.")
    if p_inst <= 0:
        raise ValueError("Oil boiler: installed_power_kw must be > 0.")

    # Tech constraints
    p_min = _get_param_numeric(tech_params, "Min power", default=0.0) or 0.0
    p_max = _get_param_numeric(tech_params, "Max power", default=None)
    if p_min and (p_inst < float(p_min)):
        raise ValueError(f"Oil boiler: installed power ({p_inst} kW) below techno min power ({p_min} kW).")
    if (p_max is not None) and (float(p_max) > 0):
        p_avail = min(p_inst, float(p_max))
    else:
        p_avail = p_inst

    eta = _normalize_efficiency_from_units(tech_params)

    dt_h = float(infer_dt_hours_from_index(demand_th_kwh_ts.index))
    if not (dt_h > 0):
        raise ValueError("Oil boiler: could not infer dt_h from demand index.")

    demand = pd.to_numeric(demand_th_kwh_ts, errors="coerce").fillna(0.0).clip(lower=0.0)
    q_max = float(p_avail) * dt_h

    heat_out = demand.clip(upper=q_max)
    unmet = (demand - heat_out).clip(lower=0.0)

    fuel_in = heat_out / float(eta)
    losses = (fuel_in - heat_out).clip(lower=0.0)

    price = _get_param_numeric(tech_params, "Energy cost", default=0.0) or 0.0
    co2_factor = _get_param_numeric(tech_params, "CO2 emissions", default=0.0) or 0.0

    cost_chf = fuel_in * float(price)
    co2_kg = fuel_in * float(co2_factor)
    
    # -----------------------------
    # Economics (standardized totals)
    # LCOE is recalculated from simulated production and costs.
    # Convention: fuel cost is included in OPEX (opex_annual_CHF).
    # -----------------------------
    capex_chf_per_kw = _get_param_numeric(tech_params, "CAPEX", default=0.0) or 0.0
    opex_fixed_chf_per_year = _get_param_numeric(tech_params, "OPEX", default=0.0) or 0.0
    lifetime_years = _get_param_numeric(tech_params, "Lifetime", default=None)
    
    # CAPEX total
    capex_total_chf = float(p_inst) * float(capex_chf_per_kw)
    
    # Variable OPEX: fuel
    opex_fuel_chf = float(cost_chf.sum())
    
    # Total annual OPEX = fixed + fuel (MVP convention)
    opex_annual_chf = float(opex_fixed_chf_per_year) + float(opex_fuel_chf)
    
    # Production for LCOE denominator (useful heat delivered)
    production_machine_kwh = float(heat_out.sum())
    
    # Annualized CAPEX
    if lifetime_years is None or lifetime_years <= 0:
        # Keep traceability: if missing, set to None and avoid computing LCOE
        lifetime_years = None
        capex_annual_chf = None
    else:
        capex_annual_chf = float(capex_total_chf) / float(lifetime_years)
    
    # LCOE
    if production_machine_kwh > 0 and capex_annual_chf is not None:
        lcoe_machine_chf_per_kwh = (float(capex_annual_chf) + float(opex_annual_chf)) / float(production_machine_kwh)
    else:
        lcoe_machine_chf_per_kwh = None

    totals = {
        "demand_th_kWh": float(demand.sum()),
        "heat_out_th_kWh": float(heat_out.sum()),
        "fuel_in_kWh": float(fuel_in.sum()),
        "losses_kWh": float(losses.sum()),
        "unmet_th_kWh": float(unmet.sum()),
        "efficiency": float(eta),
        "installed_power_kW": float(p_inst),
        "available_power_kW": float(p_avail),
        "dt_h_med": float(dt_h),
    
        # Operational costs & CO2 (fuel-based)
        "fuel_cost_CHF": float(cost_chf.sum()),
        "fuel_scope1_kgCO2e": float(co2_kg.sum()),
    
        # --- Standardized economics for compute_economics_from_flow_blocks ---
        "capex_total_CHF": float(capex_total_chf),
        "opex_fixed_annual_CHF": float(opex_fixed_chf_per_year),
        "opex_fuel_annual_CHF": float(opex_fuel_chf),
    
        # Convention: fuel included in OPEX (for LCOE + aggregation)
        "opex_annual_CHF": float(opex_annual_chf),
    
        "lifetime_years": lifetime_years,
        "production_machine_kWh": float(production_machine_kwh),
        "lcoe_machine_CHF_kWh": lcoe_machine_chf_per_kwh,
    }


    fuel_id = node_fuel_oil()
    prod_id = node_heat_prod("OIL_BOILER", index=int(producer_index))
    load_id = node_heat_load()
    loss_id = node_heat_loss()

    flow_block = {
        "name": f"Thermal supply – {label} ({batiment_nom})",
        "type": "thermal_boiler_oil",
        "meta": {
            "batiment_id": batiment_id,
            "batiment_nom": batiment_nom,
            "techno": label,
            "engine": "boiler_oil",
            "installed_power_kW": float(p_inst),
            "available_power_kW": float(p_avail),
            "efficiency": float(eta),
            "techno_params_used": tech_params or {},
            "units": {"fuel_in": "kWh_fuel", "heat_out": "kWh_th"},
        },
        "nodes": [
            {"id": fuel_id, "label": "Fuel oil", "group": "fuel"},
            {"id": prod_id, "label": "Oil boiler", "group": "prod_th"},
            {"id": load_id, "label": "Thermal demand", "group": "demande"},
            {"id": loss_id, "label": "Thermal losses", "group": "loss"},
        ],
        "links": [
            {"source": fuel_id, "target": prod_id, "value": totals["fuel_in_kWh"]},
            {"source": prod_id, "target": load_id, "value": totals["heat_out_th_kWh"]},
            {"source": prod_id, "target": loss_id, "value": totals["losses_kWh"]},
        ],
        "totals": totals,
    }

    return {
        "profiles": {
            "demand_th_kWh": demand.values.tolist(),
            "heat_out_th_kWh": heat_out.values.tolist(),
            "fuel_in_kWh": fuel_in.values.tolist(),
            "losses_kWh": losses.values.tolist(),
            "unmet_th_kWh": unmet.values.tolist(),
            "fuel_cost_CHF": cost_chf.values.tolist(),
            "fuel_scope1_kgCO2e": co2_kg.values.tolist(),
        },
        "totals": totals,
        "flow_block": flow_block,
    }
