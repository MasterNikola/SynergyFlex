# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:18:02 2025

@author: infor
"""

# core/economics.py
from __future__ import annotations
from typing import Dict, Any, Iterable


def get_param_numeric(
    tech_params: Dict[str, Any],
    candidate_keys: Iterable[str],
    default: float = 0.0,
) -> float:
    """
    Récupère une valeur numérique dans tech_params en testant plusieurs clés possibles.

    tech_params ressemble à :
    {
        "Capex (CHF/kW)": {"valeur": 1800, "unite": "CHF/kW"},
        "Opex [CHF/an]": {"valeur": 100, "unite": "CHF/an"},
        ...
    }
    """
    if not tech_params:
        return float(default)

    for key in candidate_keys:
        raw = tech_params.get(key)
        if raw is None:
            continue

        if isinstance(raw, dict):
            val = raw.get("valeur", None)
        else:
            val = raw

        if val is None:
            continue

        try:
            return float(val)
        except Exception:
            continue

    return float(default)


def compute_tech_economics(
    annual_prod_kwh: float,
    installed_kw: float,
    tech_params: Dict[str, Any],
    default_lifetime_years: float = 25.0,
) -> Dict[str, float]:
    """
    Calcule les indicateurs économiques génériques pour une machine.
    """

    annual_prod_kwh = float(annual_prod_kwh or 0.0)
    installed_kw = float(installed_kw or 0.0)

    if annual_prod_kwh <= 0.0 or installed_kw <= 0.0:
        return {
            "capex_total_CHF": 0.0,
            "opex_annual_CHF": 0.0,
            "lifetime_years": float(default_lifetime_years),
            "lcoe_machine_CHF_kWh": 0.0,
        }

    # --- CAPEX ---
    capex_specific_kw = get_param_numeric(
        tech_params,
        [
            "Capex (CHF/kW)",
            "Capex [CHF/kW]",
            "CAPEX (CHF/kW)",
            "Investissement spécifique (CHF/kW)",
            "Capex spécifique (CHF/kW)",
            "Capex",  # ⚠️ chez toi: 1800 CHF/kW
        ],
        default=0.0,
    )
    capex_total_direct = get_param_numeric(
        tech_params,
        [
            "Capex (CHF)",
            "Capex total (CHF)",
            "Investissement total (CHF)",
            # ⚠️ NE PAS mettre "Capex" ici, sinon on le prend pour un montant total
        ],
        default=0.0,
    )

    if capex_total_direct > 0:
        capex_total = capex_total_direct
    elif capex_specific_kw > 0 and installed_kw > 0:
        capex_total = capex_specific_kw * installed_kw
    else:
        capex_total = 0.0

    # --- OPEX ---
    opex_global = get_param_numeric(
        tech_params,
        [
            "Opex (CHF/an)", "Opex [CHF/an]", "OPEX (CHF/an)",
            "Coûts fixes (CHF/an)",
            "Opex",  # ⚠️ ton Excel
        ],
        default=0.0,
    )
    opex_specific_kw_an = get_param_numeric(
        tech_params,
        [
            "Opex (CHF/kW/an)", "Opex [CHF/kW/an]",
            "Opex spécifique (CHF/kW/an)",
        ],
        default=0.0,
    )
    opex_specific_kw = get_param_numeric(
        tech_params,
        [
            "Opex (CHF/kW)", "Opex [CHF/kW]",
            "Opex spécifique (CHF/kW)",
        ],
        default=0.0,
    )

    opex_annual = 0.0
    if opex_global > 0:
        opex_annual += opex_global
    if opex_specific_kw_an > 0 and installed_kw > 0:
        opex_annual += opex_specific_kw_an * installed_kw
    if opex_specific_kw > 0 and installed_kw > 0:
        # On interprète ça comme des coûts annuels proportionnels à la puissance.
        opex_annual += opex_specific_kw * installed_kw

    # --- Durée de vie ---
    lifetime_years = get_param_numeric(
        tech_params,
        [
            "Durée de vie (ans)", "Durée de vie [ans]",
            "Lifetime (a)", "Durée de vie",  # ⚠️ ton Excel
        ],
        default=default_lifetime_years,
    )
    if lifetime_years <= 0:
        lifetime_years = float(default_lifetime_years)

    # --- LCOE (simplifié, pas d’actualisation) ---
    if annual_prod_kwh > 0:
        lcoe = (capex_total / lifetime_years + opex_annual) / annual_prod_kwh
    else:
        lcoe = 0.0

    return {
        "capex_total_CHF": float(capex_total),
        "opex_annual_CHF": float(opex_annual),
        "lifetime_years": float(lifetime_years),
        "lcoe_machine_CHF_kWh": float(lcoe),
    }

