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
            val = raw.get("Values", None)
        else:
            val = raw

        if val is None:
            continue

        try:
            return float(val)
        except Exception:
            continue

    return float(default)

def _apply_replacements_to_cashflows(
    cashflows: list,
    capex_total_chf: float,
    lifetime_years: float,
    horizon_years: int,
    replacement_cost_factor: float,
) -> list:
    """
    Ajoute des coûts de remplacement CAPEX sur une série de cashflows.

    Convention:
      - cashflows[y] = cashflow de l'année y
      - année 0 = investissement initial
      - remplacement comptabilisé au début de l'année suivante:
        y = lifetime + 1, 2*lifetime + 1, ... <= horizon_years
    """
    if capex_total_chf <= 0:
        return cashflows

    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be > 0 to apply replacement costs.")

    if horizon_years <= 0:
        return cashflows

    if replacement_cost_factor <= 0:
        raise ValueError("replacement_cost_factor must be > 0 to apply replacement costs.")

    step = int(round(float(lifetime_years)))
    if step <= 0:
        raise ValueError(f"Invalid lifetime after rounding: lifetime_years={lifetime_years}")

    replacement_cost = float(capex_total_chf) * float(replacement_cost_factor)

    y = step + 1
    while y <= horizon_years:
        if y < len(cashflows):
            cashflows[y] -= replacement_cost
        y += step

    return cashflows

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
            "Capex",
            "CAPEX" # ⚠️ chez toi: 1800 CHF/kW
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
            "Opex",
            "OPEX" # ⚠️ ton Excel
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
            "Lifetime (a)", "Durée de vie", "Lifetime" # ⚠️ ton Excel
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

def compute_pv_cashflow_series(
    capex_total_chf: float,
    opex_annual_chf: float,
    annual_selfc_kwh: float,
    annual_inj_kwh: float,
    price_buy_chf_kwh: float,
    price_sell_chf_kwh: float,
    horizon_years: int,
    ru_amount_chf: float = 0.0,
    ru_year: int | None = 0,   # 0 = année investissement, 2 = versement à l’année 2
):
    annual_benefit = (annual_selfc_kwh * price_buy_chf_kwh) + (annual_inj_kwh * price_sell_chf_kwh) - opex_annual_chf

    years = list(range(0, horizon_years + 1))
    cashflows = [0.0] * (horizon_years + 1)

    # année 0 : investissement
    cashflows[0] = -capex_total_chf

    # années 1..N : bénéfices annuels
    for y in range(1, horizon_years + 1):
        cashflows[y] = annual_benefit

    # rétribution ponctuelle
    if ru_amount_chf and (ru_year is not None) and (0 <= ru_year <= horizon_years):
        cashflows[ru_year] += ru_amount_chf

    cashflow_cum = []
    s = 0.0
    for cf in cashflows:
        s += cf
        cashflow_cum.append(s)

    payback_year = None
    for y, cum in zip(years, cashflow_cum):
        if cum >= 0:
            payback_year = y
            break

    return {
        "years": years,
        "cashflow_CHF": cashflows,
        "cashflow_cum_CHF": cashflow_cum,
        "payback_year": payback_year,
        "ru_amount_chf": ru_amount_chf,
        "ru_year": ru_year,
    }

def compute_battery_cashflow_series(
    capex_total_chf: float,
    opex_annual_chf: float,
    pv_to_batt_kwh: float,
    grid_to_batt_kwh: float,
    batt_to_load_kwh: float,
    batt_to_grid_kwh: float,
    price_buy_chf_kwh: float,
    price_sell_chf_kwh: float,
    horizon_years: int,
    lifetime_years: float,
    replacement_cost_factor: float = 1.0,   # 1.0 = remplacement complet au même CAPEX
):
    """
    Cashflow batterie (annuel) basé sur flux mesurés/simulés.
    Convention:
      - Année 0: investissement initial (CAPEX)
      - Années 1..horizon: revenus - coûts (OPEX) + remplacements éventuels
    Hypothèses (simples, présentables) :
      - Gains: batt->load * prix_achat + batt->grid * prix_vente
      - Coût charge réseau: grid->batt * prix_achat (on le paie)
      - PV->batt est "gratuit" en cash direct ici (pas de coût d'opportunité PV), MVP pour demain.
    """

    # --- checks
    if horizon_years <= 0:
        raise ValueError("horizon_years must be > 0")
    if capex_total_chf < 0:
        raise ValueError("capex_total_chf must be >= 0")
    if opex_annual_chf < 0:
        raise ValueError("opex_annual_chf must be >= 0")
    if price_buy_chf_kwh <= 0:
        raise ValueError("price_buy_chf_kwh must be > 0")
    if price_sell_chf_kwh < 0:
        raise ValueError("price_sell_chf_kwh must be >= 0")
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be > 0")

    # --- année 0 = CAPEX
    cashflows = [0.0] * (horizon_years + 1)
    cashflows[0] = -float(capex_total_chf)

    # --- cash annuel (années 1..N)
    gain_self = float(batt_to_load_kwh) * float(price_buy_chf_kwh)
    gain_inj = float(batt_to_grid_kwh) * float(price_sell_chf_kwh)

    # coût énergie achetée pour charger
    cost_grid_charge = float(grid_to_batt_kwh) * float(price_buy_chf_kwh)

    annual_net = (gain_self + gain_inj) - cost_grid_charge - float(opex_annual_chf)

    for y in range(1, horizon_years + 1):
        cashflows[y] = float(annual_net)

    # --- remplacements (durée de vie + 1)
    cashflows = _apply_replacements_to_cashflows(
        cashflows=cashflows,
        capex_total_chf=float(capex_total_chf),
        lifetime_years=float(lifetime_years),
        horizon_years=int(horizon_years),
        replacement_cost_factor=float(replacement_cost_factor),
    )

    # cumul + payback + bénéfice
    cashflow_cum = []
    s = 0.0
    payback_year = None
    for y, cf in enumerate(cashflows):
        s += float(cf)
        cashflow_cum.append(s)
        if payback_year is None and y > 0 and s >= 0:
            payback_year = y

    out = {
        "capex_total_chf" : capex_total_chf,
        "years": list(range(0, horizon_years + 1)),
        "cashflow_CHF": cashflows,
        "cashflow_cum_CHF": cashflow_cum,
        "payback_year": payback_year,
        "benefit_horizon_CHF": float(cashflow_cum[-1]) if cashflow_cum else 0.0,
        "assumptions": {
            "annual_gain_self_CHF": gain_self,
            "annual_gain_inj_CHF": gain_inj,
            "annual_cost_grid_charge_CHF": cost_grid_charge,
            "annual_opex_CHF": float(opex_annual_chf),
            "annual_net_CHF": float(annual_net),
            "note": "MVP: no opportunity cost considered for PV → battery flows. To be refined later.",

        },
    }
    return out
