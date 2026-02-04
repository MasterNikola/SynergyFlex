# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 08:12:24 2025

@author: infor
"""

# core/bat_li_ion_module.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, Optional, List
import math
import pandas as pd
from core.storage_dispatch import dispatch_electric_storage
from core.nodes_ontology import node_battery, node_pv, node_elec_grid, node_elec_load

def _num(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace(",", ".")
        return float(s)
    except Exception:
        return float(default)


def _get_param(params: Dict[str, Any], key: str, default=None):
    """
    params is expected like: { "Capacité min": {"Values": ..., "unite": ...}, ... }
    Returns params[key]["Values"] if present else default.
    """
    if not params:
        return default
    item = params.get(key)
    if not item:
        return default
    return item.get("Values", default)


def _find_block(flow_blocks: List[Dict[str, Any]], block_type: str) -> Optional[Dict[str, Any]]:
    for b in flow_blocks or []:
        if b.get("type") == block_type:
            return b
    return None


def _get_monthly_series(block: Dict[str, Any], series_key_candidates: List[str]) -> Optional[List[float]]:
    """
    Tries to find a monthly series in a block totals or profiles.
    We keep it tolerant because your structure may vary slightly.
    Returns list[12] or None.
    """
    if not block:
        return None

    # try block["profiles"]
    prof = block.get("profiles") or {}
    for k in series_key_candidates:
        if k in prof and isinstance(prof[k], list) and len(prof[k]) == 12:
            return [float(_num(v, 0.0)) for v in prof[k]]

    # try block["totals"]
    tot = block.get("totals") or {}
    for k in series_key_candidates:
        v = tot.get(k)
        if isinstance(v, list) and len(v) == 12:
            return [float(_num(x, 0.0)) for x in v]

    return None


def compute_battery_li_ion_flow_block(
    demand_block: Dict[str, Any],
    pv_block: Optional[Dict[str, Any]],
    storage_cfg: Dict[str, Any],
    dt_hours: float = 730.0 / 12.0,  # ~60.8h; used only if you later convert power->energy monthly
) -> Dict[str, Any]:
    """
    V1 monthly energy-balance model (simple but consistent).
    Dispatch priority:
      - charge from PV surplus (PV - load)
      - discharge to cover deficit (load - PV)
    Optional:
      - grid charge/discharge is ignored in V1 (kept in cfg but not used), unless you ask later.

    Returns a flow_block:
      type="storage_elec"
      totals: charged_kWh, discharged_kWh, losses_kWh, soc_min_kWh, soc_max_kWh, etc.
      profiles: soc_kWh (12), charged_kWh (12), discharged_kWh (12)
      impacts: delta_grid_import_kWh, delta_grid_injection_kWh, delta_self_consumption_kWh
    """
    params = storage_cfg.get("parametres") or {}
    mapping = storage_cfg.get("mapping") or {}
    charge_sources = set(mapping.get("charge_sources") or ["PV"])
    discharge_sinks = set(mapping.get("discharge_sinks") or ["LOAD"])
  
    grid_charge_allowed = bool(storage_cfg.get("grid_charge_allowed", False)) and ("GRID" in charge_sources)
    grid_discharge_allowed = bool(storage_cfg.get("grid_discharge_allowed", False)) and ("GRID" in discharge_sinks)
  
    # V1: on supporte PV->BAT et BAT->LOAD.
    # GRID charge/discharge sera activé plus tard (peak shaving / arbitrage),
    # mais on garde les flags propres.

    # --- sizing / limits
    C = _num(storage_cfg.get("capacity_kwh"), 0.0)
    if C <= 0:
        return {"type": "storage_elec", "totals": {"error": "capacity_kwh <= 0"}, "profiles": {}}
    
    # --- SOC window (no hardcode): priority to explicit cfg, else techno params
    soc_min_frac = storage_cfg.get("soc_min_frac", None)
    soc_max_frac = storage_cfg.get("soc_max_frac", None)
    
    if soc_min_frac is None or soc_max_frac is None:
        # depuis Excel techno (parametres), typiquement en %
        soc_min_raw = _get_param(params, "SOC min", None)
        soc_max_raw = _get_param(params, "SOC max", None)
    
        if soc_min_raw is None or soc_max_raw is None:
            raise ValueError("Batterie: SoC min/max manquants (ni dans storage_cfg, ni dans techno).")
    
        soc_min_frac = _num(soc_min_raw, None)
        soc_max_frac = _num(soc_max_raw, None)
    
        # normalisation % -> fraction
        if soc_min_frac is None or soc_max_frac is None:
            raise ValueError("Batterie: SoC min/max non numériques.")
    
        if soc_min_frac > 1.0: soc_min_frac = soc_min_frac / 100.0
        if soc_max_frac > 1.0: soc_max_frac = soc_max_frac / 100.0
    
    soc_min_frac = float(soc_min_frac)
    soc_max_frac = float(soc_max_frac)
    
    if not (0.0 <= soc_min_frac <= 1.0) or not (0.0 <= soc_max_frac <= 1.0):
        raise ValueError(f"Batterie: SoC min/max hors bornes: min={soc_min_frac}, max={soc_max_frac}")
    if soc_min_frac >= soc_max_frac:
        raise ValueError(f"Batterie: SoC min doit être < SoC max: min={soc_min_frac}, max={soc_max_frac}")

    soc_min = soc_min_frac * C
    soc_max = soc_max_frac * C

    eta = _num(_get_param(params, "Efficiency", 1.0), 1.0)
    # If rendement is given in %, accept 25 -> 0.25
    if eta > 1.0:
        eta = eta / 100.0
    # For V1, we split evenly charge/discharge (sqrt) to avoid double counting
    eta_ch = math.sqrt(eta) if eta > 0 else 1.0
    eta_dis = math.sqrt(eta) if eta > 0 else 1.0

    ## C-rate style parameters: kW/kWh (ex 0.5)
    pch_spec = _num(_get_param(params, "Max charge C-rate", None), None)
    pdis_spec = _num(_get_param(params, "Max discharge C-rate", None), None)
    
    # Backward-compatible aliases (si tu renommes plus tard)
    if pch_spec is None:
        pch_spec = _num(_get_param(params, "Puissance charge max.", 999.0), 999.0)
    if pdis_spec is None:
        pdis_spec = _num(_get_param(params, "Puissance décharge max.", 999.0), 999.0)


    # Monthly energy throughput limits from power (optional V1)
    # energy_max_per_step = Pmax * hours_in_step
    # Here: Pmax = spec * C
    pch_max_kw = pch_spec * C
    pdis_max_kw = pdis_spec * C
    e_ch_max = max(0.0, pch_max_kw * dt_hours)
    e_dis_max = max(0.0, pdis_max_kw * dt_hours)

    standby_loss_per_day = _num(_get_param(params, "Pertes standby", 0.0), 0.0)
    if standby_loss_per_day > 1.0:
        standby_loss_per_day = standby_loss_per_day / 100.0

    # --- get monthly load and PV
    load_kwh_12 = _get_monthly_series(demand_block, ["demand_elec_kWh_monthly", "load_monthly_kWh", "monthly_kWh"])
    if not load_kwh_12:
        # fallback: distribute annual equally
        L = _num((demand_block.get("totals") or {}).get("demand_elec_kWh"), 0.0)
        load_kwh_12 = [L / 12.0] * 12

    pv_kwh_12 = None
    if pv_block:
        pv_kwh_12 = _get_monthly_series(pv_block, ["pv_prod_kWh_monthly", "pv_monthly_kWh", "monthly_kWh"])

    if not pv_kwh_12:
        PV = _num((pv_block or {}).get("totals", {}).get("pv_prod_kWh"), 0.0)
        pv_kwh_12 = [PV / 12.0] * 12

    # --- simulation monthly
    soc = soc_min  # start conservative
    soc_series = []
    ch_series = []
    dis_series = []
    loss_series = []
    delta_import_series = []
    delta_inj_series = []
    delta_sc_series = []

    for m in range(12):
        load = max(0.0, float(load_kwh_12[m]))
        pv = max(0.0, float(pv_kwh_12[m]))

        surplus = max(pv - load, 0.0) if ("PV" in charge_sources) else 0.0
        deficit = max(load - pv, 0.0) if ("LOAD" in discharge_sinks) else 0.0

        # standby loss (approx): loss = soc * loss_per_day * days_in_month
        days = 365.0 / 12.0
        standby_loss = soc * standby_loss_per_day * days
        soc = max(soc_min, soc - standby_loss)

        # charge from PV surplus
        charge_possible = min(surplus, (soc_max - soc) / max(eta_ch, 1e-9), e_ch_max)
        soc += charge_possible * eta_ch

        # discharge to cover deficit
        discharge_possible = min(deficit / max(eta_dis, 1e-9), (soc - soc_min), e_dis_max)
        soc -= discharge_possible
        delivered = discharge_possible * eta_dis

        # accounting
        charged_kwh = charge_possible
        discharged_kwh = delivered
        losses_kwh = standby_loss + (charged_kwh * (1 - eta_ch)) + (discharge_possible * (1 - eta_dis))

        # impacts vs no battery
        # - battery reduces grid import by delivered (if deficit exists)
        # - battery reduces grid injection by charged_kwh (if surplus exists)
        # - battery increases self-consumption by min(charged, surplus) ≈ charged_kwh
        delta_import = min(deficit, delivered)
        delta_inj = min(surplus, charged_kwh)
        delta_sc = delta_inj  # charging from PV surplus becomes self-consumed later (approx)

        soc_series.append(float(soc))
        ch_series.append(float(charged_kwh))
        dis_series.append(float(discharged_kwh))
        loss_series.append(float(losses_kwh))
        delta_import_series.append(float(delta_import))
        delta_inj_series.append(float(delta_inj))
        delta_sc_series.append(float(delta_sc))

    totals = {
        "energy_charged_kWh": sum(ch_series),
        "energy_discharged_kWh": sum(dis_series),
        "losses_kWh": sum(loss_series),
        "soc_min_kWh": soc_min,
        "soc_max_kWh": soc_max,
        "capacity_kWh": C,
        "pch_max_kw": pch_max_kw,
        "pdis_max_kw": pdis_max_kw,
        "eta_charge": eta_ch,
        "eta_discharge": eta_dis,
    }

    impacts = {
        "delta_grid_import_kWh": sum(delta_import_series),
        "delta_grid_injection_kWh": sum(delta_inj_series),
        "delta_self_consumption_kWh": sum(delta_sc_series),
    }
    
    # ----------------------------
    # ECONOMICS (pour l'agrégation globale)
    # ----------------------------
    # CAPEX spécifique batterie: CHF/kWh
    capex_spec = None
    for k in [
        "Capex (CHF/kWh)", "CAPEX (CHF/kWh)", "Capex spécifique (CHF/kWh)", "CAPEX"
        "CAPEX spécifique (CHF/kWh)", "Capex spécifique", "CAPEX spécifique",
    ]:
        v = _get_param(params, k, None)
        if v is not None:
            capex_spec = _num(v, None)
            break
    
    # OPEX: soit en % du CAPEX, soit CHF/an
    opex_pct = None
    for k in ["Opex (%)", "OPEX (%)", "Opex annuel (%)", "OPEX annuel (%)", "OPEX"]:
        v = _get_param(params, k, None)
        if v is not None:
            opex_pct = _num(v, None)
            break
    
    opex_fixed = None
    for k in ["Opex (CHF/an)", "OPEX (CHF/an)", "Opex annuel (CHF/an)", "OPEX annuel (CHF/an)"]:
        v = _get_param(params, k, None)
        if v is not None:
            opex_fixed = _num(v, None)
            break
    
    lifetime = None
    for k in ["Durée de vie (ans)", "Durée de vie", "Lifetime (years)", "Lifetime"]:
        v = _get_param(params, k, None)
        if v is not None:
            lifetime = _num(v, None)
            break
    if not lifetime or lifetime <= 0:
        lifetime = 10.0  # <-- règle de base batterie à remplacer
    
    capex_total = 0.0
    if capex_spec is not None:
        capex_total = float(capex_spec) * float(C)
    
    # opex: priorité au fixe si défini, sinon % capex si défini
    opex_annual = 0.0
    if opex_fixed is not None:
        opex_annual = float(opex_fixed)
    elif opex_pct is not None and capex_total > 0:
        # accepte 2 ou 0.02
        pct = float(opex_pct)
        if pct > 1.0:
            pct = pct / 100.0
        opex_annual = capex_total * pct
    
    prod_useful = float(totals.get("energy_discharged_kWh", 0.0) or 0.0)
    
    # LCOE "machine" (utile pour agrégation multi-technos)
    if prod_useful > 0 and lifetime > 0:
        lcoe = (capex_total / lifetime + opex_annual) / prod_useful
    else:
        lcoe = 0.0
    
    totals["capex_total_CHF"] = float(capex_total)
    totals["opex_annual_CHF"] = float(opex_annual)
    totals["lifetime_years"] = float(lifetime)
    totals["production_machine_kWh"] = float(prod_useful)
    totals["lcoe_machine_CHF_kWh"] = float(lcoe)
    
    series = {
        "charged_kWh": ch_series,          # 12 points (mensuel)
        "discharged_kWh": dis_series,      # 12 points
        "losses_kWh": loss_series,         # 12 points
        "soc_kWh": soc_series,             # 12 points
        "delta_grid_import_kWh": delta_import_series,   # 12
        "delta_grid_injection_kWh": delta_inj_series,   # 12
        "delta_self_consumption_kWh": delta_sc_series,  # 12
    }

    return {
        "type": "storage_elec",
        "subtype": "battery_li_ion",
        "totals": totals,
        "profiles": {
            "soc_kWh_monthly": soc_series,
            "charged_kWh_monthly": ch_series,
            "discharged_kWh_monthly": dis_series,
            "losses_kWh_monthly": loss_series,
        },
        "series": series,
        "impacts": impacts,
        "meta": {
            "model": "monthly_simple_dispatch",
            "priority": "PV->Battery->Load",
        },
    }





def compute_battery_li_ion_flow_block_timeseries(
    load_kwh: pd.Series,   # kWh/step, DatetimeIndex
    pv_kwh: pd.Series,     # kWh/step, DatetimeIndex
    storage_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Batterie Li-ion au pas fin (timeseries dispatch).

    Entrées:
      - load_kwh / pv_kwh : Series kWh par pas, même index (sinon erreur)
      - storage_cfg : dict issu de Phase 1 (capacity_kwh + parametres + mapping)

    Conventions impacts (IMPORTANT pour calculations.py):
      - delta_grid_import_kWh = import_final - import_base   (négatif si import réduit)
      - delta_grid_injection_kWh = inj_final - inj_base      (négatif si injection réduite)
      - delta_self_consumption_kWh = autoconsommation_final - autoconsommation_base (positif)

    Retour:
      flow_block type="storage_elec" avec nodes/links + totals + profiles + impacts.
    """
    # --- guards / alignment
    load_kwh = (load_kwh if load_kwh is not None else pd.Series(dtype=float)).fillna(0.0)
    pv_kwh = (pv_kwh if pv_kwh is not None else pd.Series(dtype=float)).fillna(0.0)

    if not isinstance(load_kwh.index, pd.DatetimeIndex) or not isinstance(pv_kwh.index, pd.DatetimeIndex):
        return {"type": "storage_elec", "totals": {"error": "load/pv must be DatetimeIndex"}, "profiles": {}}

    if not load_kwh.index.equals(pv_kwh.index):
        # on refuse ici : le ré-alignment est fait dans calculations.py (patch 2/3)
        raise ValueError("compute_battery_li_ion_flow_block_timeseries: load_kwh and pv_kwh must have the same index")

    params = storage_cfg.get("parametres") or {}
    mapping = storage_cfg.get("mapping") or {}
    charge_sources = set(mapping.get("charge_sources") or ["PV"])
    discharge_sinks = set(mapping.get("discharge_sinks") or ["LOAD"])

    # --- sizing / limits
    C = _num(storage_cfg.get("capacity_kwh"), 0.0)
    if C <= 0:
        return {"type": "storage_elec", "totals": {"error": "capacity_kwh <= 0"}, "profiles": {}}

    # --- SOC window (no hardcode): priority to explicit cfg, else techno params
    soc_min_frac = storage_cfg.get("soc_min_frac", None)
    soc_max_frac = storage_cfg.get("soc_max_frac", None)
    
    if soc_min_frac is None or soc_max_frac is None:
        # depuis Excel techno (parametres), typiquement en %
        soc_min_raw = _get_param(params, "SOC min", None)
        soc_max_raw = _get_param(params, "SOC max", None)
    
        if soc_min_raw is None or soc_max_raw is None:
            raise ValueError("Battery: missing SoC min/max (neither in storage_cfg nor in technology parameters).")

    
        soc_min_frac = _num(soc_min_raw, None)
        soc_max_frac = _num(soc_max_raw, None)
    
        # normalisation % -> fraction
        if soc_min_frac is None or soc_max_frac is None:
            raise ValueError("Battery: SoC min/max are not numeric.")
    
        if soc_min_frac > 1.0: soc_min_frac = soc_min_frac / 100.0
        if soc_max_frac > 1.0: soc_max_frac = soc_max_frac / 100.0
    
    soc_min_frac = float(soc_min_frac)
    soc_max_frac = float(soc_max_frac)
    
    if not (0.0 <= soc_min_frac <= 1.0) or not (0.0 <= soc_max_frac <= 1.0):
        raise ValueError(f"Battery: SoC min/max out of bounds: min={soc_min_frac}, max={soc_max_frac}")
    if soc_min_frac >= soc_max_frac:
        raise ValueError(f"Battery: SoC min must be < SoC max: min={soc_min_frac}, max={soc_max_frac}")


    # Rendement global (souvent en %) dans ton Excel techno
    eta = _num(_get_param(params, "Efficiency", 0.90), 0.90)
    if eta > 1.0:
        eta = eta / 100.0
    eta = max(min(float(eta), 1.0), 0.0)

    eta_ch = math.sqrt(eta) if eta > 0 else 1.0
    eta_dis = math.sqrt(eta) if eta > 0 else 1.0

    # Puissances max via C-rate * capacité (kW/kWh * kWh = kW)
    c_ch = _num(_get_param(params, "Max charge C-rate", 0.5), 0.5)
    c_dis = _num(_get_param(params, "Max discharge C-rate", 0.5), 0.5)
    p_ch_max_kw = max(float(c_ch) * C, 0.0)
    p_dis_max_kw = max(float(c_dis) * C, 0.0)

    grid_charge_allowed = bool(storage_cfg.get("grid_charge_allowed", False)) and ("GRID" in charge_sources)
    grid_discharge_allowed = bool(storage_cfg.get("grid_discharge_allowed", False)) and ("GRID" in discharge_sinks)

    # --- base (sans batterie) au pas fin
    base_auto = pv_kwh.combine(load_kwh, min)
    import_base = float((load_kwh - base_auto).clip(lower=0.0).sum())
    inj_base = float((pv_kwh - base_auto).clip(lower=0.0).sum())
    auto_base = float(base_auto.sum())

    # --- dispatch
    disp = dispatch_electric_storage(
        prod_series=pv_kwh,
        load_series=load_kwh,
        capacity_kwh=C,
        p_charge_max_kw=p_ch_max_kw,
        p_discharge_max_kw=p_dis_max_kw,
        eta_charge=eta_ch,
        eta_discharge=eta_dis,
        soc_min_frac=soc_min_frac,
        soc_max_frac=soc_max_frac,
        mapping={"charge_sources": list(charge_sources), "discharge_sinks": list(discharge_sinks)},
        grid_charge_allowed=grid_charge_allowed,
        grid_discharge_allowed=grid_discharge_allowed,
    )
    
    tot = (disp.get("totals") or {})
    prof = (disp.get("profiles") or {})
    
    pv_to_batt   = float(tot.get("pv_to_batt", 0.0) or 0.0)
    grid_to_batt = float(tot.get("grid_to_batt", 0.0) or 0.0)
    
    # dispatch renvoie séparé PV vs GRID pour la décharge -> on somme
    batt_to_load = float(tot.get("batt_to_load_pv", 0.0) or 0.0) + float(tot.get("batt_to_load_grid", 0.0) or 0.0)
    batt_to_grid = float(tot.get("batt_to_grid_pv", 0.0) or 0.0) + float(tot.get("batt_to_grid_grid", 0.0) or 0.0)
    
    losses = float(tot.get("losses", 0.0) or 0.0)
    
    # ----------------------------------------------------------
    # PATCH: pertes batterie par bilan énergétique (robuste)
    # losses = E_in - E_out - ΔSOC
    # ----------------------------------------------------------
    soc_start = float(disp.get("soc_start_kWh", 0.0) or 0.0)
    soc_end = float(disp.get("soc_end_kWh", 0.0) or 0.0)
    delta_soc = soc_end - soc_start
    
    e_in = pv_to_batt + grid_to_batt
    e_out = batt_to_load + batt_to_grid
    
    losses_bilan = e_in - e_out - delta_soc
    
    # tolérance numérique
    if losses_bilan < 0 and abs(losses_bilan) < 1e-6:
        losses_bilan = 0.0
    
    # si le dispatcher ne remonte pas les pertes (ou incohérent), on force le bilan
    if losses <= 1e-9 and losses_bilan > 1e-9:
        losses = float(losses_bilan)
    
    # (optionnel) debug dans totals
    losses_bilan = float(max(losses_bilan, 0.0))


    # --- final (avec batterie)
    # import_final = import_base - batt_to_load + grid_to_batt
    # inj_final    = inj_base - pv_to_batt + batt_to_grid
    import_final = max(import_base - batt_to_load + grid_to_batt, 0.0)
    inj_final = max(inj_base - pv_to_batt + batt_to_grid, 0.0)

    # autoconsommation finale (V1):
    # - si charge réseau interdite => pv_to_batt provient PV surplus, donc sc augmente ~ min(batt_to_load, pv_to_batt)
    # - si charge réseau autorisée, on ne peut pas savoir l'origine (V1) -> on prend min(batt_to_load, pv_to_batt)
    delta_sc = min(batt_to_load, pv_to_batt)

    auto_final = min(auto_base + delta_sc, float(load_kwh.sum()))

    impacts = {
        "delta_grid_import_kWh": float(import_final - import_base),
        "delta_grid_injection_kWh": float(inj_final - inj_base),
        "delta_self_consumption_kWh": float(auto_final - auto_base),
    }

    # --- sankey nodes/links (battery is internal)
    pv_id = node_pv()
    grid_id = node_elec_grid()
    batt_id = node_battery()
    load_id = node_elec_load()
    
    nodes = [
        {"id": pv_id, "label": "PV", "group": "prod_elec"},
        {"id": grid_id, "label": "Electric grid", "group": "reseau"},
        {"id": batt_id, "label": "Battery", "group": "storage_elec"},
        {"id": load_id, "label": "Building electricity consumption", "group": "demande"},
    ]
    # --- node pertes batterie (si > 0)
    loss_id = None
    if losses > 1e-6:
        loss_id = "ELEC_STORAGE_LOSSES"
        nodes.append({"id": loss_id, "label": "Battery losses", "group": "losses"})

    links = []
    if pv_to_batt > 0:
        links.append({"source": pv_id, "target": batt_id, "value": pv_to_batt})
    if grid_to_batt > 0:
        links.append({"source": grid_id, "target": batt_id, "value": grid_to_batt})
    if batt_to_load > 0:
        links.append({"source": batt_id, "target": load_id, "value": batt_to_load})
    if batt_to_grid > 0:
        links.append({"source": batt_id, "target": grid_id, "value": batt_to_grid})
    # ==========================================================
    # AUDIT "liens réellement créés" -> pertes/non affecté
    # (même logique que ton patch global, mais local au storage_elec)
    # ==========================================================
    def _sum_incoming(node_id: str) -> float:
        s = 0.0
        for l in links:
            if l.get("target") == node_id:
                try:
                    s += float(l.get("value") or 0.0)
                except Exception:
                    pass
        return s

    def _sum_outgoing(node_id: str) -> float:
        s = 0.0
        for l in links:
            if l.get("source") == node_id:
                try:
                    s += float(l.get("value") or 0.0)
                except Exception:
                    pass
        return s

    batt_in_links = _sum_incoming(batt_id)   # PV->Batt + Grid->Batt (si présent)
    batt_out_links = _sum_outgoing(batt_id)  # Batt->Load + Batt->Grid (si présent)

    # Si on ne veut pas gérer ΔSOC ici, on considère "non affecté" = in - out
    batt_unmapped = batt_in_links - batt_out_links

    if batt_unmapped > 1e-6:
        loss_id = "ELEC_STORAGE_LOSSES"
        if not any(n.get("id") == loss_id for n in nodes):
            nodes.append({"id": loss_id, "label": "Electrical losses / unassigned", "group": "losses"})
        links.append({"source": batt_id, "target": loss_id, "value": float(batt_unmapped)})
        
    if loss_id is not None:
        links.append({"source": batt_id, "target": loss_id, "value": losses})


    return {
        "type": "storage_elec",
        "subtype": "battery_li_ion",
        "name": storage_cfg.get("techno", "Battery Li-ion"),
        "nodes": nodes,
        "links": links,
        "totals": {
            "capacity_kWh": C,
            "soc_min_kWh": float(C * soc_min_frac),
            "soc_max_kWh": float(C * soc_max_frac),
            "p_charge_max_kW": float(p_ch_max_kw),
            "p_discharge_max_kW": float(p_dis_max_kw),
            "eta_charge": float(eta_ch),
            "eta_discharge": float(eta_dis),
            "pv_to_batt_kWh": pv_to_batt,
            "grid_to_batt_kWh": grid_to_batt,
            "batt_to_load_kWh": batt_to_load,
            "batt_to_grid_kWh": batt_to_grid,
            "losses_kWh": float(losses),
            "delta_soc_kWh": float(delta_soc),
            "losses_bilan_kWh": float(losses_bilan),
            "soc_start_kWh": float(disp.get("soc_start_kWh", 0.0) or 0.0),
            "soc_end_kWh": float(disp.get("soc_end_kWh", 0.0) or 0.0),
        },
        "profiles": {
            "time_index": [t.isoformat() for t in load_kwh.index],
        
            # séries SOC / pertes
            "soc_kWh": list(prof.get("soc_kWh", [])),
            "losses": list(prof.get("losses", [])),  # clé brute du dispatcher
        
            # séries flux batterie (clés du dispatcher storage_dispatch)
            "pv_to_batt": list(prof.get("pv_to_batt", [])),
            "grid_to_batt": list(prof.get("grid_to_batt", [])),
            "batt_to_load": list(prof.get("batt_to_load", [])),
            "batt_to_grid": list(prof.get("batt_to_grid", [])),
        },


        "impacts": impacts,
        "meta": {
            "model": "timeseries_dispatch",
            "grid_charge_allowed": grid_charge_allowed,
            "grid_discharge_allowed": grid_discharge_allowed,
        },
    }


def compute_storage_blocks_for_batiment(
    flow_blocks: List[Dict[str, Any]],
    stockages_cfg: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Entry-point used by calculations.py for one building.
    Takes existing flow_blocks list and appends storage blocks.
    """
    demand = _find_block(flow_blocks, "demand_elec")
    pv = _find_block(flow_blocks, "pv")

    if not demand:
        return []

    out = []
    for sto in stockages_cfg or []:
        if (sto.get("type_general") or "") != "Electrique":
            continue
        # engine must match Li-ion module, else ignore in V1
        # You can relax later.
        engine = sto.get("engine") or ""
        techno = (sto.get("techno") or "").lower()
        if engine not in {"bat_li_ion", "bat_liion"} and "li" not in techno:
            continue

        out.append(compute_battery_li_ion_flow_block(demand, pv, sto))

    return out
