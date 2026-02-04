# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 10:42:59 2026

@author: infor
"""

import pandas as pd
from typing import Dict, Any, List
from core.utils.timebase import infer_dt_hours_from_index


def dispatch_electric_storage(
    prod_series: pd.Series,   # PV production kWh/step
    load_series: pd.Series,   # load kWh/step
    capacity_kwh: float,
    p_charge_max_kw: float,
    p_discharge_max_kw: float,
    eta_charge: float,
    eta_discharge: float,
    soc_min_frac: float,
    soc_max_frac: float,
    mapping: Dict[str, List[str]],
    grid_charge_allowed: bool,
    grid_discharge_allowed: bool,
) -> Dict[str, Any]:
    """
    Dispatch batterie électrique au pas fin.

    Convention énergie:
      - prod_series, load_series : kWh par pas (pas forcément régulier)
      - SoC en kWh
      - puissance max en kW -> convertie en kWh via dt_h

    Tracking origine:
      - soc_pv : part du SoC provenant du PV
      - soc_grid : part provenant du réseau
      - Décharge: on vide d'abord soc_pv (maximise autoconsommation), puis soc_grid.
    """
    # Alignement strict: même index attendu
    prod_series = prod_series.fillna(0.0)
    load_series = load_series.fillna(0.0)

    index = prod_series.index
    if not index.equals(load_series.index):
        raise ValueError(
            "dispatch_electric_storage: prod_series et load_series doivent avoir le même index"
        )


    dt_h = infer_dt_hours_from_index(index)

    C = float(capacity_kwh or 0.0)
    if C <= 0:
        return {
            "totals": {"pv_to_batt": 0.0, "grid_to_batt": 0.0, "batt_to_load_pv": 0.0, "batt_to_load_grid": 0.0,
                       "batt_to_grid_pv": 0.0, "batt_to_grid_grid": 0.0, "losses": 0.0},
            "profiles": {"soc_kWh": [], "pv_to_batt": [], "grid_to_batt": [], "batt_to_load": [], "batt_to_grid": [], "losses": []},
        }

    soc_min = C * float(soc_min_frac)
    soc_max = C * float(soc_max_frac)

    # Initialisation SoC au minimum (choix conservatif)
    soc_pv = 0.0
    soc_grid = 0.0
    soc = soc_pv + soc_grid
    if soc < soc_min:
        soc_grid += (soc_min - soc)
        soc = soc_min

    charge_sources = set(mapping.get("charge_sources") or ["PV"])
    discharge_sinks = set(mapping.get("discharge_sinks") or ["LOAD"])

    pv_to_batt = []
    grid_to_batt = []
    batt_to_load_pv = []
    batt_to_load_grid = []
    batt_to_grid_pv = []
    batt_to_grid_grid = []
    losses = []
    soc_profile = []

    for pv_kwh, load_kwh in zip(prod_series.values, load_series.values):
        pv_kwh = float(pv_kwh or 0.0)
        load_kwh = float(load_kwh or 0.0)

        base_auto = min(pv_kwh, load_kwh)
        surplus = max(pv_kwh - base_auto, 0.0)    # injection potentielle
        deficit = max(load_kwh - base_auto, 0.0)  # import potentiel

        # ---------------- CHARGE ----------------
        ch_from_pv = 0.0
        ch_from_grid = 0.0
        e_max_p = float(p_charge_max_kw) * dt_h
        e_max_soc_in = max(soc_max - (soc_pv + soc_grid), 0.0)  # kWh qui peuvent entrer (après rendement)
        # énergie brute à injecter avant rendement
        e_max_brut = e_max_soc_in / max(float(eta_charge), 1e-9)

        if surplus > 0 and ("PV" in charge_sources):
            ch_from_pv = min(surplus, e_max_p, e_max_brut)

        if grid_charge_allowed and ("GRID" in charge_sources):
            # Charge réseau = optionnelle, on ne la fait que si pas de PV dispo
            if ch_from_pv <= 1e-12:
                ch_from_grid = min(e_max_p, e_max_brut)

        ch_total = ch_from_pv + ch_from_grid
        e_ch_net = ch_total * float(eta_charge)
        e_ch_loss = ch_total - e_ch_net

        # Mise à jour SoC (origine)
        if e_ch_net > 0:
            if ch_total > 0:
                # répartir l'énergie nette au prorata des sources
                if ch_from_pv > 0:
                    soc_pv += e_ch_net * (ch_from_pv / ch_total)
                if ch_from_grid > 0:
                    soc_grid += e_ch_net * (ch_from_grid / ch_total)

        # ---------------- DECHARGE ----------------
        dis_to_load = 0.0
        dis_to_grid = 0.0
        e_max_p_dis = float(p_discharge_max_kw) * dt_h

        # énergie brute qu'on peut sortir du SoC (avant rendement)
        e_avail = max((soc_pv + soc_grid) - soc_min, 0.0)
        dis_brut = 0.0

        if deficit > 0 and ("LOAD" in discharge_sinks):
            dis_brut = min(deficit / max(float(eta_discharge), 1e-9), e_max_p_dis, e_avail)
            dis_to_load = dis_brut * float(eta_discharge)
        elif grid_discharge_allowed and ("GRID" in discharge_sinks):
            dis_brut = min(e_max_p_dis, e_avail)
            dis_to_grid = dis_brut * float(eta_discharge)

        e_dis_loss = dis_brut - (dis_to_load + dis_to_grid)

        # Retirer du SoC en priorité PV, puis GRID
        dis_pv = min(soc_pv, dis_brut)
        soc_pv -= dis_pv
        dis_grid = dis_brut - dis_pv
        soc_grid = max(soc_grid - dis_grid, 0.0)

        # Répartir l’énergie NETTE délivrée selon l’origine réellement retirée
        net_total = (dis_to_load + dis_to_grid)
        pv_share = (dis_pv / dis_brut) if dis_brut > 1e-12 else 0.0
        grid_share = 1.0 - pv_share if dis_brut > 1e-12 else 0.0

        b2l_pv = dis_to_load * pv_share
        b2l_grid = dis_to_load * grid_share
        b2g_pv = dis_to_grid * pv_share
        b2g_grid = dis_to_grid * grid_share

        pv_to_batt.append(ch_from_pv)
        grid_to_batt.append(ch_from_grid)
        batt_to_load_pv.append(b2l_pv)
        batt_to_load_grid.append(b2l_grid)
        batt_to_grid_pv.append(b2g_pv)
        batt_to_grid_grid.append(b2g_grid)
        losses.append(e_ch_loss + e_dis_loss)

        soc_profile.append(soc_pv + soc_grid)

    totals = {
        "pv_to_batt": float(sum(pv_to_batt)),
        "grid_to_batt": float(sum(grid_to_batt)),
        "batt_to_load_pv": float(sum(batt_to_load_pv)),
        "batt_to_load_grid": float(sum(batt_to_load_grid)),
        "batt_to_grid_pv": float(sum(batt_to_grid_pv)),
        "batt_to_grid_grid": float(sum(batt_to_grid_grid)),
        "losses": float(sum(losses)),
        "soc_start_kWh": float(soc_profile[0] if soc_profile else soc_min),
        "soc_end_kWh": float(soc_profile[-1] if soc_profile else soc_min),
    }

    profiles = {
        "soc_kWh": soc_profile,
        "pv_to_batt": pv_to_batt,
        "grid_to_batt": grid_to_batt,
        "batt_to_load": [a + b for a, b in zip(batt_to_load_pv, batt_to_load_grid)],
        "batt_to_grid": [a + b for a, b in zip(batt_to_grid_pv, batt_to_grid_grid)],
        "losses": losses,
    }

    return {"totals": totals, "profiles": profiles}

