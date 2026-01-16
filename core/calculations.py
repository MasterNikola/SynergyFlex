# core/calculations.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List

import pandas as pd
import math
from core.pv_module import (
    compute_pv_flow_block,
    compute_pv_monthly_energy_per_kw,   # ✅ pour le PV “théorique standard”
    compute_pv_economics,
    compute_ru_pv_added_weighted,
    reconstruct_pv_to_load_proportional_bounded)

from core.nodes_ontology import (
    node_elec_grid,
    node_elec_load,
    node_heat_load,
    node_heat_prod,
    node_pv,
    node_battery,
    # plus tard : node_battery, node_device, etc.
)
from core.economics import get_param_numeric, compute_battery_cashflow_series
from core.technologies import load_producer_technologies, infer_engine_from_type_and_techno
from core.bat_li_ion_module import compute_battery_li_ion_flow_block, compute_battery_li_ion_flow_block_timeseries
from core.utils.timebase import (
    build_timeseries_kwh,
    pick_finest_index,
    common_sim_window_intersection,
    align_energy_series_to_index,
    infer_dt_hours_from_index,
    build_monthly_kwh_from_df,
    extract_time_info)



# ----------------------------------------------------------------------
# Utilitaire : lire une valeur numérique dans les paramètres techno
# ----------------------------------------------------------------------


def _get_param_numeric(
    tech_params: Dict[str, Any],
    candidate_keys,
    default: float = 0.0
) -> float:
    """
    Cherche une valeur numérique dans tech_params en testant plusieurs clés possibles.
    tech_params ressemble à :
        { "Puissance du module": {"valeur": 0.42, "unite": "kW"}, ... }
    """
    if not tech_params:
        return default

    for key in candidate_keys:
        raw = tech_params.get(key)
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

    return default

def _get_price_buy_sell_and_horizon(project: dict):
    params = project.get("params") or {}

    # On exige que Phase 1 ait rempli ces valeurs dans params.
    # (les defaults UI peuvent être 0.25/0.06/25, mais on ne fallback pas ici)
    if "horizon_analyse_ans" not in params:
        raise ValueError("Paramètre manquant: project.params.horizon_analyse_ans")
    if "price_buy_chf_kwh" not in params:
        raise ValueError("Paramètre manquant: project.params.price_buy_chf_kwh")
    if "price_sell_chf_kwh" not in params:
        raise ValueError("Paramètre manquant: project.params.price_sell_chf_kwh")

    horizon_years = int(params["horizon_analyse_ans"])
    price_buy = float(params["price_buy_chf_kwh"])
    price_sell = float(params["price_sell_chf_kwh"])

    horizon_years = max(1, horizon_years)
    return price_buy, price_sell, horizon_years

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
        raise ValueError("lifetime_years doit être > 0 pour appliquer des remplacements.")

    if horizon_years <= 0:
        return cashflows

    if replacement_cost_factor <= 0:
        raise ValueError("replacement_cost_factor doit être > 0 pour appliquer des remplacements.")

    step = int(round(float(lifetime_years)))
    if step <= 0:
        raise ValueError(f"Durée de vie invalide (arrondie): lifetime_years={lifetime_years}")

    replacement_cost = float(capex_total_chf) * float(replacement_cost_factor)

    y = step + 1
    while y <= horizon_years:
        if y < len(cashflows):
            cashflows[y] -= replacement_cost
        y += step

    return cashflows



def compute_economics_from_flow_blocks(flow_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agrège les indicateurs économiques provenant des flow_blocks.

    Convention :
      - Un flow_block "machine" contient dans totals au minimum :
          - capex_total_CHF
          - opex_annual_CHF
          - lifetime_years
          - production_machine_kWh
          - lcoe_machine_CHF_kWh

      - Les blocs purement énergétiques / de demande ne sont pas pris
        en compte (pas de capex_total_CHF dans totals).

    Retourne un dict avec :
      - capex_total_CHF
      - opex_annual_CHF
      - production_totale_kWh
      - lcoe_global_CHF_kWh
      - by_type : breakdown par type de machine (pv, pac, boiler, ...)
    """

    total_capex = 0.0
    total_opex = 0.0
    total_prod_kwh = 0.0
    total_lcoe_numerateur = 0.0  # Σ (CAPEX_i / life_i + OPEX_i)

    by_type: Dict[str, Dict[str, float]] = {}

    for fb in flow_blocks:
        totals = fb.get("totals", {}) or {}

        # On considère que c'est une "machine" s'il y a explicitement un CAPEX
        if "capex_total_CHF" not in totals:
            continue

        capex_i = float(totals.get("capex_total_CHF", 0.0) or 0.0)
        opex_i = float(totals.get("opex_annual_CHF", 0.0) or 0.0)
        life_i = float(totals.get("lifetime_years", 25.0) or 25.0)
        prod_i = float(totals.get("production_machine_kWh", 0.0) or 0.0)

        if life_i <= 0:
            life_i = 25.0

        total_capex += capex_i
        total_opex += opex_i
        total_prod_kwh += prod_i
        total_lcoe_numerateur += (capex_i / life_i + opex_i)

        mtype = fb.get("type", "unknown") or "unknown"
        stats = by_type.setdefault(
            mtype,
            {
                "capex_total_CHF": 0.0,
                "opex_annual_CHF": 0.0,
                "production_machine_kWh": 0.0,
                "lcoe_machine_CHF_kWh": 0.0,  # LCOE moyen pondéré du type
            },
        )
        stats["capex_total_CHF"] += capex_i
        stats["opex_annual_CHF"] += opex_i
        stats["production_machine_kWh"] += prod_i

    # LCOE global (simple) : Σ(CAPEX_i / life_i + OPEX_i) / Σ(prod_i)
    if total_prod_kwh > 0:
        lcoe_global = total_lcoe_numerateur / total_prod_kwh
    else:
        lcoe_global = 0.0

    # LCOE moyen par type (si utile)
    for mtype, stats in by_type.items():
        prod = stats.get("production_machine_kWh", 0.0)
        if prod > 0:
            # Ici, on approxime en utilisant la même formule,
            # car on n'a pas stocké life_i par type de manière détaillée.
            cap = stats.get("capex_total_CHF", 0.0)
            opx = stats.get("opex_annual_CHF", 0.0)
            # Hypothèse : lifetime "moyenne" = 25 ans pour affichage
            stats["lcoe_machine_CHF_kWh"] = (cap / 25.0 + opx) / prod
        else:
            stats["lcoe_machine_CHF_kWh"] = 0.0

    return {
        "capex_total_CHF": float(total_capex),
        "opex_annual_CHF": float(total_opex),
        "production_totale_kWh": float(total_prod_kwh),
        "lcoe_global_CHF_kWh": float(lcoe_global),
        "by_type": by_type,
    }



def _build_monthly_series_from_df(
    df: pd.DataFrame,
    energy_col: str,
    time_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    Wrapper (PATCH 1) : délègue la construction mensuelle au module timebase.
    """
    return build_monthly_kwh_from_df(df=df, energy_col=energy_col, time_col=time_col)

def _normalize_efficiency(eta_raw: float) -> float:
    """
    Normalise un rendement:
      - 0.9 -> 0.9
      - 90  -> 0.9
    Refuse tout fallback silencieux.
    """
    if eta_raw is None:
        raise ValueError("Batterie proposée: eta_global manquant.")
    eta = float(eta_raw)
    if eta <= 0:
        raise ValueError(f"Batterie proposée: eta_global invalide (<=0): {eta_raw}")
    if eta > 1.0:
        eta = eta / 100.0
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"Batterie proposée: eta_global invalide après normalisation: {eta}")
    return eta


def _pack_capacities(total_kwh: float, e_min: float, e_max: float) -> list:
    """
    Packing techno:
      - si total <= e_max: [max(total, e_min)]
      - sinon: n*e_max + reste
        - si reste < e_min: on ajoute e_min (cas pénible)
    """
    total_kwh = float(total_kwh)
    e_min = float(e_min)
    e_max = float(e_max)

    if total_kwh <= 0:
        raise ValueError("Batterie proposée: capacité totale <= 0.")
    if e_min <= 0 or e_max <= 0 or e_min > e_max:
        raise ValueError("Batterie proposée: capacity_min/max invalides.")

    if total_kwh <= e_max:
        return [max(total_kwh, e_min)]

    n = int(total_kwh // e_max)
    r = total_kwh - n * e_max
    caps = [e_max] * n

    if r <= 1e-9:
        return caps

    if r < e_min:
        caps.append(e_min)
    else:
        caps.append(r)

    return caps


# ----------------------------------------------------------------------
# ORCHESTRATEUR PRINCIPAL
# ----------------------------------------------------------------------
def run_calculations(
    project: Dict[str, Any],
    excel_sheets: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """
    Orchestrateur principal des calculs SynergyFlex.

    - project : dict normalisé issu de la Phase 1
      (doit contenir batiments, ouvrages, producteurs, etc.)
    - excel_sheets : dictionnaire {sheet_name: DataFrame}
      construit dans la Phase 1 (page Validation) à partir
      de tous les fichiers Excel des ouvrages.
    """

    excel_sheets = excel_sheets or {}

    batiments = project.get("batiments", [])
    ouvrages = project.get("ouvrages", [])
    categories_ouvrages = project.get("categories_ouvrages", [])
    meta_global = project.get("meta", {}) or {}
    

    # Station météo choisie en Phase 1 (page_meta)
    station_meteo = meta_global.get("station_meteo")

    results: Dict[str, Any] = {
        "flows": {
            # batiment_id -> [flow_blocks...]
            "batiments": {},
        },
        "summaries": {},
        "pv_standard": [],   # ✅ pour la comparaison standard PV
        "pv_proposed_by_batiment": {},  # ✅ nouveau
    
        # ✅ Interne uniquement (jamais affiché en Sankey)
        "internal": {
            # bilans par ouvrage (clé bat_id -> liste d'enregistrements)
            "energy_global_base": {},    # sans stockage
            "energy_global_final": {},   # avec stockage (celui réellement affiché)
        },
    }


    # ------------------------------------------------------------------
    # Utilitaire : retrouver le nom du bâtiment
    # ------------------------------------------------------------------
    def _get_bat_name(bat_id: Any) -> str:
        if isinstance(bat_id, int) and 0 <= bat_id < len(batiments):
            return batiments[bat_id].get("nom", f"Bâtiment {bat_id+1}")
        if isinstance(bat_id, str):
            for b in batiments:
                if b.get("id") == bat_id:
                    return b.get("nom", bat_id)
        return str(bat_id)
    
    def _get_default_pv_tech_params() -> dict:
        techs = load_producer_technologies("Electrique") or {}
    
        # Cas 1 (le plus probable chez toi) : techs = {"Photovoltaic panel": {...}, "Wind turbine": {...}}
        if "Photovoltaic panel" in techs:
            return techs["Photovoltaic panel"] or {}
    
        # Cas 2 : structure imbriquée {"Electrique": {"Photovoltaic panel": {...}}}
        elec = techs.get("Electrique", {}) or {}
        if "Photovoltaic panel" in elec:
            return elec["Photovoltaic panel"] or {}
    
        # fallback : cherche un onglet qui ressemble à du PV
        for techno_name, params in (elec or techs).items():
            name = str(techno_name).lower()
            if "photovolta" in name or "pv" == name.strip():
                return params or {}
    
        return {}

    # Catégorie d'ouvrage (infos Excel) associée à l'index d'ouvrage
    def _get_cat_for_ouvrage(index_ouv: int) -> Dict[str, Any]:
        if 0 <= index_ouv < len(categories_ouvrages):
            return categories_ouvrages[index_ouv]
        return {}

    # ------------------------------------------------------------------
    # BOUCLE PRINCIPALE SUR LES OUVRAGES
    # ------------------------------------------------------------------
    storage_by_batiment = {}
    ts_load_by_bat = {}   # bat_id -> list[pd.Series kWh/step]
    ts_pv_by_bat = {}     # bat_id -> list[pd.Series kWh/step]
    pv_ts_by_bat = {}    # bat_id -> list[pd.Series]
    load_ts_by_bat = {}  # bat_id -> list[pd.Series]


    for oi, ouv in enumerate(ouvrages):
        bat_id = ouv.get("batiment_id")
        bat_nom = _get_bat_name(bat_id)
        ouv_nom = ouv.get("nom", "Ouvrage")
        storage_by_batiment.setdefault(bat_id, [])
        storage_by_batiment[bat_id].extend(ouv.get("stockages", []) or [])

        # Liste de blocs de flux pour ce bâtiment
        bat_flows = results["flows"]["batiments"].setdefault(bat_id, [])

        # ---- Accumulateurs locaux pour cet ouvrage ----
        local_demand_elec_kwh = 0.0
        local_demand_th_kwh = 0.0
        local_pv_prod_kwh = 0.0
        local_pv_auto_kwh = 0.0
        local_pv_inj_kwh = 0.0
        
        # ---- Accumulateurs stockage (toujours définis, même sans stockage) ----
        sto_delta_import = 0.0
        sto_delta_inj = 0.0
        sto_delta_sc = 0.0
        sto_ch = 0.0
        sto_dis = 0.0
        sto_losses = 0.0

        # ------------------ DEMANDE (à partir Excel) -------------------
        cat = _get_cat_for_ouvrage(oi)
        sheet_name = cat.get("sheet_name")
        time_col = cat.get("time_col")
        conso_elec_col = cat.get("conso_elec_col")
        conso_th_col = cat.get("conso_th_col")
        
        df_demand = None
        if sheet_name and sheet_name in excel_sheets:
            df_demand = excel_sheets[sheet_name]
            results.setdefault("timebase_debug", {})
            info = extract_time_info(df_demand, time_col)
            if info is not None:
                # JSON-friendly : pas de datetime brute
                results["timebase_debug"][f"demand::{sheet_name}"] = info

        # 1) FLOW_BLOCK – DEMANDE ELECTRIQUE (conservation pour analyse fine)
        if df_demand is not None and conso_elec_col and conso_elec_col in df_demand.columns:
            serie_elec = pd.to_numeric(df_demand[conso_elec_col], errors="coerce")
            local_demand_elec_kwh = float(serie_elec.sum(skipna=True) or 0.0)
            # PATCH 2 : série fine load (kWh/step) pour bilan au pas fin
            ts_load_kwh = None
            try:
                ts_load_kwh = build_timeseries_kwh(df_demand, conso_elec_col, time_col=time_col)
                store_ts = bool(((project.get("meta") or {}).get("debug") or {}).get("store_timeseries", False))

                def _ts_bucket(results, bat_id, oi, bat_nom, ouv_nom, index):
                    results.setdefault("timeseries_store", {})
                    results["timeseries_store"].setdefault("batiments", {})
                    b = results["timeseries_store"]["batiments"].setdefault(str(bat_id), {})
                    ovs = b.setdefault("ouvrages", {})
                    o = ovs.setdefault(str(oi), {})
                    o.setdefault("meta", {})
                    o["meta"].update({
                        "batiment_id": bat_id,
                        "batiment_nom": bat_nom,
                        "ouvrage_index": oi,
                        "ouvrage_nom": ouv_nom,
                        "dt_h_med": float(infer_dt_hours_from_index(index)) if index is not None else None,
                    })
                    return o
                
                if store_ts and (ts_load_kwh is not None):
                    o = _ts_bucket(results, bat_id, oi, bat_nom, ouv_nom, ts_load_kwh.index)
                    o["index"] = [str(x) for x in ts_load_kwh.index]
                    o["load_kWh"] = ts_load_kwh.values.tolist()
            except Exception:
                ts_load_kwh = None

            # debug JSON-friendly
            results.setdefault("checks", {})
            results["checks"].setdefault("patch2_timeseries", {})
            if ts_load_kwh is not None:
                results["checks"]["patch2_timeseries"][f"load::{bat_id}::{oi}"] = {
                    "n": int(len(ts_load_kwh)),
                    "dt_h_med": float(infer_dt_hours_from_index(ts_load_kwh.index)),
                    "sum_kWh": float(ts_load_kwh.sum()),
                    "start": ts_load_kwh.index.min().isoformat(),
                    "end": ts_load_kwh.index.max().isoformat(),
                }

            # PATCH 2b : stocker la série load au niveau bâtiment (pour dispatch batterie éventuel)
            if ts_load_kwh is not None:
                ts_load_by_bat.setdefault(bat_id, []).append(ts_load_kwh)

            if local_demand_elec_kwh > 0:
                grid_id = node_elec_grid()
                load_id = node_elec_load()

                # --- Profil mensuel DEMANDE (kWh/mois) pour les courbes phase 2 ---
                monthly_elec = None
                try:
                    serie_mois = _build_monthly_series_from_df(df_demand, conso_elec_col, time_col=time_col)
                    if serie_mois is not None:
                        monthly_elec = [float(x) for x in serie_mois.tolist()]
                except Exception:
                    monthly_elec = None
                
                flow_block_elec = {
                    "name": f"Demande élec – {bat_nom} / {ouv_nom}",
                    "type": "demand_elec",
                    "meta": {
                        "batiment_id": bat_id,
                        "batiment_nom": bat_nom,
                        "ouvrage_nom": ouv_nom,
                        "ouvrage_type_usage": ouv.get("type_usage", ""),
                        "sheet_name": sheet_name,
                        "time_col": time_col,
                        "conso_elec_col": conso_elec_col,
                    },
                    "nodes": [
                        {"id": grid_id, "label": "Réseau électrique", "group": "reseau"},
                        {"id": load_id, "label": "Consommation électrique", "group": "demande"},
                    ],
                    "links": [
                        {"source": grid_id, "target": load_id, "value": local_demand_elec_kwh},
                    ],
                    "totals": {
                        "demand_elec_kWh": local_demand_elec_kwh,
                        "monthly_kWh": monthly_elec,  # <-- AJOUT
                    },
                }

                bat_flows.append(flow_block_elec)

        # 2) FLOW_BLOCK – DEMANDE THERMIQUE (conservation pour analyse fine)
        if df_demand is not None and conso_th_col and conso_th_col in df_demand.columns:
            serie_th = pd.to_numeric(df_demand[conso_th_col], errors="coerce")
            local_demand_th_kwh = float(serie_th.sum(skipna=True) or 0.0)

            if local_demand_th_kwh > 0:
                heat_src_id = node_heat_prod("UNKNOWN", index=1)
                heat_load_id = node_heat_load()

                flow_block_th = {
                    "name": f"Demande thermique – {bat_nom} / {ouv_nom}",
                    "type": "demand_th",
                    "meta": {
                        "batiment_id": bat_id,
                        "batiment_nom": bat_nom,
                        "ouvrage_nom": ouv_nom,
                        "ouvrage_type_usage": ouv.get("type_usage", ""),
                        "sheet_name": sheet_name,
                        "time_col": time_col,
                        "conso_th_col": conso_th_col,
                    },
                    "nodes": [
                        {
                            "id": heat_src_id,
                            "label": "Système de chauffage",
                            "group": "prod_th",
                        },
                        {
                            "id": heat_load_id,
                            "label": "Consommation thermique",
                            "group": "demande",
                        },
                    ],
                    "links": [
                        {
                            "source": heat_src_id,
                            "target": heat_load_id,
                            "value": local_demand_th_kwh,
                        },
                    ],
                    "totals": {
                        "demand_th_kWh": local_demand_th_kwh,
                    },
                }
                bat_flows.append(flow_block_th)

        # ------------------ PRODUCTEURS (PV, etc.) ---------------------
        # PATCH 2 : accumulateurs séries PV au niveau de l’ouvrage
        pv_prod_series_list = []
        pv_auto_series_list = []
        pv_default_selfc_pct = None  # optionnel (si tu veux fallback plus tard)
        producteurs = ouv.get("producteurs", [])
        if producteurs:
            for prod in producteurs:
                techno = prod.get("techno", "")
                type_general = prod.get("type_general", "")
                engine = prod.get("engine")  # ex: "pv"
                

                # Pour l'instant, on ne gère que les PV électriques
                is_pv = (
                    engine == "pv"
                    or (
                        type_general == "Electrique"
                        and isinstance(techno, str)
                        and "photovoltaic panel" in techno.lower()
                    )
                )
                if not is_pv:
                    # Les futurs modules (PAC, batterie, etc.) viendront ici
                    continue

                sheet_name_prod = prod.get("prod_profile_sheet")
                col_prod = prod.get("prod_profile_col")

                if not sheet_name_prod or not col_prod:
                    # Producteur PV sans profil de production -> on saute
                    continue

                df_prod = None
                # Cas 1 : le producteur a son propre fichier Excel (custom)
                if prod.get("source_data_mode") == "Importer un fichier":
                    df_sheets = (prod.get("custom_excel_data") or {}).get("sheets", {})
                    df_prod = df_sheets.get(sheet_name_prod)

                # Cas 2 : le producteur réutilise un fichier d’ouvrage / bâtiment / projet
                if df_prod is None:
                    df_prod = excel_sheets.get(sheet_name_prod)

                if df_prod is None or col_prod not in df_prod.columns:
                    continue
                # PATCH 2 : séries fines PV (kWh/step)
                try:
                    ts_pv_prod = build_timeseries_kwh(
                        df_prod,
                        col_prod,
                        time_col=prod.get("prod_profile_time_col"),
                    )
                    if store_ts and (ts_pv_prod is not None):
                        o = _ts_bucket(results, bat_id, oi, bat_nom, ouv_nom, ts_pv_prod.index)
                        # si index déjà là, idéalement aligner à un index commun ; sinon on stocke brut
                        o["pv_prod_kWh"] = ts_pv_prod.values.tolist()
                        
                except Exception:
                    ts_pv_prod = None

                pv_cfg = prod.get("pv_flux_config", {}) or {}
                col_pv_auto = pv_cfg.get("col_pv_auto")

                try:
                    ts_pv_auto = (
                        build_timeseries_kwh(df_prod, col_pv_auto, time_col=prod.get("prod_profile_time_col"))
                        if (col_pv_auto and col_pv_auto in df_prod.columns)
                        else None
                    )
                except Exception:
                    ts_pv_auto = None

                # on accumule (au cas où plusieurs PV sur l’ouvrage)
                if ts_pv_prod is not None:
                    pv_prod_series_list.append(ts_pv_prod)
                if ts_pv_auto is not None:
                    pv_auto_series_list.append(ts_pv_auto)
                
                if pv_default_selfc_pct is None:
                    pv_default_selfc_pct = pv_cfg.get("default_selfc_pct", None)


                contexte = {
                    "name": f"PV_{bat_nom}_{ouv_nom}",
                    "batiment_id": bat_id,
                    "batiment_nom": bat_nom,
                    "ouvrage_nom": ouv_nom,
                    "producer_techno": techno,
                    "pv_label": f"PV – {ouv_nom}",
                    "elec_use_label": "Usages élec.",
                    "grid_inj_label": "Injection réseau",
                    # on peut aussi passer puissance_kw ici si besoin
                    "puissance_kw": prod.get("puissance_kw", 0.0),
                }

                # ---------- 2.1 Bloc de flux PV (production / auto / injection)
                try:
                    flow_block_pv = compute_pv_flow_block(
                        df_prod,    
                        col_prod,
                        prod,
                        contexte,
                    )
                except Exception:
                    # TODO : logger / stocker l'erreur si besoin
                    flow_block_pv = None

                if flow_block_pv is not None:
                    bat_flows.append(flow_block_pv)

                    t = flow_block_pv.get("totals", {})
                    local_pv_prod_kwh += float(t.get("pv_prod_kWh", 0.0))
                    local_pv_auto_kwh += float(t.get("pv_auto_kWh", 0.0))
                    local_pv_inj_kwh += float(t.get("pv_inj_kWh", 0.0))

                # ---------- 2.2 Préparation des données pour COMPARAISON STANDARD PV
                #            (mais pas d'affichage ici)
                # On stocke tout ce qu'il faut dans results["pv_standard"].
                pv_cfg = prod.get("pv_flux_config", {}) or {}
                orientation = pv_cfg.get("orientation_deg")
                inclinaison = pv_cfg.get("inclinaison_deg")
                
                tech_params = prod.get("parametres", {}) or {}
                p_module_kw = _get_param_numeric(
                    tech_params,
                    ["Puissance du module", "Puissance du module (kW)", "Puissance du module [kW]", "Puissance module [kW]", "Puissance module"]
                )
                area_module_m2 = _get_param_numeric(
                    tech_params,
                    ["Surface du module", "Surface du module (m2)", "Surface module (m2)", "surface du module (m2)"]
                )
                eta_mod_pct = _get_param_numeric(
                    tech_params,
                    ["Rendement", "Rendement (%)", "Rendement module (%)"]
                )
                
                installed_kw = float(prod.get("puissance_kw", 0.0) or 0.0)

                # ---- Profil mensuel MESURÉ (kWh/kW)
                measured_profile = None
                measured_annual = None
                if installed_kw > 0:
                    serie_mois = _build_monthly_series_from_df(
                        df_prod,
                        col_prod,
                        time_col=prod.get("prod_profile_time_col"),
                    )
                    if serie_mois is not None:
                        measured_profile = (serie_mois / installed_kw).tolist()
                        measured_annual = float(serie_mois.sum() / installed_kw)

                # ---- Vérification des champs ----
                missing = []
                if orientation is None:  missing.append("orientation")
                if inclinaison is None:  missing.append("inclinaison")
                if p_module_kw <= 0:     missing.append("Puissance du module")
                if area_module_m2 <= 0:  missing.append("Surface du module")
                if eta_mod_pct <= 0:     missing.append("Rendement du module")
                
                # ---- Construction de l’enregistrement PV_STANDARD ----
                pv_standard_record = {
                    "batiment_id": bat_id,
                    "batiment_nom": bat_nom,
                    "ouvrage_nom": ouv_nom,
                    "producer_techno": techno,
                    "installed_kw": installed_kw,
                    "orientation_deg": orientation,
                    "inclinaison_deg": inclinaison,
                    "p_module_kw": p_module_kw,
                    "area_module_m2": area_module_m2,
                    "eta_mod_pct": eta_mod_pct,
                    "missing_fields": missing,
                    "theoretical_profile_kWh_per_kW": None,
                    "theoretical_annual_kWh_per_kW": None,
                    "measured_profile_kWh_per_kW": measured_profile,
                    "measured_annual_kWh_per_kW": measured_annual,
                }
                
                # ---- Si tout est OK → calcul théorique mensuel ----
                if not missing:
                    station = project.get("meta", {}).get("station_meteo")
                    try:
                        serie = compute_pv_monthly_energy_per_kw(
                            station_name=station,
                            orientation_deg=orientation,
                            tilt_deg=inclinaison,
                            p_module_kw=p_module_kw,
                            area_module_m2=area_module_m2,
                            panel_efficiency=eta_mod_pct,
                        )
                        pv_standard_record["theoretical_profile_kWh_per_kW"] = serie.tolist()
                        pv_standard_record["theoretical_annual_kWh_per_kW"] = float(serie.sum())
                    except Exception as e:
                        pv_standard_record["calc_error"] = str(e)
                
                # ---- On stocke le résultat ----
                results.setdefault("pv_standard", []).append(pv_standard_record)
                
                try:
                    ts_pv_kwh = build_timeseries_kwh(
                        df_prod,
                        col_prod,
                        time_col=prod.get("prod_profile_time_col"),
                    )
                    if ts_pv_kwh is not None:
                        ts_pv_by_bat.setdefault(bat_id, []).append(ts_pv_kwh)
                except Exception:
                    pass

        storage_blocks = []
        sto_delta_import = 0.0
        sto_delta_inj = 0.0
        sto_delta_sc = 0.0
        sto_pv_to_batt = 0.0
        sto_grid_to_batt = 0.0
        sto_batt_to_load = 0.0
        sto_batt_to_grid = 0.0
        sto_losses = 0.0
        
        results.setdefault("storage_economics_by_batiment", {})
        batt_econ_entries = []   # liste des batteries de ce bâtiment
        price_buy, price_sell, horizon_years = _get_price_buy_sell_and_horizon(project)
        # On calcule le stockage uniquement si on a une série load
        have_load_ts = ("ts_load_kwh" in locals()) and (ts_load_kwh is not None)
        
        # PV peut être absent -> pv série = 0
        have_pv_ts = ("pv_prod_series_list" in locals()) and (len(pv_prod_series_list) > 0)
        
        if have_load_ts and (ouv.get("stockages") or []):
            # construire index commun au pas le plus fin disponible
            idx_list = [ts_load_kwh.index]
            if have_pv_ts:
                idx_list += [s.index for s in pv_prod_series_list]
        
            common_index = common_sim_window_intersection(idx_list)
        
            if common_index is not None and len(common_index) > 0:
                load_ts = align_energy_series_to_index(ts_load_kwh, common_index)
        
                if have_pv_ts:
                    pv_ts = sum(align_energy_series_to_index(s, common_index) for s in pv_prod_series_list)
                else:
                    pv_ts = pd.Series(0.0, index=common_index)
                
                # ==========================================================
                # HYBRID: banque timeseries optionnelle (export séparé)
                # ==========================================================
                store_ts = bool(((project.get("meta") or {}).get("debug") or {}).get("store_timeseries", False))
                
                if store_ts:
                    results.setdefault("timeseries_store", {})
                    results["timeseries_store"].setdefault("batiments", {})
                    results["timeseries_store"]["batiments"].setdefault(str(bat_id), {})
                    results["timeseries_store"]["batiments"][str(bat_id)].setdefault("ouvrages", {})
                    key_ouv = f"{oi}"
                    results["timeseries_store"]["batiments"][str(bat_id)]["ouvrages"].setdefault(key_ouv, {})
                
                    ts_bucket = results["timeseries_store"]["batiments"][str(bat_id)]["ouvrages"][key_ouv]
                    ts_bucket["meta"] = {
                        "batiment_id": bat_id,
                        "batiment_nom": bat_nom,
                        "ouvrage_index": oi,
                        "ouvrage_nom": ouv_nom,
                        "dt_h_med": float(infer_dt_hours_from_index(common_index)),
                    }
                    ts_bucket["index"] = [str(x) for x in common_index]
                    ts_bucket["load_kWh"] = load_ts.values.tolist()
                    ts_bucket["pv_prod_kWh"] = pv_ts.values.tolist()

                
                for si, sto in enumerate(ouv.get("stockages") or []):
                    if (sto.get("type_general") or "") != "Electrique":
                        continue
        
                    engine = (sto.get("engine") or "")
                    techno = (sto.get("techno") or "").lower()
                    if engine not in {"bat_li_ion", "bat_liion"} and "li" not in techno:
                        continue
        
                    # ------------------------------
                    # MODE MESURÉ : on lit Excel -> flow_block
                    # ------------------------------
                    if (sto.get("data_mode") or "") == "measured":
                        mp = (sto.get("measured_profile") or {})
                        sheet = mp.get("sheet")
                        time_col = mp.get("time_col")
                    
                        # Rétro-compat (1 colonne) + nouvelle logique multi-col
                        col_ch = mp.get("e_charge_col")         # ex: pv_to_batt_kWh
                        col_dis = mp.get("e_discharge_col")     # ex: batt_to_load_kWh
                        col_soc = mp.get("soc_col")             # ex: soc_kWh (optionnel)
                    
                        cols_ch = mp.get("e_charge_cols") or ([col_ch] if col_ch else [])
                        cols_dis = mp.get("e_discharge_cols") or ([col_dis] if col_dis else [])
                    
                        # Mapping manuel (colonne -> origine/destination)
                        charge_map = mp.get("charge_col_map") or {}        # {col: "PV"|"GRID"}
                        discharge_map = mp.get("discharge_col_map") or {}  # {col: "LOAD"|"GRID"}
                    
                        df_sto = None
                    
                        # Cas 1 : stockage a son propre fichier
                        if sto.get("source_data_mode") == "Importer un fichier":
                            df_sheets = (sto.get("custom_excel_data") or {}).get("sheets", {})
                            df_sto = df_sheets.get(sheet)
                    
                        # Cas 2 : réutilisation ouvrage/bâtiment/projet -> on utilise excel_sheets (comme PV)
                        if df_sto is None:
                            df_sto = excel_sheets.get(sheet)
                    
                        # Skip proprement si mapping/feuille/time manquants
                        if (
                            df_sto is None
                            or not sheet
                            or not time_col
                            or (not cols_ch and not cols_dis and not mp.get("p_signed_col"))
                        ):
                                
                            sb = {
                                "name": f"Stockage mesuré – {bat_nom} / {ouv_nom}",
                                "type": "storage_elec",
                                "meta": {"mode": "measured", "reason": "missing_mapping_or_sheet"},
                                "nodes": [],
                                "links": [],
                                "totals": {},
                                "impacts": {},
                            }
                    
                        else:
                            # --- Filtrer colonnes valides
                            cols_ch = [c for c in cols_ch if c and c in df_sto.columns]
                            cols_dis = [c for c in cols_dis if c and c in df_sto.columns]
                    
                            # --- Séries kWh/step par flux (mappées)
                            pv_to_batt = pd.Series(0.0, index=common_index)
                            grid_to_batt = pd.Series(0.0, index=common_index)
                            batt_to_load = pd.Series(0.0, index=common_index)
                            batt_to_grid = pd.Series(0.0, index=common_index)
                    
                            # CHARGE : somme par origine (PV vs GRID)
                            for c in cols_ch:
                                try:
                                    s = build_timeseries_kwh(df_sto, c, time_col=time_col)
                                    s = align_energy_series_to_index(s, common_index).clip(lower=0.0)
                                except Exception:
                                    s = pd.Series(0.0, index=common_index)
                    
                                src = (charge_map.get(c) or "PV").upper()
                                if src == "GRID":
                                    grid_to_batt = grid_to_batt.add(s, fill_value=0.0)
                                else:
                                    pv_to_batt = pv_to_batt.add(s, fill_value=0.0)
                    
                            # DÉCHARGE : somme par destination (LOAD vs GRID)
                            for c in cols_dis:
                                try:
                                    s = build_timeseries_kwh(df_sto, c, time_col=time_col)
                                    s = align_energy_series_to_index(s, common_index).clip(lower=0.0)
                                except Exception:
                                    s = pd.Series(0.0, index=common_index)
                    
                                dst = (discharge_map.get(c) or "LOAD").upper()
                                if dst == "GRID":
                                    batt_to_grid = batt_to_grid.add(s, fill_value=0.0)
                                else:
                                    batt_to_load = batt_to_load.add(s, fill_value=0.0)
                    
                            # SOC (optionnel)
                            ts_soc = None
                            if col_soc and col_soc in df_sto.columns:
                                try:
                                    ts_soc = build_timeseries_kwh(df_sto, col_soc, time_col=time_col)
                                    ts_soc = align_energy_series_to_index(ts_soc, common_index)
                                except Exception:
                                    ts_soc = None
                    
                            pv_to_batt_kWh = float(pv_to_batt.sum())
                            grid_to_batt_kWh = float(grid_to_batt.sum())
                            batt_to_load_kWh = float(batt_to_load.sum())
                            batt_to_grid_kWh = float(batt_to_grid.sum())
                    
                            charge_total_kWh = pv_to_batt_kWh + grid_to_batt_kWh
                            discharge_total_kWh = batt_to_load_kWh + batt_to_grid_kWh
                    
                            # --- Pertes via bilan énergétique (+ SOC si présent)
                            losses_kWh = 0.0
                            soc_start = None
                            soc_end = None
                    
                            # énergie "non restituée" (pertes + ΔSOC) -> cohérent même si charge vient de GRID
                            gap = float(charge_total_kWh - discharge_total_kWh)
                    
                            if ts_soc is not None and len(ts_soc) > 0:
                                soc_start = float(ts_soc.iloc[0])
                                soc_end = float(ts_soc.iloc[-1])
                                delta_soc = soc_end - soc_start
                                losses_kWh = float(max(gap - delta_soc, 0.0))
                            else:
                                # Pas de SOC: on ne peut pas distinguer pertes vs ΔSOC.
                                # Pour debug Sankey, on considère le gap comme "pertes/non affecté".
                                losses_kWh = float(max(gap, 0.0))
                    
                            # --- Impacts (MVP : basé sur batt->load et pv->batt)
                            grid_to_load_base_local = float((load_ts - (pv_ts.combine(load_ts, min))).clip(lower=0.0).sum())
                            pv_to_grid_base_local = float((pv_ts - pv_ts.combine(load_ts, min)).clip(lower=0.0).sum())
                    
                            delta_imp = min(batt_to_load_kWh, grid_to_load_base_local)
                            delta_inj = min(pv_to_batt_kWh, pv_to_grid_base_local)
                            delta_sc = min(batt_to_load_kWh, float(load_ts.sum()))
                    
                            # --- Nodes/links pour Phase 2 (Sankey storage)
                            pv_node_id = node_pv(index=1)
                            grid_node_id = node_elec_grid()
                            load_node_id = node_elec_load()
                            batt_node_id = node_battery(index=1)
                    
                            nodes = [
                                {"id": pv_node_id, "label": "PV", "group": "prod_elec"},
                                {"id": grid_node_id, "label": "Réseau élec.", "group": "reseau"},
                                {"id": batt_node_id, "label": "Batterie", "group": "storage_elec"},
                                {"id": load_node_id, "label": "Usages élec.", "group": "demande"},
                            ]
                    
                            links = []
                            if pv_to_batt_kWh > 0:
                                links.append({"source": pv_node_id, "target": batt_node_id, "value": pv_to_batt_kWh})
                            if grid_to_batt_kWh > 0:
                                links.append({"source": grid_node_id, "target": batt_node_id, "value": grid_to_batt_kWh})
                            if batt_to_load_kWh > 0:
                                links.append({"source": batt_node_id, "target": load_node_id, "value": batt_to_load_kWh})
                            if batt_to_grid_kWh > 0:
                                links.append({"source": batt_node_id, "target": grid_node_id, "value": batt_to_grid_kWh})
                    
                            if losses_kWh > 1e-6:
                                loss_node_id = f"ELEC_STORAGE_LOSSES_BAT{bat_id}_OUV{oi}"
                                nodes.append({"id": loss_node_id, "label": "Pertes batterie", "group": "losses"})
                                links.append({"source": batt_node_id, "target": loss_node_id, "value": float(losses_kWh)})
                    
                            # --- Timeseries store (debug)
                            ts_payload = None
                            if store_ts:
                                ts_payload = {
                                    "pv_to_batt_kWh": pv_to_batt.values.tolist(),
                                    "grid_to_batt_kWh": grid_to_batt.values.tolist(),
                                    "batt_to_load_kWh": batt_to_load.values.tolist(),
                                    "batt_to_grid_kWh": batt_to_grid.values.tolist(),
                                    "soc_kWh": (ts_soc.values.tolist() if ts_soc is not None else None),
                                }

                            sb = {
                                "name": f"Stockage (mesuré) – {bat_nom} / {ouv_nom}",
                                "type": "storage_elec",
                                "meta": {
                                    "mode": "measured",
                                    "batiment_id": bat_id,
                                    "ouvrage_nom": ouv_nom,
                                    "sheet": sheet,
                                    "time_col": time_col,
                                    # rétro compat + debug
                                    "e_charge_col": col_ch,
                                    "e_discharge_col": col_dis,
                                    "e_charge_cols": cols_ch,
                                    "e_discharge_cols": cols_dis,
                                    "charge_col_map": charge_map,
                                    "discharge_col_map": discharge_map,
                                    "soc_col": col_soc,
                                },
                                "nodes": nodes,
                                "links": links,
                                "totals": {
                                    "pv_to_batt_kWh": pv_to_batt_kWh,
                                    "grid_to_batt_kWh": grid_to_batt_kWh,
                                    "batt_to_load_kWh": batt_to_load_kWh,
                                    "batt_to_grid_kWh": batt_to_grid_kWh,
                                    "losses_kWh": losses_kWh,
                                    "soc_start": soc_start,
                                    "soc_end": soc_end,
                                },
                                "impacts": {
                                    "delta_grid_import_kWh": float(delta_imp),
                                    "delta_grid_injection_kWh": float(delta_inj),
                                    "delta_self_consumption_kWh": float(delta_sc),
                                },
                            }
                    
                            if store_ts and ts_payload is not None:
                                try:
                                    ts_bucket = results["timeseries_store"]["batiments"][str(bat_id)]["ouvrages"][f"{oi}"]
                                    ts_bucket.update({k: v for k, v in ts_payload.items() if v is not None})
                                except Exception:
                                    pass

                        # =========================================
                        # Économie batterie (cashflow) — uniquement stockage électrique
                        # =========================================
                        try:
                            if (sto.get("type_general") == "Electrique") and (sb.get("type") == "storage_elec"):
                                tech_params = sto.get("parametres") or {}

                                # --- Capex (valeur + unité)
                                capex_item = tech_params.get("Capex")
                                if not isinstance(capex_item, dict) or capex_item.get("valeur") in (None, "", 0):
                                    raise ValueError("Batterie: 'Capex' manquant dans techno (Excel).")
                                
                                capex_val = float(capex_item["valeur"])
                                capex_unit = (capex_item.get("unite") or "").strip().lower()
                                
                                cap_kwh = sto.get("capacity_kwh")
                                if cap_kwh in (None, "", 0):
                                    raise ValueError("Batterie: capacity_kwh manquant — nécessaire pour calculer CAPEX total.")
                                cap_kwh = float(cap_kwh)
                                
                                # Conversion CAPEX -> total
                                if "/kwh" in capex_unit:
                                    capex_total_chf = capex_val * cap_kwh
                                elif capex_unit in ("chf", "chf total", "chf/an") or capex_unit == "chf":  # tolérant
                                    capex_total_chf = capex_val
                                else:
                                    raise ValueError(f"Batterie: unité Capex non supportée: '{capex_item.get('unite')}'. Attendu CHF/kWh ou CHF.")

                        
                                opex_annual_chf = _get_param_numeric(
                                    tech_params,
                                    ["Opex"],
                                    default=None
                                )
                                if opex_annual_chf is None:
                                    raise ValueError("Batterie: OPEX annuel manquant dans techno (Excel).")
                        
                                lifetime_years = _get_param_numeric(
                                    tech_params,
                                    ["Durée de vie", "Lifetime", "lifetime_years", "Vie utile", "Durée de vie (ans)"],
                                    default=None
                                )
                                if lifetime_years in (None, "", 0):
                                    raise ValueError("Batterie: durée de vie manquante dans techno (Excel).")
                        
                                # TODO: méthode pour calculer le cost factor (pour demain on force 1.0)
                                replacement_cost_factor = 1.0  # TODO: définir une méthode (dégradation/indice/prix)
                        
                                t = sb.get("totals") or {}
                                econ_batt = compute_battery_cashflow_series(
                                    capex_total_chf=float(capex_total_chf),
                                    opex_annual_chf=float(opex_annual_chf),
                                    pv_to_batt_kwh=float(t.get("pv_to_batt_kWh", 0.0) or 0.0),
                                    grid_to_batt_kwh=float(t.get("grid_to_batt_kWh", 0.0) or 0.0),
                                    batt_to_load_kwh=float(t.get("batt_to_load_kWh", 0.0) or 0.0),
                                    batt_to_grid_kwh=float(t.get("batt_to_grid_kWh", 0.0) or 0.0),
                                    price_buy_chf_kwh=float(price_buy),
                                    price_sell_chf_kwh=float(price_sell),
                                    horizon_years=int(horizon_years),
                                    lifetime_years=float(lifetime_years),
                                    replacement_cost_factor=float(replacement_cost_factor),
                                )
                        
                                batt_econ_entries.append({
                                    "batiment_id": bat_id,
                                    "ouvrage_nom": ouv_nom,
                                    "storage_index": si,
                                    "mode": (sto.get("data_mode") or ""),
                                    "techno": sto.get("techno"),
                                    "engine": sto.get("engine"),
                                    "capacity_kwh": sto.get("capacity_kwh"),
                                    "economics": econ_batt,
                                })
                        except Exception as e:
                            # pas de fallback silencieux → on stoppe net (tu préfères voir l'erreur)
                            raise

                    
                    # ------------------------------
                    # MODE SIMULÉ : dispatch PV/load existant
                    # ------------------------------
                    else:
                        sb = compute_battery_li_ion_flow_block_timeseries(load_ts, pv_ts, sto)

                    storage_blocks.append(sb)
                    
                    # HYBRID: stocker timeseries batterie dans timeseries_store
                    if store_ts and (sto.get("data_mode") or "") == "measured":
                        try:
                            ts_bucket = results["timeseries_store"]["batiments"][str(bat_id)]["ouvrages"][f"{oi}"]
                            ts_bucket["pv_to_batt_kWh"] = pv_to_batt.values.tolist()
                            ts_bucket["batt_to_load_kWh"] = batt_to_load.values.tolist()
                            if ts_soc is not None:
                                ts_bucket["soc_kWh"] = ts_soc.values.tolist()
                        except Exception:
                            pass

        
                    imp = sb.get("impacts") or {}
                    sto_delta_import += float(imp.get("delta_grid_import_kWh", 0.0) or 0.0)
                    sto_delta_inj += float(imp.get("delta_grid_injection_kWh", 0.0) or 0.0)
                    sto_delta_sc += float(imp.get("delta_self_consumption_kWh", 0.0) or 0.0)
                    t_sb = sb.get("totals") or {}
                    sto_pv_to_batt += float(t_sb.get("pv_to_batt_kWh", 0.0) or 0.0)
                    sto_grid_to_batt += float(t_sb.get("grid_to_batt_kWh", 0.0) or 0.0)
                    sto_batt_to_load += float(t_sb.get("batt_to_load_kWh", 0.0) or 0.0)
                    sto_batt_to_grid += float(t_sb.get("batt_to_grid_kWh", 0.0) or 0.0)
                    sto_losses += float(t_sb.get("losses_kWh", 0.0) or 0.0)
        
        # Append storage blocks to bat flows (Phase 2 affichera ces links)
        for sb in storage_blocks:
            bat_flows.append(sb)

        if (
            local_demand_elec_kwh > 0
            or local_pv_prod_kwh > 0
            or local_demand_th_kwh > 0
        ):
            # ==========================================================
            # PATCH 2 : recalcul "base" (load / pv / auto / inj) au pas fin
            # Objectif : si séries datetime existent, on recalcule les totaux
            # de manière physiquement cohérente (min(pv,load) pas-à-pas).
            # ==========================================================
            results.setdefault("checks", {})
            results["checks"].setdefault("patch2_energy", {})
          
            have_load = ("ts_load_kwh" in locals()) and (ts_load_kwh is not None)
            have_pv = (len(pv_prod_series_list) > 0)
          
            if have_load and have_pv:
                pv_prod_raw = pv_prod_series_list
                pv_auto_raw = pv_auto_series_list  # peut être vide
          
                # Fenêtre temporelle commune (intersection) + index le plus fin
                common_index = common_sim_window_intersection(
                    [ts_load_kwh.index] + [s.index for s in pv_prod_raw]
                )
          
                if common_index is not None and len(common_index) > 0:
                    load = align_energy_series_to_index(ts_load_kwh, common_index)
                    pv_prod = sum(align_energy_series_to_index(s, common_index) for s in pv_prod_raw)
          
                    # Si l'utilisateur a donné une colonne "PV auto" -> on la prend
                    if len(pv_auto_raw) > 0:
                        pv_auto = sum(align_energy_series_to_index(s, common_index) for s in pv_auto_raw)
                    else:
                        # ----------------------------------------------------------
                        # Reconstruction PV->Load proportionnelle + bornage strict
                        # via helper pv_module (source unique de vérité)
                        # ----------------------------------------------------------
                                     
                        pv_auto, meta_selfc = reconstruct_pv_to_load_proportional_bounded(
                            pv_ts=pv_prod,
                            load_ts=load,
                            selfc_pct_input=pv_default_selfc_pct,
                        )
                    
                        # Stockage meta pour Phase 2 (warning saturation)
                        results.setdefault("internal", {})
                        results["internal"].setdefault("pv_selfc_meta", [])
                        results["internal"]["pv_selfc_meta"].append(meta_selfc)

          
                    # Invariants (physique)
                    pv_auto = pv_auto.clip(lower=0.0)
                    pv_auto = pv_auto.where(pv_auto <= pv_prod, pv_prod)
                    pv_auto = pv_auto.where(pv_auto <= load, load)
          
                    pv_inj = (pv_prod - pv_auto).clip(lower=0.0)
                    
                    # ----------------------------------------------------------
                    # Contrat Phase 2 : timeseries mesurées (JSON-friendly)
                    # ----------------------------------------------------------
                    results.setdefault("timeseries_by_batiment", {})
                    bat_bucket = results["timeseries_by_batiment"].setdefault(str(bat_id), {})
                    
                    # index commun (string) pour reconstruire côté Phase 2
                    bat_bucket["index"] = [str(x) for x in common_index]
                    
                    measured = bat_bucket.setdefault("measured", {})
                    measured.update({
                        "load_kWh": load.values.tolist(),
                        "pv_prod_kWh": pv_prod.values.tolist(),
                        "pv_to_load_kWh": pv_auto.values.tolist(),
                        "pv_to_grid_kWh": pv_inj.values.tolist(),
                    })
                    
                    # meta autoconsommation (warning Phase 2)
                    if "meta_selfc" in locals():
                        measured["pv_selfc_meta"] = meta_selfc


                    
                    # ==========================================================
                    # HYBRID – stockage des timeseries RESULTATS (load / PV / grid)
                    # ==========================================================
                    store_ts = bool(((project.get("meta") or {}).get("debug") or {}).get("store_timeseries", False))
                    
                    if store_ts:
                        def _ts_bucket(results, bat_id, oi, bat_nom, ouv_nom, index):
                            results.setdefault("timeseries_store", {})
                            results["timeseries_store"].setdefault("batiments", {})
                            b = results["timeseries_store"]["batiments"].setdefault(str(bat_id), {})
                            ovs = b.setdefault("ouvrages", {})
                            o = ovs.setdefault(str(oi), {})
                            o.setdefault("meta", {})
                            o["meta"].update({
                                "batiment_id": bat_id,
                                "batiment_nom": bat_nom,
                                "ouvrage_index": oi,
                                "ouvrage_nom": ouv_nom,
                                "dt_h_med": float(infer_dt_hours_from_index(index)) if index is not None else None,
                            })
                            return o
                    
                    if store_ts and have_load and have_pv and (common_index is not None) and len(common_index) > 0:
                        o = _ts_bucket(results, bat_id, oi, bat_nom, ouv_nom, common_index)
                        o["index"] = [str(x) for x in common_index]
                        o["load_kWh"] = load.values.tolist()
                        o["pv_prod_kWh"] = pv_prod.values.tolist()
                        o["pv_auto_kWh"] = pv_auto.values.tolist()
                        o["pv_to_grid_kWh"] = pv_inj.values.tolist()
                        o["grid_to_load_kWh"] = (load - pv_auto).clip(lower=0.0).values.tolist()


                    # Overwrite totaux "base" (ceux utilisés ensuite)
                    local_demand_elec_kwh = float(load.sum())
                    local_pv_prod_kwh = float(pv_prod.sum())
                    local_pv_auto_kwh = float(pv_auto.sum())
                    local_pv_inj_kwh = float(pv_inj.sum())
          
                    # Debug / contrôles
                    results["checks"]["patch2_energy"][f"bat{bat_id}::ouv{oi}"] = {
                        "mode": "fine",
                        "n": int(len(common_index)),
                        "dt_h_med": float(infer_dt_hours_from_index(common_index)),
                        "pv_balance_abs": float(abs(local_pv_prod_kwh - (local_pv_auto_kwh + local_pv_inj_kwh))),
                    }
                else:
                    results["checks"]["patch2_energy"][f"bat{bat_id}::ouv{oi}"] = {
                        "mode": "fallback",
                        "reason": "no_common_index",
                    }
            else:
                results["checks"]["patch2_energy"][f"bat{bat_id}::ouv{oi}"] = {
                    "mode": "fallback",
                    "reason": "missing_timeseries",
                }
          
            # ==========================================================
            # BASE (doit venir APRÈS PATCH 2)
            # ==========================================================
            pv_auto_used_base = min(local_pv_auto_kwh, local_demand_elec_kwh)
            grid_to_load_base = max(local_demand_elec_kwh - pv_auto_used_base, 0.0)
            pv_to_grid_base = max(local_pv_inj_kwh, 0.0)

            # ------------------------------
            # FINAL (avec stockage) — état physique réel
            # (si pas de stockage, deltas = 0 et final == base)
            # ------------------------------
            try:
                _d_imp = float(sto_delta_import or 0.0)
            except Exception:
                _d_imp = 0.0
            try:
                _d_inj = float(sto_delta_inj or 0.0)
            except Exception:
                _d_inj = 0.0
            try:
                _d_sc = float(sto_delta_sc or 0.0)
            except Exception:
                _d_sc = 0.0

            # ------------------------------
            # FINAL (avec stockage) — mode mesuré : on utilise les flux physiques
            # ------------------------------
            if (sto_batt_to_load > 0) or (sto_pv_to_batt > 0) or (sto_grid_to_batt > 0) or (sto_batt_to_grid > 0):
                # PV->Load = PV direct (PAS l'auto-conso totale)
                pv_to_load_final = float(local_pv_auto_kwh or 0.0)  # chez toi local_pv_auto_kwh doit correspondre à pv_to_load
                # Si tu n'es pas sûr, mets explicitement: pv_to_load_final = float(local_pv_prod_kwh) - float(pv_to_grid) - float(sto_pv_to_batt)
            
                pv_auto_used = min(pv_to_load_final, float(local_demand_elec_kwh or 0.0))
            
                # Grid->Load = demande - PV direct - batt->load
                grid_to_load = max(float(local_demand_elec_kwh or 0.0) - pv_auto_used - float(sto_batt_to_load or 0.0), 0.0)
            
                # PV->Grid = PV prod - PV->Load - PV->Batt
                pv_to_grid = max(float(local_pv_prod_kwh or 0.0) - pv_auto_used - float(sto_pv_to_batt or 0.0), 0.0)
            
            else:
                # fallback : logique delta (utile pour "batterie proposée" plus tard)
                grid_to_load = max(grid_to_load_base - _d_imp, 0.0)
                pv_to_grid   = max(pv_to_grid_base - _d_inj, 0.0)
                pv_auto_used = min(pv_auto_used_base + max(_d_sc, 0.0), local_demand_elec_kwh)


            # ------------------------------
            # Stockage interne (base vs final) — pas pour le Sankey
            # ------------------------------
            rec_base = {
                "ouvrage_index": oi,
                "ouvrage_nom": ouv_nom,
                "demand_elec_kWh": float(local_demand_elec_kwh),
                "pv_prod_kWh": float(local_pv_prod_kwh),
                "pv_auto_used_kWh": float(pv_auto_used_base),
                "grid_to_load_kWh": float(grid_to_load_base),
                "pv_to_grid_kWh": float(pv_to_grid_base),
            }
            rec_final = {
                "ouvrage_index": oi,
                "ouvrage_nom": ouv_nom,
                "demand_elec_kWh": float(local_demand_elec_kwh),
                "pv_prod_kWh": float(local_pv_prod_kwh),
                "pv_auto_used_kWh": float(pv_auto_used),
                "grid_to_load_kWh": float(grid_to_load),
                "pv_to_grid_kWh": float(pv_to_grid),
                "delta_grid_import_kWh": float(_d_imp),
                "delta_grid_injection_kWh": float(_d_inj),
                "delta_self_consumption_kWh": float(_d_sc),
            }

            results["internal"]["energy_global_base"].setdefault(bat_id, []).append(rec_base)
            results["internal"]["energy_global_final"].setdefault(bat_id, []).append(rec_final)

            # --- Nœuds de base (ontologie élec) ---
            pv_node_id = node_pv(index=1)      # agrégé PV pour l’instant
            grid_node_id = node_elec_grid()
            load_node_id = node_elec_load()

            nodes_global = [
                {"id": pv_node_id,   "label": "PV", "group": "prod_elec"},
                {"id": grid_node_id, "label": "Réseau élec.", "group": "reseau"},
                {
                    "id": load_node_id,
                    "label": "Consommation élec. bâtiment",
                    "group": "demande",
                },
            ]
            links_global: List[Dict[str, Any]] = []
            if (sto_pv_to_batt > 0) or (sto_grid_to_batt > 0) or (sto_batt_to_load > 0) or (sto_batt_to_grid > 0):
                batt_node_id = node_battery(index=1)
                nodes_global.append({"id": batt_node_id, "label": "Batterie", "group": "storage_elec"})
            
                if sto_pv_to_batt > 0:
                    links_global.append({"source": pv_node_id, "target": batt_node_id, "value": sto_pv_to_batt})
            
                if sto_grid_to_batt > 0:
                    links_global.append({"source": grid_node_id, "target": batt_node_id, "value": sto_grid_to_batt})
            
                if sto_batt_to_load > 0:
                    links_global.append({"source": batt_node_id, "target": load_node_id, "value": sto_batt_to_load})
            
                if sto_batt_to_grid > 0:
                    links_global.append({"source": batt_node_id, "target": grid_node_id, "value": sto_batt_to_grid})

            # --- Liens électriques PV + Réseau -> conso bâtiment ---
            if pv_auto_used > 0:
                links_global.append(
                    {
                        "source": pv_node_id,
                        "target": load_node_id,
                        "value": pv_auto_used,
                    }
                )

            if grid_to_load > 0:
                links_global.append(
                    {
                        "source": grid_node_id,
                        "target": load_node_id,
                        "value": grid_to_load,
                    }
                )

            # Injection réseau depuis le PV
            if pv_to_grid > 0:
                grid_export_id = grid_node_id  # même nœud, mais conceptuellement export
                links_global.append(
                    {
                        "source": pv_node_id,
                        "target": grid_export_id,
                        "value": pv_to_grid,
                    }
                )

            # --- Liens thermiques (si on a une demande) ---
            if local_demand_th_kwh > 0:
                heat_src_id = node_heat_prod("UNKNOWN", index=1)
                heat_load_id = node_heat_load()

                nodes_global.extend(
                    [
                        {
                            "id": heat_src_id,
                            "label": "Système de chauffage thermique",
                            "group": "prod_th",
                        },
                        {
                            "id": heat_load_id,
                            "label": "Consommation thermique",
                            "group": "demande",
                        },
                    ]
                )
                links_global.append(
                    {
                        "source": heat_src_id,
                        "target": heat_load_id,
                        "value": local_demand_th_kwh,
                    }
                )

            # --- Nœud de demande finale ELECTRIQUE uniquement ---
            final_demand_id = "ELEC_FINAL_DEMAND"
            nodes_global.append(
                {
                    "id": final_demand_id,
                    "label": "Demande électrique",
                    "group": "final",
                }
            )

            # Toute la conso élec. bâtiment -> demande électrique finale
            if local_demand_elec_kwh > 0:
                links_global.append(
                    {
                        "source": load_node_id,
                        "target": final_demand_id,
                        "value": local_demand_elec_kwh,
                    }
                )
            # ==========================================================
            # Pertes / non affecté (2 nodes distincts : ELEC + TH) — par ouvrage
            # ==========================================================
            loss_elec_id = f"ELEC_LOSSES_BAT{bat_id}_OUV{oi}"
            loss_th_id   = f"TH_LOSSES_BAT{bat_id}_OUV{oi}"
            
            def _ensure_loss_node(_id: str, label: str):
                if not any(isinstance(n, dict) and n.get("id") == _id for n in nodes_global):
                    nodes_global.append({"id": _id, "label": label, "group": "losses"})
            
            def _add_loss_link(source_id: str, loss_id: str, loss_label: str, value: float):
                v = float(value or 0.0)
                if v <= 1e-6:
                    return
                _ensure_loss_node(loss_id, loss_label)
                links_global.append({"source": source_id, "target": loss_id, "value": v})
            
            # --------------------------
            # A) PERTES ELECTRIQUES
            # --------------------------
            
            # 1) Pertes batterie (si batterie présente)
            # (tu as batt_node_id défini uniquement si les flux batterie > 0)
            if ("batt_node_id" in locals()) and (batt_node_id is not None):
                _add_loss_link(
                    source_id=batt_node_id,
                    loss_id=loss_elec_id,
                    loss_label="Pertes / non affecté – électricité",
                    value=sto_losses,
                )
            
            # 2) PV non réparti (PV prod doit aller vers load + grid + batt)
            pv_allocated = float(pv_auto_used or 0.0) + float(pv_to_grid or 0.0) + float(sto_pv_to_batt or 0.0)
            pv_unmapped = float(local_pv_prod_kwh or 0.0) - pv_allocated
            _add_loss_link(
                source_id=pv_node_id,
                loss_id=loss_elec_id,
                loss_label="Pertes / non affecté – électricité",
                value=pv_unmapped,
                )   
            # --------------------------
            # B) PERTES THERMIQUES
            # --------------------------
            # Ici on fait simple et robuste : si tu as un système thermique "UNKNOWN" qui alimente la demande,
            # alors (input - demande) = pertes/non affecté.
            # Pour l’instant ton modèle thermique est juste 1 lien heat_src -> heat_load.
            # Donc pertes = 0 par défaut, mais on met la structure pour éviter le futur "flux fantôme".
            
            if local_demand_th_kwh > 0:
                # heat_src_id et heat_load_id existent dans ton code quand tu ajoutes le thermique
                # Si un jour tu ajoutes d’autres flux (ex: chaudière->pertes->...), ce node servira.
                th_input = float(local_demand_th_kwh or 0.0)  # actuellement tu mets input == demande
                th_unmapped = th_input - float(local_demand_th_kwh or 0.0)
            
                # Si tu ajoutes plus tard un vrai th_input (ex: production), remplace th_input par ce total.
                _add_loss_link(
                    source_id=heat_src_id,
                    loss_id=loss_th_id,
                    loss_label="Pertes / non affecté – thermique",
                    value=th_unmapped,
                )

            # --- Résidu bilan élec vers la demande (toujours défini) ---
            elec_to_load = float(pv_auto_used or 0.0) + float(sto_batt_to_load or 0.0) + float(grid_to_load or 0.0)
            elec_balance_residual = elec_to_load - float(local_demand_elec_kwh or 0.0)
            
            flow_block_bilan_global = {
                "name": f"Bilan énergétique – {bat_nom} / {ouv_nom}",
                "type": "energy_global",
                "meta": {
                    "batiment_id": bat_id,
                    "batiment_nom": bat_nom,
                    "ouvrage_nom": ouv_nom,
                    "ouvrage_type_usage": ouv.get("type_usage", ""),
                },
                "nodes": nodes_global,
                "links": links_global,
                "totals": {
                    "demand_elec_kWh": local_demand_elec_kwh,
                    "demand_th_kWh": local_demand_th_kwh,
                    "pv_prod_kWh": local_pv_prod_kwh,
                    "pv_auto_used_kWh": pv_auto_used,
                    "grid_to_load_kWh": grid_to_load,
                    "pv_to_grid_kWh": pv_to_grid,
                    "final_demand_kWh": local_demand_elec_kwh,
                    "storage_charge_kWh": float(sto_pv_to_batt + sto_grid_to_batt),
                    "storage_discharge_kWh": float(sto_batt_to_load + sto_batt_to_grid),
                    "storage_losses_kWh": float(sto_losses),
                    "delta_grid_import_kWh": sto_delta_import,
                    "delta_grid_injection_kWh": sto_delta_inj,
                    "delta_self_consumption_kWh": sto_delta_sc,
                    "losses_unmapped_kWh": float(max(pv_unmapped, 0.0) + max(sto_losses, 0.0) + max((elec_to_load - float(local_demand_elec_kwh or 0.0)), 0.0)),


                },
            }
            bat_flows.append(flow_block_bilan_global)
  

    # ------------------------------------------------------------------
    # RÉSUMÉS GLOBAUX (PV + demandes)
    # ------------------------------------------------------------------
    total_pv = 0.0
    total_auto = 0.0
    total_inj = 0.0
    total_demand_elec = 0.0
    total_demand_th = 0.0

    for bat_id, blocks in results["flows"]["batiments"].items():
        for block in blocks:
            btype = block.get("type")

            if btype == "pv":
                t = block.get("totals", {})
                total_pv += float(t.get("pv_prod_kWh", 0.0))
                total_auto += float(t.get("pv_auto_kWh", 0.0))
                total_inj += float(t.get("pv_inj_kWh", 0.0))

            elif btype == "demand_elec":
                t = block.get("totals", {})
                total_demand_elec += float(t.get("demand_elec_kWh", 0.0))

            elif btype == "demand_th":
                t = block.get("totals", {})
                total_demand_th += float(t.get("demand_th_kWh", 0.0))

            elif btype == "energy_global":
                # On pourrait aussi utiliser ce bloc pour les résumés,
                # mais pour l'instant on garde la logique simple ci-dessus.
                pass
    results["storage_economics_by_batiment"][str(bat_id)] = batt_econ_entries
    results["summaries"]["pv_prod_kWh_total"] = total_pv
    results["summaries"]["pv_auto_kWh_total"] = total_auto
    results["summaries"]["pv_inj_kWh_total"] = total_inj
    results["summaries"]["demand_elec_kWh_total"] = total_demand_elec
    results["summaries"]["demand_th_kWh_total"] = total_demand_th


    # ------------------------------------------------------------------
    # ÉCONOMIE PAR BÂTIMENT (agrégation des flow_blocks du bâtiment)
    # ------------------------------------------------------------------
    economics_by_batiment: Dict[Any, Dict[str, float]] = {}

    for bat_id, blocks in results["flows"]["batiments"].items():
        economics_by_batiment[bat_id] = compute_economics_from_flow_blocks(blocks)

    results["economics_by_batiment"] = economics_by_batiment

    # ==========================================================
    # PV PROPOSÉ (toiture) – par bâtiment  (clé = batiment_id)
    # ==========================================================
    pv_defaults = _get_default_pv_tech_params()
    station_name = (project.get("meta") or {}).get("station_meteo") or (project.get("meta") or {}).get("station_name")
    
    # Paramètres module (venant de ton Excel techno)
    p_module_kw = _get_param_numeric(pv_defaults, ["Puissance du module", "Puissance module (kW)", "Puissance min", "P_module (kW)", "p_module_kw"])
    area_module_m2 = _get_param_numeric(pv_defaults, ["Surface du module", "Surface du module (m2)", "Surface module (m2)", "area_module_m2"])
    eta_mod_pct = _get_param_numeric(pv_defaults, ["Rendement", "Rendement (%)", "eta_mod_pct"]) or 100.0
    
    # Interdit: fallback silencieux sur techno PV
    if not p_module_kw or not area_module_m2:
        raise ValueError(
            "PV proposé: techno PV incomplète (Excel). "
            "Paramètres requis manquants: 'Puissance du module' (p_module_kw) et/ou 'Surface du module' (area_module_m2). "
            "Corriger la techno PV dans l'Excel (sheet techno PV) et relancer."
        )
    
    panel_eff = float(eta_mod_pct or 100.0)
    
    results.setdefault("pv_proposed_by_batiment", {})
    
    # ==========================================================
    # BATTERIE PROPOSÉE (PV-only, niveau bâtiment)
    # ==========================================================
    # Objectif:
    # - Dimensionnement heuristique (P_target = PV installé du bâtiment)
    # - Dispatch via compute_battery_li_ion_flow_block_timeseries (=> dispatch_electric_storage)
    # - Stockage des timeseries bâtiment pour "avant/après" dans show_variantes
    #
    # TODO ARCHI:
    # Aujourd'hui l'UX batterie est sous PV (PV-only).
    # À terme, config stockage indépendante, sources/puits multi-ouvrages/bâtiments/projet/cluster.
    
    results.setdefault("battery_proposed_by_batiment", {})
    results.setdefault("timeseries_by_batiment", {})  # pour show_variantes (courbes avant/après)
    
    def _sum_pv_installed_kw_for_bat(bat_id: Any) -> float:
        total = 0.0
        for ouv in (project.get("ouvrages") or []):
            if ouv.get("batiment_id") != bat_id:
                continue
            for prod in (ouv.get("producteurs") or []):
                # robuste: engine ou techno
                if (prod.get("engine") == "pv") or ("photovolta" in str(prod.get("techno") or "").lower()):
                    total += float(prod.get("puissance_kw") or 0.0)
        if total <= 0:
            raise ValueError(f"Batterie proposée: aucune puissance PV trouvée pour batiment_id={bat_id}.")
        return total
    
    
    for bi, bat in enumerate(batiments):
        bat_id = bat.get("id") or bat.get("batiment_id") or bi
        bat_nom = bat.get("nom") or f"Bâtiment {bi+1}"
    
        batt_cfg = (bat.get("pv_battery_proposed") or {})
        if not batt_cfg.get("enabled"):
            continue
    
        # --- récupérer séries bâtiment (kWh/step) déjà accumulées durant la boucle ouvrages
        load_list = ts_load_by_bat.get(bat_id, []) or ts_load_by_bat.get(bi, []) or []
        pv_list = ts_pv_by_bat.get(bat_id, []) or ts_pv_by_bat.get(bi, []) or []
    
        if not load_list:
            raise ValueError(f"Batterie proposée: série load manquante pour {bat_nom} (bat_id={bat_id}).")
        if not pv_list:
            raise ValueError(f"Batterie proposée: série PV manquante pour {bat_nom} (bat_id={bat_id}).")
    
        idx_list = [s.index for s in load_list] + [s.index for s in pv_list]
        common_index = common_sim_window_intersection(idx_list)
        if common_index is None or len(common_index) == 0:
            raise ValueError(f"Batterie proposée: index commun vide pour {bat_nom}.")
    
        load_ts = sum(align_energy_series_to_index(s, common_index) for s in load_list)
        pv_ts = sum(align_energy_series_to_index(s, common_index) for s in pv_list)
    
        # ----------------------------------------------------------
        # Lire config Phase 1 (strict: pas de fallback silencieux)
        # ----------------------------------------------------------
        sizing = (batt_cfg.get("sizing") or {})
        techc = (batt_cfg.get("tech_constraints") or {})
    
        hours_target = float(sizing.get("hours_target") or 0.0)
        if hours_target <= 0:
            raise ValueError(f"Batterie proposée: hours_target manquant/<=0 ({bat_nom}).")
    
        soc_min_frac = float(sizing.get("soc_min_frac") if sizing.get("soc_min_frac") is not None else 0.0)
        soc_max_frac = float(sizing.get("soc_max_frac") if sizing.get("soc_max_frac") is not None else 0.0)
        soc_window = soc_max_frac - soc_min_frac
        if soc_window <= 0:
            raise ValueError(f"Batterie proposée: fenêtre SOC invalide ({bat_nom}).")
    
        e_min = techc.get("capacity_min_kwh")
        e_max = techc.get("capacity_max_kwh")
        c_rate_ch = techc.get("c_rate_charge_kw_per_kwh")
        c_rate_dis = techc.get("c_rate_discharge_kw_per_kwh")
        eta_global_raw = techc.get("eta_global")
    
        if e_min is None or e_max is None:
            raise ValueError(f"Batterie proposée: capacity_min_kwh/capacity_max_kwh manquants ({bat_nom}).")
        if c_rate_ch is None or c_rate_dis is None:
            raise ValueError(f"Batterie proposée: C-rate manquant ({bat_nom}).")
    
        e_min = float(e_min); e_max = float(e_max)
        c_rate_ch = float(c_rate_ch); c_rate_dis = float(c_rate_dis)
        if c_rate_ch <= 0 or c_rate_dis <= 0:
            raise ValueError(f"Batterie proposée: C-rate <=0 ({bat_nom}).")
    
        # conversion rendement (90 -> 0.9) dans les calculs
        eta_global = _normalize_efficiency(float(eta_global_raw))
    
        # ----------------------------------------------------------
        # Dimensionnement heuristique (validé)
        # ----------------------------------------------------------
        p_target_kw = _sum_pv_installed_kw_for_bat(bat_id)
    
        cap_energy = (p_target_kw * hours_target) / soc_window
        cap_power = max(p_target_kw / c_rate_ch, p_target_kw / c_rate_dis)
        cap_total = max(cap_energy, cap_power)
    
        pack = _pack_capacities(cap_total, e_min=e_min, e_max=e_max)
        cap_final = float(sum(pack))
    
        # ----------------------------------------------------------
        # Construire un storage_cfg compatible bat_li_ion_module
        # ----------------------------------------------------------
        sto_cfg = {
            "type_general": "Electrique",
            "engine": "bat_li_ion",
            "techno": batt_cfg.get("techno") or "Battery (proposed)",
            "data_mode": "simulated",
    
            "capacity_kwh": cap_final,
            "soc_min_frac": soc_min_frac,
            "soc_max_frac": soc_max_frac,
    
            # PV-only (pour l’instant)
            "grid_charge_allowed": False,
            "grid_discharge_allowed": False,
    
            "parametres": {
                # la fonction bat module lit "Rendement" -> on met déjà en fraction
                "Rendement": {"valeur": eta_global, "unite": "-"},
                "C-rate charge max.": {"valeur": c_rate_ch, "unite": "kW/kWh"},
                "C-rate décharge max.": {"valeur": c_rate_dis, "unite": "kW/kWh"},
            },
    
            "_sizing_debug": {
                "p_target_kw": p_target_kw,
                "hours_target": hours_target,
                "soc_window": soc_window,
                "cap_energy_kwh": cap_energy,
                "cap_power_kwh": cap_power,
                "cap_total_kwh": cap_total,
                "pack_capacities_kwh": pack,
                "cap_final_kwh": cap_final,
                "eta_global_input": float(eta_global_raw),
                "eta_global_norm": eta_global,
            },
        }
    
        # ----------------------------------------------------------
        # Dispatch + flow_block via dispatch_electric_storage
        # ----------------------------------------------------------
        sb = compute_battery_li_ion_flow_block_timeseries(load_ts, pv_ts, sto_cfg)
    
        # marquage “proposé” pour que show_variantes sache filtrer
        sb.setdefault("meta", {})
        sb["meta"].update({
            "batiment_id": bat_id,
            "batiment_nom": bat_nom,
            "mode": "proposed",
            "pack_capacities_kwh": pack,
            "capacity_kwh": cap_final,
            "p_target_kw": p_target_kw,
            "hours_target": hours_target,
        })
    
        # Injecter dans flows bâtiment
        results["flows"]["batiments"].setdefault(bat_id, [])
        results["flows"]["batiments"][bat_id].append(sb)
    
        # ----------------------------------------------------------
        # BASE (AVANT) = RÉALITÉ MESURÉE (pas de recalcul)
        # - on réutilise pv_to_load_kWh déjà construit plus haut (reconstruction proportionnelle bornée)
        # ----------------------------------------------------------
        ts_measured = (
            results.get("timeseries_by_batiment", {})
            .get(bat_id, {})
            .get("measured", {})
        )
        
        pv_to_load_measured_ts = ts_measured.get("pv_to_load_kWh")
        
        if pv_to_load_measured_ts is None:
            # fallback strict minimal (devrait rarement arriver si PATCH 2 est en place)
            pv_to_load_base_ts = pv_ts.clip(upper=load_ts)
        else:
            # align sur common_index
            pv_to_load_base_ts = align_energy_series_to_index(pv_to_load_measured_ts, common_index)
        
        # ----------------------------------------------------------
        # BEFORE = MONDE RÉEL (measured) : ne pas recalculer ici
        # ----------------------------------------------------------
        def _as_series_on_index(x, idx):
            if x is None:
                return None
            if isinstance(x, pd.Series):
                return x.reindex(idx).fillna(0.0)
            # list / array-like
            try:
                arr = list(x)
            except Exception:
                return None
            s = pd.Series(arr, index=idx[:len(arr)])
            if len(s) < len(idx):
                s = s.reindex(idx, fill_value=0.0)
            else:
                s = s.iloc[:len(idx)]
                s.index = idx
            return s.fillna(0.0)
        
        ts_bat = (results.get("timeseries_by_batiment", {}) or {}).get(str(bat_id), {}) or {}
        ts_meas = (ts_bat.get("measured") or {})
        
        pv_to_load_meas = _as_series_on_index(ts_meas.get("pv_to_load_kWh"), common_index)
        
        # Fallback ultime (si measured absent) : cohérent physiquement
        if pv_to_load_meas is None:
            pv_to_load_base_ts = pv_ts.clip(upper=load_ts)
        else:
            pv_to_load_base_ts = pv_to_load_meas
        
        # Bornage physique strict (au cas où)
        pv_to_load_base_ts = pv_to_load_base_ts.clip(lower=0.0)
        pv_to_load_base_ts = pv_to_load_base_ts.where(pv_to_load_base_ts <= pv_ts, pv_ts)
        pv_to_load_base_ts = pv_to_load_base_ts.where(pv_to_load_base_ts <= load_ts, load_ts)
        
        grid_to_load_base_ts = (load_ts - pv_to_load_base_ts).clip(lower=0.0)
        pv_to_grid_base_ts = (pv_ts - pv_to_load_base_ts).clip(lower=0.0)

        
        grid_to_load_base_ts = (load_ts - pv_to_load_base_ts).clip(lower=0.0)
        pv_to_grid_base_ts = (pv_ts - pv_to_load_base_ts).clip(lower=0.0)

    
        t_sb = sb.get("totals") or {}
        pv_to_batt_kwh = float(t_sb.get("pv_to_batt_kWh", 0.0) or 0.0)
        batt_to_load_kwh = float(t_sb.get("batt_to_load_kWh", 0.0) or 0.0)
        losses_kwh = float(t_sb.get("losses_kWh", 0.0) or 0.0)
    
        # PV-only => la charge batterie vient du surplus PV, donc pv_to_load_direct reste min(pv, load)
        pv_to_load_final_kwh = float(pv_to_load_base_ts.sum())
        grid_to_load_final_kwh = max(float(load_ts.sum()) - pv_to_load_final_kwh - batt_to_load_kwh, 0.0)
    
        # injection finale = surplus PV - pv_to_batt
        pv_surplus_kwh = float((pv_ts - pv_to_load_base_ts).clip(lower=0.0).sum())
        pv_to_grid_final_kwh = max(pv_surplus_kwh - pv_to_batt_kwh, 0.0)
    
        results["battery_proposed_by_batiment"][bat_id] = {
            "batiment_id": bat_id,
            "batiment_nom": bat_nom,
            "capacity_kwh": cap_final,
            "pack_capacities_kwh": pack,
            "p_target_kw": p_target_kw,
            "hours_target": hours_target,
            "eta_global": eta_global,
            "totals_before": {
                "demand_kwh": float(load_ts.sum()),
                "pv_prod_kwh": float(pv_ts.sum()),
                "pv_to_load_kwh": float(pv_to_load_base_ts.sum()),
                "grid_to_load_kwh": float(grid_to_load_base_ts.sum()),
                "pv_to_grid_kwh": float(pv_to_grid_base_ts.sum()),
            },
            "totals_after": {
                "demand_kwh": float(load_ts.sum()),
                "pv_prod_kwh": float(pv_ts.sum()),
                "pv_to_load_kwh": pv_to_load_final_kwh,
                "pv_to_batt_kwh": pv_to_batt_kwh,
                "batt_to_load_kwh": batt_to_load_kwh,
                "grid_to_load_kwh": float(grid_to_load_final_kwh),
                "pv_to_grid_kwh": float(pv_to_grid_final_kwh),
                "losses_kwh": losses_kwh,
            },
        }
    
        # ----------------------------------------------------------
        # Timeseries bâtiment pour courbes "before/after" (batterie proposée)
        # IMPORTANT : ne pas écraser measured déjà présent
        # ----------------------------------------------------------
        results.setdefault("timeseries_by_batiment", {})
        results["timeseries_by_batiment"].setdefault(str(bat_id), {})
        ts_b = results["timeseries_by_batiment"][str(bat_id)]
        
        # métadonnées
        ts_b.setdefault("meta", {})
        ts_b["meta"].update({
            "batiment_id": bat_id,
            "batiment_nom": bat_nom,
            "dt_h_med": float(infer_dt_hours_from_index(common_index)),
        })
        
        # index commun pour les plots before/after
        ts_b["index"] = [str(x) for x in common_index]
        
        # BEFORE = monde réel (measured / reconstruit dans PATCH 2 plus haut)
        ts_b["before"] = {
            "load_kWh": load_ts.values.tolist(),
            "pv_prod_kWh": pv_ts.values.tolist(),
            "pv_to_load_kWh": pv_to_load_base_ts.values.tolist(),
            "grid_to_load_kWh": grid_to_load_base_ts.values.tolist(),
            "pv_to_grid_kWh": pv_to_grid_base_ts.values.tolist(),
        }
        
        # AFTER = séries issues du dispatch batterie (sb["profiles"])
        try:
            prof = sb.get("profiles") or {}
        
            pv_to_batt_list = prof.get("pv_to_batt")          # list
            batt_to_load_list = prof.get("batt_to_load")      # list
            soc_list = prof.get("soc_kWh")                    # list
            losses_list = prof.get("losses")                  # list
        
            def _ensure_len(x):
                if not isinstance(x, list):
                    return None
                return x if len(x) == len(common_index) else None
        
            ts_b.setdefault("after", {})
            ts_b["after"].update({
                "pv_to_batt_kWh": _ensure_len(pv_to_batt_list),
                "batt_to_load_kWh": _ensure_len(batt_to_load_list),
                "soc_kWh": _ensure_len(soc_list),
                "losses_kWh": _ensure_len(losses_list),
            })
        
        except Exception:
            # si ton module ne renvoie pas les séries, on ne plante pas ici
            pass


    
        # ----------------------------------------------------------
        # Économie batterie proposée (cashflow)
        # ----------------------------------------------------------
        price_buy, price_sell, horizon_years = _get_price_buy_sell_and_horizon(project)
    
        capex_spec = techc.get("capex_chf_per_kwh")
        opex_annual = techc.get("opex_chf_per_year")
        lifetime_years = techc.get("lifetime_years")
    
        if capex_spec is None:
            raise ValueError(f"Batterie proposée: capex_chf_per_kwh manquant ({bat_nom}).")
        if opex_annual is None:
            raise ValueError(f"Batterie proposée: opex_chf_per_year manquant ({bat_nom}).")
        if lifetime_years is None or float(lifetime_years) <= 0:
            raise ValueError(f"Batterie proposée: lifetime_years manquant/invalide ({bat_nom}).")
    
        capex_total = float(capex_spec) * float(cap_final)
        replacement_cost_factor = 1.0  # TODO dégradation/learning
    
        econ_batt = compute_battery_cashflow_series(
            capex_total_chf=float(capex_total),
            opex_annual_chf=float(opex_annual),
            pv_to_batt_kwh=float(pv_to_batt_kwh),
            grid_to_batt_kwh=0.0,
            batt_to_load_kwh=float(batt_to_load_kwh),
            batt_to_grid_kwh=0.0,
            price_buy_chf_kwh=float(price_buy),
            price_sell_chf_kwh=float(price_sell),
            horizon_years=int(horizon_years),
            lifetime_years=float(lifetime_years),
            replacement_cost_factor=float(replacement_cost_factor),
        )
    
        results.setdefault("storage_economics_by_batiment", {})
        results["storage_economics_by_batiment"].setdefault(bat_id, [])
        results["storage_economics_by_batiment"][bat_id].append({
            "batiment_id": bat_id,
            "batiment_nom": bat_nom,
            "mode": "proposed",
            "techno": batt_cfg.get("techno") or "Battery (proposed)",
            "engine": "bat_li_ion",
            "capacity_kwh": cap_final,
            "economics": econ_batt,
        })

    
    # ==========================================================
    # PV PROPOSÉ (toiture) – par bâtiment
    #   - utilise params (⚙️) en priorité
    #   - fallback legacy meta.electricity / meta.pv_defaults
    # ==========================================================
    
    for bi, bat in enumerate(batiments):
        # IMPORTANT: on force une clé cohérente avec results["flows"]["batiments"][bat_id]
        bat_id = bat.get("id") or bat.get("batiment_id") or bi
        bat_nom = bat.get("nom") or f"Bâtiment {bi+1}"
    
        pv_flag = bat.get("pv_exists")
    
        # On calcule le PV proposé si :
        # - PV inexistant (False) -> ok
        # - PV partiel ("partial") -> ok
        # On ne calcule pas si PV déjà plein (True)
        if pv_flag not in (False, "partial"):
            continue
    
        roofs = bat.get("pv_roofs") or []
        if not roofs:
            continue
    
        eligible = []
        monthly_total = None
        p_tot_kw = 0.0
        s_usable_tot = 0.0
    
        for r in roofs:
            ori = r.get("orientation_deg") if r.get("orientation_deg") is not None else r.get("orientation_code")
            tilt = r.get("inclinaison_deg")
            s_use = float(
                r.get("surface_usable_m2")
                or r.get("surface utilisiable")
                or r.get("surface_utilisable_m2")
                or 0.0
            )
    
            # --- normalisation tilt / orientation ---
            # Toit plat: si tilt manquant -> 0 ; et orientation devient "Sud" (0) par défaut
            tilt = 0.0 if tilt in (None, "") else float(tilt)
    
            # Orientation: si manquante -> 0
            if ori in (None, ""):
                ori = 0.0
            else:
                ori = float(ori)
    
            # 👉 Règle spéciale toit plat:
            # si inclinaison = 0 et orientation est "bidon" (souvent 180), on force Sud (=0)
            # (tu peux élargir à 180/-180/360 si tes données le font)
            if abs(tilt) < 1e-9 and (abs(ori) == 180.0 or abs(ori) == 360.0):
                ori = 0.0
    
            # --- contrôles de base ---
            if s_use <= 0:
                continue
    
            # filtre orientation (ta règle ±120°) -> on garde les toits plats car ori est maintenant 0
            if abs(ori) > 120:
                continue
    
            # puissance installable
            p_kw = s_use * (float(p_module_kw) / float(area_module_m2))
            if p_kw <= 0:
                continue
    
            serie_kwh_per_kw = compute_pv_monthly_energy_per_kw(
                station_name=station_name,
                orientation_deg=float(ori),
                tilt_deg=float(tilt),
                p_module_kw=float(p_module_kw),
                area_module_m2=float(area_module_m2),
                panel_efficiency=float(panel_eff),
            )
    
            serie_kwh = serie_kwh_per_kw * p_kw
    
            if monthly_total is None:
                monthly_total = serie_kwh.copy()
            else:
                monthly_total = monthly_total.add(serie_kwh, fill_value=0.0)
    
            p_tot_kw += p_kw
            s_usable_tot += s_use
            eligible.append({**r, "p_kw": p_kw})
    
        if monthly_total is None or p_tot_kw <= 0:
            continue
    
        annual_kwh = float(monthly_total.sum())
    
        # -----------------------------
        # Paramètres scénario (⚙️) - source unique: project["params"]
        # -----------------------------
        # -----------------------------
        # Autoconsommation PV proposé (source : bâtiment)
        # -----------------------------
        pv_prop_cfg = bat.get("pv_proposed_config") or {}

        if "default_selfc_pct" in pv_prop_cfg and pv_prop_cfg["default_selfc_pct"] not in (None, ""):
            selfc_pct = float(pv_prop_cfg["default_selfc_pct"])
        else:
            raise ValueError(
                "PV proposé: autoconsommation manquante. "
                "Attendu: bat.pv_proposed_config.default_selfc_pct "
                "(configuré dans Phase 1 > Bâtiments / Ouvrages)."
            )
        
        selfc_pct = min(max(selfc_pct, 0.0), 1.0)

        
        selfc_kwh = annual_kwh * selfc_pct
        inj_kwh = max(annual_kwh - selfc_kwh, 0.0)
        
        pv_lifetime_years = _get_param_numeric(
        pv_defaults,
        ["lifetime_years", "Lifetime (years)", "Durée de vie", "Duree de vie", "Lifetime"],
        default=0.0
        )
        if pv_lifetime_years <= 0:
            raise ValueError(
                "PV proposé: durée de vie manquante dans la techno PV (Excel). "
                "Attendu une clé type 'lifetime_years' / 'Durée de vie'."
            )
        
        # PV replacement cost factor (TEMP)
        # TODO: remplacer ce 1.0 par un paramètre techno/éco (CAPEX replacement vs initial, onduleur séparé, inflation, learning rate, etc.)
        pv_replacement_factor = 1.0


        
        # -----------------------------
        # Économie techno PV (Capex/Opex/Lifetime depuis pv_defaults techno)
        # -----------------------------
        econ = compute_pv_economics(
            annual_prod_kwh=annual_kwh,
            installed_kw=p_tot_kw,
            tech_params=pv_defaults,  # techno PV (Excel->JSON), PAS meta
        )
        
        # -----------------------------
        # Cashflow : économies réseau + rétribution injection (prix/horizon depuis ⚙️)
        # -----------------------------
        price_buy, price_sell, horizon_years = _get_price_buy_sell_and_horizon(project)
        
        capex_total = float(econ.get("capex_total_CHF", 0.0) or 0.0)
        opex_annual = float(econ.get("opex_annual_CHF", 0.0) or 0.0)

    
        revenue_selfc = float(selfc_kwh) * price_buy
        revenue_inj = float(inj_kwh) * price_sell
        annual_benefit = revenue_selfc + revenue_inj - opex_annual
    
        # -----------------------------
        # RU (constante modèle : année d'application)
        # -----------------------------
        ru_year = 1  # Constante modèle (règle RU). À documenter.
        
        # Montant RU manuel (si tu veux garder une override)
        meta = project.get("meta") or {}
        # RU manuel : support meta["ru_amount_chf"] OU legacy meta["pv_defaults"]["ru_amount_chf"]
        ru_amount_manual = meta.get("ru_amount_chf", None)
        if ru_amount_manual in (None, ""):
            pv_meta_legacy = meta.get("pv_defaults") or {}
            ru_amount_manual = pv_meta_legacy.get("ru_amount_chf", None)
        
        ru_amount_manual = float(ru_amount_manual) if ru_amount_manual not in (None, "") else None


        if ru_amount_manual is not None and ru_amount_manual != 0.0:
            ru_amount = ru_amount_manual
            ru_details = {"source": "manual"}
        else:
            tilt_weighted_sum = 0.0
            p_sum = 0.0
            for r in eligible:
                tilt = r.get("inclinaison_deg")
                p_kw = r.get("p_kw", 0.0)
                if tilt is None or p_kw is None:
                    continue
                try:
                    tilt = float(tilt)
                    p_kw = float(p_kw)
                except Exception:
                    continue
                if p_kw <= 0:
                    continue
                tilt_weighted_sum += tilt * p_kw
                p_sum += p_kw
    
            inclination_for_ru = (tilt_weighted_sum / p_sum) if p_sum > 0 else None
    
            altitude_for_ru = meta.get("altitude_m", None)
            parking_for_ru = bool(meta.get("pv_parking", False))
    
            ru_out = compute_ru_pv_added_weighted(
                p_kw=float(p_tot_kw),
                inclination_deg=inclination_for_ru,
                altitude_m=altitude_for_ru,
                parking=parking_for_ru,
            )
            ru_amount = float(ru_out.get("ru_total_chf", 0.0) or 0.0)
            ru_details = ru_out
    
        # -----------------------------
        # Séries cashflow + indicateurs
        # -----------------------------
        cashflows = [-capex_total] + [annual_benefit] * horizon_years

        # Remplacements (CAPEX) si horizon > lifetime
        cashflows = _apply_replacements_to_cashflows(
            cashflows=cashflows,
            capex_total_chf=capex_total,
            lifetime_years=pv_lifetime_years,
            horizon_years=horizon_years,
            replacement_cost_factor=pv_replacement_factor,
        )
        
        # RU à l'année ru_year (année 0 = investissement)
        if ru_amount != 0.0 and 0 <= ru_year <= horizon_years:
            cashflows[ru_year] += ru_amount

    
        cashflow_cum = []
        s = 0.0
        for cf in cashflows:
            s += float(cf)
            cashflow_cum.append(s)
    
        payback_year = next((y for y, cum in enumerate(cashflow_cum) if cum >= 0), None)
        benefit_25 = cashflow_cum[min(25, horizon_years)]
    
        econ["years"] = list(range(0, horizon_years + 1))
        econ["cashflows_CHF"] = cashflows
        econ["cashflow_cum_CHF"] = cashflow_cum
        econ["ru_amount_CHF"] = ru_amount
        econ["ru_year"] = ru_year
        econ["ru_details"] = ru_details
        econ["payback_year"] = payback_year
        econ["benefit_25_CHF"] = benefit_25
        econ["pv_lifetime_years"] = float(pv_lifetime_years)
        econ["pv_lifetime_years"] = float(pv_lifetime_years)
        econ["pv_replacement_cost_factor"] = float(pv_replacement_factor)
        econ["pv_replacement_cost_factor_source"] = "TEMP_HARDCODED_1.0_TODO"


        econ["capex_breakdown"] = {
            "capex_total_CHF": capex_total,
            "capex_spec_CHF_kW": float(econ.get("capex_spec_CHF_kW", 0.0) or 0.0),
            "installed_kw": float(p_tot_kw),
        }
    
        results["pv_proposed_by_batiment"][bat_id] = {
            "batiment_id": bat_id,
            "batiment_nom": bat_nom,
            "station_name": station_name,
            "eligible_roofs": eligible,
            "surface_usable_total_m2": s_usable_tot,
            "p_tot_kw": p_tot_kw,
            "monthly_kwh": [float(x) for x in monthly_total.tolist()],
            "annual_kwh": annual_kwh,
            "selfc_pct": selfc_pct,
            "selfc_kwh": selfc_kwh,
            "inj_kwh": inj_kwh,
            "economics": econ,
        }
    
    # Expose au projet (pour Phase 2 / show_variantes)
    project["pv_simulated"] = results.get("pv_proposed_by_batiment", {})
    
    # === Simuler la production PV si demandé ===
    params_simu = project.get("params") or project.get("simu_params") or {}
    if bool(params_simu.get("simulate_new_pv", False)):
        from core.pv_module import compute_simulated_rooftop_pv_per_building
        simulated_pv_results = compute_simulated_rooftop_pv_per_building(project)
        project["pv_simulated"] = simulated_pv_results

        
    # ==========================================================
    # BATTERY IMPACT — KPI COMPARAISON (sans vs avec stockage)
    # ==========================================================
    
    battery_impact_by_batiment: Dict[Any, Dict[str, float]] = {}
    
    base_all = results.get("internal", {}).get("energy_global_base", {})
    final_all = results.get("internal", {}).get("energy_global_final", {})
    
    for bat_id, base_records in base_all.items():
        final_records = final_all.get(bat_id, [])
        if not base_records or not final_records:
            continue
    
        # Agrégation simple (somme annuelle)
        def _sum(key: str, records: list) -> float:
            return float(sum(r.get(key, 0.0) for r in records))
    
        base_grid_import = _sum("grid_to_load_kWh", base_records)
        final_grid_import = _sum("grid_to_load_kWh", final_records)
    
        base_injection = _sum("pv_to_grid_kWh", base_records)
        final_injection = _sum("pv_to_grid_kWh", final_records)
    
        base_selfc = _sum("pv_auto_used_kWh", base_records)
        final_selfc = _sum("pv_auto_used_kWh", final_records)
    
        demand = _sum("demand_elec_kWh", final_records)
    
        # Deltas (positif = amélioration)
        delta_grid_import = final_grid_import - base_grid_import
        delta_injection = base_injection - final_injection
        delta_selfc = final_selfc - base_selfc
    
        # Autoconsommation [%]
        selfc_pct_base = (base_selfc / demand * 100.0) if demand > 0 else 0.0
        selfc_pct_final = (final_selfc / demand * 100.0) if demand > 0 else 0.0
        delta_selfc_pct = selfc_pct_final - selfc_pct_base
    
        battery_impact_by_batiment[bat_id] = {
            "grid_import_base_kWh": base_grid_import,
            "grid_import_final_kWh": final_grid_import,
            "delta_grid_import_kWh": delta_grid_import,
    
            "pv_injection_base_kWh": base_injection,
            "pv_injection_final_kWh": final_injection,
            "delta_pv_injection_kWh": delta_injection,
    
            "pv_selfc_base_kWh": base_selfc,
            "pv_selfc_final_kWh": final_selfc,
            "delta_pv_selfc_kWh": delta_selfc,
    
            "self_consumption_base_pct": selfc_pct_base,
            "self_consumption_final_pct": selfc_pct_final,
            "delta_self_consumption_pct": delta_selfc_pct,
        }
    
    # Exposé au projet (Phase 2)
    results["summaries"]["battery_impact_by_batiment"] = battery_impact_by_batiment


    return results

