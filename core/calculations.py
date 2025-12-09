# core/calculations.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List

import pandas as pd

from core.pv_module import (
    compute_pv_flow_block,
    compute_pv_monthly_energy_per_kw,   # ✅ pour le PV “théorique standard”
)
from core.nodes_ontology import (
    node_elec_grid,
    node_elec_load,
    node_heat_load,
    node_heat_prod,
    node_pv,
    # plus tard : node_battery, node_device, etc.
)
from core.economics import get_param_numeric



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


def _build_monthly_series_from_df(df: pd.DataFrame, energy_col: str) -> Optional[pd.Series]:
    """
    Essaie de construire une série mensuelle (12 mois) à partir d'un DataFrame
    et d'une colonne d'énergie (kWh).

    Retourne une Series indexée 1..12 (mois) ou None si impossible.
    """
    if energy_col not in df.columns:
        return None

    # Cas 1 : colonne 'Mois' déjà présente (1..12)
    if "Mois" in df.columns:
        mois = pd.to_numeric(df["Mois"], errors="coerce")
        s = pd.to_numeric(df[energy_col], errors="coerce")
        grouped = s.groupby(mois).sum()
        idx = range(1, 13)
        grouped = grouped.reindex(idx).fillna(0.0)
        grouped.index = list(idx)
        return grouped

    # Cas 2 : on a une vraie date -> on regroupe par mois
    for date_col in ["Date_periode", "Date", "date", "Datetime", "datetime"]:
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            s = pd.to_numeric(df[energy_col], errors="coerce")
            mask = dates.notna()
            if not mask.any():
                continue
            grouped = s[mask].groupby(dates[mask].dt.month).sum()
            idx = range(1, 13)
            grouped = grouped.reindex(idx).fillna(0.0)
            grouped.index = list(idx)
            return grouped

    # Sinon : on ne sait pas construire un profil mensuel proprement
    return None


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
        # On ajoutera une clé "economics_by_batiment" plus bas
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

    # Catégorie d'ouvrage (infos Excel) associée à l'index d'ouvrage
    def _get_cat_for_ouvrage(index_ouv: int) -> Dict[str, Any]:
        if 0 <= index_ouv < len(categories_ouvrages):
            return categories_ouvrages[index_ouv]
        return {}

    # ------------------------------------------------------------------
    # BOUCLE PRINCIPALE SUR LES OUVRAGES
    # ------------------------------------------------------------------
    for oi, ouv in enumerate(ouvrages):
        bat_id = ouv.get("batiment_id")
        bat_nom = _get_bat_name(bat_id)
        ouv_nom = ouv.get("nom", "Ouvrage")

        # Liste de blocs de flux pour ce bâtiment
        bat_flows = results["flows"]["batiments"].setdefault(bat_id, [])

        # ---- Accumulateurs locaux pour cet ouvrage ----
        local_demand_elec_kwh = 0.0
        local_demand_th_kwh = 0.0
        local_pv_prod_kwh = 0.0
        local_pv_auto_kwh = 0.0
        local_pv_inj_kwh = 0.0

        # ------------------ DEMANDE (à partir Excel) -------------------
        cat = _get_cat_for_ouvrage(oi)
        sheet_name = cat.get("sheet_name")
        time_col = cat.get("time_col")
        conso_elec_col = cat.get("conso_elec_col")
        conso_th_col = cat.get("conso_th_col")

        df_demand = None
        if sheet_name and sheet_name in excel_sheets:
            df_demand = excel_sheets[sheet_name]

        # 1) FLOW_BLOCK – DEMANDE ELECTRIQUE (conservation pour analyse fine)
        if df_demand is not None and conso_elec_col and conso_elec_col in df_demand.columns:
            serie_elec = pd.to_numeric(df_demand[conso_elec_col], errors="coerce")
            local_demand_elec_kwh = float(serie_elec.sum(skipna=True) or 0.0)

            if local_demand_elec_kwh > 0:
                grid_id = node_elec_grid()
                load_id = node_elec_load()

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
                        {
                            "id": grid_id,
                            "label": "Réseau électrique",
                            "group": "reseau",
                        },
                        {
                            "id": load_id,
                            "label": "Consommation électrique",
                            "group": "demande",
                        },
                    ],
                    "links": [
                        {
                            "source": grid_id,
                            "target": load_id,
                            "value": local_demand_elec_kwh,
                        },
                    ],
                    "totals": {
                        "demand_elec_kWh": local_demand_elec_kwh,
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
                    ["Puissance du module (kW)", "Puissance du module [kW]", "Puissance module [kW]", "Puissance module", "Puissance du module"]
                )
                area_module_m2 = _get_param_numeric(
                    tech_params,
                    ["surface du module (m2)", "Surface du module (m2)", "Surface du module", "Surface module (m2)"]
                )
                eta_mod_pct = _get_param_numeric(
                    tech_params,
                    ["Rendement (%)", "Rendement", "Rendement module (%)"]
                )
                
                installed_kw = float(prod.get("puissance_kw", 0.0) or 0.0)

                # ---- Profil mensuel MESURÉ (kWh/kW)
                measured_profile = None
                measured_annual = None
                if installed_kw > 0:
                    serie_mois = _build_monthly_series_from_df(df_prod, col_prod)
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


        # ------------------------------------------------------------------
        # 3) BLOC AGGRÉGÉ – BILAN ENERGETIQUE GLOBAL (élec + thermique)
        # ------------------------------------------------------------------
        if (
            local_demand_elec_kwh > 0
            or local_pv_prod_kwh > 0
            or local_demand_th_kwh > 0
        ):
            # Part auto-consommée limitée par la demande élec
            pv_auto_used = min(local_pv_auto_kwh, local_demand_elec_kwh)
            # Part de la demande élec couverte par le réseau
            grid_to_load = max(local_demand_elec_kwh - pv_auto_used, 0.0)
            # Injection réseau (on utilise la valeur fournie par le module PV)
            pv_to_grid = max(local_pv_inj_kwh, 0.0)

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

    return results

