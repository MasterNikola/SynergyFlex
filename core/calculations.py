# core/calculations.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional

import pandas as pd
from core.pv_module import compute_pv_flow_block


# 🔀 Dispatch des fonctions de calcul par ID 'engine'
PRODUCER_COMPUTE_DISPATCH = {
    "pv": compute_pv_flow_block,
    # "pac_ae": compute_pac_flow_block,
    # "bat_li_ion": compute_battery_flow_block,
}


def run_calculations(
    project: Dict[str, Any],
    excel_sheets: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Any]:
    """
    Orchestrateur principal des calculs SynergyFlex.

    - project : dict normalisé issu de la Phase 1
      (doit contenir batiments, ouvrages, producteurs, etc.)
    - excel_sheets : dictionnaire {sheet_name: DataFrame}
      permettant de récupérer les profils temporels.

    Retourne un dict `results` structuré avec :
      - flows["batiments"][batiment_id] = [flow_block1, flow_block2, ...]
      - summaries[...] : indicateurs globaux
    """

    excel_sheets = excel_sheets or {}

    batiments = project.get("batiments", [])
    ouvrages = project.get("ouvrages", [])

    results: Dict[str, Any] = {
        "flows": {
            "batiments": {},   # batiment_id -> [flow_blocks...]
        },
        "summaries": {},
    }

    # -----------------------
    #  Utilitaire noms de bat
    # -----------------------
    def _get_bat_name(bat_id: Any) -> str:
        # Cas index entier (compatible avec _build_flat_project_for_calculations)
        if isinstance(bat_id, int) and 0 <= bat_id < len(batiments):
            return batiments[bat_id].get("nom", f"Bâtiment {bat_id+1}")
        # Cas ID string (si plus tard tu ajoutes des IDs explicites)
        if isinstance(bat_id, str):
            for b in batiments:
                if b.get("id") == bat_id:
                    return b.get("nom", bat_id)
        return str(bat_id)

    # =======================
    #   BOUCLE PRINCIPALE
    # =======================

    for ouvrage in ouvrages:
        bat_id = ouvrage.get("batiment_id")
        ouv_nom = ouvrage.get("nom", "Ouvrage ?")
        producteurs = ouvrage.get("producteurs", [])

        for prod in producteurs:
            # -----------------------
            #  1) Détermination engine
            # -----------------------
            engine = prod.get("engine")  # ex. "pv", "pac_ae", ...
            techno_lower = (prod.get("techno") or "").lower()

            # Compat ascendante : si engine absent mais techno PV -> force "pv"
            if not engine:
                if "photovolta" in techno_lower or "pv" in techno_lower:
                    engine = "pv"
                    prod["engine"] = engine

            compute_func = PRODUCER_COMPUTE_DISPATCH.get(engine)
            if not compute_func:
                # Aucun module de calcul associé à cet engine -> on ignore
                continue

            # -----------------------
            #  2) Récupération des données Excel
            # -----------------------
            sheet_name = prod.get("prod_profile_sheet")
            col_prod = prod.get("prod_profile_col")

            if not sheet_name or not col_prod:
                # Données incomplètes -> on saute ce producteur
                continue

            df = excel_sheets.get(sheet_name)
            if df is None or df.empty:
                # Feuille Excel manquante ou vide -> on saute
                continue

            # -----------------------
            #  3) Construction du contexte
            # -----------------------
            bat_nom = _get_bat_name(bat_id)

            contexte = {
                "batiment_id": bat_id,
                "batiment_nom": bat_nom,
                "ouvrage_nom": ouv_nom,
                "producteur_engine": engine,
                "producteur_techno": prod.get("techno"),
                "producteur_type_general": prod.get("type_general"),
                # Nom lisible pour le flow_block (utilisé dans le Sankey Phase 2)
                "name": f"{(engine or 'bloc').upper()}_{bat_nom}_{ouv_nom}",
            }

            # -----------------------
            #  4) Appel du module producteur
            # -----------------------
            try:
                flow_block = compute_func(
                    df=df,
                    prod_profile_col=col_prod,
                    prod=prod,
                    contexte=contexte,
                )
            except Exception:
                # TODO : tu pourras logger ici si tu veux analyser les erreurs
                continue

            # -----------------------
            #  5) Stockage du flow_block
            # -----------------------
            results["flows"]["batiments"].setdefault(bat_id, []).append(flow_block)

    # =======================
    #   RÉSUMÉS GLOBAUX
    # =======================

    results["summaries"]["nb_batiments"] = len(batiments)
    results["summaries"]["nb_ouvrages"] = len(ouvrages)

    # Exemple de synthèse PV globale à partir des flow_blocks
    total_pv = 0.0
    total_auto = 0.0
    total_inj = 0.0

    for _bat_id, blocks in results["flows"]["batiments"].items():
        for block in blocks:
            if block.get("type") == "pv":
                t = block.get("totals", {})
                total_pv += t.get("pv_prod_kWh", 0.0)
                total_auto += t.get("pv_auto_kWh", 0.0)
                total_inj += t.get("pv_inj_kWh", 0.0)

    results["summaries"]["pv_prod_kWh_total"] = total_pv
    results["summaries"]["pv_auto_kWh_total"] = total_auto
    results["summaries"]["pv_inj_kWh_total"] = total_inj

    return results
