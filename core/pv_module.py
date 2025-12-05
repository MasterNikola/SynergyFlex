# core/pv_module.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st


def render_pv_extra_params(
    df: pd.DataFrame,
    existing_config: Optional[Dict[str, Any]] = None,
    widget_key_prefix: str = "",
) -> Dict[str, Any]:
    """
    UI simple pour les paramètres PV supplémentaires :
    - colonne d'autoconsommation SI elle existe
    - sinon, taux d'autoconsommation estimé [%]

    Retourne un dict du type :
    {
        "col_pv_auto": "NomColonne" ou None,
        "default_selfc_pct": 20 ou autre,
    }
    """
    st.markdown("##### Paramètres PV supplémentaires")

    cfg = existing_config.copy() if existing_config else {}

    col_pv_auto = cfg.get("col_pv_auto")
    default_selfc_pct = cfg.get("default_selfc_pct", 20)

    # 1) Checkbox : est-ce qu'il a une colonne d'autocons ?
    has_pv_auto_default = col_pv_auto is not None
    has_pv_auto = st.checkbox(
        "Avez-vous une colonne d'autoconsommation PV ?",
        value=has_pv_auto_default,
        key=f"{widget_key_prefix}_has_pv_auto",
    )

    if has_pv_auto:
        cols_num = df.select_dtypes(include=["number"]).columns.tolist()

        if not cols_num:
            st.warning("Aucune colonne numérique trouvée dans le fichier.")
            col_pv_auto = None
            default_selfc_pct = None
        else:
            if col_pv_auto not in cols_num:
                col_pv_auto = cols_num[0]

            col_pv_auto = st.selectbox(
                "Sélectionnez la colonne d'autoconsommation PV :",
                options=cols_num,
                index=cols_num.index(col_pv_auto) if col_pv_auto in cols_num else 0,
                key=f"{widget_key_prefix}_pv_auto_col",
            )
            # si colonne réelle → plus besoin d'un % par défaut
            default_selfc_pct = None
    else:
        col_pv_auto = None
        # 2) Slider pour le taux d'autoconsommation estimé
        default_selfc_pct = st.slider(
            "Taux d'autoconsommation PV estimé [%]",
            min_value=0,
            max_value=100,
            value=int(default_selfc_pct),
            step=1,
            key=f"{widget_key_prefix}_pv_selfc_pct",
        )

    return {
        "col_pv_auto": col_pv_auto,
        "default_selfc_pct": default_selfc_pct,
    }


def compute_pv_flow_block(
    df: pd.DataFrame,
    prod_profile_col: str,
    prod: Dict[str, Any],
    contexte: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calcule un bloc de flux PV (flow_block) pour un producteur donné.

    - df : DataFrame (profil temporel PV de ce producteur)
    - prod_profile_col : nom de la colonne de prod PV (choisie en Phase 1)
    - prod : dict du producteur (doit contenir 'pv_flux_config')
    - contexte : infos sur bâtiment / ouvrage / producteur (optionnel)

    Retourne un dict du type :
    {
      "name": "PV_..._...",
      "type": "pv",
      "meta": {...},
      "nodes": [...],
      "links": [...],
      "totals": {...},
    }
    """

    if df is None or df.empty:
        raise ValueError("Impossible de calculer un bloc PV : df vide.")

    if prod_profile_col not in df.columns:
        raise KeyError(f"Colonne de production PV '{prod_profile_col}' introuvable dans le DataFrame.")

    pv_cfg = prod.get("pv_flux_config", {}) or {}
    col_pv_auto = pv_cfg.get("col_pv_auto")
    default_selfc_pct = pv_cfg.get("default_selfc_pct", 20)

    # Production totale PV
    prod_pv = float(df[prod_profile_col].astype(float).sum())

    if prod_pv <= 0:
        raise ValueError("Production PV totale nulle ou négative, bloc non pertinent.")

    # Autoconsommation
    if col_pv_auto and col_pv_auto in df.columns:
        pv_auto = float(df[col_pv_auto].astype(float).sum())
    else:
        pv_auto = prod_pv * (float(default_selfc_pct) / 100.0)

    # Injection (pour l'instant : le reste)
    pv_inj = max(prod_pv - pv_auto, 0.0)

    meta = (contexte or {}).copy()
    taux_autocons_pct = float(pv_auto / prod_pv * 100) if prod_pv > 0 else 0.0

    flow_block = {
        "name": meta.get("name", "PV_block"),
        "type": "pv",
        "meta": meta,
        "nodes": [
            {"id": "PV",       "label": meta.get("pv_label", "PV"),                   "group": "prod_elec"},
            {"id": "ELEC_USE", "label": meta.get("elec_use_label", "Usages élec."),   "group": "use_elec"},
            {"id": "GRID_INJ", "label": meta.get("grid_inj_label", "Injection réseau"), "group": "reseau"},
        ],
        "links": [
            {"source": "PV", "target": "ELEC_USE", "value": pv_auto},
            {"source": "PV", "target": "GRID_INJ", "value": pv_inj},
        ],
        "totals": {
            "pv_prod_kWh": prod_pv,
            "pv_auto_kWh": pv_auto,
            "pv_inj_kWh": pv_inj,
            "taux_autocons_pct": taux_autocons_pct,
        },
    }

    return flow_block
