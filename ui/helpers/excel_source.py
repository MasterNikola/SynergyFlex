# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:19:50 2026

@author: infor
"""

# ui/helpers/excel_source.py
from typing import Dict, Any, Optional, Tuple
import streamlit as st

def choose_excel_source(
    *,
    label: str,
    key_prefix: str,
    state: Dict[str, Any],
    building_files: Dict[str, Any],
    ouvrage_files: Dict[str, Any],
    project_files: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """
    UI standard pour choisir une source Excel (upload / reuse).
    Retourne: (selected_excel_sheets_dict, source_label)
    Modifie 'state' (ex: state["source_data_mode"]) de manière explicite.
    """
    def _unwrap_sheets(x: Any) -> Dict[str, Any]:
        """
        Normalise vers: {sheet_name: DataFrame}
        Accepte:
          - {sheet: df}
          - {"sheets": {sheet: df}}
        """
        if not x:
            return {}
        if isinstance(x, dict) and "sheets" in x and isinstance(x["sheets"], dict):
            return x["sheets"]
        if isinstance(x, dict):
            return x
        return {}

    options = [
        "Aucune",
        "Importer un fichier",
        "Réutiliser un fichier du bâtiment",
        "Réutiliser un fichier de l’ouvrage",
        "Réutiliser un fichier du projet",
    ]

    state["source_data_mode"] = st.radio(
        label,
        options=options,
        index=options.index(state.get("source_data_mode", "Aucune")) if state.get("source_data_mode") in options else 0,
        key=f"{key_prefix}_source_mode",
    )

    source_label = ""
    sheets: Dict[str, Any] = {}

    if state["source_data_mode"] == "Aucune":
        return sheets, "Aucune"

    if state["source_data_mode"] == "Importer un fichier":
        up = st.file_uploader(
            "Importer un fichier Excel",
            type=["xlsx", "xlsm", "xls"],
            key=f"{key_prefix}_uploader",
        )
        if up is None:
            return {}, "Upload en attente"
        # Ici tu ne lis pas le fichier (parce que tu as déjà ton loader ailleurs)
        # On stocke juste un flag; le loader existant remplira st.session_state["excel_sheets"][...]
        state["uploaded_file_present"] = True
        source_label = "Upload"
        # Les sheets réelles doivent venir de ton pipeline existant (excel_loader)
        # => on retourne {} et ton code appelant doit récupérer les sheets déjà chargées
        return {}, source_label

    if state["source_data_mode"] == "Réutiliser un fichier du bâtiment":
        return _unwrap_sheets(building_files), "Bâtiment"


    if state["source_data_mode"] == "Réutiliser un fichier de l’ouvrage":
        return _unwrap_sheets(ouvrage_files), "Ouvrage"
    
    if state["source_data_mode"] == "Réutiliser un fichier du projet":
        return _unwrap_sheets(project_files), "Projet"


    raise ValueError(f"Mode source_data_mode inattendu: {state['source_data_mode']}")
