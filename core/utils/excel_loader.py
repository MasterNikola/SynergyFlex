# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:52:14 2025

@author: vujic
"""

# core_utils/excel_loader.py

import pandas as pd
import streamlit as st
from typing import Dict, Optional

def handle_excel_upload(
    container: Dict,
    label: str,
    key_prefix: str,
    reuse_sources: Dict[str, Optional[Dict]] = {}
) -> Optional[Dict]:
    """
    Centralise la gestion de l'import Excel pour projet, bâtiment, ouvrage, producteur.
    
    :param container: dictionnaire de destination (ex: projet["meta"], batiment, etc.)
    :param label: texte pour l'utilisateur (ex: "du bâtiment", "de l’ouvrage")
    :param key_prefix: préfixe unique pour les widgets streamlit
    :param reuse_sources: sources disponibles pour réutilisation (projet, bâtiment, ouvrage)
    :return: dict contenant {"file_name": ..., "sheets": {...}} ou None
    """
    container["data_mode"] = container.get("data_mode", "Aucune")
    container["excel_data"] = container.get("excel_data", {})

    options = ["Aucune"]
    if reuse_sources.get("projet"):
        options.append("Réutiliser fichier du projet")
    if reuse_sources.get("bâtiment"):
        options.append("Réutiliser fichier du bâtiment")
    if reuse_sources.get("ouvrage"):
        options.append("Réutiliser fichier de l’ouvrage")
    options.append("Importer un fichier")

    container["data_mode"] = st.radio(
        f"📁 Source de données {label} :",
        options=options,
        index=options.index(container["data_mode"]) if container["data_mode"] in options else 0,
        key=f"{key_prefix}_data_mode"
    )

    if container["data_mode"] == "Importer un fichier":
        uploaded = st.file_uploader(f"Importer un fichier Excel/CSV {label}", type=["xlsx", "xls", "csv"], key=f"{key_prefix}_upload")
        if uploaded:
            try:
                if uploaded.name.lower().endswith((".xlsx", ".xls")):
                    xls = pd.ExcelFile(uploaded)
                    sheets = {name: xls.parse(name) for name in xls.sheet_names}
                else:
                    df = pd.read_csv(uploaded)
                    sheets = {"Données": df}
                container["excel_data"] = {
                    "file_name": uploaded.name,
                    "sheets": sheets
                }
                st.success(f"Fichier {label} chargé : {uploaded.name}")
            except Exception as e:
                st.error(f"Erreur de lecture du fichier : {e}")

    elif container["data_mode"] == "Réutiliser fichier du projet" and reuse_sources.get("projet"):
        container["excel_data"] = reuse_sources["projet"]

    elif container["data_mode"] == "Réutiliser fichier du bâtiment" and reuse_sources.get("bâtiment"):
        container["excel_data"] = reuse_sources["bâtiment"]

    elif container["data_mode"] == "Réutiliser fichier de l’ouvrage" and reuse_sources.get("ouvrage"):
        container["excel_data"] = reuse_sources["ouvrage"]

    # Aperçu si fichier présent
    if container["excel_data"].get("sheets"):
        sheets = container["excel_data"]["sheets"]
        sheet_names = list(sheets.keys())
        default_sheet = container.get("excel_sheet_name", sheet_names[0])

        selected_sheet = st.selectbox(
            f"Feuille à utiliser {label}",
            options=sheet_names,
            index=sheet_names.index(default_sheet) if default_sheet in sheet_names else 0,
            key=f"{key_prefix}_sheet_select"
        )

        container["excel_sheet_name"] = selected_sheet
        st.caption(f"Aperçu de la feuille **{selected_sheet}**")
        st.dataframe(sheets[selected_sheet].head())

        return container["excel_data"]
    
    return None


from typing import Dict, Union
from io import BytesIO



def load_excel_file(uploaded_file) -> Dict[str, pd.DataFrame]:
    """
    Charge un fichier Excel ou CSV depuis un Streamlit file_uploader
    et retourne un dictionnaire {nom_feuille: DataFrame}.
    """
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded_file)
            sheets = {name: xls.parse(name) for name in xls.sheet_names}
        elif file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            sheets = {"Données": df}
        else:
            raise ValueError("Format de fichier non supporté.")

        return sheets

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        raise
