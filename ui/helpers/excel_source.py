# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:19:50 2026

@author: infor
"""

# ui/helpers/excel_source.py
from typing import Dict, Any, Tuple
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
    Standard UI to select an Excel data source (upload / reuse).
    Returns: (selected_excel_sheets_dict, source_label)
    Explicitly updates 'state' (e.g. state["source_data_mode"]).
    """

    def _unwrap_sheets(x: Any) -> Dict[str, Any]:
        """
        Normalize to: {sheet_name: DataFrame}
        Accepts:
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
        "None",
        "Upload a file",
        "Reuse building file",
        "Reuse unit file",
        "Reuse project file",
    ]

    state["source_data_mode"] = st.radio(
        label,
        options=options,
        index=options.index(state.get("source_data_mode", "None"))
        if state.get("source_data_mode") in options
        else 0,
        key=f"{key_prefix}_source_mode",
    )

    source_label = ""
    sheets: Dict[str, Any] = {}

    if state["source_data_mode"] == "None":
        return sheets, "None"

    if state["source_data_mode"] == "Upload a file":
        up = st.file_uploader(
            "Upload Excel file",
            type=["xlsx", "xlsm", "xls"],
            key=f"{key_prefix}_uploader",
        )
        if up is None:
            return {}, "Upload pending"

        # File reading is handled elsewhere (existing loader)
        state["uploaded_file_present"] = True
        source_label = "Upload"

        # Actual sheets must come from the existing pipeline (excel_loader)
        return {}, source_label

    if state["source_data_mode"] == "Reuse building file":
        return _unwrap_sheets(building_files), "Building"

    if state["source_data_mode"] == "Reuse unit file":
        return _unwrap_sheets(ouvrage_files), "Unit"

    if state["source_data_mode"] == "Reuse project file":
        return _unwrap_sheets(project_files), "Project"

    raise ValueError(f"Unexpected source_data_mode: {state['source_data_mode']}")
