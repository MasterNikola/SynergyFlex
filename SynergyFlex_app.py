# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:22:05 2025

@author: vujic
"""
import streamlit as st
from ui.phase1_import import render_phase1
from ui.phase2_dashboard import render_phase2
from core.io_project import export_project_dict, import_project_from_json


def init_session_state():
    ss = st.session_state
    ss.setdefault("project", None)
    ss.setdefault("validated", False)


def sidebar_menu():
    st.sidebar.header("Gestion du projet")

    if st.sidebar.button("🆕 Nouveau projet"):
        st.session_state["project"] = None
        st.session_state["validated"] = False
        st.experimental_rerun()

    uploaded = st.sidebar.file_uploader("📂 Charger un projet (.json)", type=["json"])
    if uploaded:
        st.session_state["project"] = import_project_from_json(uploaded)
        st.session_state["validated"] = True
        st.success("Projet chargé !")
        st.experimental_rerun()

    if st.session_state["project"] is not None:
        data = export_project_dict(st.session_state["project"])
        st.sidebar.download_button(
            "💾 Exporter le projet",
            data=data,
            file_name="projet_synergyflex.json",
            mime="application/json",
        )


def render_header():
    """Affiche le logo SynergyFlex centré en haut de la page."""
    # 3 colonnes pour centrer proprement
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.image("logo_synergyflex.png", use_column_width=True)


def main():
    # Config générale de la page (titre + icône d’onglet)
    st.set_page_config(
        page_title="SynergyFlex",
        layout="wide",
        page_icon="logo_synergyflex.png",
    )

    init_session_state()
    sidebar_menu()

    # --- HEADER COMMUN (logo) ---
    render_header()
    st.write("")  # petit espace

    # --- ROUTAGE PHASE 1 / PHASE 2 ---
    if not st.session_state["validated"]:
        st.subheader("Phase 1 – Préparation des données")
        render_phase1()
    else:
        st.subheader("Phase 2 – Analyse et visualisation")
        render_phase2()


if __name__ == "__main__":
    main()
