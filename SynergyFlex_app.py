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
        st.rerun()


    uploaded = st.sidebar.file_uploader("📂 Charger un projet (.json)", type=["json"])
    if uploaded:
        st.session_state["project"] = import_project_from_json(uploaded)
        st.session_state["validated"] = True
        st.success("Projet chargé !")
        st.rerun()


    if st.session_state["project"] is not None:
        data_light = export_project_dict(st.session_state["project"], include_results=False)
        st.sidebar.download_button(
            "💾 Exporter le projet",
            data=data_light,
            file_name="projet_synergyflex.json",
            mime="application/json",
        )
    
        data_full = export_project_dict(st.session_state["project"], include_results=True, include_timeseries=False)

        if data_full is None:
            st.sidebar.error("Export debug: export_project_dict a renvoyé None (voir io_project.py).")
        else:
            st.sidebar.download_button(
                "🐞 Export complet (debug résumé)",
                data=data_full,
                file_name="synergyflex_results_summary.json",
                mime="application/json",
            )

        # Export timeseries séparé
        ts_store = ((st.session_state["project"].get("results") or {}).get("timeseries_store") or {})
        if ts_store:
            import pandas as pd
            rows = []
            bats = (ts_store.get("batiments") or {})
            for bat_id, bd in bats.items():
                ouvs = (bd.get("ouvrages") or {})
                for oi, rec in ouvs.items():
                    idx = rec.get("index") or []
                    # colonnes possibles
                    cols = {k: v for k, v in rec.items() if isinstance(v, list) and k != "index"}
                    for i, t in enumerate(idx):
                        row = {
                            "batiment_id": bat_id,
                            "ouvrage_index": oi,
                            "time": t,
                        }
                        for k, v in cols.items():
                            row[k] = v[i] if i < len(v) else None
                        rows.append(row)

            df_ts = pd.DataFrame(rows)
            csv_bytes = df_ts.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button(
                "📈 Export timeseries (CSV)",
                data=csv_bytes,
                file_name="synergyflex_timeseries.csv",
                mime="text/csv",
            )


def render_header():
    """Affiche le logo SynergyFlex centré en haut de la page."""
    # 3 colonnes pour centrer proprement
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.image("logo_synergyflex.png", use_container_width=True)


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
