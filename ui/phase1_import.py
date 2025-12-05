# ui/phase1_import.py
# -*- coding: utf-8 -*-
import streamlit as st
from typing import Dict, Any, List
from core.calculations import run_calculations
from core.utils.excel_loader import load_excel_file
import pandas as pd
import plotly.graph_objects as go
from core.technologies import (
    get_available_producer_types,
    get_label_for_producer,
    load_producer_technologies,
    infer_engine_from_type_and_techno,
)
from core.pv_module import render_pv_extra_params
from core.climate_data import load_climate_monthly
import plotly.express as px  




def _extract_ouvrages_with_data(data: dict) -> list:
    categories = []
    for bi, bat in enumerate(data.get("batiments", [])):
        for oi, ouv in enumerate(bat.get("ouvrages", [])):
            cat = {
                "nom": ouv.get("nom"),
                "type": ouv.get("type_usage"),
                "sheet_name": ouv.get("excel_sheet_name"),
                "sre": ouv.get("sre_m2"),
                "surface_enveloppe": ouv.get("surface_enveloppe_m2"),
                "time_col": ouv.get("time_col"),
                "conso_elec_col": ouv.get("conso_elec_col"),
                "conso_th_col": ouv.get("conso_th_col"),
            }
            categories.append(cat)
    return categories



def _default_phase1_data() -> Dict[str, Any]:
    return {
        "meta": {
            "nom_projet": "",
            "type_projet": "",
            "commune": "",
            "zone_climatique": "",
        },
        "batiments": [],
        "simu_params": {
            "pas_temps": "Horaire",
            "duree_simulation_ans": 5,
            "horizon_analyse_ans": 10,
            "methode_optimisation": "Aucune (simulation simple)",
            "critere_prioritaire": "Équilibré",
        },
    }


def init_phase1_state():
    ss = st.session_state
    ss.setdefault("phase1_data", _default_phase1_data())
    ss.setdefault("phase1_page", "📋 Données initiales")


# ========== PAGE MÉTA PROJET ==========

def page_meta(data: Dict[str, Any]):
    all_climate = load_climate_monthly()
    st.header("Données initiales du projet")
    meta = data["meta"]

    meta["nom_projet"] = st.text_input("Nom du projet", value=meta.get("nom_projet", ""))
    meta["type_projet"] = st.text_input("Type de projet (optionnel)", value=meta.get("type_projet", ""))

    # --- Choix du canton et station météo associée ---
    st.subheader("Station météo de référence")
    
    # Dictionnaire des cantons et leurs stations météo
    stations_par_canton = {
        "Argovie": ["Basel-Binningen", "Buchs Aarau"],
        "Appenzell Rhodes-Intérieures": ["St-Gallen"],
        "Appenzell Rhodes-Extérieures": ["St-Gallen"],
        "Bâle-Campagne": ["Basel-Binningen"],
        "Bâle ville": ["Basel-Binningen"],
        "Berne": ["Adelboden", "Bern Libefeld"],
        "Fribourg": ["Adelboden", "Bern Libefeld"],
        "Genève": ["Genève"],
        "Glaris": ["Glarus"],
        "Grisons": ["Chur", "Davos", "Disentis", "Robbia", "Samedan", "Schuls"],
        "Jura": ["Bern Libefeld", "La Chaux-de-Fonds"],
        "Lucerne": ["Luzern"],
        "Neuchâtel": ["La Chaux-de-Fonds", "Neuchâtel"],
        "Nidwald": ["Engelberg", "Luzern"],
        "Obwald": ["Engelberg", "Luzern"],
        "St-Gall": ["St-Gallen"],
        "Schaffhouse": ["Schuffhausen"],
        "Schwytz": ["Luzern", "Zürich-MeteoSchweiz"],
        "Soleure": ["Wynau"],
        "Thurgovie": ["Güttingen"],
        "Tessin": ["Locarno_monti", "Lugano", "Magadino", "Robbia", "San Bernardino"],
        "Uri": ["Altdorf"],
        "Valais": ["Sion", "Montana", "Zermatt", "Grand st. Bernard", "Aigle"],
        "Vaud": ["Pully", "Aigle", "Payerne", "La Chaux-de-Fonds", "Adelboden"],
        "Zoug": ["Luzern"],
        "Zürich": ["Zürich-MeteoSchweiz", "Zürich-Kloten"],
        "Linchtenstein": ["Vaduz", "Engelberg"],
        "Spécial": [
            "Adelboden", "Aigle", "Altdorf", "Basel-Binningen", "Bern-Liebefeld", "Buchs-Aarau",
            "Chur", "Davos", "Disentis", "Engelberg", "Genève-Cointrin", "Glarus",
            "Grand-St-Bernard", "Güttingen", "Interlaken", "La Chaux-de-Fonds", "La Frétaz",
            "Locarno-Monti", "Lugano", "Luzern", "Magadino", "Montana", "Neuchâtel", "Payerne",
            "Piotta", "Pully", "Robbia", "Rünenberg", "Samedan", "San Bernardino", "St. Gallen",
            "Schaffhausen", "Scuol", "Sion", "Ulrichen", "Vaduz", "Wynau",
            "Zermatt", "Zürich-Kloten", "Zürich-MeteoSchweiz"
        ]
    }
    
    # Liste des cantons avec tri manuel (optionnel)
    liste_cantons = list(stations_par_canton.keys())
    
    # Valeur actuelle ou valeur par défaut
    selected_canton = meta.get("canton", liste_cantons[0])
    selected_canton = st.selectbox("Canton", options=liste_cantons, index=liste_cantons.index(selected_canton) if selected_canton in liste_cantons else 0)
    meta["canton"] = selected_canton
    
    # Mise à jour des stations météo en fonction du canton
    stations_disponibles = stations_par_canton.get(selected_canton, [])
    selected_station = meta.get("station_meteo", stations_disponibles[0] if stations_disponibles else "")
    
    if stations_disponibles:
        selected_station = st.selectbox(
            "Station météo associée",
            options=stations_disponibles,
            index=stations_disponibles.index(selected_station) if selected_station in stations_disponibles else 0
        )
        meta["station_meteo"] = selected_station
    else:
        st.info("Aucune station disponible pour ce canton.")
        
    if "station_meteo" in meta and meta["station_meteo"]:
        selected_station = meta["station_meteo"]
    
        if selected_station in all_climate:
            meta["climate_data"] = all_climate[selected_station].to_dict(orient="list")
        else:
            meta["climate_data"] = None

    st.markdown("### 📈 Données climatiques – Station sélectionnée")

    station = meta.get("station_meteo")

    if station and station in all_climate:
        df_clim = all_climate[station]

        # 👉 VERSION SIMPLE : PAS DE .style, PAS DE FORMAT, JUSTE LE DATAFRAME
        st.dataframe(
            df_clim,
            use_container_width=True,
            height=350,
        )

        # On stocke le climat dans les métadonnées pour les phases suivantes
        meta["climate_data"] = df_clim.to_dict(orient="list")

         # (Optionnel) graphique de température
        if {"Mois", "T_ext"}.issubset(df_clim.columns):
            fig_t = px.line(
                df_clim,
                x="Mois",
                y="T_ext",
                markers=True,
                title=f"Température extérieure moyenne – {station}",
            )
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Aucune donnée climatique trouvée pour cette station.")
        meta["climate_data"] = None

    st.markdown("---")
    st.subheader("Source de données du projet")

    meta["global_data_mode"] = st.radio(
        "Données globales pour le projet :",
        options=["Aucune", "Importer un fichier"],
        index=["Aucune", "Importer un fichier"].index(meta.get("global_data_mode", "Aucune")),
        key="meta_data_mode"
    )

    if meta["global_data_mode"] == "Importer un fichier":
        uploaded_global = st.file_uploader("Importer un fichier Excel global", type=["xlsx", "xls", "csv"], key="global_excel_upload")
        if uploaded_global:
            try:
                sheets = load_excel_file(uploaded_global)
                meta["global_excel"] = {
                    "file_name": uploaded_global.name,
                    "sheets": sheets
                }
                st.success(f"Fichier importé : {uploaded_global.name}")
            except Exception as e:
                st.error(f"Erreur lors de l'import : {e}")

    if meta.get("global_excel", {}).get("sheets"):
        st.caption("Aperçu du fichier global")
        first_sheet = list(meta["global_excel"]["sheets"].keys())[0]
        st.dataframe(meta["global_excel"]["sheets"][first_sheet].head())

def page_batiments_ouvrages(data: Dict[str, Any]):
    st.header("Bâtiments & ouvrages")
    batiments: List[Dict[str, Any]] = data["batiments"]

    if st.button("➕ Ajouter un bâtiment"):
        batiments.append({"nom": "", "adresse": "", "ouvrages": []})

    if not batiments:
        st.info("Aucun bâtiment défini.")
        return

    for bi, bat in enumerate(batiments):
        with st.expander(f"Bâtiment {bi+1} – {bat.get('nom') or '(sans nom)'}", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                bat["nom"] = st.text_input("Nom du bâtiment", value=bat.get("nom", ""), key=f"bat_nom_{bi}")
                bat["adresse"] = st.text_input("Adresse", value=bat.get("adresse", ""), key=f"bat_adresse_{bi}")
            with col2:
                if st.button("🗑️ Supprimer ce bâtiment", key=f"bat_del_{bi}"):
                    batiments.pop(bi)
                    st.experimental_rerun()

            # -------- Import Excel bâtiment --------
            bat["data_mode"] = bat.get("data_mode", "Aucune")
            bat["excel_data"] = bat.get("excel_data", {})

            bat["data_mode"] = st.radio(
                "Source de données pour ce bâtiment :",
                options=["Aucune", "Réutiliser fichier du projet", "Importer un fichier"],
                index=["Aucune", "Réutiliser fichier du projet", "Importer un fichier"].index(bat["data_mode"]),
                key=f"bat_data_mode_{bi}"
            )

            if bat["data_mode"] == "Importer un fichier":
                bat_excel = st.file_uploader("📥 Fichier Excel du bâtiment", type=["xlsx", "xls", "csv"], key=f"bat_excel_{bi}")
                if bat_excel:
                    try:
                        sheets = load_excel_file(bat_excel)
                        bat["excel_data"]["file_name"] = bat_excel.name
                        bat["excel_data"]["sheets"] = sheets
                        st.success(f"Fichier bâtiment chargé : {bat_excel.name}")
                    except Exception as e:
                        st.error(f"Erreur lors de l'import bâtiment : {e}")

            elif bat["data_mode"] == "Réutiliser fichier du projet":
                meta = st.session_state["phase1_data"]["meta"]
                bat["excel_data"] = meta.get("global_excel", {})

            # -------- Ouvrages --------
            st.markdown("---")
            st.markdown("##### Ouvrages de ce bâtiment")

            if st.button("➕ Ajouter un ouvrage", key=f"ouv_add_{bi}"):
                bat["ouvrages"].append({
                    "nom": "",
                    "type_usage": "",
                    "sre_m2": 0.0,
                    "surface_enveloppe_m2": 0.0,
                    "excel_data": {},
                    "producteurs": [],
                    "stockages": [],
                })

            ouvrages: List[Dict[str, Any]] = bat.get("ouvrages", [])
            if not ouvrages:
                st.info("Aucun ouvrage pour ce bâtiment.")
                continue

            for oi, ouv in enumerate(ouvrages):
                st.markdown("---")
                st.markdown(f"#### 🧱 Ouvrage {oi+1} – {ouv.get('nom') or '(sans nom)'}")

                col_o1, col_o2 = st.columns([3, 1])
                with col_o1:
                    ouv["nom"] = st.text_input("Nom de l’ouvrage", ouv.get("nom", ""), key=f"ouv_nom_{bi}_{oi}")
                    type_options = [
                        "", "Habitat individuel", "Habitat collectif", "Commerce", "Administration",
                        "Ecole", "Restauration", "Lieux de rassemblement", "Hôpitaux", "Industrie",
                        "Dépôts", "Installations sportives", "Piscines couvertes"
                    ]
                    ouv["type_usage"] = st.selectbox(
                        "Type d’usage", options=type_options,
                        index=type_options.index(ouv.get("type_usage", "")) if ouv.get("type_usage", "") in type_options else 0,
                        key=f"ouv_type_{bi}_{oi}"
                    )
                with col_o2:
                    if st.button("🗑️ Supprimer cet ouvrage", key=f"ouv_del_{bi}_{oi}"):
                        ouvrages.pop(oi)
                        st.experimental_rerun()

                # ------ Source de données Excel Ouvrage ------
                ouv["data_mode"] = ouv.get("data_mode", "Aucune")
                ouv["excel_data"] = ouv.get("excel_data", {})

                ouv["data_mode"] = st.radio(
                    "Source de données de l’ouvrage :",
                    options=["Aucune", "Réutiliser fichier du bâtiment", "Réutiliser fichier du projet", "Importer un fichier"],
                    index=["Aucune", "Réutiliser fichier du bâtiment", "Réutiliser fichier du projet", "Importer un fichier"].index(ouv["data_mode"]),
                    key=f"ouv_data_mode_{bi}_{oi}"
                )

                if ouv["data_mode"] == "Importer un fichier":
                    uploaded = st.file_uploader("Importer un fichier Excel/CSV", type=["xlsx", "xls", "csv"], key=f"ouv_excel_{bi}_{oi}")
                    if uploaded:
                        try:
                            sheets = load_excel_file(uploaded)
                            ouv["excel_data"]["file_name"] = uploaded.name
                            ouv["excel_data"]["sheets"] = sheets
                            st.success(f"Fichier ouvrage chargé : {uploaded.name}")
                        except Exception as e:
                            st.error(f"Erreur lors de la lecture du fichier : {e}")

                elif ouv["data_mode"] == "Réutiliser fichier du bâtiment":
                    ouv["excel_data"] = bat.get("excel_data", {})
                elif ouv["data_mode"] == "Réutiliser fichier du projet":
                    meta = st.session_state["phase1_data"]["meta"]
                    ouv["excel_data"] = meta.get("global_excel", {})

                # ------- Aperçu Excel si dispo -------
                excel_data = ouv.get("excel_data") or {}
                sheets = excel_data.get("sheets", {})
                if sheets:
                    sheet_names = list(sheets.keys())
                    selected_sheet = ouv.get("excel_sheet_name", sheet_names[0])
                    selected_sheet = st.selectbox(
                        "Feuille Excel à utiliser pour cet ouvrage",
                        options=sheet_names,
                        index=sheet_names.index(selected_sheet) if selected_sheet in sheet_names else 0,
                        key=f"ouv_sheet_{bi}_{oi}"
                    )
                    ouv["excel_sheet_name"] = selected_sheet
                    df = sheets[selected_sheet]
                    
                    if "Annee" in df.columns and "Mois" in df.columns:
                        if "Date_periode" not in df.columns:
                            df["Date_periode"] = pd.to_datetime(
                                df["Annee"].astype(int).astype(str) + "-" +
                                df["Mois"].astype(int).astype(str) + "-01",
                                errors="coerce"
                            )
                 
                    st.caption(f"Aperçu de **{selected_sheet}**")
                    st.dataframe(df.head())
                    
                    cols = df.columns.tolist()

                    ouv["sre_m2"] = st.number_input(
                        "SRE de l’ouvrage [m²]",
                        min_value=0.0,
                        value=float(ouv.get("sre_m2") or 0.0),
                        step=10.0,
                        key=f"ouv_sre_{bi}_{oi}"
                    )
                    ouv["surface_enveloppe_m2"] = st.number_input(
                        "Surface d’enveloppe thermique [m²] (facultatif)",
                        min_value=0.0,
                        value=float(ouv.get("surface_enveloppe_m2") or 0.0),
                        step=10.0,
                        key=f"ouv_env_{bi}_{oi}"
                    )

                    # --- Sélection des colonnes de base ---
                    ouv["time_col"] = st.selectbox(
                        "Colonne de temps",
                        options=cols,
                        index=cols.index(ouv.get("time_col", cols[0])) if ouv.get("time_col") in cols else 0,
                        key=f"time_col_{bi}_{oi}"
                    )

                    # Consommation électrique
                    elec_options = ["(aucune)"] + cols
                    current_elec = ouv.get("conso_elec_col")
                    elec_index = elec_options.index(current_elec) if current_elec in cols else 0
                    selected_elec = st.selectbox(
                        "Colonne de consommation électrique [kWh]",
                        options=elec_options,
                        index=elec_index,
                        key=f"conso_elec_col_{bi}_{oi}",
                    )
                    ouv["conso_elec_col"] = selected_elec if selected_elec != "(aucune)" else None

                    # Consommation thermique
                    th_options = ["(aucune)"] + cols
                    current_th = ouv.get("conso_th_col")
                    th_index = th_options.index(current_th) if current_th in cols else 0
                    selected_th = st.selectbox(
                        "Colonne de consommation thermique [kWh]",
                        options=th_options,
                        index=th_index,
                        key=f"conso_th_col_{bi}_{oi}",
                    )
                    ouv["conso_th_col"] = selected_th if selected_th != "(aucune)" else None

                    # --- Aperçu graphique des consommations ---
                    try:
                        time_col = ouv.get("time_col")
                        conso_elec_col = ouv.get("conso_elec_col")
                        conso_th_col = ouv.get("conso_th_col")

                        if time_col and (conso_elec_col or conso_th_col):
                            fig = go.Figure()

                            x_raw = df[time_col]

                            # Reconstruction d'une date si "Mois" + "Annee"
                            if time_col.lower().startswith("mois") and "Annee" in df.columns:
                                x = pd.to_datetime(
                                    df["Annee"].astype(int).astype(str) + "-" +
                                    df["Mois"].astype(int).astype(str) + "-01",
                                    errors="coerce"
                                )
                            else:
                                x = pd.to_datetime(x_raw, errors="ignore")

                            if conso_elec_col and conso_elec_col in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=x,
                                    y=pd.to_numeric(df[conso_elec_col], errors="coerce"),
                                    mode="lines",
                                    name=f"Conso élec – {conso_elec_col}",
                                    fill="tozeroy",
                                ))

                            if conso_th_col and conso_th_col in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=x,
                                    y=pd.to_numeric(df[conso_th_col], errors="coerce"),
                                    mode="lines",
                                    name=f"Conso th – {conso_th_col}",
                                    fill="tozeroy",
                                ))

                            if fig.data:
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=40, r=20, t=10, b=40),
                                    xaxis_title="Temps",
                                    yaxis_title="Puissance / Énergie",
                                    hovermode="x unified",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Aucune colonne de consommation valide sélectionnée pour le graphique.")
                        else:
                            st.info("Sélectionne la colonne de temps et au moins une colonne de consommation pour le graphique.")
                    except Exception as e:
                        st.warning(f"Erreur d'affichage du graphique : {e}")
                else:
                    st.info("Aucune donnée Excel chargée pour cet ouvrage.")
                    
                render_producteurs(bi, oi, ouv, bat_excel=bat.get("excel_data", {}), project_excel=data["meta"].get("global_excel", {}))
                render_stockages(bi, oi, ouv)
# ========== PAGE PARAMS SIMULATION ==========

def page_simu_params(data: Dict[str, Any]):
    st.header("Paramétrage de la simulation")
    params = data["simu_params"]

    params["pas_temps"] = st.selectbox("Pas de temps", ["Horaire", "15 min", "Journalier", "Saisonnier"])
    
    col1, col2 = st.columns(2)
    with col1:
        params["duree_simulation_ans"] = st.slider("Durée de la simulation [années]", 1, 30, value=params.get("duree_simulation_ans", 5))
    with col2:
        params["horizon_analyse_ans"] = st.slider("Horizon d’analyse [années]", 1, 50, value=params.get("horizon_analyse_ans", 10))

    params["methode_optimisation"] = st.selectbox("Méthode d’optimisation", [
        "Aucune (simulation simple)", "Optimisation coûts", "Optimisation CO₂",
        "Optimisation exergie", "Optimisation multi-critères"
    ])

    params["critere_prioritaire"] = st.selectbox("Critère prioritaire", [
        "Énergie", "Finances", "CO₂", "Exergie", "Équilibré"
    ])


# ========== PAGE VALIDATION ==========

def page_validation(data: Dict[str, Any]):
    st.header("Validation du projet et passage à la Phase 2")

    meta = data["meta"]
    bats = data["batiments"]
    params = data["simu_params"]

    st.subheader("Résumé du projet")
    st.write("Nom du projet :", meta.get("nom_projet", ""))
    st.write("Commune :", meta.get("commune", ""))
    st.write("Zone climatique :", meta.get("zone_climatique", ""))

    st.subheader("Bâtiments et ouvrages")
    st.write(f"Nombre de bâtiments : {len(bats)}")
    nb_ouvrages = sum(len(b.get("ouvrages", [])) for b in bats)
    st.write(f"Nombre total d’ouvrages : {nb_ouvrages}")

    st.subheader("Paramètres de simulation")
    st.write(params)

    erreurs = []
    if not meta.get("nom_projet"):
        erreurs.append("Le nom du projet est vide.")
    if len(bats) == 0:
        erreurs.append("Aucun bâtiment défini.")
    if nb_ouvrages == 0:
        erreurs.append("Aucun ouvrage défini.")

    if erreurs:
        st.error("Merci de corriger les points suivants avant de passer à la Phase 2 :")
        for e in erreurs:
            st.markdown(f"- {e}")
        valide = False
    else:
        st.success("Les données semblent cohérentes pour lancer la Phase 2.")
        valide = True

    if st.button("✅ Valider et lancer les calculs (passer à la Phase 2)") and valide:
        project = _build_flat_project_for_calculations(data)
    
        # Injecte les catégories d’ouvrages avec leurs infos Excel
        project["categories_ouvrages"] = _extract_ouvrages_with_data(data)
        
        # Sauvegarde aussi dans la session pour Phase 2 (fallback)
        st.session_state["categories_ouvrages"] = project["categories_ouvrages"]
    
        # Facultatif : injecte aussi les feuilles en cache pour Phase 2
        all_sheets = {}
        for bat in data.get("batiments", []):
            for ouv in bat.get("ouvrages", []):
                sheets = (ouv.get("excel_data") or {}).get("sheets", {})
                all_sheets.update(sheets)
        st.session_state["excel_sheets"] = all_sheets
    
        # Lancer les calculs (si tu as un calcul derrière)
        excel_sheets = st.session_state.get("excel_sheets", {})
        results = run_calculations(project, excel_sheets)
        project["results"] = results
    
        # Enregistre projet dans la session
        st.session_state["project"] = project
        st.session_state["validated"] = True
        st.success("Projet validé et calculs lancés. Passage à la Phase 2.")
        st.rerun()


def _build_flat_project_for_calculations(data: Dict[str, Any]) -> Dict[str, Any]:
    bats_flat = []
    ouvr_flat = []

    for bi, bat in enumerate(data["batiments"]):
        bats_flat.append({
            "nom": bat.get("nom", f"Bâtiment {bi+1}"),
            "adresse": bat.get("adresse", ""),
        })
        for oi, ouv in enumerate(bat.get("ouvrages", [])):
            ouvr_flat.append({
                "batiment_id": bi,
                "nom": ouv.get("nom", f"Ouvrage {oi+1}"),
                "type_usage": ouv.get("type_usage", ""),
                "producteurs": ouv.get("producteurs", []),
                "stockages": ouv.get("stockages", []),
            })

    return {
        "nom": data["meta"].get("nom_projet", "Projet sans nom"),
        "batiments": bats_flat,
        "ouvrages": ouvr_flat,
        "params": data["simu_params"],
    }


# ========== MAIN RENDER ENTRY ==========
def render_producteurs(bi, oi, ouv, bat_excel={}, project_excel={}):
    st.markdown("---")
    st.markdown("##### ⚡ Producteurs")

    if st.button("➕ Ajouter un producteur", key=f"prod_add_{bi}_{oi}"):
        ouv["producteurs"].append({
            "type_general": "",
            "techno": "",
            "parametres": {},
            "puissance_kw": 0.0,
        })

    producteurs = ouv.get("producteurs", [])
    if not producteurs:
        st.info("Aucun producteur pour cet ouvrage.")
        return

    for pi, prod in enumerate(producteurs):
        st.markdown(f"**Producteur {pi+1}**")

        with st.container():
            # Sélection du type
            type_options = get_available_producer_types()
            prod["type_general"] = st.selectbox(
                "Type de producteur",
                options=type_options,
                index=type_options.index(prod.get("type_general", "")) if prod.get("type_general", "") in type_options else 0,
                key=f"type_general_{bi}_{oi}_{pi}"
            )

            try:
                # Charger les techno dynamiquement
                techno_label = get_label_for_producer(prod["type_general"])
                techno_dict = load_producer_technologies(prod["type_general"])
                techno_list = list(techno_dict.keys())

                previous_tech = prod.get("techno", "")
                selected_tech = st.selectbox(
                    techno_label,
                    options=techno_list,
                    index=techno_list.index(previous_tech) if previous_tech in techno_list else 0,
                    key=f"techno_select_{bi}_{oi}_{pi}"
                )

                if selected_tech != previous_tech:
                    default_params = techno_dict.get(selected_tech, {})
                    prod["parametres"] = {
                        k: {
                            "valeur": v["valeur"] if v["valeur"] != "-" else 0.0,
                            "unite": v["unite"]
                        } for k, v in default_params.items()
                    }

                prod["techno"] = selected_tech
                
                type_general = prod.get("type_general", "")
                techno_name = prod.get("techno", "")
                
                # 🔑 Déterminer l'ID interne (engine) à partir du type + techno
                engine = infer_engine_from_type_and_techno(type_general, techno_name)
                prod["engine"] = engine
                
                if engine is None:
                    st.warning(
                        f"Aucun moteur associé à la techno '{techno_name}' "
                        f"({type_general}). Complète ENGINE_BY_TECHNO dans core/technologies.py."
                    )


                # Données principales
                prod["puissance_kw"] = st.number_input(
                    "Puissance installée [kW]",
                    min_value=0.0,
                    step=1.0,
                    value=prod.get("puissance_kw", 0.0) or 0.0,
                    key=f"prod_puissance_{bi}_{oi}_{pi}"
                )

                # 📁 Source de données
                prod["source_data_mode"] = st.radio(
                    "Choisir comment alimenter ce producteur :",
                    options=[
                        "Aucune",
                        "Importer un fichier",
                        "Réutiliser un fichier d’ouvrage",
                        "Réutiliser un fichier du bâtiment",
                        "Réutiliser un fichier du projet"
                    ],
                    index=[
                        "Aucune",
                        "Importer un fichier",
                        "Réutiliser un fichier d’ouvrage",
                        "Réutiliser un fichier du bâtiment",
                        "Réutiliser un fichier du projet"
                    ].index(prod.get("source_data_mode", "Aucune")),
                    key=f"prod_data_mode_{bi}_{oi}_{pi}"
                )

                sheets = {}
                source_label = ""
                if prod["source_data_mode"] == "Importer un fichier":
                    uploaded_prod = st.file_uploader(
                        "Importer un fichier Excel/CSV",
                        type=["xlsx", "xls", "csv"],
                        key=f"prod_upload_{bi}_{oi}_{pi}"
                    )
                    if uploaded_prod:
                        try:
                            sheets = load_excel_file(uploaded_prod)
                            prod["custom_excel_data"] = {
                                "file_name": uploaded_prod.name,
                                "sheets": sheets
                            }
                            source_label = uploaded_prod.name
                        except Exception as e:
                            st.error(f"Erreur de lecture : {e}")
                    sheets = prod.get("custom_excel_data", {}).get("sheets", {})

                elif prod["source_data_mode"] == "Réutiliser un fichier d’ouvrage":
                    sheets = ouv.get("excel_data", {}).get("sheets", {})
                    source_label = ouv.get("excel_data", {}).get("file_name", "")

                elif prod["source_data_mode"] == "Réutiliser un fichier du bâtiment":
                    sheets = bat_excel.get("sheets", {})
                    source_label = bat_excel.get("file_name", "")

                elif prod["source_data_mode"] == "Réutiliser un fichier du projet":
                    sheets = project_excel.get("sheets", {})
                    source_label = project_excel.get("file_name", "")

                if sheets:
                    st.caption(f"📄 Source : {source_label}")
                    sheet_names = list(sheets.keys())
                    selected_sheet = st.selectbox(
                        "Feuille à utiliser",
                        options=sheet_names,
                        key=f"prod_sheet_{bi}_{oi}_{pi}"
                    )
                    df = sheets[selected_sheet]
                    cols = df.columns.tolist()
                    selected_col = st.selectbox(
                        "Colonne de production à utiliser",
                        options=cols,
                        key=f"prod_col_{bi}_{oi}_{pi}"
                    )
                    prod["prod_profile_sheet"] = selected_sheet
                    prod["prod_profile_col"] = selected_col
                    # Test techno
                    engine = prod.get("engine")
                    is_pv = engine == "pv"

                    
                    if is_pv:
                    
                        widget_key_prefix = f"pv_{bi}_{oi}_{pi}"
                        existing_cfg = prod.get("pv_flux_config")
                    
                        pv_cfg = render_pv_extra_params(
                            df=df,
                            existing_config=existing_cfg,
                            widget_key_prefix=widget_key_prefix,
                        )
                    
                        prod["pv_flux_config"] = pv_cfg

                    st.dataframe(df[[selected_col]].head())
                else:
                    if prod["source_data_mode"] != "Aucune":
                        st.warning("Aucune donnée Excel trouvée pour cette source.")

                # BLACK BOX
                st.markdown("### 📦 Black Box")
                st.markdown(f"**{selected_tech}** – {prod['puissance_kw']} kW")

                # Modification des paramètres
                state_key = f"show_params_{bi}_{oi}_{pi}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = False

                if st.button("✏️ Modifier les paramètres", key=f"toggle_params_{bi}_{oi}_{pi}"):
                    st.session_state[state_key] = not st.session_state[state_key]

                if st.session_state[state_key]:
                    st.markdown("#### Paramètres modifiables")
                    for param_key, param in prod["parametres"].items():
                        if param_key.lower() in ["puissance min", "puissance max"]:
                            continue
                        val = param.get("valeur", 0.0)
                        updated_val = st.number_input(
                            f"{param_key} ({param.get('unite', '')})",
                            value=float(val) if val is not None else 0.0,
                            key=f"param_{bi}_{oi}_{pi}_{param_key}"
                        )
                        prod["parametres"][param_key]["valeur"] = updated_val

            except Exception as e:
                st.error(f"Erreur lors du chargement de la technologie : {e}")

        if st.button("🗑️ Supprimer ce producteur", key=f"prod_del_{bi}_{oi}_{pi}"):
            producteurs.pop(pi)
            st.experimental_rerun()

def render_stockages(bi: int, oi: int, ouv: Dict[str, Any]):
    st.markdown("---")
    st.markdown("##### 🧊 Stockages")

    if st.button("➕ Ajouter un stockage", key=f"sto_add_{bi}_{oi}"):
        ouv["stockages"].append({
            "id_tech_stockage": "",
            "capacite_kwh": 0.0,
            "puissance_kw": 0.0,
            "params_custom": ""
        })

    stockages = ouv.get("stockages", [])
    if not stockages:
        st.info("Aucun stockage défini.")
        return

    for si, sto in enumerate(stockages):
        st.markdown(f"**Stockage {si+1}**")
        sto["id_tech_stockage"] = st.text_input(
            "ID technologie (stockage_db.xlsx)",
            value=sto.get("id_tech_stockage", ""),
            key=f"sto_id_{bi}_{oi}_{si}"
        )
        col1, col2 = st.columns(2)
        with col1:
            sto["capacite_kwh"] = st.number_input(
                "Capacité utile [kWh]",
                value=sto.get("capacite_kwh", 0.0),
                min_value=0.0,
                step=1.0,
                key=f"sto_capacite_{bi}_{oi}_{si}"
            )
            sto["puissance_kw"] = st.number_input(
                "Puissance charge/décharge [kW]",
                value=sto.get("puissance_kw", 0.0),
                min_value=0.0,
                step=1.0,
                key=f"sto_puissance_{bi}_{oi}_{si}"
            )
        with col2:
            sto["params_custom"] = st.text_area(
                "Paramètres personnalisés (texte libre)",
                value=sto.get("params_custom", ""),
                key=f"sto_params_{bi}_{oi}_{si}"
            )

        if st.button("🗑️ Supprimer ce stockage", key=f"sto_del_{bi}_{oi}_{si}"):
            stockages.pop(si)
            st.experimental_rerun()


def render_phase1():
    init_phase1_state()
    data = st.session_state["phase1_data"]

    pages = {
        "📋 Données initiales": page_meta,
        "🏢 Bâtiments & ouvrages": page_batiments_ouvrages,
        "⚙️ Paramétrage simulation": page_simu_params,
        "✅ Validation": page_validation,
    }

    current_label = st.session_state.get("phase1_page", "📋 Données initiales")
    if current_label not in pages:
        current_label = "📋 Données initiales"

    page_label = st.sidebar.radio("Phase 1 – Navigation", list(pages.keys()), index=list(pages.keys()).index(current_label))
    st.session_state["phase1_page"] = page_label

    pages[page_label](data)
