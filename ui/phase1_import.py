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
    get_available_storage_types,
    get_label_for_storage,
    load_storage_technologies,
    infer_engine_from_type_and_techno,
)
from core.pv_module import render_pv_extra_params
from core.climate_data import load_climate_monthly
import plotly.express as px  
from core.apis.sonnendach_api import search_addresses, fetch_roof_by_feature_id
import streamlit.components.v1 as components
from ui.helpers.excel_source import choose_excel_source



def guess_time_col(df: pd.DataFrame):
    """
    Devine automatiquement une colonne temps dans un DataFrame.
    - priorité sur noms standards
    - fallback sur une colonne parsable en datetime (>=80% non-NaT)
    - dernier recours : première colonne
    """
    if df is None or df.empty or len(df.columns) == 0:
        return None

    # priorités par nom
    preferred = ["time", "date", "datetime", "timestamp", "date_periode", "date", "datetime"]
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for key in preferred:
        real = lower_map.get(key.strip().lower())
        if real:
            return real

    # fallback: première colonne parsable en datetime avec un bon taux
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce", utc=False)
            if len(dt) and float(dt.notna().mean()) >= 0.8:
                return c
        except Exception:
            pass

    # dernier recours: première colonne
    return df.columns[0]



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
            "duree_simulation_ans": 25,
            "horizon_analyse_ans": 25,
            "methode_optimisation": "Aucune (simulation simple)",
            "critere_prioritaire": "Équilibré",
            "price_buy_chf_kwh": 0.25,
            "price_sell_chf_kwh": 0.06,
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
        "Argovie": ["Basel-Binningen", "Buchs-Aarau"],
        "Appenzell Rhodes-Intérieures": ["St. Gallen"],
        "Appenzell Rhodes-Extérieures": ["St. Gallen"],
        "Bâle-Campagne": ["Basel-Binningen"],
        "Bâle ville": ["Basel-Binningen"],
        "Berne": ["Adelboden", "Bern-Liebefeld"],
        "Fribourg": ["Adelboden", "Bern-Liebefeld"],
        "Genève": ["Genève"],
        "Glaris": ["Glarus"],
        "Grisons": ["Chur", "Davos", "Disentis", "Robbia", "Samedan", "Schuls"],
        "Jura": ["Bern-Liebefeld", "La Chaux-de-Fonds"],
        "Lucerne": ["Luzern"],
        "Neuchâtel": ["La Chaux-de-Fonds", "Neuchâtel"],
        "Nidwald": ["Engelberg", "Luzern"],
        "Obwald": ["Engelberg", "Luzern"],
        "St-Gall": ["St. Gallen"],
        "Schaffhouse": ["Schuffhausen"],
        "Schwytz": ["Luzern", "Zürich-MeteoSchweiz"],
        "Soleure": ["Wynau"],
        "Thurgovie": ["Güttingen"],
        "Tessin": ["Locarno_monti", "Lugano", "Magadino", "Robbia", "San Bernardino"],
        "Uri": ["Altdorf"],
        "Valais": ["Sion", "Montana", "Zermatt", "Grand-St-Bernard", "Aigle"],
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
        
        # dans meta (phase 1)
    meta.setdefault("electricity", {})
    meta.setdefault("debug", {})
    meta["debug"].setdefault("store_timeseries", False)
  
    meta["debug"]["store_timeseries"] = st.checkbox(
        "🐞 Stocker les séries temporelles (export séparé)",
        value=bool(meta["debug"].get("store_timeseries", False)),
        help="Active le stockage des timeseries (load/PV/batterie/grid) dans results['timeseries_store'] pour export séparé. Peut être volumineux.",
        )
    
    meta.setdefault("pv_defaults", {})
    meta["pv_defaults"].setdefault("ru_amount_chf", 0.0)  # montant RU en CHF
    meta["pv_defaults"].setdefault("ru_year", 1)          # versement en année 2 (année 0 = investissement)


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

            # =========================
            # COLONNE 1 : infos + PV
            # =========================
            with col1:
                bat["nom"] = st.text_input("Nom du bâtiment", value=bat.get("nom", ""), key=f"bat_nom_{bi}")
                # -------------------------
                # Adresse + autocomplete (geo.admin)
                # -------------------------
                bat.setdefault("geo", {})  # {x,y,sr,label}
                addr_query = st.text_input(
                    "Adresse",
                    value=bat.get("adresse", ""),
                    key=f"bat_adresse_{bi}",
                    help="Tapez quelques caractères pour obtenir des suggestions (geo.admin).",
                )
                bat["adresse"] = addr_query
                
                suggestions = []
                try:
                    suggestions = search_addresses(addr_query, limit=8, lang="fr", sr=2056) if len((addr_query or "").strip()) >= 3 else []
                except Exception as e:
                    st.caption(f"⚠️ Autocomplete adresse indisponible: {e}")
                    suggestions = []
                
                if suggestions:
                    labels = [s["label"] for s in suggestions]
                    sel = st.selectbox(
                        "Suggestions d’adresses (geo.admin)",
                        options=[""] + labels,
                        index=0,
                        key=f"bat_adresse_suggest_{bi}",
                        help="Sélectionnez une suggestion puis cliquez sur 'Utiliser'.",
                    )
                
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        use_sel = st.button("✅ Utiliser", key=f"bat_adresse_use_{bi}")
                    with c2:
                        if bat.get("geo") and bat["geo"].get("x") and bat["geo"].get("y"):
                            st.caption(f"Coordonnées (LV95): x={bat['geo'].get('x'):.1f}, y={bat['geo'].get('y'):.1f}")
                
                    if use_sel and sel:
                        chosen = next((s for s in suggestions if s.get("label") == sel), None)
                        if chosen:
                            bat["adresse"] = chosen["label"]
                            bat["geo"] = {
                                "x": float(chosen["x"]),
                                "y": float(chosen["y"]),
                                "sr": int(chosen.get("sr", 2056)),
                                "label": chosen["label"],
                            }
                    
                            
                # --- PV state au niveau bâtiment ---
                bat.setdefault("pv_exists", None)                 # True/False/None
                bat.setdefault("pv_roofs", [])                    # liste de pans importés
                bat.setdefault("pv_roof_feature_id_input", "")    # input texte

                st.markdown("### ☀️ Photovoltaïque du bâtiment")

                # radio -> stockage bool
                choice = st.radio(
                    "Avez-vous déjà du PV sur ce bâtiment ?",
                    options=["(choisir)", "Oui", "Non", "Partiellement"],
                    index=["(choisir)", "Oui", "Non", "Partiellement"].index(
                        "Oui" if bat.get("pv_exists") is True else
                        "Non" if bat.get("pv_exists") is False else
                        "Partiellement" if bat.get("pv_exists") == "partial" else
                        "(choisir)"
                    ),
                    key=f"bat_pv_exists_{bi}",
                )
                
                if choice == "Oui":
                    bat["pv_exists"] = True
                elif choice == "Non":
                    bat["pv_exists"] = False
                elif choice == "Partiellement":
                    bat["pv_exists"] = "partial"
                else:
                    bat["pv_exists"] = None
                
                if bat.get("pv_exists") is True :
                    # ==========================================================
                    # 🔋 Batterie proposée (PV-only)
                    # ==========================================================
                    st.markdown("#### 🔋 Batterie proposée (basée sur le PV proposé)")
                    
                    # TODO ARCHI (future) :
                    # Aujourd'hui l'UX batterie est placée sous PV pour simplifier le cas d'usage
                    # "augmentation autoconsommation PV".
                    # À terme, une batterie doit devenir un stockage autonome configurable dans une
                    # section "Stockages", avec définition explicite :
                    # - sources de charge (PV, réseau, autres producteurs),
                    # - puits de décharge (load, réseau, autres),
                    # - scope (ouvrage / bâtiment / projet/cluster multi-bâtiments),
                    # car les sources/puits peuvent venir d'ouvrages/bâtiments différents.
                    
                    # force la clé dans le bâtiment (évite les cas où Streamlit rerun reset)
                    bat.setdefault("pv_battery_proposed", {})
                    bat["pv_battery_proposed"]["enabled"] = bool(bat["pv_battery_proposed"].get("enabled", False))

                    
                    bat["pv_battery_proposed"]["enabled"] = st.checkbox(
                        "Ajouter une batterie électrique proposée (PV-only)",
                        value=bool(bat["pv_battery_proposed"].get("enabled", False)),
                        key=f"bat_pv_batt_enable_{bi}",
                        help="Batterie proposée uniquement pour augmenter l’autoconsommation du PV proposé. "
                             "Pour l’instant : charge PV uniquement, décharge vers load uniquement."
                    )
                    
                    if bat["pv_battery_proposed"]["enabled"]:
                    
                        # --- Lecture techno depuis electricity_storage.xlsx ---
                        # Cache session pour éviter de relire à chaque rerun
                        if "_storage_techs_elec" not in st.session_state:
                            try:
                                st.session_state["_storage_techs_elec"] = load_storage_technologies("Electrique")
                            except Exception as e:
                                st.error(f"Impossible de charger les technologies de stockage électrique : {e}")
                                st.stop()
                    
                        storage_techs = st.session_state["_storage_techs_elec"]
                        if not storage_techs:
                            st.error("Aucune technologie trouvée dans electricity_storage.xlsx (stockage électrique).")
                            st.stop()
                    
                        techno_names = list(storage_techs.keys())
                        bat["pv_battery_proposed"].setdefault("techno", "Battery Li-ion" if "Battery Li-ion" in techno_names else techno_names[0])
                        #TODO gérer ça en foncton de Critère prioritaire dans ⚙️ Paramétrage simulation
                        bat["pv_battery_proposed"]["techno"] = st.selectbox(
                            "Technologie batterie (Excel : electricity_storage.xlsx)",
                            options=techno_names,
                            index=techno_names.index(bat["pv_battery_proposed"]["techno"]) if bat["pv_battery_proposed"]["techno"] in techno_names else 0,
                            key=f"bat_pv_batt_techno_{bi}",
                        )
                    
                        # Mapping techno -> engine (optionnel mais utile)
                        bat["pv_battery_proposed"]["engine"] = infer_engine_from_type_and_techno(
                            "Electrique", bat["pv_battery_proposed"]["techno"]
                        )
                    
                        # Param techno (pour affichage)
                        tech_params = storage_techs.get(bat["pv_battery_proposed"]["techno"], {}) or {}
                    
                        # Helpers locaux pour extraire une valeur numérique depuis tech_params
                        def _tech_num(keys, required=True):
                            for k in keys:
                                if k in tech_params:
                                    try:
                                        return float(tech_params[k].get("valeur"))
                                    except Exception:
                                        pass
                            if required:
                                raise ValueError(f"Paramètre techno manquant ou non numérique: {keys}")
                            return None
                    
                        # Lire paramètres indispensables (pas de fallback silencieux)
                        try:
                            capex_chf_kwh = _tech_num(["Capex", "CAPEX", "CAPEX [CHF/kWh]", "CAPEX_CHF_per_kWh"])
                            opex_chf_yr = _tech_num(["Opex", "OPEX", "OPEX [CHF/an]", "OPEX_CHF_per_year"])
                            lifetime_yr = _tech_num(["Durée de vie", "Lifetime", "Lifetime [an]", "lifetime_years"])
                            eta_global = _tech_num(["Rendement", "Efficiency", "Efficiency [-]"], required=True)
                            e_min_kwh = _tech_num(["Capacité min", "Capacité min [kWh]", "E_min_kWh"])
                            e_max_kwh = _tech_num(["Capacité max", "Capacité max [kWh]", "E_max_kWh"])
                            c_rate_ch = _tech_num(["C-rate charge max.", "C-rate charge", "C_rate_charge_max"])
                            c_rate_dis = _tech_num(["C-rate décharge max.", "C-rate décharge", "C_rate_discharge_max"])
                            volume_density = _tech_num(["Densité"])
                            CO2_rate = _tech_num(["Emission de CO2"])
                        except Exception as e:
                            st.error(f"Techno batterie incomplète dans Excel : {e}")
                            st.stop()
                    
                        show_tech = st.checkbox(
                            "🔍 Afficher paramètres techno batterie (extraits Excel)",
                            value=False,
                            key=f"bat_batt_show_tech_{bi}",
                        )
                        
                        if show_tech:
                            st.write(f"CAPEX : {capex_chf_kwh:.1f} CHF/kWh")
                            st.write(f"OPEX : {opex_chf_yr:.1f} CHF/an")
                            st.write(f"Lifetime : {lifetime_yr:.1f} ans")
                            st.write(f"Rendement global : {eta_global:.3f} [-]")
                            st.write(f"Capacité min/max : {e_min_kwh:.1f} / {e_max_kwh:.1f} kWh")
                            st.write(f"C-rate charge/décharge : {c_rate_ch:.3f} / {c_rate_dis:.3f} kW/kWh")

                    
                        # --- Méthode de sizing ---
                        bat["pv_battery_proposed"].setdefault("sizing", {})
                        bat["pv_battery_proposed"]["sizing"].setdefault("method", "heuristic")
                    
                        method = st.radio(
                            "Méthode de dimensionnement",
                            options=["heuristic", "manual"],
                            format_func=lambda x: "Heuristique (autoconsommation PV)" if x == "heuristic" else "Manuel",
                            index=0 if bat["pv_battery_proposed"]["sizing"].get("method") != "manual" else 1,
                            key=f"bat_pv_batt_sizing_method_{bi}",
                        )
                        bat["pv_battery_proposed"]["sizing"]["method"] = method
                    
                        # SOC (hardcodé dans ton moteur aujourd'hui, mais on le stocke pour traçabilité)
                        bat["pv_battery_proposed"]["sizing"]["soc_min_frac"] = 0.20
                        bat["pv_battery_proposed"]["sizing"]["soc_max_frac"] = 0.80
                    
                        if method == "heuristic":
                            # Paramètres heuristiques
                            bat["pv_battery_proposed"]["sizing"].setdefault("hours_target", 4.0)
                            bat["pv_battery_proposed"]["sizing"].setdefault("p_target_mode", "pv_installed")
                            bat["pv_battery_proposed"]["sizing"].setdefault("p_target_kw_override", None)
                    
                            st.caption("Heuristique : capacité dimensionnée pour augmenter l’autoconsommation, "
                                       "en tenant compte de la fenêtre SOC (20–80%) et du C-rate.")
                    
                            bat["pv_battery_proposed"]["sizing"]["hours_target"] = float(st.number_input(
                                "Durée cible (hours_target) [h]",
                                min_value=0.25,
                                max_value=24.0,
                                value=float(bat["pv_battery_proposed"]["sizing"].get("hours_target", 4.0)),
                                step=0.25,
                                key=f"bat_pv_batt_hours_target_{bi}",
                                help="Interprétation : la batterie doit pouvoir restituer la puissance cible pendant cette durée "
                                     "(hypothèse conservatrice hiver).",
                            ))
                    
                        else:
                            # Manuel : capacité totale imposée, Phase 2 n'aura plus qu'à simuler le dispatch
                            bat["pv_battery_proposed"]["sizing"].setdefault("capacity_total_kwh", max(e_min_kwh, 5.0))
                    
                            bat["pv_battery_proposed"]["sizing"]["capacity_total_kwh"] = float(st.number_input(
                                "Capacité totale batterie [kWh]",
                                min_value=float(e_min_kwh),
                                max_value=float(max(e_min_kwh, e_max_kwh * 50)),  # limite UI, le packing se fait en Phase 2
                                value=float(bat["pv_battery_proposed"]["sizing"].get("capacity_total_kwh", max(e_min_kwh, 5.0))),
                                step=1.0,
                                key=f"bat_pv_batt_capacity_manual_{bi}",
                            ))
                    
                        # --- Dispatch : PV-only strict (pour l'instant) ---
                        bat["pv_battery_proposed"].setdefault("dispatch", {})
                        bat["pv_battery_proposed"]["dispatch"]["strategy"] = "self_consumption_maximization"
                        bat["pv_battery_proposed"]["dispatch"]["allow_grid_charge"] = False
                        bat["pv_battery_proposed"]["dispatch"]["allow_grid_discharge"] = False
                    
                        st.info("PV-only : la batterie se charge uniquement sur surplus PV et se décharge uniquement vers la charge (pas de réseau).")
                    
                        # Stocker aussi les bornes techno (utile en Phase 2 pour packing)
                        bat["pv_battery_proposed"].setdefault("tech_constraints", {})
                        bat["pv_battery_proposed"]["tech_constraints"] = {
                            "capacity_min_kwh": float(e_min_kwh),
                            "capacity_max_kwh": float(e_max_kwh),
                            "c_rate_charge_kw_per_kwh": float(c_rate_ch),
                            "c_rate_discharge_kw_per_kwh": float(c_rate_dis),
                            "eta_global": float(eta_global),
                            "capex_chf_per_kwh": float(capex_chf_kwh),
                            "opex_chf_per_year": float(opex_chf_yr),
                            "lifetime_years": float(lifetime_yr),
                        }

                
                # --- Sonnendach seulement si PV = NON ---
                if (bat.get("pv_exists") is False) or (bat.get("pv_exists") == "partial"):
                    st.markdown("#### 🏠 Toiture (geo.admin / Sonnendach) – pour proposer une installation PV")
                
                    # --- Geo (LV95) depuis l’adresse sélectionnée ---
                    geo = bat.get("geo") or {}
                    has_xy = (geo.get("x") is not None) and (geo.get("y") is not None)
                
                    if not has_xy:
                        st.warning("Adresse non géocodée. Choisis une suggestion puis clique ✅ Utiliser.")
                    else:
                        st.caption(
                            f"Adresse géocodée (LV95): x={float(geo['x']):.1f}, y={float(geo['y']):.1f}"
                        )
                
                    # --- Config PV proposé ---
                    bat.setdefault("pv_proposed_config", {})
                    default_selfc = bat["pv_proposed_config"].get("default_selfc_pct", 0.20)
                
                    bat["pv_proposed_config"]["default_selfc_pct"] = (
                        st.slider(
                            "Taux d’autoconsommation estimé pour le PV proposé [%]",
                            0, 100,
                            value=int(100 * float(default_selfc)),
                            step=1,
                            key=f"bat_pv_prop_selfc_{bi}",
                        ) / 100.0
                    )
                
                    from urllib.parse import quote

                    addr = (bat.get("adresse") or "").strip()
                    
                    geo_admin_url = (
                        "https://map.geo.admin.ch/#/map"
                        "?lang=fr"
                        "&topic=energie"
                        "&layers=ch.bfe.solarenergie-eignung-daecher"
                    )
                    
                    # Optionnel: centre sur le point géocodé si dispo
                    geo = bat.get("geo") or {}
                    if geo.get("x") is not None and geo.get("y") is not None:
                        geo_admin_url += f"&center={float(geo['x'])},{float(geo['y'])}&z=17"
                    
                    # Pré-remplir la recherche (adresse)
                    if addr:
                        geo_admin_url += f"&swisssearch={quote(addr)}&swisssearch_autoselect=true"

                    components.iframe(geo_admin_url, height=520, scrolling=True)
                
                    bat["pv_roof_feature_id_input"] = st.text_input(
                        "FeatureId (copier depuis geo.admin après avoir cliqué le pan de toiture)",
                        value=bat.get("pv_roof_feature_id_input", ""),
                        key=f"bat_pv_roof_fid_{bi}",
                    )
                
                    fid = (bat.get("pv_roof_feature_id_input") or "").strip()
                
                    cbtn1, cbtn2, cbtn3 = st.columns([1, 1, 1])
                    with cbtn1:
                        add_roof = st.button("➕ Ajouter", key=f"bat_pv_roof_add_{bi}")
                    with cbtn2:
                        clear_roofs = st.button("🧹 Vider", key=f"bat_pv_roof_clear_{bi}")
                    with cbtn3:
                        show_raw = st.checkbox("Debug attrs", value=False, key=f"bat_pv_roof_debug_{bi}")
                
                    if clear_roofs:
                        bat["pv_roofs"] = []
                        st.success("Liste des pans vidée.")
                
                    if add_roof:
                        if not fid.isdigit():
                            st.warning("FeatureId invalide (doit être un nombre).")
                        else:
                            fid_int = int(fid)
                            existing_ids = {r.get("feature_id") for r in bat.get("pv_roofs", [])}
                            if fid_int in existing_ids:
                                st.info("Ce featureId est déjà dans la liste.")
                            else:
                                try:
                                    attrs = fetch_roof_by_feature_id(fid_int)
                
                                    surface = float(attrs.get("flaeche") or 0.0)
                                    ratio_default = 0.75
                
                                    bat["pv_roofs"].append({
                                        "feature_id": fid_int,
                                        "surface_m2": surface,
                                        "orientation_code": -float(attrs.get("ausrichtung") or 0.0),  # ton inversion
                                        "inclinaison_deg": float(attrs.get("neigung") or 0.0),
                
                                        # ✅ nouveau : surface utilisable
                                        "ratio_usable": ratio_default,
                                        "surface_usable_m2": surface * ratio_default,
                
                                        "_raw": attrs,
                                    })
                                    st.success(f"Pan {fid_int} ajouté.")
                                except Exception as e:
                                    st.error(f"Erreur import toiture : {e}")
                
                    # ---- Affichage + édition ratio/surface utilisable ----
                    if bat.get("pv_roofs"):
                        st.markdown("##### Pans de toiture ajoutés (avec surface utilisable)")
                
                        # éditeur par pan (ratio)
                        for ri, roof in enumerate(bat["pv_roofs"]):
                            fid_i = roof.get("feature_id")
                            st.markdown(f"**Pan {fid_i}**")
                
                            c1, c2, c3 = st.columns([1.4, 1, 1.2])
                            with c1:
                                st.write(f"Surface geo.admin: **{float(roof.get('surface_m2') or 0):.1f} m²**")
                            with c2:
                                ratio = st.number_input(
                                    "Part utilisable [-]",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=float(roof.get("ratio_usable", 0.75) or 0.75),
                                    step=0.05,
                                    key=f"pv_ratio_usable_{bi}_{ri}",
                                )
                                roof["ratio_usable"] = ratio
                            with c3:
                                roof["surface_usable_m2"] = float(roof.get("surface_m2") or 0.0) * float(ratio or 0.0)
                                st.write(f"Surface utilisable: **{roof['surface_usable_m2']:.1f} m²**")
                
                        # tableau récap
                        df_roofs = pd.DataFrame(bat["pv_roofs"]).drop(columns=["_raw"], errors="ignore")
                        st.dataframe(df_roofs, use_container_width=True)
                
                        # suppression d'un pan
                        del_col1, del_col2 = st.columns([2, 1])
                        with del_col1:
                            to_del = st.selectbox(
                                "Supprimer un pan (featureId)",
                                options=[r["feature_id"] for r in bat["pv_roofs"]],
                                key=f"bat_roof_del_select_{bi}",
                            )
                        with del_col2:
                            if st.button("🗑️ Supprimer ce pan", key=f"bat_roof_del_btn_{bi}"):
                                bat["pv_roofs"] = [r for r in bat["pv_roofs"] if r.get("feature_id") != to_del]
                                st.rerun()

                
                        if show_raw:
                            st.markdown("##### Debug (attributs bruts du dernier pan)")
                            st.json(bat["pv_roofs"][-1].get("_raw", {}))
                    else:
                        st.info("Aucun pan de toiture ajouté.")


            # =========================
            # COLONNE 2 : actions bâtiment
            # =========================
            with col2:
                if st.button("🗑️ Supprimer ce bâtiment", key=f"bat_del_{bi}"):
                    batiments.pop(bi)
                    st.rerun()


            # =========================
            # Import Excel bâtiment (TOUJOURS)
            # =========================
            bat["data_mode"] = bat.get("data_mode", "Aucune")
            bat["excel_data"] = bat.get("excel_data", {})

            bat["data_mode"] = st.radio(
                "Source de données pour ce bâtiment :",
                options=["Aucune", "Réutiliser fichier du projet", "Importer un fichier"],
                index=["Aucune", "Réutiliser fichier du projet", "Importer un fichier"].index(bat["data_mode"]),
                key=f"bat_data_mode_{bi}"
            )

            if bat["data_mode"] == "Importer un fichier":
                bat_excel = st.file_uploader(
                    "📥 Fichier Excel du bâtiment",
                    type=["xlsx", "xls", "csv"],
                    key=f"bat_excel_{bi}"
                )
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

            # =========================
            # Ouvrages (TOUJOURS)
            # =========================
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
                        "Type d’usage",
                        options=type_options,
                        index=type_options.index(ouv.get("type_usage", "")) if ouv.get("type_usage", "") in type_options else 0,
                        key=f"ouv_type_{bi}_{oi}"
                    )
                    
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


                with col_o2:
                    if st.button("🗑️ Supprimer cet ouvrage", key=f"ouv_del_{bi}_{oi}"):
                        ouvrages.pop(oi)
                        st.rerun()


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
                    
                render_producteurs(
                    bi, oi, ouv,
                    bat_excel=bat.get("excel_data", {}),
                    project_excel=data["meta"].get("global_excel", {})
                )
                render_storages(
                    bi, oi, ouv,
                    bat_excel=bat.get("excel_data", {}),
                    project_excel=data["meta"].get("global_excel", {})
)
# ========== PAGE PARAMS SIMULATION ==========
def _ensure_params_initialized(data: Dict[str, Any]) -> Dict[str, Any]:
    # ✅ Source de vérité: data["params"]
    #    Fallback rétro-compat: data["simu_params"]

    if "params" not in data or not isinstance(data.get("params"), dict):
        data["params"] = (data.get("simu_params") or {}).copy()

    params = data["params"]

    # Defaults JSON (doivent exister même si ancien projet)
    # (repris tel quel — pas de nouvelles valeurs)
    params.setdefault("pas_temps", "Horaire")
    params.setdefault("duree_simulation_ans", 25)
    params.setdefault("horizon_analyse_ans", 25)
    params.setdefault("methode_optimisation", "Aucune (simulation simple)")
    params.setdefault("critere_prioritaire", "Équilibré")
    params.setdefault("price_buy_chf_kwh", 0.25)
    params.setdefault("price_sell_chf_kwh", 0.06)

    data["params"] = params
    data["simu_params"] = data["params"]  # compat temporaire
    return params

def page_simu_params(data: Dict[str, Any]):
    st.header("⚙️ Paramétrage de la simulation")
    
    # ----------------------------------------------------------
    # ✅ Source de vérité: data["params"]
    #    Fallback rétro-compat: data["simu_params"]
    # ----------------------------------------------------------
    if "params" not in data or not isinstance(data.get("params"), dict):
        data["params"] = (data.get("simu_params") or {}).copy()
    
    params = data["params"]
    
    # ----------------------------------------------------------
    # Defaults JSON (doivent exister même si ancien projet)
    # ----------------------------------------------------------
    params.setdefault("pas_temps", "Horaire")
    params.setdefault("duree_simulation_ans", 25)
    params.setdefault("horizon_analyse_ans", 25)
    params.setdefault("methode_optimisation", "Aucune (simulation simple)")
    params.setdefault("critere_prioritaire", "Équilibré")
    params.setdefault("price_buy_chf_kwh", 0.25)
    params.setdefault("price_sell_chf_kwh", 0.06)

    # ----------------------------------------------------------
    # Pas de temps
    # ----------------------------------------------------------
    params["pas_temps"] = st.selectbox(
        "Pas de temps",
        ["Horaire", "15 min", "Journalier", "Saisonnier"],
        index=["Horaire", "15 min", "Journalier", "Saisonnier"].index(
            params.get("pas_temps", "Horaire")
        ) if params.get("pas_temps") in ["Horaire", "15 min", "Journalier", "Saisonnier"] else 0,
    )


    # ----------------------------------------------------------
    # Durées (source de vérité = params)
    # ----------------------------------------------------------
    params.setdefault("duree_simulation_ans", 25)
    params.setdefault("horizon_analyse_ans", 25)
   
    # Key de projet pour éviter que Streamlit recycle l'ancien session_state
    meta = data.get("meta") or {}
    proj_key = str(meta.get("nom_projet") or "__no_name__").strip()
   
    k_duree = f"{proj_key}__k_duree_simulation_ans"
    k_horizon = f"{proj_key}__k_horizon_analyse_ans"
   
    col1, col2 = st.columns(2)
    with col1:
        duree = st.slider(
            "Durée de la simulation [années]",
            min_value=1,
            max_value=30,
            value=int(params.get("duree_simulation_ans", 25)),
            step=1,
            key=k_duree,
        )
    with col2:
        horizon = st.slider(
            "Horizon d’analyse [années]",
            min_value=1,
            max_value=50,
            value=int(params.get("horizon_analyse_ans", 25)),
            step=1,
            key=k_horizon,
        )
   
    # Sync vers params (JSON)
    params["duree_simulation_ans"] = int(duree)
    params["horizon_analyse_ans"] = int(horizon)


   # ----------------------------------------------------------
    # 💰 Prix électricité (utilisés par PV + batterie)
    # ----------------------------------------------------------
    st.subheader("💰 Prix de l’électricité")
    
    col3, col4 = st.columns(2)
    with col3:
        params["price_buy_chf_kwh"] = st.number_input(
            "Prix achat électricité [CHF/kWh]",
            min_value=0.0,
            value=float(params.get("price_buy_chf_kwh", 0.25)),
            step=0.01,
            format="%.3f",
            key="k_price_buy_chf_kwh",
        )
    with col4:
        params["price_sell_chf_kwh"] = st.number_input(
            "Prix vente / rétribution [CHF/kWh]",
            min_value=0.0,
            value=float(params.get("price_sell_chf_kwh", 0.06)),
            step=0.01,
            format="%.3f",
            key="k_price_sell_chf_kwh",
        )


    # ----------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------
    params["methode_optimisation"] = st.selectbox(
        "Méthode d’optimisation",
        [
            "Aucune (simulation simple)", "Optimisation coûts", "Optimisation CO₂",
            "Optimisation exergie", "Optimisation multi-critères"
        ],
        index=[
            "Aucune (simulation simple)", "Optimisation coûts", "Optimisation CO₂",
            "Optimisation exergie", "Optimisation multi-critères"
        ].index(params.get("methode_optimisation", "Aucune (simulation simple)"))
        if params.get("methode_optimisation") in [
            "Aucune (simulation simple)", "Optimisation coûts", "Optimisation CO₂",
            "Optimisation exergie", "Optimisation multi-critères"
        ]
        else 0,
    )

    params["critere_prioritaire"] = st.selectbox(
        "Critère prioritaire",
        ["Énergie", "Finances", "CO₂", "Exergie", "Équilibré"],
        index=["Énergie", "Finances", "CO₂", "Exergie", "Équilibré"].index(params.get("critere_prioritaire", "Équilibré"))
        if params.get("critere_prioritaire") in ["Énergie", "Finances", "CO₂", "Exergie", "Équilibré"]
        else 4,
    )

    # ----------------------------------------------------------
    # ✅ Sync final (rétro-compat)
    # ----------------------------------------------------------
    data["params"] = params
    data["simu_params"] = data["params"]  # DEPRECATED: compat temporaire, à supprimer plus tard


# ========== PAGE VALIDATION ==========

def page_validation(data: Dict[str, Any]):
    st.header("Validation du projet et passage à la Phase 2")

    meta = data["meta"]
    bats = data["batiments"]
    params = _ensure_params_initialized(data)


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
            
            # ✅ IMPORTANT : conserver PV toiture pour Phase 2
            "pv_exists": bat.get("pv_exists", None),
            "pv_roofs": bat.get("pv_roofs", []),
            "pv_proposed_config": bat.get("pv_proposed_config", {}),
            # (optionnel) si tu veux garder l’input / debug
            "pv_roof_feature_id_input": bat.get("pv_roof_feature_id_input", ""),
            "pv_battery_proposed": bat.get("pv_battery_proposed", {}),
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
        "meta": data["meta"],         
        "batiments": bats_flat,
        "ouvrages": ouvr_flat,
        "params": data.get("params") or {},
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
                sheets, source_label = choose_excel_source(
                    label="Choisir comment alimenter ce producteur :",
                    key_prefix=f"prod_{bi}_{oi}_{pi}",
                    state=prod,
                    building_files=bat_excel,
                    ouvrage_files=ouv.get("excel_sheets") or {},
                    project_files=project_excel,
                )
                if not sheets:
                    st.info("Aucune donnée disponible (choisis une source ou importe un fichier).")
                    
                if sheets:
                    st.caption(f"📄 Source : {source_label}")
                    sheet_names = list(sheets.keys())
                
                    # valeur courante (si déjà choisie auparavant)
                    current_sheet = prod.get("prod_profile_sheet", None)
                    sheet_index = sheet_names.index(current_sheet) if current_sheet in sheet_names else 0
                
                    selected_sheet = st.selectbox(
                        "Feuille à utiliser",
                        options=sheet_names,
                        index=sheet_index,
                        key=f"prod_sheet_{bi}_{oi}_{pi}",
                    )
                
                    prod["prod_profile_sheet"] = selected_sheet
                
                    df = sheets[selected_sheet]
                    # --- unwrap si df est emballé dans un dict
                    if isinstance(df, dict):
                        if "df" in df:
                            df = df["df"]
                        elif "data" in df:
                            df = df["data"]
                    
                    if df is None or not hasattr(df, "columns"):
                        st.error("Erreur: la feuille sélectionnée n'est pas un DataFrame. Vérifie le format des excel_sheets.")
                        st.stop()

                    cols = df.columns.tolist()
                    current_col = prod.get("prod_profile_col", None)
                    col_index = cols.index(current_col) if current_col in cols else 0
                    
                    selected_col = st.selectbox(
                        "Colonne de production à utiliser",
                        options=cols,
                        index=col_index,
                        key=f"prod_col_{bi}_{oi}_{pi}",
                    )
                    
                    prod["prod_profile_sheet"] = selected_sheet
                    prod["prod_profile_col"] = selected_col
                    # --- Colonne temps (important pour profils mensuels et future robustesse batterie/stockage)
                    # --- Colonne temps (important pour profils mensuels)
                    guessed = guess_time_col(df)
                    
                    # Si l'utilisateur n'a encore rien choisi, on pré-remplit
                    if guessed and not prod.get("time_col"):
                        prod["time_col"] = guessed


                    
                    time_cols = cols[:]  # on laisse tout (l'utilisateur peut choisir)
                    default_time = prod.get("prod_profile_time_col")
                    if default_time not in time_cols:
                        default_time = guess_time_col(df)
                    
                    time_index = time_cols.index(default_time) if default_time in time_cols else 0
                    
                    selected_time_col = st.selectbox(
                        "Colonne temps à utiliser",
                        options=time_cols,
                        index=time_index,
                        key=f"prod_timecol_{bi}_{oi}_{pi}",
                    )
                    
                    prod["prod_profile_time_col"] = selected_time_col

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
            st.rerun()

            
            
# --- Labels UI (ne pas utiliser dans les calculs) ---
STORAGE_MAPPING_LABELS = {
    "PV": "Production photovoltaïque",
    "LOAD": "Consommation électrique du bâtiment",
    "GRID": "Réseau électrique",
}

def _label_mapping(code: str) -> str:
    return STORAGE_MAPPING_LABELS.get(code, code)


def render_storages(bi, oi, ouv, bat_excel=None, project_excel=None):
    import streamlit as st

    st.markdown("---")
    st.markdown("##### 🔋 Stockages")

    # Init
    ouv.setdefault("stockages", [])

    # Helpers
    def _get_param_val(params: dict, key: str, default=None):
        if not params:
            return default
        item = params.get(key)
        if not item:
            return default
        v = item.get("valeur", default)
        try:
            if isinstance(v, str):
                return float(v.strip().replace(" ", "").replace(",", "."))
            return float(v) if isinstance(v, (int, float)) else v
        except Exception:
            return v

    def _get_param_unit(params: dict, key: str, default=""):
        if not params:
            return default
        item = params.get(key)
        if not item:
            return default
        return item.get("unite", default) or default

    # Add storage
    if st.button("➕ Ajouter un stockage", key=f"sto_add_{bi}_{oi}"):
        ouv["stockages"].append({
            "type_general": "",
            "techno": "",
            "engine": None,
            "parametres": {},

            # dimensionnement
            "capacity_kwh": None,
            "capacity_kwh_th": None,
            "capacity_l": None,

            # options réseau
            "grid_charge_allowed": False,
            "grid_discharge_allowed": False,

            # données
            "data_mode": "simulated",  # simulated | measured

            # mapping flux (simu)
            "mapping": {
                "charge_sources": ["PV"],
                "discharge_sinks": ["LOAD"],
            },

            # mapping données mesurées
            "measured_profile": {
                "sheet": None,
                "time_col": None,
                "p_signed_col": None,
                "p_charge_col": None,
                "p_discharge_col": None,
                "e_charge_col": None,
                "e_discharge_col": None,
                "soc_col": None,
            },
        })

    if not ouv["stockages"]:
        st.info("Aucun stockage pour cet ouvrage.")
        return

    type_options = get_available_storage_types()  # Electrique / Thermique

    for si, sto in enumerate(ouv["stockages"]):
        st.markdown(f"**Stockage {si+1}**")

        # --- Ligne Type / Techno / Delete
        with st.container():
            c1, c2 = st.columns([3, 4])

            # Type
            with c1:
                current_type = sto.get("type_general", "")
                sto["type_general"] = st.selectbox(
                    "Type de stockage",
                    options=[""] + type_options,
                    index=([""] + type_options).index(current_type) if current_type in ([""] + type_options) else 0,
                    key=f"sto_type_{bi}_{oi}_{si}"
                )

            # Techno + delete
            with c2:
                techno_options = []
                if sto["type_general"]:
                    techno_data = load_storage_technologies(sto["type_general"])
                    techno_options = list(techno_data.keys())

                current_tech = sto.get("techno", "")
                sto["techno"] = st.selectbox(
                    get_label_for_storage(sto["type_general"]),
                    options=[""] + sorted(techno_options),
                    index=([""] + sorted(techno_options)).index(current_tech) if current_tech in ([""] + sorted(techno_options)) else 0,
                    key=f"sto_tech_{bi}_{oi}_{si}"
                )

                # Charger paramètres techno UNE SEULE FOIS
                if sto["type_general"] and sto["techno"] and not sto["parametres"]:
                    techno_params = load_storage_technologies(sto["type_general"]).get(sto["techno"], {}) or {}
                    sto["parametres"] = techno_params
                    sto["engine"] = infer_engine_from_type_and_techno(sto["type_general"], sto["techno"])


        if not sto.get("type_general") or not sto.get("techno"):
            st.caption("Choisis un type et une technologie pour configurer le stockage.")
            continue

        params = sto.get("parametres") or {}

        # ---------------- Dimensionnement
        if sto["type_general"] == "Electrique":
            cap_min = _get_param_val(params, "Capacité min", 0.0) or 0.0
            cap_max = _get_param_val(params, "Capacité max", 999999.0) or 999999.0
            default = sto["capacity_kwh"] or (cap_min if cap_min > 0 else 10.0)

            sto["capacity_kwh"] = st.number_input(
                "Capacité batterie [kWh]",
                min_value=float(cap_min),
                max_value=float(cap_max),
                value=float(default),
                step=1.0,
                key=f"sto_cap_e_{bi}_{oi}_{si}"
            )

            cA, cB = st.columns(2)
            with cA:
                sto["grid_charge_allowed"] = st.checkbox(
                    "Autoriser charge depuis réseau",
                    value=bool(sto.get("grid_charge_allowed")),
                    key=f"sto_grid_ch_{bi}_{oi}_{si}"
                )
            with cB:
                sto["grid_discharge_allowed"] = st.checkbox(
                    "Autoriser décharge vers réseau",
                    value=bool(sto.get("grid_discharge_allowed")),
                    key=f"sto_grid_dis_{bi}_{oi}_{si}"
                )

        elif sto["type_general"] == "Thermique":
            unit = (_get_param_unit(params, "Capacité min") or "").lower()
            is_l = "l" in unit and "kwh" not in unit

            cap_min = _get_param_val(params, "Capacité min", 0.0) or 0.0
            cap_max = _get_param_val(params, "Capacité max", 999999.0) or 999999.0

            if is_l:
                default = sto["capacity_l"] or (cap_min if cap_min > 0 else 200.0)
                sto["capacity_l"] = st.number_input(
                    "Capacité accu [L]",
                    min_value=float(cap_min),
                    max_value=float(cap_max),
                    value=float(default),
                    step=50.0,
                    key=f"sto_cap_l_{bi}_{oi}_{si}"
                )
            else:
                default = sto["capacity_kwh_th"] or (cap_min if cap_min > 0 else 10.0)
                sto["capacity_kwh_th"] = st.number_input(
                    "Capacité stockage thermique [kWh_th]",
                    min_value=float(cap_min),
                    max_value=float(cap_max),
                    value=float(default),
                    step=1.0,
                    key=f"sto_cap_th_{bi}_{oi}_{si}"
                )

        # ---------------- Mode données
        sto["data_mode"] = st.radio(
            "Mode données du stockage",
            options=["Simuler le stockage", "Importer des mesures"],
            index=0 if sto.get("data_mode") == "simulated" else 1,
            key=f"sto_data_mode_{bi}_{oi}_{si}"
        )
        sto["data_mode"] = "simulated" if sto["data_mode"] == "Simuler le stockage" else "measured"

        # ---------------- Mapping SIMULÉ
        if sto["data_mode"] == "simulated":
            st.markdown("**Mapping énergie batterie (simulation)**")

            sto.setdefault("mapping", {})
            sto["mapping"].setdefault("charge_sources", ["PV"])
            sto["mapping"].setdefault("discharge_sinks", ["LOAD"])

            charge_opts = ["PV"] + (["GRID"] if sto.get("grid_charge_allowed") else [])
            discharge_opts = ["LOAD"] + (["GRID"] if sto.get("grid_discharge_allowed") else [])

            st.caption("La batterie se charge depuis les sources sélectionnées et se décharge vers les usages sélectionnés.")

            m1, m2 = st.columns(2)
            with m1:
                sto["mapping"]["charge_sources"] = st.multiselect(
                    "Charge depuis",
                    options=charge_opts,
                    default=[x for x in sto["mapping"]["charge_sources"] if x in charge_opts],
                    format_func=_label_mapping,
                    key=f"sto_map_ch_{bi}_{oi}_{si}"
                )

            with m2:
                sto["mapping"]["discharge_sinks"] = st.multiselect(
                    "Décharge vers",
                    options=discharge_opts,
                    default=[x for x in sto["mapping"]["discharge_sinks"] if x in discharge_opts],
                    format_func=_label_mapping,
                    key=f"sto_map_dis_{bi}_{oi}_{si}"
                )

            if not sto["mapping"]["charge_sources"]:
                sto["mapping"]["charge_sources"] = ["PV"]
            if not sto["mapping"]["discharge_sinks"]:
                sto["mapping"]["discharge_sinks"] = ["LOAD"]

        # ---------------- Mapping MESURÉ
        else:
            st.markdown("**Données batterie mesurées (Excel utilisateur)**")
            mp = sto.setdefault("measured_profile", {})

            # ==========================================================
            # 📁 Source de données (standardisé)
            # ==========================================================
            sheets, source_label = choose_excel_source(
                label="Choisir comment alimenter ce stockage :",
                key_prefix=f"sto_{bi}_{oi}_{si}",
                state=sto,  # ✅ le stockage, pas prod
                building_files=bat_excel or {},
                ouvrage_files=ouv.get("excel_sheets") or {},
                project_files=project_excel or {},
            )

            # ==========================================================
            # Sélection feuille + colonnes
            # ==========================================================
            if not sheets:
                st.info("Aucune donnée disponible (choisis une source ou importe un fichier).")
                
            if sheets:
                st.caption(f"📄 Source : {source_label}")
                sheet_names = list(sheets.keys())

                current_sheet = mp.get("sheet")
                sheet_index = sheet_names.index(current_sheet) if current_sheet in sheet_names else 0

                mp["sheet"] = st.selectbox(
                    "Feuille à utiliser",
                    options=sheet_names,
                    index=sheet_index,
                    key=f"sto_sheet_{bi}_{oi}_{si}",
                )

                df = sheets.get(mp["sheet"])
                # --- unwrap / validation type
                if isinstance(df, dict):
                    if "df" in df:
                        df = df["df"]
                    elif "data" in df:
                        df = df["data"]
                
                if df is None:
                    st.error("Feuille invalide ou non chargée.")
                    st.stop()
                
                if isinstance(df, str):
                    st.error(
                        "Erreur: la feuille sélectionnée pointe vers une chaîne (str) au lieu d'un DataFrame. "
                        "Le mapping excel_sheets n'est pas au bon format (attendu: {sheet: DataFrame})."
                    )
                    st.stop()
                
                if not hasattr(df, "columns"):
                    st.error(
                        f"Erreur: objet feuille de type {type(df)} sans attribut columns. "
                        "Vérifie le format des excel_sheets."
                    )
                    st.stop()

                cols = [""] + (list(df.columns) if df is not None else [])
                
                # Auto-détection colonne temps si rien n'est défini
                if df is not None:
                    guessed = guess_time_col(df)
                    if guessed and not mp.get("time_col"):
                        mp["time_col"] = guessed


                # Colonne temps
                mp["time_col"] = st.selectbox(
                    "Colonne temps",
                    options=cols,
                    index=cols.index(mp.get("time_col")) if mp.get("time_col") in cols else 0,
                    key=f"sto_time_{bi}_{oi}_{si}",
                )
                # --- Option A : énergie charge/décharge (kWh/step) - MULTI COLONNES ---
                default_charge = mp.get("e_charge_cols") or ([mp.get("e_charge_col")] if mp.get("e_charge_col") else [])
                default_charge = [c for c in default_charge if c in cols]
                
                mp["e_charge_cols"] = st.multiselect(
                    "Colonnes énergie CHARGE [kWh/step] (plusieurs possibles)",
                    options=[c for c in cols if c != ""],
                    default=default_charge,
                    key=f"sto_ech_cols_{bi}_{oi}_{si}",
                )
                
                default_dis = mp.get("e_discharge_cols") or ([mp.get("e_discharge_col")] if mp.get("e_discharge_col") else [])
                default_dis = [c for c in default_dis if c in cols]
                
                mp["e_discharge_cols"] = st.multiselect(
                    "Colonnes énergie DÉCHARGE [kWh/step] (plusieurs possibles)",
                    options=[c for c in cols if c != ""],
                    default=default_dis,
                    key=f"sto_edis_cols_{bi}_{oi}_{si}",
                )
                
                # rétro-compat (utile si autre code attend encore *_col)
                mp["e_charge_col"] = mp["e_charge_cols"][0] if mp["e_charge_cols"] else ""
                mp["e_discharge_col"] = mp["e_discharge_cols"][0] if mp["e_discharge_cols"] else ""
                
                # ----------------------------
                # Init mapping manuel (MEASURED)
                # ----------------------------
                mp.setdefault("charge_col_map", {})     # { "colA": "PV", "colB": "GRID", ... }
                mp.setdefault("discharge_col_map", {})  # { "colX": "LOAD", "colY": "GRID", ... }
                
                charge_source_opts = ["PV"] + (["GRID"] if sto.get("grid_charge_allowed") else [])
                discharge_sink_opts = ["LOAD"] + (["GRID"] if sto.get("grid_discharge_allowed") else [])
                
                # Nettoyage (évite de garder des mappings sur des colonnes retirées)
                mp["charge_col_map"] = {k: v for k, v in mp["charge_col_map"].items() if k in (mp.get("e_charge_cols") or [])}
                mp["discharge_col_map"] = {k: v for k, v in mp["discharge_col_map"].items() if k in (mp.get("e_discharge_cols") or [])}
                
                # ----------------------------
                # Table mapping (colonne -> origine/destination)
                # ----------------------------
                st.markdown("---")


                st.caption("Associe chaque colonne à une origine/destination pour un Sankey correct (PV/GRID/LOAD).")

                # ----------------------------
                # Mapping CHARGE
                # ----------------------------
                if mp.get("e_charge_cols"):
                    st.markdown("**Mapping des colonnes de CHARGE**")
                
                    # petit header comme une table
                    h1, h2 = st.columns([3, 2])
                    with h1:
                        st.caption("Colonne (kWh/step)")
                    with h2:
                        st.caption("Origine")
                
                    for c in mp["e_charge_cols"]:
                        r1, r2 = st.columns([3, 2])
                
                        with r1:
                            st.write(c)
                
                        with r2:
                            prev = mp["charge_col_map"].get(
                                c,
                                "PV" if "PV" in charge_source_opts else charge_source_opts[0]
                            )
                            mp["charge_col_map"][c] = st.selectbox(
                                label="",
                                options=charge_source_opts,
                                index=charge_source_opts.index(prev) if prev in charge_source_opts else 0,
                                key=f"sto_map_ch_col_{bi}_{oi}_{si}_{c}",
                                label_visibility="collapsed",
                            )
                # ----------------------------
                # Mapping DÉCHARGE
                # ----------------------------
                if mp.get("e_discharge_cols"):
                    st.markdown("**Mapping des colonnes de DÉCHARGE**")
                
                    h1, h2 = st.columns([3, 2])
                    with h1:
                        st.caption("Colonne (kWh/step)")
                    with h2:
                        st.caption("Destination")
                
                    for c in mp["e_discharge_cols"]:
                        r1, r2 = st.columns([3, 2])
                
                        with r1:
                            st.write(c)
                
                        with r2:
                            prev = mp["discharge_col_map"].get(
                                c,
                                "LOAD" if "LOAD" in discharge_sink_opts else discharge_sink_opts[0]
                            )
                            mp["discharge_col_map"][c] = st.selectbox(
                                label="",
                                options=discharge_sink_opts,
                                index=discharge_sink_opts.index(prev) if prev in discharge_sink_opts else 0,
                                key=f"sto_map_dis_col_{bi}_{oi}_{si}_{c}",
                                label_visibility="collapsed",
                            )

                
                # Nettoyage (évite de garder des mappings sur des colonnes retirées)
                mp["charge_col_map"] = {k: v for k, v in mp["charge_col_map"].items() if k in (mp.get("e_charge_cols") or [])}
                mp["discharge_col_map"] = {k: v for k, v in mp["discharge_col_map"].items() if k in (mp.get("e_discharge_cols") or [])}
                
                st.markdown("---")
                
                # Option B: puissance signée (kW) si pas d'énergie
                mp["p_signed_col"] = st.selectbox(
                    "Colonne puissance signée [kW] (optionnel si énergie charge/décharge)",
                    options=cols,
                    index=cols.index(mp.get("p_signed_col")) if mp.get("p_signed_col") in cols else 0,
                    key=f"sto_psigned_{bi}_{oi}_{si}",
                )

                # SOC (kWh ou %)
                mp["soc_col"] = st.selectbox(
                    "Colonne SOC (kWh ou %) (optionnel)",
                    options=cols,
                    index=cols.index(mp.get("soc_col")) if mp.get("soc_col") in cols else 0,
                    key=f"sto_soc_{bi}_{oi}_{si}",
                )

                # Petit aperçu
                try:
                    preview_cols = [c for c in [mp["time_col"], mp["e_charge_col"], mp["e_discharge_col"], mp["p_signed_col"], mp["soc_col"]] if c]
                    if df is not None and preview_cols:
                        st.dataframe(df[preview_cols].head())
                except Exception:
                    pass

            else:
                if sto["source_data_mode"] != "Aucune":
                    st.warning("Aucune donnée Excel trouvée pour cette source.")


        # ---------------- Paramètres modifiables (comme producteurs)
        state_key = f"sto_edit_params_{bi}_{oi}_{si}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        if st.button("✏️ Modifier les paramètres", key=f"sto_btn_params_{bi}_{oi}_{si}"):
            st.session_state[state_key] = not st.session_state[state_key]

        if st.session_state[state_key]:
            st.markdown("#### Paramètres modifiables")
            for k, p in params.items():
                if k.lower() in ["capacité min", "capacité max"]:
                    continue
                unit = p.get("unite", "")
                v = p.get("valeur", None)
                label = f"{k} ({unit})" if unit else k

                try:
                    v_num = float(str(v).replace(",", "."))
                    sto["parametres"][k]["valeur"] = st.number_input(
                        label,
                        value=v_num,
                        key=f"sto_param_{bi}_{oi}_{si}_{k}"
                    )
                except Exception:
                    sto["parametres"][k]["valeur"] = st.text_input(
                        label,
                        value="" if v is None else str(v),
                        key=f"sto_param_{bi}_{oi}_{si}_{k}"
                    )

    if st.button("🗑️ Supprimer ce stockage", key=f"sto_del_{bi}_{oi}_{si}"):
        ouv["stockages"].pop(si)
        st.rerun()   
            
        st.markdown("---")

                            



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
