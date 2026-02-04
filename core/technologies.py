# core/technologies.py

import os
import pandas as pd

# 🔁 Fonction interne commune
def _load_techno_file(file_name: str) -> dict:
    """
    Charge un fichier Excel de technologies et retourne un dict {nom_techno: {paramètres}}.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.abspath(os.path.join(current_dir, "..", "data", file_name))

    xl = pd.read_excel(file_path, sheet_name=None)
    data = {}
    for sheet_name, df in xl.items():
        df = df.dropna(subset=[df.columns[0]])
        tech_data = {
            str(row[df.columns[0]]).strip(): {
                "Values": row[df.columns[1]],
                "Units": row[df.columns[2]] if len(row) > 2 else ""
            }
            for _, row in df.iterrows()
        }
        data[sheet_name] = tech_data
    return data


# ⚡ Producteurs (électrique, Thermal, mixte)
PRODUCER_REGISTRY = {
    "Electric": {
        "label": "Electrical technology",
        "file": "electricity_producers.xlsx"
    },
    "Thermal": {
        "label": "Thermal technology",
        "file": "thermal_producers.xlsx"
    },
    "Hybrid": {
        "label": "Hybrid technology",
        "file": "mixed_producers.xlsx"
    }
}


def get_available_producer_types() -> list:
    return list(PRODUCER_REGISTRY.keys())


def get_label_for_producer(type_general: str) -> str:
    return PRODUCER_REGISTRY.get(type_general, {}).get("label", "Technologie")


def load_producer_technologies(type_general: str) -> dict:
    """
    Charge les technologies de producteurs selon le type (Electric, Thermal, Hybrid).
    """
    config = PRODUCER_REGISTRY.get(type_general)
    if not config:
        raise ValueError(f"Type de producteur inconnu: {type_general}")
    return _load_techno_file(config["file"])


# 🧊 Stockages (électrique, Thermal, mixte)
STOCKAGE_REGISTRY = {
    "Electric": {
        "label": "Electrical storage",
        "file": "electricity_storage.xlsx"
    },
    "Thermal": {
        "label": "Thermal storage",
        "file": "thermal_storage.xlsx"
    },
}


def get_available_storage_types() -> list:
    return list(STOCKAGE_REGISTRY.keys())


def get_label_for_storage(type_general: str) -> str:
    return STOCKAGE_REGISTRY.get(type_general, {}).get("label", "Stockage")


def load_storage_technologies(type_general: str) -> dict:
    """
    Charge les technologies de stockage selon le type.
    """
    config = STOCKAGE_REGISTRY.get(type_general)
    if not config:
        raise ValueError(f"Type de stockage inconnu: {type_general}")
    return _load_techno_file(config["file"])

# 🔗 Mapping techno -> engine (identifiant interne du producteur)
# Le principe : on mappe le type + nom EXACT de la techno vers un ID stable, court.
ENGINE_BY_TECHNO = {
    # "Type général"      "Nom EXACT de la techno dans l'Excel" : "ID interne"
    ("Electric", "Photovoltaic panel"): "pv",
    ("Electric", "Battery Li-ion") : "bat_li_ion",
    ("Electric", "Battery Evolium") : "bat_li_ion",

    # --- Thermal producers (tabs renamed in English) ---
    ("Thermal", "Wood stove"): "wood_stove",
    ("Thermal", "Pellet stove"): "pellet_stove",
    ("Thermal", "Log boiler"): "boiler_log",
    ("Thermal", "Pellet boiler"): "boiler_pellet",
    ("Thermal", "Gaz boiler"): "boiler_gas",
    ("Thermal", "Oil boiler"): "boiler_oil",
    ("Thermal", "Electric Heater"): "electric_heater",
    ("Thermal", "Air-to-Water Heat pump"): "hp_air_water",
    ("Thermal", "Water-to-Water Heat pump"): "hp_water_water",
    ("Thermal", "Solar Thermal"): "solar_thermal",
}


def infer_engine_from_type_and_techno(type_general: str, techno_name: str) -> str | None:
    """
    Retourne l'ID 'engine' (pv, pac_ae, bat_li_ion, ...) à partir du type général
    et du nom EXACT de la techno (tels que définis dans les Excels).
    """
    if not type_general or not techno_name:
        return None

    key = (type_general.strip(), techno_name.strip())
    return ENGINE_BY_TECHNO.get(key)


