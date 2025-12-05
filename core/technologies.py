# core/loaders/technologies.py

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
                "valeur": row[df.columns[1]],
                "unite": row[df.columns[2]] if len(row) > 2 else ""
            }
            for _, row in df.iterrows()
        }
        data[sheet_name] = tech_data
    return data


# ⚡ Producteurs (électrique, thermique, mixte)
PRODUCER_REGISTRY = {
    "Electrique": {
        "label": "Technologie électrique",
        "file": "electricity_producers.xlsx"
    },
    "Thermique": {
        "label": "Technologie thermique",
        "file": "thermal_producers.xlsx"
    },
    "Mixte": {
        "label": "Technologie mixte",
        "file": "mixed_producers.xlsx"
    }
}


def get_available_producer_types() -> list:
    return list(PRODUCER_REGISTRY.keys())


def get_label_for_producer(type_general: str) -> str:
    return PRODUCER_REGISTRY.get(type_general, {}).get("label", "Technologie")


def load_producer_technologies(type_general: str) -> dict:
    """
    Charge les technologies de producteurs selon le type (Electrique, Thermique, Mixte).
    """
    config = PRODUCER_REGISTRY.get(type_general)
    if not config:
        raise ValueError(f"Type de producteur inconnu: {type_general}")
    return _load_techno_file(config["file"])


# 🧊 Stockages (électrique, thermique, mixte)
STOCKAGE_REGISTRY = {
    "Electrique": {
        "label": "Stockage électrique",
        "file": "electric_storage.xlsx"
    },
    "Thermique": {
        "label": "Stockage thermique",
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
    ("Electrique", "Photovoltaic panel"): "pv",
    # ("Thermique", "PAC air-eau"): "pac_ae",
    # ("Electrique", "Batterie Li-ion"): "bat_li_ion",
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


