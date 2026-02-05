import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "clean", "dvf_2024_clean.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_features.parquet")


def add_basic_features(df):
    """Ajoute les variables individuelles principales."""
    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    df["log_prix_m2"] = np.log1p(df["prix_m2"])
    df["ratio_surface_terrain"] = df["Surface terrain"] / df["Surface reelle bati"]
    df["ratio_surface_terrain"] = df["ratio_surface_terrain"].fillna(0)

    df["type_local_encoded"] = df["Type local"].map({"Maison": 0, "Appartement": 1})
    return df


def add_temporal_features(df):
    """Extrait les informations temporelles : mois, trimestre."""
    df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce")
    df["mois"] = df["Date mutation"].dt.month
    df["trimestre"] = df["Date mutation"].dt.quarter
    return df


def add_commune_features(df):
    """Ajoute les statistiques locales par commune."""
    grouped = df.groupby("Commune")

    df["prix_median_commune"] = grouped["prix_m2"].transform("median")
    df["prix_m2_median_commune"] = grouped["prix_m2"].transform("median")
    df["surface_median_commune"] = grouped["Surface reelle bati"].transform("median")
    df["dynamique_volume_commune"] = grouped["Valeur fonciere"].transform("count")
    df["part_maisons_commune"] = grouped["type_local_encoded"].transform("mean")

    return df


def add_code_postal_features(df):
    """Target encoding du code postal."""
    grouped = df.groupby("Code postal")
    df["prix_median_cp"] = grouped["prix_m2"].transform("median")
    return df


def add_departement_features(df):
    """Target encoding du département."""
    grouped = df.groupby("Code departement")
    df["prix_median_departement"] = grouped["prix_m2"].transform("median")
    return df


def add_type_voie_features(df):
    """Frequency encoding du type de voie."""
    freq = df["Type de voie"].value_counts(normalize=True)
    df["freq_type_voie"] = df["Type de voie"].map(freq)
    df["freq_type_voie"] = df["freq_type_voie"].fillna(0)
    return df


def add_voie_target_encoding(df):
    """Target encoding du nom de voie (Voie)."""
    grouped = df.groupby("Voie")
    df["prix_median_voie"] = grouped["prix_m2"].transform("median")
    df["prix_median_voie"] = df["prix_median_voie"].fillna(df["prix_m2"].median())
    return df


def main():
    print("Chargement des données nettoyées")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Lignes : {len(df):,}")

    print("Ajout des variables individuelles")
    df = add_basic_features(df)

    print("Ajout des variables temporelles")
    df = add_temporal_features(df)

    print("Ajout des features communes")
    df = add_commune_features(df)

    print("Encodage code postal")
    df = add_code_postal_features(df)

    print("Encodage département")
    df = add_departement_features(df)

    print("Encodage type de voie (frequency)")
    df = add_type_voie_features(df)

    print("Encodage voie (target)")
    df = add_voie_target_encoding(df)

    print("Suppression des NaN restants")
    df = df.fillna(0)

    # Normalisation des colonnes d'identifiants en string pour éviter les erreurs Parquet
    cols_str = [
        "Code postal", "Code commune", "Code departement",
        "Commune", "Voie", "Type de voie", "No voie"
    ]

    for col in cols_str:
        if col in df.columns:
            df[col] = df[col].astype("string")

        
    # Type de voie : frequency encoding
    if "Type de voie" in df.columns:
        freq_type = df["Type de voie"].value_counts(normalize=True)
        df["freq_type_voie"] = df["Type de voie"].map(freq_type)
    else:
        df["freq_type_voie"] = 0

    # Voie : target encoding (prix médian par voie)
    if "Voie" in df.columns:
        prix_voies = df.groupby("Voie")["prix_m2"].median()
        df["prix_median_voie"] = df["Voie"].map(prix_voies)
    else:
        df["prix_median_voie"] = df["prix_m2"].median()

    # 5. Suppression des colonnes inutiles
    cols_to_drop = [
        "Nature mutation",       # toujours 'Vente'
        "No voie",               # bruit pur, inutilisable
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Sauvegarde
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Fichier final sauvegardé : {OUTPUT_PATH}")
    print(f"Nombre de colonnes finales : {len(df.columns)}")
    print("Feature engineering terminé.")


if __name__ == "__main__":
    main()
