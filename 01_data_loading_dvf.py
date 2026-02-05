import pandas as pd
import os 

# on défini les chemins 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "ValeursFoncieres-2024.txt")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "clean", "dvf_2024_prefiltre.parquet")

# on liste les colonnes utiles pour le pré filtrage
COLUMNS_TO_KEEP = [
    "Date mutation",
    "Nature mutation",
    "Valeur fonciere",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Type local",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Surface terrain",
    "Type de voie",
    "Voie",
    "No voie"
]


def load_and_prefilter_dvf(filepath, chunksize=200_000):
    """
    Charge DVF par chunks, filtre les ventes, 
    supprime les lignes vides et conserve uniquement les colonnes essentielles.
    """
    filtered_chunks = []

    for chunk in pd.read_csv(filepath, sep="|", dtype=str, chunksize=chunksize):
        # Garder uniquement les ventes
        chunk = chunk[chunk["Nature mutation"] == "Vente"]

        # Supprimer les lignes sans valeur foncière
        chunk = chunk.dropna(subset=["Valeur fonciere"])

        # Garder uniquement les colonnes essentielles
        chunk = chunk[COLUMNS_TO_KEEP]

        # Ajouter le chunk filtré à la liste
        filtered_chunks.append(chunk)

    # Concaténer tous les chunks filtrés
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)

    return df_filtered


def main():
    print("Chargement DVF 2024 en cours...")
    df = load_and_prefilter_dvf(RAW_PATH)

    print(f"Nombre de lignes après pré-filtrage : {len(df):,}")

    # Sauvegarde en parquet (plus compact et plus rapide)
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"Fichier pré-filtré sauvegardé dans : {CLEAN_PATH}")


if __name__ == "__main__":
    main()
