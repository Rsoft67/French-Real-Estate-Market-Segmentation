import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "clean", "dvf_2024_prefiltre.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "clean", "dvf_2024_clean.parquet")

def clean_dvf(df):
    """
    Effectue les étapes de nettoyage nécessaires :
    Conversion des types (prix, surfaces)
    Filtrage maisons / appartements
    Suppression des ventes multi lots
    Suppression des valeurs aberrantes
    """
    # conversion des types:
    print("conversion des colonnes numeriques")

    df["Valeur fonciere"] = (
        df["Valeur fonciere"].str.replace(",",".", regex=False).astype(float)
    )

    df["Surface reelle bati"] = pd.to_numeric(df["Surface reelle bati"], errors="coerce")
    df["Nombre pieces principales"] = pd.to_numeric(df["Nombre pieces principales"], errors="coerce")
    df["Surface terrain"] = pd.to_numeric(df["Surface terrain"], errors="coerce")

    # maison et appartement :
    print("On filtre les biens résidentiels ( maisons et appartements)")
    df = df[df["Type local"].isin(["Maison","Appartement"])]

    # suppr ventes multi lots
    # elle est définie par plusieurs lignes dans le DVF donc on concervera les lignes ou le nombre de lots est 1

    if "Nombre de lots" in df.columns:
        df["Nombre de lots"] = pd.to_numeric(df["Nombre de lots"], errors="coerce")
        df = df[df["Nombre de lots"] == 1]

    # valeurs aberrantes
    print("Suppression des valeurs aberrantes")
    # si le prix est trop faible pour etre réaliste 
    df = df[df["Valeur fonciere"] > 5000]
    # si la surface habitable minimable raisonnable 
    df = df[df["Surface reelle bati"] >= 10]
    # prix  metre carré 
    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    # filtre metre carré aberrants
    df = df[(df["prix_m2"] > 300) & (df["prix_m2"] < 20000)]
    # suppr la col m^2
    df = df.drop(columns=["prix_m2"])

    print("Nettoyage terminé")
    return df


def main():
    print("Chargement du fichier pré filtré")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Lignes avant nettoyage : {len(df):,}")
    df_clean = clean_dvf(df)
    print(f"Lignes après nettoyage : {len(df_clean):,}")
    df_clean.to_parquet(OUTPUT_PATH, index=False)
    print(f"Fichier nettoyé sauvegardé : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()