import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_features.parquet")
OUTPUT_PCA20 = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_pca20.parquet")
OUTPUT_PCA2 = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_pca2.parquet")

def select_features(df):
    """Sélectionne les colonnes numériques utiles pour PCA."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # on retire les colonnes trop influentes ou les  identifiants inutiles
    cols_to_remove = ["Valeur fonciere", "Surface terrain"]
    numeric_cols = [c for c in numeric_cols if c not in cols_to_remove]
    return df[numeric_cols], numeric_cols


def apply_standard_scaling(df):
    """Applique une standardisation."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def apply_pca(scaled_data, n_components):
    """Applique la PCA avec n composantes."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled_data)
    return reduced, pca

def main():
    print("Chargement des données enrichies")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Lignes chargées : {len(df):,}")
    
    # sélection des variables
    df_features, feature_cols = select_features(df)
    print(f"Nombre de variables utilisées pour PCA : {len(feature_cols)}")

    # standardisation
    print("Standardisation des variables")
    scaled_data, scaler = apply_standard_scaling(df_features)

    # PCA 20 composantes
    n_features = scaled_data.shape[1]
    n_components = min(20, n_features)
    print(f"Application de la PCA avec {n_components} composantes (max possible)")
    pca20_data, pca20 = apply_pca(scaled_data, n_components=n_components)

    # ajout au dataframe
    df_pca20 = pd.DataFrame(
        pca20_data,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    df_pca20["Type local"] = df["Type local"].values

    # sauvegarde
    df_pca20.to_parquet(OUTPUT_PCA20, index=False)
    print(f"PCA20 sauvegardée : {OUTPUT_PCA20}")

    # variance expliquée
    explained_var = np.sum(pca20.explained_variance_ratio_)
    print(f"Variance expliquée par {n_components} composantes : {explained_var:.4f}")
    
    # PCA 2 composantes pour visualiser
    print("Application de la PCA (visu)")
    pca2_data, pca2 = apply_pca(scaled_data, n_components=2)
    df_pca2 = pd.DataFrame(
    pca2_data,
    columns=["PC1", "PC2"]
    )
    df_pca2["type_local_encoded"] = df["type_local_encoded"].values
    df_pca2["Type local"] = df["Type local"].values

    df_pca2.to_parquet(OUTPUT_PCA2, index=False)
    print(f"PCA2 sauvegardée : {OUTPUT_PCA2}")
    print(f"Variance expliquée par les 2 premières composantes : "
          f"{np.sum(pca2.explained_variance_ratio_):.4f}")

    print("Réduction de dimension terminée.")


if __name__ == "__main__":
    main()