import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import umap

SAMPLE_SIZE = 5_000        # volontairement petit
N_PCA_UMAP = 10            # PCA intermédiaire pour UMAP
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCA20_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_pca20.parquet")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "dvf_2024_features.parquet")


def main():

    print("Chargement PCA (prétraitement)")
    df_pca = pd.read_parquet(PCA20_PATH)

    print("Chargement features (labels métier)")
    df_feat = pd.read_parquet(FEATURES_PATH)

    # Sélection des composantes PCA
    pca_cols = [c for c in df_pca.columns if c.startswith("PC")]
    pca_cols = pca_cols[:N_PCA_UMAP]

    X_pca = df_pca[pca_cols]

    print(f"Utilisation des {len(pca_cols)} premières composantes PCA")

    # ÉCHANTILLONNAGE SIMPLE
    if len(X_pca) > SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(X_pca.index, SAMPLE_SIZE, replace=False)

        X_visu = X_pca.loc[sample_idx].values
        labels = df_feat.loc[sample_idx, "type_local_encoded"].values
    else:
        X_visu = X_pca.values
        labels = df_feat["type_local_encoded"].values

    # UMAP (VISUALISATION EXPLORATOIRE)
    print("Application de UMAP (visualisation exploratoire)")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE
    )

    embedding = reducer.fit_transform(X_visu)

    # VISUALISATION
    plt.figure(figsize=(10, 7))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="viridis",
        s=4,
        alpha=0.6
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Projection UMAP (après PCA – échantillon)")
    plt.colorbar(label="Type de bien (encodé)")
    plt.tight_layout()
    plt.show()

    print("Visualisation terminée.")


if __name__ == "__main__":
    main()
