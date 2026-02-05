[Click here to view the Project Report (PDF)](./RapportProjet.pdf)

# France-Accidents-Predictor-Spark
Multidimensional analysis and prediction of road accident severity in France (2021-2023) using Apache Spark and MLlib: Identifying risk profiles through Gradient Boosted Trees

Analyse multidimensionnelle et prédiction de la gravité des accidents de la route en France (2021-2023) via Apache Spark et MLlib. Identification des profils à risque par Gradient Boosted Trees


# Analyse Exploratoire et Prédiction de la Gravité des Accidents (Spark)

[cite_start]Ce projet utilise **Apache Spark** pour analyser les accidents corporels de la route en France entre 2021 et 2023. L'objectif est d'identifier les facteurs déterminants de la gravité des accidents et de prédire le profil des usagers susceptibles d'être gravement touchés. [cite: 50, 60]

## 🎯 Objectifs du Projet
- [cite_start]Traitement de données massives (Big Data) avec **PySpark**. [cite: 70]
- [cite_start]Analyse au niveau de l'**usager** (une ligne = un usager impliqué). [cite: 61, 78]
- Classification binaire de la gravité : 
    - [cite_start]**0 :** Indemne ou blessé léger. [cite: 65]
    - [cite_start]**1 :** Blessé grave ou décédé. [cite: 65, 89]

## 📊 Données & Préparation
[cite_start]Les données proviennent de l'Open Data officielle (BAAC) via data.gouv. [cite: 72]
- [cite_start]**Volume :** 373 139 observations pour la modélisation. [cite: 197]
- [cite_start]**Feature Engineering :** Création de variables temporelles (nuit, weekend), contextuelles (nb usagers, nb véhicules) et d'interactions (nuit_pluie, sans_secu). [cite: 160, 162, 169, 174, 178]
- [cite_start]**Format :** Toutes les variables sont converties en `double` pour la compatibilité avec Spark MLlib. [cite: 193]

## 🤖 Modélisation & Performance
[cite_start]Le projet compare deux approches de classification distribuée : [cite: 206, 211]
1. [cite_start]**Régression Logistique** (Baseline) [cite: 207]
2. [cite_start]**Gradient Boosted Trees (GBT)** (Modèle final retenu) [cite: 211, 401]

| Modèle | Accuracy | F1-Score | AUC ROC |
| :--- | :--- | :--- | :--- |
| **GBTClassifier** | **0.766** | **0.775** | **0.856** |
| Logistic Regression | 0.679 | 0.687 | 0.747 |

[cite_start][cite: 219]

## 🔍 Profils à Haut Risque Identifiés
[cite_start]L'analyse du top 10% des usagers les plus à risque montre que les facteurs les plus discriminants sont : [cite: 259, 285]
- [cite_start]**L'absence de dispositif de sécurité :** Facteur le plus critique (38.5% du segment à risque vs 6.5% du reste). [cite: 277, 278, 335]
- [cite_start]**Les deux-roues motorisés :** Fortement surreprésentés. [cite: 275, 316]
- [cite_start]**La conduite de nuit :** Augmente significativement l'exposition au risque. [cite: 273, 274]
- [cite_start]**L'âge :** Le segment à risque est globalement plus jeune (médiane 30 ans). [cite: 281]


## 🛠️ Stack Technique
- [cite_start]**Traitement :** PySpark (Spark SQL & MLlib). [cite: 58, 68]
- [cite_start]**Visualisation :** Matplotlib, Seaborn. [cite: 37]
- [cite_start]**Source :** Bulletin d'Analyse des Accidents Corporels (BAAC). [cite: 72]

