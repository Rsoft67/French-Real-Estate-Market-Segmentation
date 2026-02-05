[Click here to view the Project Report (PDF)](./RapportProjet.pdf)

## 📌 Project Overview
This project focuses on segmenting the French real estate market using official **DVF (Demandes de Valeurs Foncières)** data. By applying unsupervised machine learning, we identify distinct market clusters and detect pricing anomalies to distinguish between genuine real estate opportunities and data entry errors.

## 🛠 Methodology
The analysis follows a rigorous data science pipeline:
1. **Data Ingestion & Cleaning:** Filtering raw DVF data to focus on residential properties, handling multi-lot transactions, and removing non-comparable goods.
2. **Feature Engineering:** Calculation of price per square meter (€/m²) and geographical encoding.
3. **Dimensionality Reduction:** Use of **PCA** (Principal Component Analysis) to handle multi-dimensional data.
4. **Clustering:** Implementation of **K-Means** to stratify the market into 5 homogeneous groups.
5. **Anomaly Detection:** Use of **Isolation Forest** to identify outliers with extreme price-to-surface ratios.



## 📊 Key Results
* **5 Market Clusters:** Ranging from Cluster 0 (Premium/Parisian markets) to Cluster 1 (Rural areas).
* **Outlier Identification:** Detection of extreme "discounts" (> 95%) in high-value areas, highlighting potential data inconsistencies.

## 📂 Project Structure
* **src/**: Python source files (.py) for cleaning and modeling.
* **notebooks/**: Main analysis notebook (Colab compatible).
* **rapport2_NathanW_RobinK_AlexandruN.pdf**: Full technical report.
* **README.md**: Project documentation.

## 🚀 How to use
1. **View Report:** Read the [Full PDF Report](./rapport2_NathanW_RobinK_AlexandruN.pdf).
2. **Run the Analysis:** Click the "Open in Colab" badge at the top to execute the code.

## 👥 Authors
* **Nathan W.**
* **Robin K.**
* **Alexandru N.**
