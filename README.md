# Prédiction de la santé fœtale (CTG) — Classification multiclasse

Projet de Machine Learning : prédire l'état de santé fœtale (**Normal / Suspect / Pathologique**) à partir de mesures de cardiotocographie (CTG).

**Auteurs :** Zakaria Ouakil, Marwane Boulehia — Génie Digital en Santé (2GDS), UM6SS

📓 **Notebook :** [`fetal_health_classification.ipynb`](fetal_health_classification.ipynb) (s'affiche directement sur GitHub, graphiques inclus)

## Données

- Source : UCI Machine Learning Repository — *Cardiotocography*
- 2126 examens × 21 mesures CTG, cible `fetal_health` (1 = Normal, 2 = Suspect, 3 = Pathologique)
- Classes déséquilibrées : ~78 % Normal, ~14 % Suspect, ~8 % Pathologique
- Fichier : [`fetal_health.csv`](fetal_health.csv)

## Méthodologie

1. Exploration et contrôle qualité (valeurs manquantes, doublons, outliers conservés car cliniquement significatifs)
2. Split stratifié 75/25 **avant** toute transformation, doublons supprimés du train uniquement (anti-leakage)
3. EDA : distributions, boxplots par classe, corrélations
4. Feature engineering (`accel_decel_ratio`, `variability_score`, `histogram_amplitude`) et suppression des variables redondantes
5. Sélection ANOVA, normalisation, ACP (visualisation)
6. Rééquilibrage par **SMOTE** (train uniquement)
7. Modèles de base : KNN, SVM, Decision Tree
8. Ensembles : Random Forest, AdaBoost, XGBoost, LightGBM
9. Combinaisons : Voting (hard/soft), Stacking
10. Validation croisée 10-fold
11. Explicabilité : **SHAP** et **LIME**

## Résultats

| Modèle | Test Accuracy | CV Mean | F1 Suspect | F1 Pathologique |
|---|---|---|---|---|
| KNN | 0.92 | 0.872 | 0.73 | 0.84 |
| SVM | 0.91 | 0.880 | 0.66 | 0.83 |
| Decision Tree | 0.90 | 0.885 | 0.69 | 0.88 |
| Random Forest | 0.93 | 0.895 | 0.78 | 0.91 |
| AdaBoost | 0.84 | 0.867 | 0.61 | 0.85 |
| XGBoost | 0.94 | — | 0.79 | 0.90 |
| **LightGBM** | **0.94** | **0.917** | 0.79 | 0.90 |
| Voting Hard | 0.94 | 0.895 | 0.79 | 0.88 |
| Voting Soft | 0.94 | 0.904 | 0.79 | 0.89 |
| **Stacking** | **0.94** | **0.917** | 0.79 | 0.88 |

**Meilleurs modèles : LightGBM** (meilleur individuel, rapide) et **Stacking** (meilleure généralisation en CV).

L'analyse SHAP/LIME montre que LightGBM s'appuie sur des indicateurs cliniques reconnus (accélérations, variabilité anormale, décélérations prolongées).

## Limites

- La classe **Suspect** reste difficile (F1 ≈ 0.79) : frontière floue avec Normal
- Dataset de taille moyenne (2126 lignes), à valider sur d'autres populations
- Outil d'**aide à la décision** uniquement — ne remplace pas un avis médical

## Lancer le projet

```bash
git clone https://github.com/USERNAME/fetal-health-ml.git
cd fetal-health-ml
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook fetal_health_classification.ipynb
```

## Structure

```
.
├── fetal_health_classification.ipynb
├── fetal_health.csv
├── requirements.txt
└── README.md
```
