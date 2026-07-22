# 🚗 Getaround – Delay Analysis & Pricing Optimization

**Jedha Bootcamp – Certification Bac+4 – Bloc Déploiement**

---

## Liens de production

| Livrable | URL |
|---|---|
| 📊 **Dashboard** | `https://YOUR_USERNAME-getaround-dashboard.hf.space` |
| 🤖 **API Pricing** | `https://YOUR_USERNAME-getaround-api.hf.space` |
| 📈 **MLflow** | `https://YOUR_USERNAME-getaround-mlflow.hf.space` |

---

## Contexte

Getaround est une plateforme de location de voitures entre particuliers.
Ce projet adresse deux problématiques :

1. **Retards au checkout** — les conducteurs rendent parfois les voitures en retard,
   bloquant la location suivante. Quel délai minimum imposer ? Sur quelles voitures ?
2. **Optimisation des prix** — suggestion de prix optimal par jour via Machine Learning.

---

## Structure du repo

```
getaround/                             ← repo GitHub (source de vérité)
│
├── data/
│   ├── get_around_delay_analysis.xlsx
│   └── get_around_pricing_project.csv
│
├── notebooks/
│   └── 01-Getaround_analysis.ipynb    # EDA complète + ML + MLflow tracking
│
├── api/                               # source → déployé sur HF Space API
│   ├── app.py                         #   GET /  · POST /predict · GET /docs
│   ├── model.joblib                   #   modèle embarqué (MODEL_SOURCE=local)
│   ├── encoders.joblib
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md                      #   header YAML HF Space
│
├── dashboard/                         # source → déployé sur HF Space Dashboard
│   ├── app.py
│   ├── requirements.txt
│   └── README.md                      #   header YAML HF Space
│
├── mlflow/                            # source → déployé sur HF Space MLflow
│   ├── Dockerfile                     #   NeonDB (backend) + S3 (artifacts)
│   └── README.md                      #   header YAML HF Space + guide setup
│
├── deploy.sh                          # sync source → HF Spaces (voir ci-dessous)
├── environment.yml                    # conda env de développement
├── .env.example                       # template variables d'environnement
└── .gitignore
```

---

## Installation locale

### 1. Créer l'environnement conda

```bash
conda env create -f environment.yml
conda activate getaround
python -m ipykernel install --user --name getaround --display-name "Python 3 (getaround)"
```

### 2. Configurer les variables d'environnement

```bash
cp .env.example .env
# Remplir .env avec tes credentials MLflow / AWS
```

### 3. Lancer le notebook

```bash
cd notebooks
jupyter notebook 01-Getaround_analysis.ipynb
```

### 4. Tester l'API localement

```bash
cd api
uvicorn app:app --reload --port 8000
# → http://localhost:8000/docs
```

### 5. Tester le dashboard localement

```bash
cd dashboard
streamlit run app.py
```

---

## Déploiement sur Hugging Face

### Prérequis — cloner les 3 Spaces HF (une seule fois)

```bash
mkdir -p ~/projects/hf-spaces && cd ~/projects/hf-spaces

git clone https://huggingface.co/spaces/USERNAME/getaround-mlflow
git clone https://huggingface.co/spaces/USERNAME/getaround-api
git clone https://huggingface.co/spaces/USERNAME/getaround-dashboard
```

> Ces dossiers sont **en dehors** du repo GitHub — pas de repos imbriqués.

### Déployer

```bash
# Tout déployer
./deploy.sh

# Ou un Space spécifique
./deploy.sh api
./deploy.sh dashboard
./deploy.sh mlflow
```

Le script copie les fichiers source dans le clone HF correspondant,
commit et push automatiquement.

### Ordre de déploiement initial

```
1. mlflow    → configurer les Secrets HF (NeonDB + S3 + AWS)
2. notebook  → exécuter, vérifier les runs dans l'UI MLflow
3. api       → configurer les Secrets HF si MODEL_SOURCE=registry
4. dashboard → aucun Secret nécessaire
```

---

## API – Endpoint `/predict`

```bash
curl -i -H "Content-Type: application/json" \
     -X POST \
     -d '{"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]}' \
     https://YOUR_USERNAME-getaround-api.hf.space/predict
```

```python
import requests

r = requests.post(
    "https://YOUR_USERNAME-getaround-api.hf.space/predict",
    json={"input": [[7, 140000, 120, 0, 1, 2, 1, 1, 1, 0, 1, 1, 0]]}
)
print(r.json())   # {"prediction": [119]}
```

### Source du modèle (`MODEL_SOURCE`)

| Valeur | Comportement |
|---|---|
| `local` (défaut) | `model.joblib` embarqué dans l'image Docker — garanti sans dépendance réseau |
| `registry` | Modèle `@champion` chargé depuis le MLflow Model Registry → artefact lu sur S3 |

---

## Architecture MLflow

```
Notebook local
     │  mlflow.log_params / log_metrics / log_model
     ▼
MLflow Server  (HF Space getaround-mlflow)
     │
     ├── Backend store ──► PostgreSQL NeonDB   (runs, params, metrics)
     └── Artifact store ──► AWS S3             (model.joblib, plots)
                            s3://cni-fraud-detection/getaround-mlflow/
```

Voir `mlflow/README.md` pour le guide de setup complet (NeonDB, S3, Secrets HF).

---

## Modèle ML

| Métrique | Valeur |
|---|---|
| Algorithme | Gradient Boosting Regressor |
| R² (test) | **0.736** |
| MAE (test) | **≈ 10.6 €/jour** |
| Top features | `engine_power` 43 % · `mileage` 29 % · `model_key` 7 % |

---

## Résultats clés – Analyse retards

- **57.5 %** des locations terminées sont en retard au checkout
- Médiane = **53 min** | P90 = **338 min**
- **11.8 %** des paires consécutives ont le conducteur suivant impacté
- **Recommandation PM :** threshold = **60 min**, scope = **Connect uniquement**
  → 48 % des problèmes résolus, ~2 % des revenus Connect affectés

---


