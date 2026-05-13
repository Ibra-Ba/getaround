---
title: Getaround MLflow
emoji: 📈
colorFrom: blue
colorTo: red
sdk: docker
app_port: 5000
pinned: false
---

# Getaround – MLflow Tracking Server

Serveur MLflow distant pour tracker les expériences de pricing optimization.

---

## Architecture de persistance

```
MLflow Server (ce Space)
       │
       ├── Backend store  → PostgreSQL NeonDB
       │                    métadonnées : runs, params, metrics, tags
       │
       └── Artifact store → AWS S3
                            s3://cni-fraud-detection/getaround-mlflow/
                            fichiers lourds : model.joblib, plots, etc.
```

Les deux composantes sont **externes au Space** → les données survivent
aux redémarrages et à la mise en veille du Space free tier.

---

## Setup – étapes à suivre

### 1. Créer la base PostgreSQL sur NeonDB

1. Aller sur [neon.tech](https://neon.tech) → créer un projet `getaround-mlflow`
2. Créer une base `mlflow_db`
3. Récupérer la connection string :
   ```
   postgresql://user:pass@ep-xxx.region.aws.neon.tech/mlflow_db?sslmode=require
   ```

### 2. Préparer le préfixe S3

Le bucket `cni-fraud-detection` existe déjà.
Les artefacts MLflow seront isolés sous le préfixe `getaround-mlflow/` :

```
s3://cni-fraud-detection/getaround-mlflow/
```

Aucune création de bucket nécessaire.

### 3. Configurer les Secrets du Space HF

Dans **Settings → Variables and secrets** de ce Space, ajouter :

| Secret | Valeur |
|---|---|
| `DATABASE_URL` | `postgresql://user:pass@host/mlflow_db?sslmode=require` |
| `ARTIFACT_ROOT` | `s3://cni-fraud-detection/getaround-mlflow/` |
| `AWS_ACCESS_KEY_ID` | ta clé AWS |
| `AWS_SECRET_ACCESS_KEY` | ta clé secrète AWS |
| `AWS_DEFAULT_REGION` | `eu-west-3` |

### 4. Configurer le notebook local

Remplir `.env` (copié depuis `.env.example`) :

```bash
MLFLOW_TRACKING_URI      = https://YOUR_USERNAME-getaround-mlflow.hf.space
MLFLOW_TRACKING_USERNAME = YOUR_HF_USERNAME
MLFLOW_TRACKING_PASSWORD = YOUR_HF_TOKEN
MLFLOW_EXPERIMENT_NAME   = getaround-pricing

AWS_ACCESS_KEY_ID        = xxx
AWS_SECRET_ACCESS_KEY    = xxx
AWS_DEFAULT_REGION       = eu-west-3
```

### 5. Configurer l'API (mode registry)

Dans les Secrets du Space API, ajouter :

| Secret | Valeur |
|---|---|
| `MODEL_SOURCE` | `registry` |
| `MLFLOW_TRACKING_URI` | `https://YOUR_USERNAME-getaround-mlflow.hf.space` |
| `MLFLOW_MODEL_NAME` | `getaround-pricing-gbr` |
| `MLFLOW_MODEL_ALIAS` | `champion` |
| `AWS_ACCESS_KEY_ID` | ta clé AWS |
| `AWS_SECRET_ACCESS_KEY` | ta clé secrète AWS |
| `AWS_DEFAULT_REGION` | `eu-west-3` |


---

## Promouvoir un modèle en production

Après un nouveau run dans le notebook :

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# Enregistrer dans le Registry
mv = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="getaround-pricing-gbr"
)

# Promouvoir en champion
client.set_registered_model_alias(
    name="getaround-pricing-gbr",
    alias="champion",
    version=mv.version
)
```

Puis **restart du Space API** → il charge automatiquement le nouveau champion depuis S3.

---

## Déploiement depuis le repo GitHub

```bash
git remote add origin-hf-mlflow \
    https://huggingface.co/spaces/YOUR_USERNAME/getaround-mlflow

git push mlflow origin-hf-mlflow main
```
