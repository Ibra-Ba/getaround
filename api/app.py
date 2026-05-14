"""
Getaround Pricing API
Jedha Certification – Deployment Block

MODEL_SOURCE env var (default: "local"):
  local    → model.joblib embarqué dans l'image Docker  (garanti pour le jury)
  registry → MLflow Model Registry remote sur HF Space  (démo MLOps)

Variables d'environnement pour le mode "registry" :
  MLFLOW_TRACKING_URI    URL du Space MLflow HF
  MLFLOW_MODEL_NAME      nom du modèle enregistré   (défaut: getaround-pricing-gbr)
  MLFLOW_MODEL_ALIAS     alias à charger            (défaut: champion)
"""
import os
import logging
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Getaround Pricing API", docs_url=None, redoc_url=None)

FEATURES = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color",
    "car_type", "private_parking_available", "has_gps",
    "has_air_conditioning", "automatic_car", "has_getaround_connect",
    "has_speed_regulator", "winter_tires",
]

BASE         = os.path.dirname(__file__)
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")   # "local" | "registry"


def _load_model():
    """Charge le modèle depuis le joblib local ou depuis le MLflow Registry."""
    if MODEL_SOURCE == "registry":
        import mlflow.sklearn
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_name   = os.getenv("MLFLOW_MODEL_NAME",  "getaround-pricing-gbr")
        model_alias  = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
        mlflow.set_tracking_uri(tracking_uri)
        uri = f"models:/{model_name}@{model_alias}"
        log.info("Loading model from MLflow registry → %s", uri)
        return mlflow.sklearn.load_model(uri)

    path = os.path.join(BASE, "model.joblib")
    log.info("Loading model from local file → %s", path)
    return joblib.load(path)


# Chargement unique au démarrage du serveur
model    = _load_model()
encoders = joblib.load(os.path.join(BASE, "encoders.joblib"))
log.info("Model ready  (source=%s)", MODEL_SOURCE)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictInput(BaseModel):
    input: List[List[float]]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return (
        "<html><body style='font-family:sans-serif;padding:40px'>"
        "<h1>🚗 Getaround Pricing API</h1>"
        f"<p>Model source : <code>{MODEL_SOURCE}</code></p>"
        "<p>→ <a href='/docs'>Documentation</a> &nbsp;|&nbsp; "
        "POST <code>/predict</code></p>"
        "</body></html>"
    )


@app.post("/predict")
def predict(data: PredictInput):
    """
    Prédit le prix de location par jour (€).
    Accepte des vecteurs de 11 à 13 features (zero-padding automatique à 13).
    """
    rows  = [list(r) + [0] * (13 - len(r)) for r in data.input]
    X     = pd.DataFrame([r[:13] for r in rows], columns=FEATURES)
    preds = model.predict(X)
    return {"prediction": [round(float(p)) for p in preds]}


@app.get("/docs", response_class=HTMLResponse)
def docs():
    return """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Getaround Pricing API – Documentation</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 900px; margin: 0 auto; padding: 40px 24px;
           color: #1a1a2e; background: #f8fafc; }
    h1  { color: #16213e; border-bottom: 4px solid #e94560; padding-bottom: 14px; }
    h2  { color: #0f3460; margin-top: 48px; }
    h3  { color: #e94560; margin-top: 32px; }
    code { background: #eef2f7; padding: 2px 7px; border-radius: 4px;
           font-family: 'Courier New', monospace; font-size: 0.88em; }
    pre  { background: #1a1a2e; color: #e2e8f0; padding: 20px 24px;
           border-radius: 10px; overflow-x: auto;
           font-size: 0.86em; line-height: 1.65; }
    .badge { display: inline-block; padding: 3px 12px; border-radius: 20px;
             font-size: 0.80em; font-weight: 700; margin-right: 8px; }
    .post { background: #d4edda; color: #155724; }
    .get  { background: #cce5ff; color: #004085; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; }
    td { padding: 9px 14px; border-bottom: 1px solid #e2e8f0; font-size: 0.92em; }
    tr:nth-child(even) td { background: #f0f4f8; }
    .note { background: #fff3cd; border-left: 4px solid #ffc107;
            padding: 12px 16px; border-radius: 6px; margin: 20px 0; font-size: 0.91em; }
    .info { background: #d1ecf1; border-left: 4px solid #0f3460;
            padding: 12px 16px; border-radius: 6px; margin: 20px 0; font-size: 0.91em; }
    .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 24px; margin-bottom: 32px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
    footer { margin-top: 60px; color: #aaa; font-size: 0.82em; text-align: center; }
  </style>
</head>
<body>

<h1>🚗 Getaround Pricing API</h1>
<p>API de prédiction du <strong>prix de location optimal par jour</strong>.<br>
   Modèle : <em>Gradient Boosting Regressor</em> — R² = 0.74 | MAE ≈ 10 €/jour.</p>

<div class="info">
  <strong>Source du modèle</strong> — variable d'env <code>MODEL_SOURCE</code> :<br><br>
  <code>MODEL_SOURCE=local</code> (défaut) — <code>model.joblib</code> embarqué dans
  l'image Docker. Démarrage instantané, aucune dépendance réseau.<br><br>
  <code>MODEL_SOURCE=registry</code> — modèle chargé depuis le
  <strong>MLflow Model Registry</strong> (HF Space MLflow). Permet de déployer une
  nouvelle version sans rebuilder l'image : il suffit de promouvoir un run dans le Registry.
</div>

<hr>
<h2>Endpoints</h2>

<div class="card">
  <h3><span class="badge get">GET</span> /</h3>
  <p>Page d'accueil — indique la source de modèle active.</p>
</div>

<div class="card">
  <h3><span class="badge post">POST</span> /predict</h3>
  <p>Prédit le prix de location par jour (€) pour une ou plusieurs voitures.</p>
  <p><strong>Content-Type :</strong> <code>application/json</code></p>
  <pre>{
  "input": [
    [model_key, mileage, engine_power, fuel, paint_color,
     car_type, private_parking_available, has_gps,
     has_air_conditioning, automatic_car, has_getaround_connect,
     has_speed_regulator, winter_tires],
    ...
  ]
}</pre>

  <table>
    <tr><th>Index</th><th>Feature</th><th>Type</th><th>Encodage</th></tr>
    <tr><td>0</td><td>model_key</td><td>int</td><td>LabelEncoder : 0=Alfa Romeo … 27=Yamaha</td></tr>
    <tr><td>1</td><td>mileage</td><td>int</td><td>Kilométrage (km)</td></tr>
    <tr><td>2</td><td>engine_power</td><td>int</td><td>Puissance (cv)</td></tr>
    <tr><td>3</td><td>fuel</td><td>int</td><td>0=diesel · 1=electro · 2=hybrid_petrol · 3=petrol</td></tr>
    <tr><td>4</td><td>paint_color</td><td>int</td><td>0=beige · 1=black · 2=blue · 3=brown · 4=green · 5=grey · 6=orange · 7=red · 8=silver · 9=white</td></tr>
    <tr><td>5</td><td>car_type</td><td>int</td><td>0=convertible · 1=coupe · 2=estate · 3=hatchback · 4=sedan · 5=subcompact · 6=suv · 7=van</td></tr>
    <tr><td>6</td><td>private_parking_available</td><td>0/1</td><td>Parking privé</td></tr>
    <tr><td>7</td><td>has_gps</td><td>0/1</td><td>GPS</td></tr>
    <tr><td>8</td><td>has_air_conditioning</td><td>0/1</td><td>Climatisation</td></tr>
    <tr><td>9</td><td>automatic_car</td><td>0/1</td><td>Boîte automatique</td></tr>
    <tr><td>10</td><td>has_getaround_connect</td><td>0/1</td><td>Connect activé</td></tr>
    <tr><td>11</td><td>has_speed_regulator</td><td>0/1</td><td>Régulateur de vitesse</td></tr>
    <tr><td>12</td><td>winter_tires</td><td>0/1</td><td>Pneus hiver</td></tr>
  </table>

  <p><strong>Réponse :</strong></p>
  <pre>{"prediction": [119, 145]}</pre>

  <h4>curl</h4>
  <pre>curl -i -H "Content-Type: application/json" -X POST \\
     -d '{"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]}' \\
     https://YOUR_HF_URL/predict</pre>

  <h4>Python</h4>
  <pre>import requests
r = requests.post("https://YOUR_HF_URL/predict",
                  json={"input": [[7, 140000, 120, 0, 1, 2, 1, 1, 1, 0, 1, 1, 0]]})
print(r.json())   # {"prediction": [119]}</pre>

  <div class="note">
    ⚠️ Vecteurs de 11 à 13 features acceptés — zero-padding automatique à 13.
  </div>
</div>

<div class="card">
  <h3><span class="badge get">GET</span> /docs</h3>
  <p>Cette page de documentation.</p>
</div>

<hr>
<h2>Variables d'environnement (Space HF)</h2>
<table>
  <tr><th>Variable</th><th>Défaut</th><th>Description</th></tr>
  <tr><td><code>MODEL_SOURCE</code></td><td><code>local</code></td>
      <td><code>local</code> ou <code>registry</code></td></tr>
  <tr><td><code>MLFLOW_TRACKING_URI</code></td><td>—</td>
      <td>URL du Space MLflow (requis si <code>registry</code>)</td></tr>
  <tr><td><code>MLFLOW_MODEL_NAME</code></td><td><code>getaround-pricing-gbr</code></td>
      <td>Nom du modèle dans le Registry</td></tr>
  <tr><td><code>MLFLOW_MODEL_ALIAS</code></td><td><code>champion</code></td>
      <td>Alias promu (champion, challenger…)</td></tr>
</table>

<footer>Getaround Pricing API — Jedha Bootcamp Certification — 2024</footer>
</body>
</html>"""
