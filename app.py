from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import requests


app = Flask(__name__)
CORS(app)  # Izinkan Frontend React mengakses Backend ini

MODEL_URL = os.getenv("MODEL_URL")
SCALER_URL = os.getenv("SCALER_URL")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model_rf.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

model = None
scaler = None


def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def load_artifacts():
    global model, scaler

    if model is None or scaler is None:
        download_file(MODEL_URL, MODEL_PATH)
        download_file(SCALER_URL, SCALER_PATH)

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)


# --- ROUTES ---


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "StressMonitor API is Online",
            "model": "Loaded" if model else "Not Loaded",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    load_artifacts()

    try:
        data = request.json

        # 1. Siapkan data sesuai urutan fitur saat training
        # Perhatikan urutan kolom harus SAMA PERSIS dengan saat Anda melatih model
        # Asumsi urutan: x, y, z, bvp, eda, hr (sesuaikan jika berbeda)
        features = [
            float(data["x"]),
            float(data["y"]),
            float(data["z"]),
            float(data["bvp"]),
            float(data["eda"]),
            float(data["hr"]),
            float(np.sqrt(data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2)),
        ]

        # 2. Buat DataFrame/Array
        features_array = np.array([features])

        # 3. Scaling (Standarisasi)
        features_scaled = scaler.transform(features_array)

        # 4. Prediksi
        prediction = model.predict(features_scaled)

        # 5. Konversi hasil
        result_label = int(prediction[0])

        # Opsional: Logika mapping label manual jika tidak pakai label_encoder
        risk_mapping = {0: "Baseline", 1: "Elevated", 2: "High Risk"}
        risk_text = risk_mapping.get(result_label, "Unknown")

        return jsonify(
            {
                "label": result_label,
                "riskLevel": risk_text,
                "analysis": f"Model Prediction: Class {result_label} based on HR {features[5]} and EDA {features[4]}",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Endpoint untuk mengambil data dummy/history (Opsional, menggantikan MOCK_DATASET)
@app.route("/history", methods=["GET"])
def get_history():
    try:
        # Ambil parameter query untuk pagination (opsional)
        limit = request.args.get("limit", type=int)
        offset = request.args.get("offset", default=0, type=int)

        # Jika limit tidak diberikan atau -1, ambil semua data
        if limit is None or limit == -1:
            df = pd.read_csv("data/upsample_resampled.csv")
            return jsonify(df.to_dict(orient="records"))

        # Jika limit diberikan, gunakan pagination
        df = pd.read_csv(
            "data/upsample_resampled.csv", skiprows=range(1, offset + 1), nrows=limit
        )

        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- TAMBAHKAN ROUTE BARU INI ---
@app.route("/pca", methods=["GET"])
def get_pca():
    try:
        # 1. Load Data
        df = pd.read_csv("data/upsample_resampled.csv")  # Sesuaikan path

        # 2. Fitur yang akan di-analisis (Drop timestamp & label)
        # Pastikan nama kolom sesuai dengan CSV Anda
        feature_cols = ["x", "y", "z", "bvp", "eda", "hr"]

        # Normalisasi nama kolom dataset agar lowercase
        df.columns = [c.lower() for c in df.columns]

        # Cek apakah kolom lengkap
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # 3. Standard Scaling (Sangat penting untuk PCA)
        x = df.loc[:, feature_cols].values
        x = StandardScaler().fit_transform(x)

        # 4. PCA 3 Dimensi (Komputasi x, y, z baru)
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(x)

        # 5. Gabungkan hasil dengan Label asli
        pca_data = []
        labels = df["label"].values
        ids = df["id"].values if "id" in df.columns else range(len(df))

        # Batasi jumlah data untuk mencegah timeout
        max_rows = 10000
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)

        # Sampling agar tidak terlalu berat dikirim ke JSON (Max 500 titik)
        step = max(1, len(df) // 500)

        for i in range(0, len(df), step):
            pca_data.append(
                {
                    "id": int(ids[i]),
                    "pc1": float(principalComponents[i][0]),
                    "pc2": float(principalComponents[i][1]),
                    "pc3": float(principalComponents[i][2]),
                    "label": int(labels[i]),
                }
            )

        # Hitung Explained Variance (Seberapa banyak info yang disimpan)
        variance = pca.explained_variance_ratio_

        return jsonify(
            {
                "data": pca_data,
                "variance": {
                    "pc1": float(variance[0]),
                    "pc2": float(variance[1]),
                    "pc3": float(variance[2]),
                    "total": float(sum(variance)),
                },
            }
        )

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
