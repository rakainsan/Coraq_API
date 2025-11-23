from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load ENV
load_dotenv()

# TOKEN
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ============================
#  GROQ LLM
# ============================
from groq import Groq
groq_client = Groq(api_key=GROQ_API_KEY)

def ask_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Asisten AI khusus proyek prediksi limbah."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# ============================
#  FLASK APP
# ============================

app = Flask(__name__)
CORS(app)

svr = joblib.load("model_svr.pkl")
svm = joblib.load("model_svm.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")


def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)


@app.route('/')
def home():
    return jsonify({"status": "API berjalan"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        sensor_count = float(data['sensor_count'])
        start_date = pd.to_datetime(data['date']).tz_localize(None)

        predictions = []
        for i in range(30):
            current_date = start_date + timedelta(days=i)
            dow = current_date.weekday()
            month = current_date.month
            doy = current_date.dayofyear

            X = np.array([[sensor_count, dow, month, doy]])
            X_scaled = scaler_X.transform(X)

            y_pred_scaled = svr.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )[0][0]

            anomaly = int(svm.predict(X_scaled)[0])

            predictions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_volume_liters": round(float(y_pred), 2),
                "is_anomaly": bool(anomaly)
            })

            # Alert hari pertama
            if i == 0 and y_pred > 23000:
                msg = (
                    f"🚨 *PERINGATAN!*\n"
                    f"Tanggal: {current_date.strftime('%d-%m-%Y')}\n"
                    f"Sensor: {sensor_count}\n"
                    f"Volume Prediksi: {round(y_pred, 2)} liter"
                )
                send_telegram_alert(msg)

        return jsonify({
            "sensor_count": sensor_count,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/llm', methods=['POST'])
def llm():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if prompt == "":
        return jsonify({"error": "prompt kosong"}), 400

    result = ask_groq(prompt)
    return jsonify({"response": result})


@app.route('/prediction')
def prediction_page():
    return render_template("prediction.html")


if __name__ == "__main__":
    # Untuk hosting (Railway/Render) memakai PORT otomatis
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
