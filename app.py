from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
import requests
from flask_cors import CORS
import os
from groq import Groq

app = Flask(__name__)
CORS(app)

# ====================
# LOAD ML MODELS
# ====================

svr = joblib.load("model_svr.pkl")
svm = joblib.load("model_svm.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ====================
# ENVIRONMENT VARIABLES
# ====================

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN:
    print("ERROR: BOT_TOKEN tidak ditemukan!")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY tidak ditemukan!")

# ====================
# LLM GROQ
# ====================

groq_client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt):
    system_prompt = """
    Kamu adalah CoraqBot, chatbot IoT untuk monitoring limbah air batik.
    Kamu memahami sensor pH, turbidity, TDS, suhu, prediksi limbah dan anomali.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",    "content": prompt}
            ]
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"LLM ERROR: {str(e)}"


# ====================
# TELEGRAM SENDER
# ====================

def send_telegram_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    r = requests.post(url, data=payload)


# ====================
# HOME
# ====================

@app.route("/")
def home():
    return jsonify({
        "status": "API aktif",
        "telegram_webhook": "OK",
        "groq_llm": "siap digunakan"
    })

# ====================
# ML PREDICTION ENDPOINT
# ====================

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        sensor_count = float(data['sensor_count'])
        date_str = data['date']
        start_date = pd.to_datetime(date_str).tz_localize(None)

        num_days = 30
        predictions = []

        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            dayofweek = current_date.weekday()
            month = current_date.month
            dayofyear = current_date.timetuple().tm_yday

            X = np.array([[sensor_count, dayofweek, month, dayofyear]])
            X_scaled = scaler_X.transform(X)

            y_pred_scaled = svr.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )[0][0]

            is_anomaly = int(svm.predict(X_scaled)[0])

            predictions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_volume_liters": round(float(y_pred), 2),
                "is_anomaly": bool(is_anomaly)
            })

        return jsonify({
            "start_date": start_date.strftime("%Y-%m-%d"),
            "sensor_count": sensor_count,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ====================
# LLM ASK ENDPOINT
# ====================

@app.route("/ask", methods=["POST"])
def ask():
    prompt = request.json.get("prompt", "")
    answer = ask_llm(prompt)
    return jsonify({"response": answer})

# ====================
# TELEGRAM WEBHOOK ENDPOINT
# ====================

@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    data = request.get_json()

    if "message" not in data:
        return jsonify({"status": "ignored"})

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    # Handle /start
    if text.lower() == "/start":
        welcome = (
            "👋 Halo! Aku *CoraqBot*.\n"
            "Aku terhubung dengan sistem IoT limbah batik.\n"
            "Tanya apa saja tentang sensor, limbah batik, prediksi, atau anomali!"
        )
        send_telegram_message(chat_id, welcome)
        return jsonify({"status": "ok"})

    # Pertanyaan biasa → LLM Groq
    reply = ask_llm(text)
    send_telegram_message(chat_id, reply)

    return jsonify({"status": "ok"})

# ====================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
