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

# == Load Models ==
svr = joblib.load("model_svr.pkl")
svm = joblib.load("model_svm.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# == Load Environment Variables ==
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# == Init Groq LLM ==
groq_client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt):
    """ Query LLM Groq dengan persona CoraqBot """

    system_prompt = """
    Kamu adalah CoraqBot, sebuah chatbot cerdas berbasis IoT yang dirancang khusus
    untuk mendukung sistem pengolahan limbah air batik.

    Kamu terintegrasi dengan sensor IoT seperti pH, suhu, kekeruhan, TDS,
    serta modul prediksi dan deteksi anomali limbah.

    Tugasmu:
    - Menjawab pertanyaan tentang pengolahan limbah batik
    - Menjelaskan fungsi sensor pH, suhu, turbidity, TDS, dan sensor lain
    - Memberikan edukasi tentang kualitas air limbah
    - Memberikan insight terhadap data prediksi atau anomali
    - Menjawab dengan bahasa Indonesia yang ramah dan mudah dipahami

    Jangan memberikan saran medis atau legal.
    Fokus hanya pada domain limbah air batik, IoT monitoring, dan penjelasan data.
    """

    completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message["content"]

# == Telegram Alert ==
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=payload)

@app.route('/')
def home():
    return jsonify({"status": "API jalan bro", "llm_status": "Groq siap dipakai"})

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
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

            is_anomaly = int(svm.predict(X_scaled)[0])

            predictions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_volume_liters": round(float(y_pred), 2),
                "is_anomaly": bool(is_anomaly)
            })

            if i == 0 and y_pred > 23000:
                message = (
                    f"🚨 *PERINGATAN TINGGI VOLUME LIMBAH!*\n"
                    f"Tanggal: {current_date.strftime('%d-%m-%Y')}\n"
                    f"Sensor: {sensor_count}\n"
                    f"Prediksi Volume: {round(y_pred, 2)} liter\n"
                    f"Status: ⚠️ Melebihi ambang batas 23.000 liter!"
                )
                send_telegram_alert(message)

        response = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "month": start_date.strftime("%B %Y"),
            "sensor_count": sensor_count,
            "predictions": predictions
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/ask", methods=["POST"])
def ask():
    """ API untuk query LLM Groq """
    prompt = request.json.get("prompt", "")
    answer = ask_llm(prompt)
    return jsonify({"response": answer})

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

# ===========================
# == TELEGRAM WEBHOOK API ==
# ===========================

@app.route(f"/webhook", methods=["POST"])
def telegram_webhook():
    """ Endpoint menerima pesan dari Telegram """
    data = request.get_json()

    # Pastikan itu pesan teks
    if "message" not in data:
        return jsonify({"status": "ignored"}), 200

    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"].get("text", "")

    # Jika user kirim "/start"
    if user_text == "/start":
        welcome = (
            "👋 Halo! Aku *CoraqBot*.\n"
            "Aku terhubung dengan sistem IoT limbah batik.\n"
            "Tanya apa saja tentang sensor, limbah batik, prediksi, atau anomali!"
        )
        send_telegram_alert_custom(chat_id, welcome)
        return jsonify({"status": "ok"})

    # == Kirim ke Groq ==
    try:
        answer = ask_llm(user_text)
        send_telegram_alert_custom(chat_id, answer)
    except Exception as e:
        send_telegram_alert_custom(chat_id, f"⚠️ Terjadi error LLM: {str(e)}")

    return jsonify({"status": "processed"}), 200


def send_telegram_alert_custom(chat_id, message):
    """ Fungsi kirim balasan langsung ke user """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
