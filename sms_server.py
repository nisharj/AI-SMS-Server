from flask import Flask, request
import pickle
import requests

app = Flask(__name__)

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
# Telegram Settings
TELEGRAM_BOT_TOKEN="8632577960:AAHqsgGyQGEk1X6uJNs1puTZPevv1sztGRY"
CHAT_ID="5181502452"

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    
@app.route("/sms", methods=["POST"])
def receive_sms():
    data = request.json

    # print("FULL DATA:", data)

    message = (
        data.get("message")
        or data.get("sms")
        or data.get("text")
        or data.get("body")
        or ""
    )
    
    print("Received SMS:", message)
    
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)[0]
    prob = model.predict_proba(msg_vec)[0]
    confidence = max(prob) * 100
    
    
    if prediction == 1 and confidence > 80:
        send_telegram_message(f"🚨 SPAM SMS DETECTED\n\n{message}\n\nConfidence: {confidence:.2f}%")
        print("SPAM detected! Alert sent to Telegram.")
        
    return {"status": "success", "message": "SMS processed"}

app.run(host="0.0.0.0", port=5000)