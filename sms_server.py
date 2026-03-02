from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import os
import pickle
import requests
import csv
from io import StringIO
from flask_socketio import SocketIO
from flask import Response

app = Flask(__name__)

# TEMP LOCAL LOGIN (remove before deploying)
os.environ["ADMIN_USER"] = "admin"
os.environ["ADMIN_PASS"] = "admin123"


app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sms_logs.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
    
# Telegram Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

class SMSLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(100))
    message = db.Column(db.Text)
    spam_prob = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    
with app.app_context():
    db.create_all()

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function
    
@app.route("/sms", methods=["POST"])
def receive_sms():
    data = request.json

    message = (
        data.get("message")
        or data.get("sms")
        or data.get("text")
        or data.get("body")
        or ""
    )

    sender = data.get("from", "Unknown")

    print("Received SMS:", message)

    msg_vec = vectorizer.transform([message])
    prob = model.predict_proba(msg_vec)[0]
    spam_prob = prob[1]
    confidence = spam_prob * 100

    # Risk classification
    if spam_prob > 0.90:
        risk = "HIGH"
    elif spam_prob > 0.70:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Save to database
    new_log = SMSLog(
        sender=sender,
        message=message,
        spam_prob=confidence,
        risk_level=risk
    )

    db.session.add(new_log)
    db.session.commit()

    # Telegram alert for HIGH only
    if risk == "HIGH":
        send_telegram_message(
            f"🚨 HIGH RISK SMS\n\n{message}\n\nConfidence: {confidence:.2f}%"
        )
        print("High-risk alert sent.")

    return {"status": "success"}

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        print("Entered:", username, password)
        print("Expected:", os.getenv("ADMIN_USER"), os.getenv("ADMIN_PASS"))

        if username == os.getenv("ADMIN_USER") and password == os.getenv("ADMIN_PASS"):
            session["logged_in"] = True
            print("LOGIN SUCCESS")
            return redirect(url_for("dashboard"))
        else:
            print("LOGIN FAILED")
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")



@app.route("/dashboard")
@login_required
def dashboard():
    logs = SMSLog.query.order_by(SMSLog.timestamp.desc()).limit(100).all()

    total = SMSLog.query.count()
    high = SMSLog.query.filter_by(risk_level="HIGH").count()
    medium = SMSLog.query.filter_by(risk_level="MEDIUM").count()
    low = SMSLog.query.filter_by(risk_level="LOW").count()

    return render_template(
        "dashboard.html",
        logs=logs,
        total=total,
        high=high,
        medium=medium,
        low=low,
        model_status="ACTIVE",
        db_status="CONNECTED",
        telegram_status="ENABLED" if TELEGRAM_BOT_TOKEN else "DISABLED"
    )
    

@app.route("/export")
@login_required
def export_csv():
    logs = SMSLog.query.order_by(SMSLog.timestamp.desc()).all()

    si = StringIO()
    writer = csv.writer(si)

    writer.writerow(["Time", "Sender", "Message", "Spam %", "Risk"])

    for log in logs:
        writer.writerow([
            log.timestamp,
            log.sender,
            log.message,
            f"{log.spam_prob:.2f}",
            log.risk_level
        ])

    output = si.getvalue()

    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sms_logs.csv"}
    )
    
    
# app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    socketio.run(app, debug=True)