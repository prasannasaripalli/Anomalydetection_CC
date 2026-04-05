import os, json, pickle, smtplib
from email.mime.text import MIMEText
import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

FEATURES = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
    "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_pipeline.pkl")
EMAIL_ON = os.getenv("ENABLE_EMAIL_ALERT", "true").lower() in {"1","true","yes","y","on"}
_model = None


def alert_body(inp: dict, is_fraud: bool, score_val: float) -> str:
    time_val = float(inp.get("Time", 0.0))
    amt_val = float(inp.get("Amount", 0.0))
    label = "ANOMALY (possible fraud)" if is_fraud else "NORMAL"

    return (
        "Fraud/Anomaly Alert\n\n"
        f"Anomaly score: {score_val:.6f}\n"
        f"Prediction: {label}\n\n"
        "Quick details:\n"
        f"Time: {time_val:,.2f}\n"
        f"Amount: {amt_val:,.2f}\n"
    )


def send_email(subject: str, body: str):
    sender = os.getenv("SENDER_EMAIL")
    pwd = os.getenv("SENDER_PASSWORD")
    to = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    with smtplib.SMTP(os.getenv("SMTP_SERVER", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(sender, pwd)
        s.sendmail(sender, to, msg.as_string())


@app.route(route="score", methods=["POST"])
def score(req: func.HttpRequest) -> func.HttpResponse:
    global _model
    try:
        data = req.get_json()

        miss = [f for f in FEATURES if f not in data]
        if miss:
            return func.HttpResponse(json.dumps({"error": f"Missing fields: {miss}"}), status_code=400)

        if _model is None:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)

        row = [[float(data[f]) for f in FEATURES]]
        scaler = _model.named_steps["scaler"]
        iso = _model.named_steps["model"]

        x = scaler.transform(row)
        raw = int(iso.predict(x)[0])  # +1 normal, -1 anomaly
        score_val = float(iso.decision_function(x)[0])

        is_fraud = (raw == -1)
        out = {"is_fraud": is_fraud, "anomaly_score": score_val, "email_sent": False}

        if EMAIL_ON and is_fraud:
            body = alert_body(data, is_fraud, score_val)
            send_email("Fraud/Anomaly Alert", body)
            out["email_sent"] = True

        return func.HttpResponse(json.dumps(out), mimetype="application/json", status_code=200)

    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), mimetype="application/json", status_code=500)