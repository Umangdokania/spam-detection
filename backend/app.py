from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# database
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT,
    result TEXT,
    date TEXT
)
""")

conn.commit()

@app.route("/")
def home():
    return "Spam Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    message = data["message"]

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    result = "SPAM" if prediction == 1 else "NOT SPAM"

    cursor.execute(
        "INSERT INTO history (message, result, date) VALUES (?, ?, ?)",
        (message, result, str(datetime.now()))
    )

    conn.commit()

    return jsonify({
        "result": result
    })

@app.route("/history")
def history():

    cursor.execute("SELECT message, result FROM history ORDER BY id DESC")

    rows = cursor.fetchall()

    return jsonify([
        {"message": row[0], "result": row[1]} for row in rows
    ])

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
