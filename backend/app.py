from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224  # Must match model training size
MODEL_PATH = "dermacare_model.h5"

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

classes = ['Acne', 'Drug Reaction', 'Eczema', 'Psoriasis']

suggestions = {
    "Acne": "Keep skin clean. Avoid oily food. Use salicylic acid products.",
    "Drug Reaction": "Stop medication immediately and consult a doctor.",
    "Eczema": "Moisturize regularly. Avoid harsh soaps.",
    "Psoriasis": "Use medicated creams. Consult dermatologist."
}

# ==============================
# DATABASE INIT
# ==============================
def init_db():
    conn = sqlite3.connect("dermacare.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        disease TEXT,
        confidence REAL,
        image TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ==============================
# HELPER FUNCTIONS
# ==============================
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================
# ROUTES
# ==============================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    # ✅ File type validation inside function
    if not allowed_file(file.filename):
        return render_template('index.html', prediction="Invalid file type. Only images allowed.")

    # Ensure static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Image preprocessing
    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    confidence = round(100 * np.max(prediction), 2)
    predicted_class = classes[np.argmax(prediction)]

    # Save to DB
    conn = sqlite3.connect("dermacare.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (date, disease, confidence, image)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        predicted_class,
        confidence,
        file.filename
    ))
    conn.commit()
    conn.close()

    return render_template(
        'index.html',
        prediction=predicted_class,
        confidence=confidence,
        suggestion=suggestions[predicted_class],
        img_path=filepath
    )


@app.route('/history')
def history():
    conn = sqlite3.connect("dermacare.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return render_template("history.html", data=data)


@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect("dermacare.db")
    cursor = conn.cursor()

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    row = cursor.fetchone()
    total_predictions = row[0] if row is not None else 0

    # Most common disease
    cursor.execute("SELECT disease, COUNT(*) as count FROM predictions GROUP BY disease ORDER BY count DESC LIMIT 1")
    most_common = cursor.fetchone()
    if most_common:
        most_common_disease = most_common[0]
        most_common_count = most_common[1]
    else:
        most_common_disease = "N/A"
        most_common_count = 0

    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_row = cursor.fetchone()
    avg_confidence = round(avg_row[0], 2) if avg_row and avg_row[0] is not None else 0

    # Disease frequency for chart
    cursor.execute("SELECT disease, COUNT(*) FROM predictions GROUP BY disease")
    chart_data = cursor.fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        most_common_disease=most_common_disease,
        most_common_count=most_common_count,
        avg_confidence=avg_confidence,
        chart_data=chart_data
    )

# ==============================
# RUN APP
# ==============================
if __name__ == '__main__':
    app.run(debug=True)