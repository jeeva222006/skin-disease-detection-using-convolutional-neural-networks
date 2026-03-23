from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import pymysql
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "skinsense_secret_2024"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = "uploads"
MODEL_PATH         = "model/skin_model.h5"
LABELS_PATH        = "model/labels.txt"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (64, 64)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)
# ─── DATABASE ────────────────────────────────────────────────────────────────
def get_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="skin_db",
        cursorclass=pymysql.cursors.DictCursor
    )

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
cnn_model = None
labels    = []

def load_cnn_model():
    global cnn_model, labels
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        cnn_model = load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            labels = [l.strip() for l in f.readlines()]
        print("✅ CNN Model loaded successfully!")
    else:
        print("⚠️  Model not found — run train_model.py first")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    img        = image.load_img(img_path, target_size=IMG_SIZE)
    img_array  = image.img_to_array(img) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)
    preds      = cnn_model.predict(img_array)
    class_idx  = int(np.argmax(preds[0]))
    confidence = round(float(preds[0][class_idx]) * 100, 2)
    return labels[class_idx], confidence

# ─── SERVE UPLOADED FILES ─────────────────────────────────────────────────────
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ─── HOME ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

# ─── REGISTER ────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form["name"].strip()
        email    = request.form["email"].strip()
        password = generate_password_hash(request.form["password"])
        try:
            db = get_db()
            cur = db.cursor()
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            if cur.fetchone():
                flash("Email already registered.", "danger")
            else:
                cur.execute("INSERT INTO users (name,email,password) VALUES (%s,%s,%s)",
                            (name, email, password))
                db.commit()
                flash("Account created! Please login.", "success")
                return redirect(url_for("login"))
        except Exception as e:
            flash(f"Database error: {e}", "danger")
        finally:
            db.close()
    return render_template("register.html")

# ─── LOGIN ───────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form["email"].strip()
        password = request.form["password"]
        try:
            db = get_db()
            cur = db.cursor()
            cur.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cur.fetchone()
            db.close()
            if user and check_password_hash(user["password"], password):
                session["user_id"]   = user["id"]
                session["user_name"] = user["name"]
                flash(f"Welcome back, {user['name']}!", "success")
                return redirect(url_for("upload"))
            flash("Invalid email or password.", "danger")
        except Exception as e:
            flash(f"Database error: {e}", "danger")
    return render_template("login.html")

# ─── LOGOUT ──────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    name = session.get("user_name", "")
    session.clear()
    flash(f"Goodbye, {name}!", "success")
    return redirect(url_for("home"))

# ─── UPLOAD & PREDICT ────────────────────────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    result     = None
    confidence = None
    img_name   = None
    doctors    = []

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
        elif not allowed_file(file.filename):
            flash("Only JPG, JPEG, PNG files allowed.", "danger")
        else:
            filename  = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            img_name = filename

            if cnn_model:
                result, confidence = predict_disease(save_path)
                try:
                    db = get_db()
                    cur = db.cursor()
                    cur.execute("SELECT * FROM doctors WHERE disease_class=%s", (result,))
                    doctors = cur.fetchall()
                    db.close()
                except:
                    pass
            else:
                flash("Model not loaded. Run train_model.py first.", "warning")

    return render_template("upload.html",
                           result=result,
                           confidence=confidence,
                           img_name=img_name,
                           doctors=doctors)

# ─── FEEDBACK ────────────────────────────────────────────────────────────────
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    if request.method == "POST":
        msg = request.form["message"].strip()
        if msg:
            try:
                db = get_db()
                cur = db.cursor()
                cur.execute("INSERT INTO feedback (user_id,message) VALUES (%s,%s)",
                            (session["user_id"], msg))
                db.commit()
                db.close()
                flash("Thank you for your feedback!", "success")
            except Exception as e:
                flash(f"Error: {e}", "danger")
        else:
            flash("Message cannot be empty.", "danger")
    return render_template("feedback.html")

# ─── ADMIN LOGIN ─────────────────────────────────────────────────────────────
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if session.get("admin"):
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin credentials.", "danger")
    return render_template("admin_login.html")

# ─── ADMIN DASHBOARD ─────────────────────────────────────────────────────────
@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM users ORDER BY id DESC")
        users = cur.fetchall()
        cur.execute("SELECT * FROM doctors ORDER BY id DESC")
        doctors = cur.fetchall()
        cur.execute("""
            SELECT f.id, f.message, f.created_at, u.name
            FROM feedback f
            JOIN users u ON f.user_id = u.id
            ORDER BY f.id DESC
        """)
        feedbacks = cur.fetchall()
        db.close()
    except Exception as e:
        flash(f"DB Error: {e}", "danger")
        users = doctors = feedbacks = []
    return render_template("admin_dashboard.html",
                           users=users, doctors=doctors, feedbacks=feedbacks)

# ─── ADMIN ADD DOCTOR ─────────────────────────────────────────────────────────
@app.route("/admin/add_doctor", methods=["POST"])
def add_doctor():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    name    = request.form["name"].strip()
    spec    = request.form["specialization"].strip()
    disease = request.form["disease_class"]
    contact = request.form["contact"].strip()
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("INSERT INTO doctors (name,specialization,disease_class,contact) VALUES (%s,%s,%s,%s)",
                    (name, spec, disease, contact))
        db.commit()
        db.close()
        flash("Doctor added successfully!", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("admin_dashboard"))

# ─── ADMIN DELETE DOCTOR ──────────────────────────────────────────────────────
@app.route("/admin/delete_doctor/<int:doc_id>")
def delete_doctor(doc_id):
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("DELETE FROM doctors WHERE id=%s", (doc_id,))
        db.commit()
        db.close()
        flash("Doctor removed.", "success")
    except Exception as e:
        flash(f"Error: {e}", "danger")
    return redirect(url_for("admin_dashboard"))

# ─── ADMIN LOGOUT ─────────────────────────────────────────────────────────────
@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_cnn_model()
    app.run(debug=True, port=5000)