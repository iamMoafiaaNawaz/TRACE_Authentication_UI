import os
import random
import smtplib
import datetime
import base64
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import bcrypt
import jwt
import cv2
import numpy as np
from email.mime.text import MIMEText
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
from email_validator import validate_email, EmailNotValidError
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/trace_db")
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

client = MongoClient(MONGO_URI)
db = client.get_database()

# Collections
users_collection = db["users"]
admins_collection = db["admins"]
pending_collection = db["pending_users"]
reset_collection = db["password_resets"]
analyses_collection = db["analyses"]
def load_app_timezone():
    tz_name = os.getenv("APP_TIMEZONE", "Asia/Karachi")
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        print(f"Warning: Timezone '{tz_name}' not found. Falling back to UTC.")
        return datetime.timezone.utc


APP_TIMEZONE = load_app_timezone()

# AI Model Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model/trace_model.pth'
HAIR_MODEL_PATH = 'model/chimaera_v2_final.h5'
STRICT_IMAGE_VALIDATION = os.getenv("STRICT_IMAGE_VALIDATION", "false").lower() == "true"
ALLOW_HAIR_FALLBACK = os.getenv("ALLOW_HAIR_FALLBACK", "false").lower() == "true"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# AI Model Setup
print("Loading AI Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
hair_model = None
hair_model_backend = "opencv_dullrazor"
hair_model_error = None

CLASSES = ['BCC', 'BKL', 'MEL', 'NV']

CLASS_INFO = {
    'BCC': {'name': 'Basal Cell Carcinoma', 'type': 'Malignant (Cancerous)', 'severity': 'High'},
    'BKL': {'name': 'Benign Keratosis', 'type': 'Benign (Safe)', 'severity': 'Low'},
    'MEL': {'name': 'Melanoma', 'type': 'Malignant (Cancerous)', 'severity': 'Critical'},
    'NV': {'name': 'Nevus (Mole)', 'type': 'Benign (Safe)', 'severity': 'Low'}
}

try:
    if os.path.exists(MODEL_PATH):
        loaded_obj = torch.load(MODEL_PATH, map_location=device)

        # Accept only ready-to-run nn.Module objects.
        if isinstance(loaded_obj, nn.Module):
            model = loaded_obj.to(device)
            model.eval()
            print("Model Loaded Successfully")
        elif isinstance(loaded_obj, dict) and isinstance(loaded_obj.get("model"), nn.Module):
            model = loaded_obj["model"].to(device)
            model.eval()
            print("Model Loaded Successfully (from checkpoint['model'])")
        else:
            model = None
            print(
                "Invalid model format: expected torch.nn.Module "
                "or checkpoint with key 'model' as nn.Module"
            )
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("System running with prediction disabled until model is available")
except Exception as e:
    model = None
    print(f"Error Loading Model: {e}")

try:
    if os.path.exists(HAIR_MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            hair_model = load_model(HAIR_MODEL_PATH, compile=False)
            hair_model_backend = "keras_h5"
            hair_model_error = None
            print("Hair removal model loaded successfully")
        except Exception as tf_error:
            hair_model = None
            hair_model_backend = "opencv_dullrazor_fallback"
            if "No module named 'tensorflow'" in str(tf_error):
                hair_model_error = (
                    "TensorFlow is not installed in current backend environment. "
                    "Using OpenCV fallback."
                )
            else:
                hair_model_error = str(tf_error)
            print(f"Hair model not loaded via TensorFlow/Keras. Fallback enabled: {tf_error}")
    else:
        print(f"Hair model not found at {HAIR_MODEL_PATH}. Fallback enabled.")
except Exception as e:
    hair_model = None
    hair_model_backend = "opencv_dullrazor"
    hair_model_error = str(e)
    print(f"Error preparing hair-removal model: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_auth_token_from_request():
    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not token:
        return None, (jsonify({"error": "No token provided"}), 401)

    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded, None
    except jwt.ExpiredSignatureError:
        return None, (jsonify({"error": "Token expired"}), 401)
    except jwt.InvalidTokenError:
        return None, (jsonify({"error": "Invalid token"}), 401)


def require_admin_from_request():
    decoded, error = decode_auth_token_from_request()
    if error:
        return None, error
    if decoded.get("role") != "Admin":
        return None, (jsonify({"error": "Admin access required"}), 403)
    return decoded, None


def validate_dermoscopic_image(filepath):
    """Validate uploaded image quality.
    Strict dermoscopic heuristics are optional and disabled by default because they can
    reject true dermoscopic images in real-world lighting conditions.
    """
    try:
        img = cv2.imread(filepath)
        if img is None:
            return False, "Invalid image file"

        h, w = img.shape[:2]
        if w < 96 or h < 96:
            return False, f"Image too small ({w}x{h}). Minimum 96x96 required"

        # Base quality checks (always on)
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 5.0:
            return False, "Image aspect ratio too extreme"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(np.mean(gray))
        if avg_brightness < 8:
            return False, "Image too dark"
        if avg_brightness > 247:
            return False, "Image too bright"

        # In hair-removal-only mode, avoid over-rejecting true clinical images.
        if not STRICT_IMAGE_VALIDATION:
            return True, "Quality checks passed (non-strict mode)"

        # Optional strict dermoscopic checks
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pct = (np.sum(skin_mask > 0) / (h * w)) * 100

        if skin_pct < 1:
            return False, "No skin tones detected in image"

        return True, "Strict dermoscopic checks passed"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def is_skin_lesion_prediction(probabilities, threshold=0.45):
    top_confidence = float(torch.max(probabilities).item())
    return top_confidence >= threshold, top_confidence


def remove_hair_dullrazor(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(image_bgr, hair_mask, 3, cv2.INPAINT_TELEA)
    return inpainted, hair_mask


def normalize_mask(mask_float):
    mask_float = np.nan_to_num(mask_float, nan=0.0, posinf=1.0, neginf=0.0)
    min_val = float(mask_float.min())
    max_val = float(mask_float.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(mask_float, dtype=np.uint8)
    normalized = (mask_float - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def run_hair_model_inference(image_bgr):
    # Assumes hair_model is a loaded keras model.
    input_shape = getattr(hair_model, "input_shape", None)
    if not input_shape or len(input_shape) < 4:
        raise ValueError("Unsupported hair model input shape")

    target_h = int(input_shape[1] or 256)
    target_w = int(input_shape[2] or 256)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = hair_model.predict(x, verbose=0)
    pred = np.array(pred)

    if pred.ndim == 4:
        pred = pred[0]
    if pred.ndim == 3:
        if pred.shape[-1] > 1:
            pred = pred[..., 0]
        else:
            pred = pred[..., 0]
    elif pred.ndim != 2:
        raise ValueError(f"Unexpected model output dimensions: {pred.shape}")

    pred = cv2.resize(pred.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_gray = normalize_mask(pred)
    _, mask_bin = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.medianBlur(mask_bin, 5)
    mask_bin = cv2.dilate(mask_bin, np.ones((3, 3), np.uint8), iterations=1)

    inpainted = cv2.inpaint(image_bgr, mask_bin, 3, cv2.INPAINT_TELEA)
    return inpainted, mask_bin, "keras_h5_mask_inpaint"


def get_hair_removed_image(image_bgr):
    if hair_model is not None:
        try:
            return run_hair_model_inference(image_bgr)
        except Exception as model_runtime_error:
            if not ALLOW_HAIR_FALLBACK:
                raise RuntimeError(
                    f"Hair model runtime failed and fallback is disabled: {model_runtime_error}"
                )
            fallback_img, fallback_mask = remove_hair_dullrazor(image_bgr)
            return fallback_img, fallback_mask, f"keras_runtime_fallback_dullrazor: {model_runtime_error}"

    if not ALLOW_HAIR_FALLBACK:
        raise RuntimeError(
            "Hair model is not loaded. Install TensorFlow-compatible environment or enable ALLOW_HAIR_FALLBACK=true."
        )

    fallback_img, fallback_mask = remove_hair_dullrazor(image_bgr)
    return fallback_img, fallback_mask, "opencv_dullrazor"


def encode_bgr_to_data_url(image_bgr):
    ok, encoded = cv2.imencode('.jpg', image_bgr)
    if not ok:
        return None
    b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def encode_mask_to_data_url(mask_gray):
    ok, encoded = cv2.imencode('.png', mask_gray)
    if not ok:
        return None
    b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def apply_mask_overlay(image_bgr, mask_gray):
    heat = cv2.applyColorMap(mask_gray, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image_bgr, 0.72, heat, 0.28, 0)
    return blended


def send_email(to_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False


# Authentication Routes

@app.route('/api/auth/signup', methods=['POST'])
def signup_step1():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        password = data.get('password')
        full_name = data.get('fullName')
        requested_role = data.get('role', 'Student')

        if not email or not password or not full_name:
            return jsonify({"error": "Missing fields"}), 400

        try:
            valid = validate_email(email, check_deliverability=True)
            email = valid.normalized
        except EmailNotValidError as e:
            return jsonify({"error": f"Invalid Email: {str(e)}"}), 400

        if users_collection.find_one({"email": email}) or admins_collection.find_one({"email": email}):
            return jsonify({"error": "Email already registered"}), 400

        allowed_roles = ['Student', 'Clinician', 'Doctor']
        role = requested_role if requested_role in allowed_roles else 'Student'

        otp = str(random.randint(100000, 999999))
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        pending_user = {
            "fullName": full_name,
            "email": email,
            "password": hashed_password,
            "role": role,
            "otp": otp,
            "created_at": datetime.datetime.utcnow()
        }

        pending_collection.delete_one({"email": email})
        pending_collection.insert_one(pending_user)

        if send_email(email, "TRACE - Verify Account", f"Your OTP is: {otp}\n\nExpires in 5 minutes."):
            return jsonify({"message": "OTP sent successfully"}), 200
        else:
            return jsonify({"error": "Failed to send email"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        otp = data.get('otp', '').strip()

        pending_user = pending_collection.find_one({"email": email})

        if not pending_user:
            return jsonify({"error": "User not found or Session Expired"}), 400

        if (datetime.datetime.utcnow() - pending_user['created_at']).total_seconds() > 300:
            pending_collection.delete_one({"email": email})
            return jsonify({"error": "OTP Expired. Signup again."}), 400

        if pending_user['otp'] != otp:
            return jsonify({"error": "Invalid OTP"}), 400

        new_user = {
            "fullName": pending_user['fullName'],
            "email": pending_user['email'],
            "password": pending_user['password'],
            "role": pending_user['role'],
            "created_at": datetime.datetime.utcnow()
        }

        users_collection.insert_one(new_user)
        pending_collection.delete_one({"email": email})

        return jsonify({"message": "Account Verified!"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        password = data.get('password')

        user = users_collection.find_one({"email": email})
        collection_type = "user"

        if not user:
            user = admins_collection.find_one({"email": email})
            collection_type = "admin"

        if not user:
            return jsonify({"error": "User not found"}), 404

        if bcrypt.checkpw(password.encode('utf-8'), user['password']):
            token = jwt.encode({
                'user_id': str(user['_id']),
                'role': user['role'],
                'type': collection_type,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, SECRET_KEY, algorithm="HS256")

            return jsonify({
                'message': "Login Successful",
                'token': token,
                'user': {
                    "fullName": user['fullName'],
                    "email": user['email'],
                    "role": user['role']
                }
            }), 200
        else:
            return jsonify({"error": "Invalid Password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        user = users_collection.find_one({"email": email}) or admins_collection.find_one({"email": email})

        if not user:
            return jsonify({"error": "No account found"}), 404

        otp = str(random.randint(100000, 999999))
        reset_collection.delete_one({"email": email})
        reset_collection.insert_one({
            "email": email,
            "otp": otp,
            "created_at": datetime.datetime.utcnow()
        })

        if send_email(email, "TRACE - Reset Password", f"Reset Code: {otp}\n\nExpires in 5 minutes."):
            return jsonify({"message": "Code sent"}), 200
        return jsonify({"error": "Email failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        otp = data.get('otp', '').strip()
        new_password = data.get('newPassword')

        record = reset_collection.find_one({"email": email})
        if not record or (datetime.datetime.utcnow() - record['created_at']).total_seconds() > 300:
            return jsonify({"error": "Expired"}), 400

        if record['otp'] != otp:
            return jsonify({"error": "Invalid Code"}), 400

        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        updated = users_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})
        if updated.matched_count == 0:
            admins_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})

        reset_collection.delete_one({"email": email})
        return jsonify({"message": "Password changed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Admin Routes

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    try:
        _admin, error = require_admin_from_request()
        if error:
            return error
        users = list(users_collection.find({}, {"password": 0}))
        for user in users:
            user['_id'] = str(user['_id'])
        return jsonify(users), 200
    except:
        return jsonify([]), 500


@app.route('/api/admin/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        _admin, error = require_admin_from_request()
        if error:
            return error
        users_collection.delete_one({"_id": ObjectId(user_id)})
        return jsonify({"message": "Deleted"}), 200
    except:
        return jsonify({"error": "Failed"}), 500


@app.route('/api/admin/analytics', methods=['GET'])
def get_analytics():
    try:
        _admin, error = require_admin_from_request()
        if error:
            return error
        total_users = users_collection.count_documents({})
        students = users_collection.count_documents({"role": "Student"})
        clinicians = users_collection.count_documents({"role": {"$in": ["Doctor", "Clinician"]}})
        admins = admins_collection.count_documents({})
        total_analyses = analyses_collection.count_documents({})
        hair_only = analyses_collection.count_documents({"status": "hair_processed_only"})
        flagged = analyses_collection.count_documents({"result": "Malignant (Cancerous)"})
        return jsonify({
            "totalUsers": total_users,
            "students": students,
            "clinicians": clinicians,
            "admins": admins,
            "totalAnalyses": total_analyses,
            "hairOnly": hair_only,
            "flagged": flagged
        }), 200
    except:
        return jsonify({}), 500


@app.route('/api/admin/analyses', methods=['GET'])
def get_all_analyses_admin():
    try:
        _admin, error = require_admin_from_request()
        if error:
            return error

        limit = request.args.get("limit", default=100, type=int)
        limit = max(1, min(limit, 500))
        records = list(analyses_collection.find({}).sort("created_at", -1).limit(limit))
        for rec in records:
            rec['_id'] = str(rec['_id'])
            created_at = rec.get('created_at')
            if isinstance(created_at, datetime.datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                rec['created_at_iso'] = created_at.isoformat()
                rec['date'] = created_at.astimezone(APP_TIMEZONE).strftime('%Y-%m-%d %H:%M')
            else:
                rec['created_at_iso'] = None
                rec['date'] = '-'
        return jsonify(records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/analyses/<analysis_id>', methods=['DELETE'])
def delete_analysis_admin(analysis_id):
    try:
        _admin, error = require_admin_from_request()
        if error:
            return error

        result = analyses_collection.delete_one({"_id": ObjectId(analysis_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Analysis record not found"}), 404
        return jsonify({"message": "Analysis record deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/system-status', methods=['GET'])
def admin_system_status():
    try:
        _admin, error = require_admin_from_request()
        if error:
            return error

        return jsonify({
            "timezone": str(os.getenv("APP_TIMEZONE", "Asia/Karachi")),
            "strictValidation": STRICT_IMAGE_VALIDATION,
            "allowHairFallback": ALLOW_HAIR_FALLBACK,
            "classificationModelLoaded": model is not None,
            "hairModelLoaded": hair_model is not None,
            "hairModelBackend": hair_model_backend,
            "hairModelError": hair_model_error
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# History Route

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        decoded, error = decode_auth_token_from_request()
        if error:
            return error
        user_id = decoded['user_id']

        analyses = list(analyses_collection.find({"user_id": user_id}).sort("created_at", -1))

        for analysis in analyses:
            analysis['_id'] = str(analysis['_id'])
            created_at = analysis.get('created_at')
            if isinstance(created_at, datetime.datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                local_dt = created_at.astimezone(APP_TIMEZONE)
                analysis['created_at_iso'] = created_at.isoformat()
                analysis['created_at_epoch'] = int(created_at.timestamp())
                analysis['date'] = local_dt.strftime('%Y-%m-%d %H:%M')
            else:
                analysis['created_at_iso'] = None
                analysis['created_at_epoch'] = 0
                analysis['date'] = '-'

        return jsonify(analyses), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Prediction Route

@app.route('/predict', methods=['POST'])
def predict():
    decoded, error = decode_auth_token_from_request()
    if error:
        return error

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    base_name = secure_filename(file.filename)
    filename = f"{int(datetime.datetime.utcnow().timestamp() * 1000)}_{base_name}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Validate dermoscopic image
    is_valid, reason = validate_dermoscopic_image(filepath)
    if not is_valid:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Invalid image: {reason}"}), 400

    # Hair-removal mode (classification will be integrated later)
    try:
        original_bgr = cv2.imread(filepath)
        if original_bgr is None:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Failed to decode uploaded image"}), 400

        hair_removed_bgr, hair_mask, hair_method = get_hair_removed_image(original_bgr)
        processed_image_data_url = encode_bgr_to_data_url(hair_removed_bgr)
        mask_image_data_url = encode_mask_to_data_url(hair_mask)
        overlay_data_url = encode_bgr_to_data_url(apply_mask_overlay(original_bgr, hair_mask))
        mask_coverage = round(float((hair_mask > 0).sum()) * 100.0 / float(hair_mask.size), 2)

        created_at = datetime.datetime.utcnow()
        response_payload = {
            "analysis_available": False,
            "message": "Hair removal completed successfully.",
            "result": "Hair Removal Completed",
            "diagnosis": "Preprocessing complete",
            "confidence": "N/A",
            "severity": "N/A",
            "status": "hair_processed_only",
            "hair_removal": {
                "applied": True,
                "method": hair_method,
                "model_backend": hair_model_backend,
                "model_error": hair_model_error,
                "mask_coverage_percent": mask_coverage,
                "validation_mode": "strict" if STRICT_IMAGE_VALIDATION else "non_strict"
            },
            "mask_image": mask_image_data_url,
            "mask_overlay_image": overlay_data_url,
            "processed_image": processed_image_data_url,
            "raw_scores": {}
        }

        analyses_collection.insert_one({
            "user_id": decoded["user_id"],
            "result": response_payload["result"],
            "diagnosis": response_payload["diagnosis"],
            "confidence": response_payload["confidence"],
            "severity": response_payload["severity"],
            "status": response_payload["status"],
            "raw_scores": response_payload["raw_scores"],
            "filename": filename,
            "created_at": created_at
        })

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(response_payload), 200

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
