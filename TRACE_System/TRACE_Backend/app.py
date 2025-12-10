import os
import random
import smtplib
from email.mime.text import MIMEText
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import bcrypt
import jwt
import datetime
from dotenv import load_dotenv
from bson.objectid import ObjectId
from email_validator import validate_email, EmailNotValidError 

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/trace_db")
SECRET_KEY = os.getenv("SECRET_KEY", "secret")

# ======================================================
# 👇 APKI EMAIL DETAILS (YAHAN DALEIN)
# ======================================================
SMTP_EMAIL = "mafianawaz60@gmail.com"
SMTP_PASSWORD = "sies gimw vwhz yisr" 
# ======================================================

client = MongoClient(MONGO_URI)
db = client.get_database()
users_collection = db["users"]
pending_collection = db["pending_users"] 
reset_collection = db["password_resets"] 

# --- HELPER: SEND EMAIL ---
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
        print(f"❌ Email Error: {e}")
        return False

# --- 1. SIGNUP STEP 1 ---
@app.route('/api/auth/signup', methods=['POST'])
def signup_step1():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        password = data.get('password')
        fullName = data.get('fullName')
        role = data.get('role', 'Student')

        if not email or not password or not fullName:
            return jsonify({"error": "Missing fields"}), 400

        try:
            valid = validate_email(email, check_deliverability=True)
            email = valid.normalized
        except EmailNotValidError as e:
            return jsonify({"error": f"Invalid Email: {str(e)}"}), 400

        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already registered"}), 400
        
        otp = str(random.randint(100000, 999999))
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        pending_user = {
            "fullName": fullName,
            "email": email,
            "password": hashed_password,
            "role": role,
            "otp": otp,
            "created_at": datetime.datetime.utcnow() # Time note kar liya
        }
        
        pending_collection.delete_one({"email": email}) 
        pending_collection.insert_one(pending_user)

        # CHANGE: Text updated to 5 Minutes
        if send_email(email, "TRACE - Verify Account", f"Your OTP is: {otp}\n\nThis code expires in 5 minutes."):
            return jsonify({"message": "OTP sent successfully"}), 200
        else:
            return jsonify({"error": "Failed to send email"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 2. VERIFY OTP (Signup) ---
@app.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        otp = data.get('otp', '').strip()

        pending_user = pending_collection.find_one({"email": email})

        if not pending_user:
            return jsonify({"error": "User not found or Session Expired"}), 400

        # === 🕒 LOGIC CHANGE: 5 MINUTE EXPIRATION CHECK ===
        # Current Time - Created Time
        time_diff = datetime.datetime.utcnow() - pending_user['created_at']
        
        # Agar farq 5 minute (300 seconds) se zyada hai
        if time_diff.total_seconds() > 300: 
            # Record delete karo aur error do
            pending_collection.delete_one({"email": email})
            return jsonify({"error": "OTP Expired (5 mins passed). Signup again."}), 400
        # ==================================================

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

# --- 3. LOGIN ---
@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        
        user = users_collection.find_one({"email": email})
        
        if not user:
            return jsonify({"error": "User not found"}), 404

        if bcrypt.checkpw(data.get('password').encode('utf-8'), user['password']):
            token = jwt.encode({
                'user_id': str(user['_id']),
                'role': user['role'],
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, SECRET_KEY, algorithm="HS256")

            return jsonify({
                "message": "Login Successful",
                "token": token,
                "user": {"fullName": user['fullName'], "email": user['email'], "role": user['role']}
            }), 200
        else:
            return jsonify({"error": "Invalid Password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. FORGOT PASSWORD (OTP SEND) ---
@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "No account found with this email"}), 404

        otp = str(random.randint(100000, 999999))
        
        reset_collection.delete_one({"email": email})
        reset_collection.insert_one({
            "email": email,
            "otp": otp,
            "created_at": datetime.datetime.utcnow() # Time save kiya
        })

        # CHANGE: Text updated to 5 Minutes
        if send_email(email, "TRACE - Reset Password", f"Your Password Reset Code is: {otp}\n\nExpires in 5 minutes."):
            return jsonify({"message": "Reset code sent to email"}), 200
        else:
            return jsonify({"error": "Email failed to send"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 5. RESET PASSWORD (VERIFY & CHANGE) ---
@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        otp = data.get('otp', '').strip()
        new_password = data.get('newPassword')

        record = reset_collection.find_one({"email": email})

        if not record:
            return jsonify({"error": "Request expired"}), 400

        # === 🕒 LOGIC CHANGE: 5 MINUTE EXPIRATION CHECK ===
        time_diff = datetime.datetime.utcnow() - record['created_at']
        
        if time_diff.total_seconds() > 300: # 300 seconds = 5 Minutes
            reset_collection.delete_one({"email": email})
            return jsonify({"error": "OTP Expired. Please request a new code."}), 400
        # ==================================================

        if record['otp'] != otp:
            return jsonify({"error": "Invalid Code"}), 400

        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        users_collection.update_one(
            {"email": email},
            {"$set": {"password": hashed_password}}
        )

        reset_collection.delete_one({"email": email})

        return jsonify({"message": "Password changed successfully. Please Login."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ADMIN APIs ---
@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    try:
        users = list(users_collection.find({}, {"password": 0}))
        for user in users: user['_id'] = str(user['_id'])
        return jsonify(users), 200
    except: return jsonify([]), 500

@app.route('/api/admin/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        users_collection.delete_one({"_id": ObjectId(user_id)})
        return jsonify({"message": "Deleted"}), 200
    except: return jsonify({"error": "Failed"}), 500

@app.route('/api/admin/analytics', methods=['GET'])
def get_analytics():
    try:
        total = users_collection.count_documents({})
        students = users_collection.count_documents({"role": "Student"})
        clinicians = users_collection.count_documents({"role": {"$in": ["Doctor", "Clinician"]}}) 
        admins = users_collection.count_documents({"role": "Admin"})
        return jsonify({"totalUsers": total, "students": students, "clinicians": clinicians, "admins": admins}), 200
    except: return jsonify({}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)