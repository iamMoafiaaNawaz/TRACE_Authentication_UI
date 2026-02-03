import os
import datetime
import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/trace_db")

def create_admin_user():
    try:
        client = MongoClient(MONGO_URI)
        db = client.get_database()
        
        # CHANGED: Now using 'admins' collection explicitly
        admins_collection = db["admins"]

        print("--- Create New Admin (Admins Table) ---")
        
        full_name = input("Enter Full Name: ").strip()
        email = input("Enter Email Address: ").strip().lower()
        password = input("Enter Password: ").strip()

        if not full_name or not email or not password:
            print("Error: All fields are required.")
            return

        # Check in Admins collection
        if admins_collection.find_one({"email": email}):
            print(f"Error: Admin with email '{email}' already exists.")
            return

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        admin_data = {
            "fullName": full_name,
            "email": email,
            "password": hashed_password,
            "role": "Admin",
            "created_at": datetime.datetime.utcnow()
        }

        admins_collection.insert_one(admin_data)
        print("Success: Admin account created in 'admins' collection.")
        
    except Exception as e:
        print(f"System Error: {str(e)}")

if __name__ == "__main__":
    create_admin_user()