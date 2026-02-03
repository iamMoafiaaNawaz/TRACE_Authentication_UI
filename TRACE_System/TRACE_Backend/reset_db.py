import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/trace_db")
client = MongoClient(MONGO_URI)
db = client.get_database()

def reset_database():
    print("--- WARNING: DELETING ALL DATA ---")
    confirm = input("Type 'DELETE' to confirm wiping all Users and Admins: ")
    
    if confirm == "DELETE":
        # Delete Collections
        db["users"].drop()
        db["admins"].drop()
        db["pending_users"].drop()
        db["password_resets"].drop()
        
        print("Success: Database has been completely reset.")
    else:
        print("Operation Cancelled.")

if __name__ == "__main__":
    reset_database()