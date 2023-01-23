import os
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

# Load secrets
SECRET_KEY = os.environ.get('APP_SECRET_KEY')  # replace with your own secret key
MONGO_URL = os.environ.get('MONGO_URL')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD')


def connect_to_mongo():
    client = MongoClient(MONGO_URL)
    db = client['auth_db']
    db.authenticate(MONGO_USERNAME, MONGO_PASSWORD)
    return db