import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

import redis

REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_DB = os.environ.get('REDIS_DB')

REDIS_DB_INST = redis.Redis(host=str(REDIS_HOST), port=int(REDIS_PORT), db=REDIS_DB)

# Load secrets
SECRET_KEY = os.environ.get('APP_SECRET_KEY')  # replace with your own secret key
MONGO_URL = os.environ.get('MONGO_URL')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD')
PORT = os.environ.get('PORT')
DEBUG = os.environ.get('DEBUG')