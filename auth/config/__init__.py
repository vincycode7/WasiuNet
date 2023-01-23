import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.environ.get('MONGO_URL')
MONGO_USERNAME = os.environ.get('MONGO_USERNAME')
MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD')
SECRET_KEY = os.environ.get('APP_SECRET_KEY')