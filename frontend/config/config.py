import os
from dotenv import load_dotenv
import os

load_dotenv()

# Load secrets
ML_BASE_URL = os.environ.get('ML_BASE_URL')
ML_PORT = os.environ.get('ML_PORT')