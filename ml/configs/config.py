import os

from dotenv import load_dotenv
from pymongo import MongoClient
import logging, os

logger = logging.getLogger(__name__)

load_dotenv()

import redis

MODEL_MAPPING = eval(os.environ.get("MODEL_MAPPING"))
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_DB = os.environ.get('REDIS_DB')
REDIS_PASSWD = os.environ.get('REDIS_PASSWD')
    
# Load secrets
SECRET_KEY = os.environ.get('APP_SECRET_KEY')  # replace with your own secret key
MONGO_URL = os.environ.get('MONGO_URL')
MONGO_USERNAME = os.environ.get('MONGO_INITDB_ROOT_USERNAME')
MONGO_PASSWORD = os.environ.get('MONGO_INITDB_ROOT_PASSWORD')
PORT = os.environ.get('PORT')
DEBUG = os.environ.get('DEBUG')
class RedisConfig:
    def __init__(self, host, port, password):
        self.host = host
        self.port = port
        self.password = password
        self.redis_connection = self.get_redis_connection()
        
    def get_redis_connection(self):
        try:
            redis_connection = redis.StrictRedis(host=self.host, port=self.port, password=self.password)
            redis_connection.ping()
            return redis_connection
        except redis.ConnectionError as e:
            error = "Error connecting to Redis: {}, Check if redis credentials is correct or if redis is up".format(e)
            logger.error(error)
            raise Exception(error)
        

REDIS_DB_INST = RedisConfig(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWD)
