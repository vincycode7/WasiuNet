from pymongo import MongoClient
from app.config.config import Config

class Mongo:
    def __init__(self):
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.MONGO_DBNAME]