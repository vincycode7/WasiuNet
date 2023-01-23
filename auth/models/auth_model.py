import os
from flask import Flask
from flask_mongoengine import MongoEngine
from bcrypt import checkpw, gensalt, hashpw
from datetime import datetime, timedelta
import jwt
from pymongo import MongoClient
from configs import MONGO_URL, MONGO_USERNAME, MONGO_PASSWORD, SECRET_KEY, connect_to_mongo

app = Flask(__name__)
app.config["MONGODB_SETTINGS"] = {'host': MONGO_URL}
app.config["SECRET_KEY"] = SECRET_KEY
db = MongoEngine(app)

class User(db.Document):
    email = db.StringField(required=True, unique=True)
    password = db.StringField(required=True)
    created_at = db.DateTimeField(default=datetime.utcnow)

class AuthModel:
    def __init__(self):
        # Connect to the "auth_db" database and the "revoked_tokens" collection
        self.db = connect_to_mongo()
        self.users = self.db["users"]
        self.revoked_tokens = self.db["revoked_tokens"]
        
    @staticmethod
    def register(email, password):
        hashed_password = hashpw(password.encode('utf-8'), gensalt())
        user = User(email=email, password=hashed_password)
        user.save()
        return {'status': 'success'}

    @staticmethod
    def login(email, password):
        user = User.objects(email=email).first()
        if user and checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            token = jwt.encode({'user_id': str(user.id), 'exp': datetime.utcnow() + timedelta(seconds=300)}, app.config['SECRET_KEY'], algorithm='HS256')
            return {'status': 'success', 'token': token.decode('utf-8')}
        else:
            return {'status': 'error', 'message': 'Invalid email or password.'}

    def validate_credentials(self, username, password):
        # Validate the user's credentials
        if not username or not password:
            return {'status': 'error','error': 'Username and password are required.'}, 400
        user = self.check_credentials(username, password)
        if not username:
            return {'status': 'error', 'message': 'Invalid email or password.'}, 401
        return None
        
    def check_credentials(self, username, password):
        """Check if the given username and password match a user in the database"""
        user = self.users.find_one({'username': username})
        if user:
            if bcrypt.checkpw(password.encode('utf-8'), user.get('password').encode('utf-8')):
                return user
        return None

    def insert_user(self, username, password):
        """Insert a new user with the given username and hashed password"""
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.users.insert_one({'username': username, 'password': hashed_password})
    
    def check_if_token_is_revoked(self, token):
        """Check if the given token has been revoked"""
        payload = utils.verify_token(token)
        if payload['username'] in self.revoked_tokens:
            return True
        return False

    def revoke_token(self, token):
        """Insert the given token into the revoked_tokens collection"""
        payload = utils.verify_token(token)
        if self.revoked_tokens.find_one({'username': payload['username']}):
            return {'error': 'Token has already been revoked.'}, 401
        self.revoked_tokens.insert_one({'username': payload['username']})
        