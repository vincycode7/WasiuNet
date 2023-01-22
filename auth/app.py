from flask import Flask, jsonify
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import jwt
from datetime import datetime, timedelta
from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient('mongodb://localhost:27017/')

# Connect to the "auth_db" database and the "revoked_tokens" collection
db = client["auth_db"]
revoked_tokens = db["revoked_tokens"]

app = Flask(__name__)
api = Api(app)

secret_key = 'secret_key'  # replace with your own secret key
revoked_tokens = set()  # keep track of revoked tokens

class GetToken(Resource):
    def post(self):
        # Get the user's credentials from the request
        credentials = request.json
        username = credentials.get('username')
        password = credentials.get('password')
        # Validate the user's credentials
        if not username or not password:
            return {'error': 'Username and password are required.'}, 400
        if username != 'test' or password != 'test':
            return {'error': 'Invalid username or password.'}, 401
        # Create a JWT
        payload = {'username': username, 'exp': datetime.utcnow() + timedelta(seconds=300)}
        token = jwt.encode(payload, secret_key)
        return {'token': token.decode('utf-8')}, 200
    


@app.route("/health")
def health():
    return jsonify(status="UP")

class VerifyToken(Resource):
    def post(self):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            token = auth_header.split()[1]
            try:
                # Verify and decode the token
                payload = jwt.decode(token, secret_key)
                return {'message': 'Token is valid.'}, 200
            except jwt.ExpiredSignatureError:
                return {'error': 'Token has expired.'}, 401
            except jwt.InvalidTokenError:
                return {'error': 'Invalid token.'}, 401
        else:
            return {'error': 'Authorization header is missing.'}, 401

class RefreshToken(Resource):
    def post(self):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            token = auth_header.split()[1]
            try:
                # Verify and decode the token
                payload = jwt.decode(token, secret_key)
                if payload['username'] in revoked_tokens:
                    return {'error': 'Token has been revoked.'}, 401
                # Create a new JWT
                new_payload = {'username': payload['username'], 'exp': datetime.utcnow() + timedelta(seconds=300)}
                new_token = jwt.encode(new_payload, secret_key)
                return {'token': new_token.decode('utf-8')}, 200
            except jwt.ExpiredSignatureError:
                return {'error': 'Token has expired.'}, 401
            except jwt.InvalidTokenError:
                return {'error': 'Invalid token.'}, 401
        else:
            return {'error': 'Authorization header is missing.'}, 401

class RevokeToken(Resource):
    def post(self):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            token = auth_header.split()[1]
            try:
                # Verify and decode the token
                payload = jwt.decode(token, secret_key)
                if revoked_tokens.find_one({'username': payload['username']}):
                    return {'error': 'Token has already been revoked.'}, 401
                # Insert the revoked token into the "revoked_tokens" collection
                revoked_tokens.insert_one({'username': payload['username']})
                return {'message': 'Token has been revoked.'}, 200
            except jwt.ExpiredSignatureError:
                return {'error': 'Token has expired.'}, 401
            except jwt.InvalidTokenError:
                return {'error': 'Invalid token.'}, 401
        else:
            return {'error': 'Authorization header is missing.'}, 401
                
api.add_resource(VerifyToken, '/verify')


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)