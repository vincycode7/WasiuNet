from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from datetime import datetime, timedelta
from utils import create_token, verify_token
from models.auth_model import AuthModel
from models.schemas import RegisterSchema

class Register(Resource):
    def post(self):
        # validate input
        schema = RegisterSchema()
        errors = schema.validate(request.get_json())
        if errors:
            return {"error": errors}, 400

        # register user
        data = schema.load(request.get_json())
        return AuthModel.register(data.get("email"), data.get("password"))

class Login(Resource):
    def post(self):
        data = request.get_json()
        return AuthModel.login(data["email"], data["password"])

class GetToken(Resource):
    def post(self):
        # Get the user's credentials from the request
        credentials = request.json
        username = credentials.get('username')
        password = credentials.get('password')

        # Validate the user's credentials
        auth_model = AuthModel()
        if not auth_model.validate_credentials(username, password):
            # Create a JWT
            payload = {'username': username, 'exp': datetime.utcnow() + timedelta(seconds=300)}
            token = create_token(payload)
            return {'token': token}, 200
        else:
            return {'error': 'Invalid username or password.'}, 401

class HealthCheck(Resource):
    def get(self):
        return jsonify(status="UP"), 200
    

class VerifyToken(Resource):
    def post(self):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            token = auth_header.split()[1]
            payload = verify_token(token)
            if payload:
                return {'message': 'Token is valid.'}, 200
            else:
                return payload
        else:
            return {'error': 'Authorization header is missing.'}, 401

class RefreshToken(Resource):
    def post(self):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            token = auth_header.split()[1]
            payload = verify_token(token)
            auth_model = AuthModel()
            if payload:
                if auth_model.check_if_token_is_revoked(token):
                    return {'error': 'Token has been revoked.'}, 401
                # Create a new JWT
                new_payload = {'username': payload['username'], 'exp': datetime.utcnow() + timedelta(seconds=300)}
                new_token = create_token(new_payload)
                return {'token': new_token}, 200
            else:
                return payload
        else:
            return {'error': 'Authorization header is missing.'}, 401
        
        
class RevokeToken(Resource):
    def post(self):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Extract token from "Bearer <token>" format
            token = auth_header.split()[1]
            payload = verify_token(token)
            auth_model = AuthModel()
            if payload:
                if auth_model.check_if_token_is_revoked(token):
                    return {'error': 'Token has already been revoked.'}, 401
                else:
                    # Insert the revoked token into the "revoked_tokens" collection
                    auth_model.revoked_tokens.insert_one({'username': payload['username']})
                    return {'message': 'Token has been revoked.'}, 200
            else:
                return payload
        else:
            return {'error': 'Authorization header is missing.'}, 401




