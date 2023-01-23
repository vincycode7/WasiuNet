from flask import jsonify
from controllers.auth_controller import RegisterController, LoginController, GetTokenController, VerifyTokenController, RefreshTokenController, RevokeTokenController, HealthCheckController
from flask_restful import Api
from flask_restful import Resource
from flask import request
from models.schemas import RegisterSchema
from models.auth_model import AuthModel

class AuthView:
    def __init__(self, app):
        self.api = Api(app)
        self.api.add_resource(RegisterController, '/register')
        self.api.add_resource(LoginController, '/login')
        self.api.add_resource(GetTokenController, '/get_token')
        self.api.add_resource(VerifyTokenController, '/verify_token')
        self.api.add_resource(RefreshTokenController, '/refresh_token')
        self.api.add_resource(RevokeTokenController, '/revoke_token')
        self.api.add_resource(HealthCheckController, '/health')