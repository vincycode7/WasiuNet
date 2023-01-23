from flask import jsonify
from controllers.auth_controller import Register, Login, GetToken, VerifyToken, RefreshToken, RevokeToken, HealthCheck
from flask_restful import Api

class AuthView:
    def __init__(self, app):
        self.api = Api(app)
        self.api.add_resource(Register, '/register')
        self.api.add_resource(Login, '/login')
        self.api.add_resource(GetToken, '/get_token')
        self.api.add_resource(VerifyToken, '/verify_token')
        self.api.add_resource(RefreshToken, '/refresh_token')
        self.api.add_resource(RevokeToken, '/revoke_token')
        self.api.add_resource(HealthCheck, '/health')