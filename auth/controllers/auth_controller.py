from flask import Blueprint, request
from app.controllers.base_controller import BaseController
from app.models.auth_model import AuthModel

auth_blueprint = Blueprint("auth", __name__)

class AuthController(BaseController):
    def __init__(self):
        self.model = AuthModel()
    
    @auth_blueprint.route("/register", methods=["POST"])
    def register():
        data = request.get_json()
        return self.model.register(data)

    @auth_blueprint.route("/login", methods=["POST"])
    def login():
        data = request.get_json()
        return self.model.login(data["email"], data["password"])