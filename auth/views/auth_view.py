from flask import jsonify, make_response
from app.controllers.auth_controller import auth_blueprint

class AuthView:
    @staticmethod
    def register():
        """
        Handle the registration of a new user.
        """
        data = request.get_json()
        response = auth_blueprint.register(data)
        return make_response(jsonify(response), 200)

    @staticmethod
    def login():
        """
        Handle the login of a user.
        """
        data = request.get_json()
        response = auth_blueprint.login(data)
        return make_response(jsonify(response), 200)