from flask_restful import Resource, Api
from flask import request
from marshmallow import ValidationError
from schemas.prediction_schema import PredictionSchema
from models.prediction_model import Prediction
from controllers.prediction_controller import PredictionController, HealthCheckController


from flask import jsonify, request
from flask_restful import Resource
from predict_controller import PredictionController
from predict_model import PredictionModel
import requests

class PredictionResource(Resource):
    def __init__(self):
        self.controller = PredictionController(PredictionModel())

    def post(self):
        data = request.get_json()
        prediction = self.controller.predict(data)
        # Send the token to the auth service
        token = request.headers.get("Authorization")
        auth_response = requests.post("http://auth-service.com/verify_token", headers={"Authorization": token})
        return jsonify({"prediction": prediction, "auth_status": auth_response.json()})
    
class HealthCheckResource(Resource):
    def get(self):
        data = request.get_json()
        # prediction = self.controller.predict(data)
        # Send the token to the auth service
        # token = request.headers.get("Authorization")
        # auth_response = requests.post("http://auth-service.com/verify_token", headers={"Authorization": token})
        return jsonify({"status":"UP"}),200