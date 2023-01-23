from flask_restful import Resource
from flask_restful import Api
from flask import request
from marshmallow import ValidationError
from schemas.prediction_schema import PredictionSchema
from models.prediction import Prediction
from controllers.prediction_controller import PredictionController, HealthCheckController
from utils import jwt_required

class PredictionView:
    def __init__(self, app):
        self.api = Api(app)
        self.api.add_resource(PredictionController, '/predict')
        self.api.add_resource(HealthCheckController, '/health')