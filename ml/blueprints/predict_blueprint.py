from flask import Blueprint
from flask_restful import Api
from resources.prediction_resources import PredictionResource, HealthCheckResource

pred_bp = Blueprint("predict", __name__)
api = Api(pred_bp)
api.add_resource(PredictionResource, '/predict')
api.add_resource(HealthCheckResource, '/healthz')