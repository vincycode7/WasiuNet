from flask_restful import Resource
from flask import request, jsonify
from flask import make_response
from flasgger import swag_from
from marshmallow import fields

class PredictionController:
    def __init__(self, model):
        self.model = model
    
    @swag_from("template/predict_swagger.yml")
    def predict(self, pred_datetime, asset):
        try:
            return self.model.run_prediction(pred_datetime, asset)
        except Exception as e:
            raise ValueError(f"Could not predict because of exception {e} in predict controller.")
    
class HealthCheckController(Resource):
    @swag_from("template/healthcheck_swagger.yml")
    def get_health_status(self):
        response = {"status":"UP"}
        return response