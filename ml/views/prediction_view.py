from flask_restful import Resource
from flask import request
from marshmallow import ValidationError
from ..schemas.prediction_schema import PredictionSchema
from ..models.prediction import Prediction
from ..controllers.prediction_controller import PredictionController
from ..utils import jwt_required

prediction_schema = PredictionSchema()
prediction_controller = PredictionController()

class PredictionView(Resource):
    @jwt_required
    def post(self):
        json_data = request.get_json()
        if not json_data:
            return {'message': 'No input data provided'}, 400
        try:
            data = prediction_schema.load(json_data)
        except ValidationError as err:
            return err.messages, 422
        prediction_data = Prediction(data)
        prediction_result = prediction_controller.get_prediction(prediction_data)
        return prediction_result, 201