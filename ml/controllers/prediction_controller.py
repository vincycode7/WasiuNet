from flask_restful import Resource
from flask import request
from marshmallow import ValidationError
from schemas.prediction_schema import PredictionSchema
from utils.prediction_utils import verify_token, run_prediction
            
class PredictionController(Resource):
    def get(self):
        # Verify the token
        auth_header = request.headers.get("Authorization")
        if auth_header:
            token = auth_header.split(" ")[1]
        else:
            return {'error': 'Authorization header not provided'}, 401
        if not verify_token(token):
            return {'error': 'Invalid or expired token'}, 401

        # Validate and parse the input data
        data = request.args.to_dict()
        try:
            data = PredictionSchema().load(data)
        except ValidationError as err:
            return {'error': err.messages}, 400

        # Run the prediction and return the output
        prediction = run_prediction(data['date'], data['time'], data['asset'])
        return {'prediction': prediction}
    
class HealthCheckController(Resource):
    def get(self):
        return jsonify(status="UP"), 200