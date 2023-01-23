from flask_restful import Resource
from flask import request
from marshmallow import ValidationError
from ...schemas import PredictionSchema
from ...models import Prediction
from ...utils import verify_token

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
        prediction = Prediction.run_prediction(data['date'], data['time'], data['asset'])
        return {'prediction': prediction}