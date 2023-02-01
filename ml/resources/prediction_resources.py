from flask_restful import Resource, Api
from flask import request
from marshmallow import ValidationError
from schemas.prediction_schema import PredictionSchema
from models.prediction_model import PredictionModel
from controllers.prediction_controller import PredictionController, HealthCheckController
from marshmallow import ValidationError
from schemas.prediction_schema import PredictionSchema

from flask import jsonify, request, make_response
from flask_restful import Resource
from controllers.prediction_controller import PredictionController
from models.prediction_model import PredictionModel
from schemas.prediction_schema import PredictionSchema
import requests, json
from flasgger import swag_from

class PredictionResource(Resource):
    def __init__(self):
        self.controller = PredictionController(PredictionModel())
        self.schema = PredictionSchema()
    
    @swag_from('../templates/predict_swagger.yml')
    def post(self):
        # Verify the token
        # auth_header = request.headers.get("Authorization")
        # if auth_header:
        #     token = auth_header.split(" ")[1]
        # else:
        #     return {'error': 'Authorization header not provided'}, 401
        
        # Send the token to the auth service
        token = request.headers.get("Authorization", None)
        # auth_response = requests.post("http://auth-service.com/verify_token", headers={"Authorization": token})

        # Validate and parse the input data
        data = data = json.loads(request.data)
        print(f"resource data--> ",data)
        try:
            data = self.schema.load(data)
            pred_datetime=data.get('pred_datetime') # expect a datatime, convert to string to prediction
            asset=data.get('asset')
        except ValidationError as err:
            return {'error': err.messages}, 400

        # Run the prediction and return the output    
        prediction = self.controller.predict(pred_datetime=data.get('pred_datetime'), asset=data.get('asset'))
        
        return make_response(jsonify({"prediction": prediction, "auth_status": "auth_response.json()"}),200)
    
class HealthCheckResource(Resource):
    def __init__(self):
        self.controller = HealthCheckController()
        
    @swag_from('../templates/healthcheck_swagger.yml')
    def get(self):
        # data = request.get_json()
        # prediction = self.controller.predict(data)
        # Send the token to the auth service
        # token = request.headers.get("Authorization")
        # auth_response = requests.post("http://auth-service.com/verify_token", headers={"Authorization": token})
        return  make_response(jsonify(self.controller.get_health_status()), 200)