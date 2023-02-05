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
import asyncio

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
        try:
            data = self.schema.load(data)
            pred_datetime=data.get('pred_datetime') # expect a datatime, convert to string to prediction
            asset=data.get('asset')
        except ValidationError as err:
            return {'error': err.messages}, 400

        # Run the prediction and return the output    
        prediction = self.controller.predict(pred_datetime=pred_datetime, asset=asset)
        
        return make_response(jsonify({"prediction": prediction, "auth_status": "auth_response.json()"}),200)
    
class Predict(Resource):
    async def post(self):
        data = request.get_json()
        redis_conn = redis.Redis(host='localhost', port=6379, db=0)

        prediction_key = data.get('prediction_key', None)
        if not prediction_key:
            # Generate a unique prediction key
            prediction_key = generate_unique_key()

        # Check if the prediction result is already available in the cache
        result = redis_conn.get(prediction_key)
        if result:
            # Return the result from the cache
            return result

        # If the result is not in the cache, make the prediction asynchronously
        loop = asyncio.get_event_loop()
        task = loop.create_task(make_prediction(data, prediction_key))
        await task
        result = redis_conn.get(prediction_key)

        return {"prediction_key": prediction_key}

class Result(Resource):
    async def get(self, prediction_key):
        redis_conn = redis.Redis(host='localhost', port=6379, db=0)
        result = redis_conn.get(prediction_key)
        if result:
            return {"prediction": result}
        else:
            return {"message": "Result not found"}, 404

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