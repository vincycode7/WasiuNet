# from flask_restful import Resource, Api
# from flask import request
# from marshmallow import ValidationError
# from schemas.prediction_schema import PredictionSchema
# from models.prediction_model import PredictionModel
# from controllers.prediction_controller import PredictionController, HealthCheckController
# from marshmallow import ValidationError
# from schemas.prediction_schema import PredictionSchema

# from flask import jsonify, request, make_response
# from flask_restful import Resource
# from controllers.prediction_controller import PredictionController
# from models.prediction_model import PredictionModel
# from schemas.prediction_schema import PredictionSchema
# import requests, json
# from flasgger import swag_from
# import asyncio

import asyncio
import json
import redis
from flask import request, make_response, jsonify
from flask_restful import Resource
from marshmallow import ValidationError
from prediction_controller import PredictionController
from prediction_schema import PredictionSchema

class PredictionResource(Resource):

    def __init__(self):
        self.controller = PredictionController(PredictionModel())
        self.schema = PredictionSchema()
        self.redis_conn = redis.Redis(host='localhost', port=6379, db=0)

    @swag_from('../templates/predict_swagger.yml')
    async def post(self):
        # Verify the token
        # auth_header = request.headers.get("Authorization")
        # if auth_header:
        #     token = auth_header.split(" ")[1]
        # else:
        #     return {'error': 'Authorization header not provided'}, 401
        
        # Send the token to the auth service
        # token = request.headers.get("Authorization", None)
        # auth_response = requests.post("http://auth-service.com/verify_token", headers={"Authorization": token})

        # Validate and parse the input data
        data = data = json.loads(request.data)
        try:
            data = self.schema.load(data)
            pred_datetime=data.get('pred_datetime') # expect a datatime, convert to string to prediction
            asset=data.get('asset')
        except ValidationError as err:
            return {'error': err.messages}, 400

        # Check if the prediction result is already available in the cache
        prediction_key = f"prediction:{asset}:{pred_datetime}"
        result = self.redis_conn.get(prediction_key)
        if result:
            # Return the result from the cache
            return make_response(jsonify({"prediction": result.decode(), "auth_status": "auth_response.json()"}), 200)

        # If the result is not in the cache, make the prediction
        result = await self.make_prediction(data, prediction_key)
        if result is not None:
            # Return the result
            return make_response(jsonify({"prediction": result.decode(), "auth_status": "auth_response.json()"}), 200)
        else:
            # Return a message indicating that the prediction is being processed
            return {"message": "Prediction is already being processed"}


    async def make_prediction(self, data, prediction_key, max_retries=3):
        redis_conn = redis.Redis(host='localhost', port=6379, db=0)

        # Check if the result for the same prediction key is already available in the cache
        result = redis_conn.get(prediction_key)
        if result:
            # Return the result from the cache
            return result

        # Check if the same prediction key is being processed by another task
        lock = redis_conn.get(prediction_key + '_lock')
        if lock:
            # Return None to indicate that the task is already in progress
            return None

        # Acquire the lock to prevent multiple tasks from processing the same prediction key
        redis_conn.set(prediction_key + '_lock', True, ex=60)

        # Check if the prediction has failed before
        failed_count = redis_conn.get(prediction_key + '_failed')
        if failed_count:
            failed_count = int(failed_count)
            if failed_count >= max_retries:
                redis_conn.delete(prediction_key + '_lock')
                logging.error(f"Prediction task with key {prediction_key} has failed {failed_count} times and will not be retried again.")
                return None

        try:
            # Run the prediction
            prediction = self.controller.predict(**data)
        except Exception as e:
            # Release the lock
            redis_conn.delete(prediction_key + '_lock')

            # Increment the failed count
            failed_count = redis_conn.get(prediction_key + '_failed')
            if failed_count:
                failed_count = int(failed_count) + 1
            else:
                failed_count = 1
            redis_conn.set(prediction_key + '_failed', failed_count)
            redis_conn.set(prediction_key + '_max_failed_retries', max_retries)

            logging.error(f"Prediction task with key {prediction_key} failed with error {str(e)}")
            return None

        # Store the result in the cache
        redis_conn.set(prediction_key, prediction)

        # Release the lock
        redis_conn.delete(prediction_key + '_lock')

        # Reset the failed count
        redis_conn.delete(prediction_key + '_failed')

        return prediction


class Result(Resource):
    async def get(self, prediction_key):
        redis_conn = redis.Redis(host='localhost', port=6379, db=0)

        # Check if the result is available in the cache
        result = redis_conn.get(prediction_key)
        if result:
            return {"prediction": result}

        # Check if the prediction is being processed
        lock = redis_conn.get(prediction_key + '_lock')
        if lock:
            return {"status": "processing"}

        # Check if the prediction failed
        failed = redis_conn.get(prediction_key + '_failed')
        failed_max_failed_retries = redis_conn.get(prediction_key + '_max_failed_retries')
        if failed:
            if failed_max_failed_retries and failed < failed_max_failed_retries:
                return {"status": "failed","message": f"retrying failed job with id {prediction_key}"}
            if failed_max_failed_retries and failed >= failed_max_failed_retries:
                return {"status": "failed", "message": f"failed job with id {prediction_key} has failed for {failed_max_failed_retries} amount of time(s) and won't be retried."}
            return {"status": "failed"}

        # Return not found if the result is not available and not being processed
        return {"status": "not found"}, 404


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