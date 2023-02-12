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
# import asyncio

import asyncio
import json
import redis
from flask import request, make_response, jsonify
from flask_restful import Resource
from marshmallow import ValidationError
from controllers.prediction_controller import PredictionController
from schemas.prediction_schema import PredictionSchema
from models.prediction_model import PredictionModel
from flasgger import swag_from
from utils.helper import hash_string
import logging, os
import threading, time
import concurrent.futures

logger = logging.getLogger(__name__)

class PredictionResource(Resource):

    def __init__(self):
        self.controller = PredictionController(PredictionModel())
        self.schema = PredictionSchema()
        self.redis_conn = redis.Redis(host='localhost', port=6379, db=0)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        
    def get(self, prediction_key):
        result = self.get_prediction(prediction_key)
        if result.get("status") in ["success","processing"]:
            return make_response(jsonify(result), 200)
        elif result.get("status") == "failed":
            return make_response(jsonify(result), 401)
        elif result.get("status") == "not_found":
            return make_response(jsonify(result), 404)
        
    def get_prediction(self, prediction_key):

        # Check if the result is available in the cache
        result = self.redis_conn.get(prediction_key)
        if result:
            return {"prediction": result, "status": "success"}

        # Check if the prediction is being processed
        lock = self.redis_conn.get(prediction_key + '_lock')
        if lock:
            return {"status": "processing"}

        # Check if the prediction failed
        failed = self.redis_conn.get(prediction_key + '_failed')
        failed_max_failed_retries = self.redis_conn.get(prediction_key + '_max_failed_retries')
        if failed:
            if failed_max_failed_retries and int(failed_max_failed_retries) and (int(failed) < int(failed_max_failed_retries)):
                return {"status": "failed","message": f"retrying failed job with id {prediction_key}, total_retries: {int(failed)}"}
            if failed_max_failed_retries and  int(failed_max_failed_retries) and (int(failed) >= int(failed_max_failed_retries)):
                return {"status": "failed", "message": f"failed job with id {prediction_key} has failed for {int(failed_max_failed_retries)} amount of time(s) and won't be retried."}
            return {"status": "failed"}

        # Return not found if the result is not available and not being processed
        return {"status": "not_found"}
    
    # def run_prediction_in_background(self, data, prediction_key):
    #     asyncio.create_task(self.make_prediction(data, prediction_key))
    
    def run_prediction_in_background(self, data, prediction_key):
        # start a new thread to handle the request
        self.thread_pool.submit(self.make_prediction, data, prediction_key)

    @swag_from('../templates/predict_swagger.yml')
    def post(self):
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
        prediction_key = hash_string(f"prediction:{asset}:{pred_datetime}")
        # result = asyncio.run(self.get_prediction(prediction_key))
        result = self.get_prediction(prediction_key)
        
        # Return the result from the cache or is currently available in the cache running in the background
        if result.get("status") in ["success","processing"]:
            return make_response(jsonify(result), 200)
        elif result.get("status") == "failed":
            return make_response(jsonify(result), 401)
        elif result.get("status") == "not_found":     
            self.run_prediction_in_background(data, prediction_key) # Run solution in background
        return make_response(jsonify({"message": "Prediction is already being processed", "prediction_key":prediction_key, "status": "running"}), 200)
            
    def make_prediction(self, data, prediction_key, max_retries=3):
        # redis_conn = redis.Redis(host='localhost', port=6379, db=0)

        # Check if the result for the same prediction key is already available in the cache
        result = self.redis_conn.get(prediction_key)
        if result:
            # Return the result from the cache
            return result

        # Check if the same prediction key is being processed by another task
        lock = self.redis_conn.get(prediction_key + '_lock')
        print(f"\n\nHere is the lock for the prediction: {lock} \n\n")
        if lock:
            # Return None to indicate that the task is already in progress
            return None

        # Acquire the lock to prevent multiple tasks from processing the same prediction key
        self.redis_conn.set(prediction_key + '_lock', str(True), ex=60)

        # Check if the prediction has failed before
        failed_count = self.redis_conn.get(prediction_key + '_failed')
        if failed_count:
            failed_count = int(failed_count)
            if failed_count >= max_retries:
                logger.error(f"Prediction task with key {prediction_key} has failed {failed_count} times and will not be retried again.")
                
                # Release the lock
                self.redis_conn.delete(prediction_key + '_lock')

                # Reset the failed count
                self.redis_conn.delete(prediction_key + '_failed')
                return None

        try:
            # Run the prediction
            prediction = self.controller.predict(**data)
        except Exception as e:
            # Release the lock
            self.redis_conn.delete(prediction_key + '_lock')

            # Increment the failed count and push the task back onto the queue
            failed_count = self.redis_conn.get(prediction_key + '_failed')
            logger.error(f"failed_count Before: {failed_count}")
            
            if failed_count:
                failed_count = int(failed_count) + 1
            else:
                failed_count = 1
                
            logger.error(f"failed_count After: {failed_count}")
            self.redis_conn.set(prediction_key + '_failed', failed_count)
            self.redis_conn.set(prediction_key + '_max_failed_retries', max_retries)
            logger.error(f"Prediction task with key {prediction_key} failed with error {str(e)}")
            
            # Delay for a bit before retrying
            # await asyncio.sleep(failed_count-1)
            time.sleep(failed_count**2)
            logger.error(f"awaited retry time: {failed_count}")
            prediction = self.make_prediction(data, prediction_key, max_retries=max_retries) # Run solution in background
                
            return None

        # Store the result in the cache
        self.redis_conn.set(prediction_key, prediction)

        # Release the lock
        self.redis_conn.delete(prediction_key + '_lock')

        # Reset the failed count
        self.redis_conn.delete(prediction_key + '_failed')

        return prediction

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