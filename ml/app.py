from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import aiohttp
import asyncio

app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    async def post(self):
        auth_token = request.headers.get('Authorization')
        
        async with aiohttp.ClientSession() as session:
            # Authenticate the user
            verified = verify_token(auth_token)
            if not verified:
                return jsonify({'error': 'Invalid token'}), 401
            else:
                input_data = request.json.get('input')
                
                # Input validation
                if input_data is None:
                    return {'error': 'Input data is required.'}, 400
                
                # Get a prediction
                prediction_response = await get_prediction(session, input_data, auth_token)
                prediction = prediction_response['prediction']
                return {'prediction': prediction}
            
class HealthCheck(Resource):
    def get(self):
        return jsonify(status="UP"), 200
    
api.add_resource(Prediction, '/predict')
api.add_resource(HealthCheck, '/health')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)