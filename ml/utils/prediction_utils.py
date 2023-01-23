# prediction_utils.py

import jwt

def run_prediction(date, time, asset):
    # code for running prediction goes here
    prediction_result = 'Predicted value: {}'.format(asset)
    return prediction_result

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return {'error': 'Token expired. Please login again.'}, 401
    except jwt.InvalidTokenError:
        return {'error': 'Invalid token. Please login again.'}, 401
