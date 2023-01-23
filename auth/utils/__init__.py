import jwt
from datetime import datetime, timedelta
from config import  SECRET_KEY

def verify_token(token):
    try:
        # Verify and decode the token
        payload = jwt.decode(token, SECRET_KEY)
        return payload
    except jwt.ExpiredSignatureError:
        return {'error': 'Token has expired.'}, 401
    except jwt.InvalidTokenError:
        return {'error': 'Invalid token.'}, 401


def create_token(payload):
    token = jwt.encode(payload, SECRET_KEY)
    return token.decode('utf-8')
