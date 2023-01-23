import bcrypt
import jwt
from app.models.base_model import BaseModel

class AuthModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.collection = self.db.users

    def register(self, data):
        """
        Handle the registration of a new user.
        """
        # check if the email already exists
        user = self.collection.find_one({'email': data.get('email')})
        if user:
            return {'status': 'error', 'message': 'Email already exists.'}

        # hash the password
        hashed_password = bcrypt.hashpw(data.get('password').encode('utf-8'), bcrypt.gensalt())

        # insert the user into the database
        user_id = self.collection.insert_one({
            'name': data.get('name'),
            'email': data.get('email'),
            'password': hashed_password
        }).inserted_id

        return {'status': 'success', 'user_id': user_id}

    def login(self, email, password):
        """
        Handle the login of a user.
        """
        user = self.collection.find_one({'email': email})
        if not user:
            return {'status': 'error', 'message': 'Invalid email or password.'}

        if bcrypt.checkpw(password.encode('utf-8'), user.get('password').encode('utf-8')):
            token = jwt.encode({'user_id': str(user.get('_id'))}, SECRET_KEY, algorithm='HS256')
            return {'status': 'success', 'token': token.decode('utf-8')}
        else:
            return {'status': 'error', 'message': 'Invalid email or password.'}