from marshmallow import Schema, fields

class UserSchema(ma.ModelSchema):
    class Meta:
        from auth_model import User
        model = User