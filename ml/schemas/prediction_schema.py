from marshmallow import Schema, fields

class PredictionSchema(Schema):
    date = fields.DateTime(required=True)
    time = fields.Time(required=True)
    asset = fields.Str(required=True)