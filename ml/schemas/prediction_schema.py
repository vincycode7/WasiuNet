from marshmallow import Schema, fields, pre_load
from datetime import datetime

from marshmallow import ValidationError, validates

class PredictionSchema(Schema):
    pred_datetime = fields.DateTime(required=True, format="%Y-%m-%d-%H-%M-%S")
    asset = fields.Str(required=True)

    @pre_load
    def process_input_data(self, data, **kwargs):
        try:
            datetime.strptime(data["pred_datetime"], "%Y-%m-%d-%H-%M-%S")
        except ValueError:
            raise ValidationError("Invalid date format. Please use the format YYYY-MM-DD-HH-MM-SS.")
        return data