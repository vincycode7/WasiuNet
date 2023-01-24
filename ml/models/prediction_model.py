from mongoengine import Document, StringField, DateTimeField
from utils.prediction_utils import run_prediction
from datetime import datetime

class PredictionModel(Document):
    asset = StringField(required=True)
    date_time = DateTimeField(required=True)
    prediction = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'predictions'}
    
    def run_prediction(self,datetime, asset):
        # code to run prediction using the provided date, time and asset
        prediction = run_prediction(datetime, asset) #"Some predicted value"
        return prediction

    def save_prediction(prediction, date, time, asset):
        pred = Prediction(prediction=prediction, date=date, time=time, asset=asset)
        pred.save()
        return pred

    def get_prediction_by_date_time_asset(date, time, asset):
        return Prediction.objects(date=date, time=time, asset=asset).first()