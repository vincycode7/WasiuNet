from mongoengine import Document, StringField, DateTimeField

class Prediction(Document):
    asset = StringField(required=True)
    date = StringField(required=True)
    time = StringField(required=True)
    prediction = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'predictions'}
    
    def run_prediction(date, time, asset):
        # code to run prediction using the provided date, time and asset
        prediction = "Some predicted value"
        return prediction

    def save_prediction(prediction, date, time, asset):
        pred = Prediction(prediction=prediction, date=date, time=time, asset=asset)
        pred.save()
        return pred

    def get_prediction_by_date_time_asset(date, time, asset):
        return Prediction.objects(date=date, time=time, asset=asset).first()