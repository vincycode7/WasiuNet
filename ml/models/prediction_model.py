from mongoengine import Document, StringField, DateTimeField
from utils.prediction_utils import run_prediction, load_wasiunet_model, load_wasiunet_data
from datetime import datetime
    
class PredictionModel(Document):
    asset = StringField(required=True)
    pred_datetime = DateTimeField(required=True)
    prediction = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'predictions'}
    
    def run_prediction(self,pred_datetime, asset):
        # Load wasiunet data obj
        dataframe_inst = load_wasiunet_data(asset=asset, pred_datetime=pred_datetime) 
        
        # Load wasiunet model
        wasiunet_model_trainer = load_wasiunet_model()
        
        if wasiunet_model_trainer == None:
            raise Exception("Could not load wasiunet model")
        
        # code to run prediction using the provided date, time and asset
        prediction = run_prediction(wasiunet_dataframe = dataframe_inst[0], wasiunet_model_trainer=wasiunet_model_trainer) #"Some predicted value"
        return prediction

    def save_prediction(prediction, date, time, asset):
        pred = Prediction(prediction=prediction, date=date, time=time, asset=asset)
        pred.save()
        return pred

    def get_prediction_by_date_time_asset(date, time, asset):
        return Prediction.objects(date=date, time=time, asset=asset).first()