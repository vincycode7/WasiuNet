from mongoengine import Document, StringField, DateTimeField
from utils.prediction_utils import run_prediction, load_wasiunet_model, load_wasiunet_data
from datetime import datetime
import logging, os
logger = logging.getLogger(__name__)
class PredictionModel(Document):
    asset = StringField(required=True)
    pred_datetime = DateTimeField(required=True)
    prediction = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'predictions'}
    
    def run_prediction(self,pred_datetime, asset):
        try:
            # Load wasiunet data obj
            dataframe_inst = load_wasiunet_data(asset=asset, pred_datetime=pred_datetime) 
            dataframe_1 = dataframe_inst[0]
            logger.info(f"dataframe_1 = {dataframe_1}\n\n")
            # Load wasiunet model
            wasiunet_model_trainer = load_wasiunet_model()
            
            # if wasiunet_model_trainer == None:
            #     raise Exception("Could not load wasiunet model")
            
            # code to run prediction using the provided date, time and asset
            # prediction = run_prediction(wasiunet_dataframe = dataframe_inst[0], wasiunet_model_trainer=wasiunet_model_trainer) #"Some predicted value"
            # return prediction
        except Exception as e:
            raise ValueError(f"Could not make prediction because of exception {e} in run_prediction method")

    def save_prediction(prediction, date, time, asset):
        """
            Saves a prediction of a specific asset.
            
            Args:
                prediction (float): The prediction value.
                date (datetime.date): The date the prediction was made.
                time (datetime.time): The time the prediction was made.
                asset (str): The asset being predicted.
            
            Returns:
                pred (Prediction): The saved Prediction object.
        """
        pred = Prediction(prediction=prediction, date=date, time=time, asset=asset)
        pred.save()
        return pred

    def get_prediction_by_date_time_asset(date, time, asset):
        return Prediction.objects(date=date, time=time, asset=asset).first()