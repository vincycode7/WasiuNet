from mongoengine import Document, StringField, DateTimeField
from utils.prediction_utils import run_prediction, wasiunet_model_cache
from datetime import datetime


class WasiunetModel:
    @wasiunet_model_cache
    def load_wasiunet_model(self, *args, **kwargs):
        # Code to load the model using the passed arguments
        # Wasiu Model
        asset = "BTC-USD"
        resolution = 60
        fea_output_per_data_slice = 120
        fea_data_slice = 12
        glob_time_step_forwards= 180
        batch_size = 50
        num_worker = 2
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        embedding_dim = 120 #240 #765 #120 # This is the num of input features expected into encoder and decoder.
        patch_size = 16
        in_channels = X.shape[-3] # num of timeframe : 12
        space_seq_len = X.shape[-2]
        expand_HW = 120 #240 #120 #
        inp_feat = X.shape[-1]
        out_feat = Y.shape[-1]
        nhead = 12 #9 #15 #12
        num_encoder_layers = 2 #2 #2 #12 # This is the num of Encoder transformers
        num_decoder_layers = 2 #2 #2 #12 # This is the num of Decoder transformers
        dim_feedforward = 1026 #2052 #3072 #1026 # # This is the num of feed forward output from the encoder decoder network
        dropout = 0.1
        max_len = fea_output_per_data_slice * fea_data_slice # 120 * 12
        trans_activation = "gelu"
        trans_norm_first = False
        trans_batch_first = True
        feat_map_dict = btc_usd_train.return_all_output_col_as_dict()
        
        wasiunet_model = WasiuNet(embedding_dim=embedding_dim, inp_feat=inp_feat, out_feat=out_feat, 
                        in_channels = in_channels, patch_size=patch_size,space_seq_len=space_seq_len,expand_HW=expand_HW,
                        nhead=nhead, num_encoder_layers=num_encoder_layers, 
                        num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, 
                        dropout=dropout, max_len=max_len, device=device, trans_activation=trans_activation,
                        trans_norm_first=trans_norm_first,trans_batch_first=trans_batch_first,
                        feat_map_dict=feat_map_dict
                        ).to(device)

        # Wasiu Model Attached to a Pytorch Lightening Trainer
        wasiunet_model_trainer = WasiuNetTrainer(model=wasiunet_model, lr=learning_rate).to(device)
        
        return None
    
class PredictionModel(Document):
    asset = StringField(required=True)
    pred_datetime = DateTimeField(required=True)
    prediction = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {'collection': 'predictions'}
    
    def run_prediction(self,pred_datetime, asset):
        # code to run prediction using the provided date, time and asset
        prediction = run_prediction(pred_datetime, asset, wasiunet_model) #"Some predicted value"
        return prediction

    def save_prediction(prediction, date, time, asset):
        pred = Prediction(prediction=prediction, date=date, time=time, asset=asset)
        pred.save()
        return pred

    def get_prediction_by_date_time_asset(date, time, asset):
        return Prediction.objects(date=date, time=time, asset=asset).first()