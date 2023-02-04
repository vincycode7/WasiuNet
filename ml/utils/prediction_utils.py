# prediction_utils.py
from utils.data_eng.data_util import historicCryptoBackend
from configs.config import REDIS_DB_INST, MODEL_MAPPING
from functools import wraps
from hashlib import sha256
import pickle
import torch

def wasiunet_model_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a unique key based on the arguments
        key = sha256(str((func.__name__, args, kwargs)).encode()).hexdigest()
        # Check if the result is already in cache
        result = REDIS_DB_INST.get(key)
        if result:
            # If yes, return the result
            return pickle.loads(result)
        else:
            # If not, call the function, cache the result and return it
            result = func(*args, **kwargs)
            REDIS_DB_INST.set(key, pickle.dumps(result))
            return result
    return wrapper

def load_wasiunet_data(asset, pred_datetime):
    assert type(pred_datetime) != type(None), "pred_datetime is not expected to be 'None' please fill in the correct value."
    pred_datetime = historicCryptoBackend.convert_date_to_backend_format(date=pred_datetime) # Expects a datatime, simply converts it to string
    
    ta_config = [
              # SMA - Simple Moving Average
              {"ta_func_name":"SMA", 'ta_func_config':{'window':60,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':50,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':45,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':20,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':10,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':7,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':5,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':3,'fillna':False}},

              # RSI
              {"ta_func_name":"RSI", 'ta_func_config':{'window':60,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':50,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':45,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':20,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':10,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':7,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':5,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':3,'fillna':False}},

              # STC
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':60,'window_slow':5, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':50,'window_slow':4, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':45,'window_slow':3, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':20,'window_slow':2, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':10,'window_slow':5, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':7,'window_slow':4, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':5,'window_slow':3, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':3,'window_slow':2, 'fillna':False}}
             
             ]
    # CCXT - Will use this for trading and analysis
    # Download data (this is just used to download the data and save it to current folder)
    # Data hyperparamters
    technical_analysis_config=ta_config
    resolution = 60
    fea_output_per_data_slice = 120
    fea_data_slice = 12
    glob_time_step_forwards= 180
    batch_size = 50
    num_worker = 2
    asset = "BTC-USD"
    start_date = pred_datetime
    end_date = pred_datetime
    # file_path = "inputs/historicCryptoBackend_asset=BTC-USD,resolution=60,start_date=2016-01-14 00:00:00, end_date=2023-01-08 00:00:00"        
    
    # Load data (Done)
    dataframe_inst =  historicCryptoBackend(start_date=start_date, end_date=end_date, asset=asset, resolution=resolution,technical_analysis_config=ta_config,fea_output_per_data_slice=fea_output_per_data_slice, fea_data_slice=fea_data_slice,glob_time_step_forwards=glob_time_step_forwards, verbose=True)
    return dataframe_inst

@wasiunet_model_cache
def load_wasiunet_model(*args, **kwargs):
    # Code to load the model using the passed arguments
    # Wasiu Model (TODO: implement check for all the arguments and throw exception)
    asset = kwargs.get('asset',"BTC-USD")
    resolution = kwargs.get('resolution',60)
    fea_output_per_data_slice = kwargs.get('fea_output_per_data_slice',120)
    fea_data_slice = kwargs.get('fea_data_slice',12)
    glob_time_step_forwards= kwargs.get('glob_time_step_forwards',180)
    batch_size = kwargs.get('batch_size',1)
    num_worker = kwargs.get('num_worker',2)
    model_url = kwargs.get('model_url', "https://drive.google.com/drive/u/0/folders/<file_id>")
    
    model_name = kwargs.get('model_name',"wasiunet_model")
    model_path = kwargs.get('model_path',"models/wasiunet_model")
    model_url = MODEL_MAPPING.get(model_name, None)
    enforce_cpu_use = kwargs.get('enforce_cpu_use', True)
    
    if enforce_cpu_use:
        device_name = "cpu"
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
    device = torch.device(device_name)
    embedding_dim = kwargs.get("embedding_dim",120) #240 #765 #120 # This is the num of input features expected into encoder and decoder.
    patch_size = kwargs.get("patch_size",16)
    in_channels = kwargs.get("fea_data_slice",12)#X.shape[-3] # num of timeframe : 12
    space_seq_len = kwargs.get("fea_output_per_data_slice",120) #X.shape[-2]
    expand_HW = kwargs.get("expand_HW", 120) #240 #120 #
    inp_feat = kwargs.get("X_input_feat", ) #X.shape[-1]
    out_feat = kwargs.get("Y_output_feaat", ) #Y.shape[-1]
    nhead = kwargs.get("nhead",12) #9 #15 #12
    num_encoder_layers = kwargs.get("num_encoder_layers",2) #2 #2 #12 # This is the num of Encoder transformers
    num_decoder_layers = kwargs.get("num_decoder_layers", 2) #2 #2 #12 # This is the num of Decoder transformers
    dim_feedforward = kwargs.get("dim_feedforward",1026) #2052 #3072 #1026 # # This is the num of feed forward output from the encoder decoder network
    dropout = kwargs.get("dropout", 0.1)
    max_len = fea_output_per_data_slice * fea_data_slice # 120 * 12
    trans_activation = kwargs.get("trans_activation","gelu")
    trans_norm_first = kwargs.get("trans_norm_first",False)
    trans_batch_first = kwargs.get("trans_batch_first",True)
    feat_map_dict = kwargs.get("feat_map_dict",None) #btc_usd_train.return_all_output_col_as_dict()
    
    try:
        if not os.path.exists(model_path):
            print("Downloading model...")
            if model_url is None:
                raise Exception("Model not found")
            else:
                gdown.download(model_url, model_path, quiet=False)
    except Exception as e:
        print(e)
        raise Exception(f"Model not found: {e}" % model_path)
    else:
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
        
        try:
            print(f"Loading... File path: {model_path}")
            wasiunet_model_trainer = wasiunet_model_trainer.load_from_checkpoint(model_path,model=wasiunet_model)
        except Exception as e:
            print(f"Error {e} occured while loading model checkpoint")
            return None
        else:
            print("Model loaded successfully.")

    return wasiunet_model_trainer
 

def run_prediction(wasiunet_dataframe, wasiunet_model_trainer):


    # Run Predictions on 
    wasiunet_model_pred = wasiunet_model_trainer.predict(wasiunet_dataframe)
    
    # cache ml model
    # run prediction
    # return prediction
    # code for running prediction goes here
    prediction_result = 'Predicted value: {}'.format(1243)
    return prediction_result