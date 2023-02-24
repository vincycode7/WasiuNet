# prediction_utils.py
from utils.data_eng.data_util import historicCryptoBackend
from configs.config import REDIS_DB_INST, MODEL_MAPPING
from functools import wraps
from hashlib import sha256
import pickle
import torch
from .model import WasiuNetTrainer
import logging, os
import gdown

logger = logging.getLogger(__name__)

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

# @wasiunet_model_cache
def load_wasiunet_model(model_name, **kwargs):
    # Code to load the model using the passed arguments
    # Wasiu Model (TODO: implement check for all the arguments and throw exception)
    logger.info(f"kwargs {kwargs}")
    
    asset = kwargs.get('asset',"BTC-USD")
    resolution = kwargs.get('resolution',60)
    fea_output_per_data_slice = kwargs.get('fea_output_per_data_slice',120)
    fea_data_slice = kwargs.get('fea_data_slice',12)
    glob_time_step_forwards= kwargs.get('glob_time_step_forwards',180)
    batch_size = kwargs.get('batch_size',1)
    num_worker = kwargs.get('num_worker',2)
    # model_name = kwargs.get('model_name',"DEFAULT_MODEL_URL")
    # print(f"MODEL_MAPPING: {MODEL_MAPPING}")
    model_url = MODEL_MAPPING.get(model_name)
    model_ext = kwargs.get('ext',".ckpt")
    model_path = kwargs.get('model_path',"model_checkpoints/wasiunet_model")
    full_model_path = model_path+"/"+model_name+model_ext
    enforce_cpu_use = kwargs.get('enforce_cpu_use', True)
    
    if enforce_cpu_use:
        device_name = "cpu"
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
    # device = torch.device(device_name)

    try:
        if not os.path.exists(model_path):
            logger.info(f"Creating directory {model_path}")
            os.makedirs(model_path)
        if not os.path.isfile(full_model_path):
            logger.info(f"Downloading model... {model_url}")
            if model_url is None:
                raise Exception("URL not found")
            else:
                gdown.download(model_url, full_model_path, quiet=False)
        else:
            logger.info(f"Model found at {model_path}. Skipping download.")
            # print(f"Model found at {model_path}. Skipping download.")
    except Exception as e:
        err_msg = f"Error {e} while loading model from path {full_model_path}"
        logger.error(err_msg)
        raise Exception(err_msg)
    else:
        try:
            print(f"Loading... File path: {full_model_path}")
            wasiunet_model_trainer = WasiuNetTrainer.load_from_checkpoint(full_model_path,device_name=device_name)
        except Exception as e:
            err = f"Error {e} occured while loading model checkpoint: {full_model_path}"
            raise ValueError(err)
        else:
            print("Model loaded successfully.")

    #     # Wasiu Model Attached to a Pytorch Lightening Trainer
    #     wasiunet_model_trainer = WasiuNetTrainer(model=wasiunet_model, lr=learning_rate).to(device)
        
    #     try:
    #         print(f"Loading... File path: {model_path}")
    #         wasiunet_model_trainer = wasiunet_model_trainer.load_from_checkpoint(model_path,model=wasiunet_model)
    #     except Exception as e:
    #         print(f"Error {e} occured while loading model checkpoint")
    #         return None
    #     else:
    #         print("Model loaded successfully.")

    # return wasiunet_model_trainer
 

def run_prediction(wasiunet_dataframe, wasiunet_model_trainer):


    # Run Predictions on 
    wasiunet_model_pred = wasiunet_model_trainer.predict(wasiunet_dataframe)
    
    # cache ml model
    # run prediction
    # return prediction
    # code for running prediction goes here
    prediction_result = 'Predicted value: {}'.format(1243)
    return prediction_result