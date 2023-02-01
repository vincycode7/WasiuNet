# prediction_utils.py
from utils.data_eng.data_util import historicCryptoBackend
def run_prediction(pred_datetime, asset):
    
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
    # Load data
    dataframe =  historicCryptoBackend(start_date=start_date, end_date=end_date, asset=asset, resolution=resolution,technical_analysis_config=ta_config,fea_output_per_data_slice=fea_output_per_data_slice, fea_data_slice=fea_data_slice,glob_time_step_forwards=glob_time_step_forwards, verbose=True)

    # Load ml model
    # cache ml model
    # run prediction
    # return prediction
    # code for running prediction goes here
    prediction_result = 'Predicted value: {}'.format(asset)
    return prediction_result