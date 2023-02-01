import requests, json, time, sys, aiohttp, torch, cProfile, random, gc

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl

from datetime import datetime, timedelta
from torch.utils.data import random_split
from ast import Return
from pytorch_lightning.utilities.model_summary import ModelSummary
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy, multiclass_precision, multiclass_recall
from torch.utils.data import DataLoader
from datetime import datetime
from random import randint
from collections import OrderedDict
from torchinfo import summary
from typing import Dict, Union, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

np.seterr(divide = 'ignore') 

class HistoricalData(object):
    """
    This class provides methods for gathering historical price data of a specified
    Cryptocurrency between user specified time periods. The class utilises the CoinBase Pro
    API to extract historical data, providing a performant method of data extraction.
    
    Please Note that Historical Rate Data may be incomplete as data is not published when no 
    ticks are available (Coinbase Pro API Documentation).
    :param: ticker: a singular Cryptocurrency ticker. (str)
    :param: granularity: the price data frequency in seconds, one of: 60, 300, 900, 3600, 21600, 86400. (int)
    :param: start_date: a date string in the format YYYY-MM-DD-HH-MM. (str)
    :param: end_date: a date string in the format YYYY-MM-DD-HH-MM,  Default=Now. (str)
    :param: verbose: printing during extraction, Default=True. (bool)
    :returns: data: a Pandas DataFrame which contains requested cryptocurrency data. (pd.DataFrame)
    """
    def __init__(self,
                 ticker,
                 granularity,
                 start_date,
                 end_date=None,
                 verbose=True):

        if verbose:
            print("Checking input parameters are in the correct format.")
        if not all(isinstance(v, str) for v in [ticker, start_date]):
            raise TypeError("The 'ticker' and 'start_date' arguments must be strings or None types.")
        if not isinstance(end_date, (str, type(None))):
            raise TypeError("The 'end_date' argument must be a string or None type.")
        if not isinstance(verbose, bool):
            raise TypeError("The 'verbose' argument must be a boolean.")
        if isinstance(granularity, int) is False:
            raise TypeError("'granularity' must be an integer object.")
        if granularity not in [60, 300, 900, 3600, 21600, 86400]:
            raise ValueError("'granularity' argument must be one of 60, 300, 900, 3600, 21600, 86400 seconds.")

        if not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d-%H-%M")
        else:
            end_date_datetime = datetime.strptime(end_date, '%Y-%m-%d-%H-%M')
            start_date_datetime = datetime.strptime(start_date, '%Y-%m-%d-%H-%M')
            while start_date_datetime >= end_date_datetime:
                raise ValueError("'end_date' argument cannot occur prior to the start_date argument.")

        self.ticker = ticker
        self.granularity = granularity
        self.start_date = start_date
        self.start_date_string = None
        self.end_date = end_date
        self.end_date_string = None
        self.verbose = verbose

    async def _ticker_checker(self):
        """This helper function checks if the ticker is available on the CoinBase Pro API."""
        if self.verbose:
            print("Checking if user supplied is available on the CoinBase Pro API.")

        async with aiohttp.ClientSession() as session:

              hist_url = "https://api.pro.coinbase.com/products".format(
                    self.ticker,
                    self.start_date_string,
                    self.end_date_string,
                    self.granularity)
              
              async with session.get(hist_url) as resp:
                tkr_response = resp
                text0 = await tkr_response.text()
                  # response = await resp.json()
              # tkr_response = resp
              # tkr_response = await session.get(hist_url)


        # tkr_response = requests.get("https://api.pro.coinbase.com/products")
        if tkr_response.status in [200, 201, 202, 203, 204]:
            if self.verbose:
                print('Connected to the CoinBase Pro API.')
            
            response_data = pd.json_normalize(json.loads(text0))
            ticker_list = response_data["id"].tolist()

        elif tkr_response.status in [400, 401, 404]:
            if self.verbose:
                print("Status Code: {}, malformed request to the CoinBase Pro API.".format(tkr_response.status))
            # sys.exit()
        elif tkr_response.status in [403, 500, 501]:
            if self.verbose:
                print("Status Code: {}, could not connect to the CoinBase Pro API.".format(tkr_response.status))
            # sys.exit()
        else:
            if self.verbose:
                print("Status Code: {}, error in connecting to the CoinBase Pro API.".format(tkr_response.status))
            # sys.exit()

        if self.ticker in ticker_list:
            if self.verbose:
                print("Ticker '{}' found at the CoinBase Pro API, continuing to extraction.".format(self.ticker))
        else:
            raise ValueError("""Ticker: '{}' not available through CoinBase Pro API. Please use the Cryptocurrencies 
            class to identify the correct ticker.""".format(self.ticker))

    def _date_cleaner(self, date_time: (datetime, str)):
        """This helper function presents the input as a datetime in the API required format."""
        if not isinstance(date_time, (datetime, str)):
            raise TypeError("The 'date_time' argument must be a datetime type.")
        if isinstance(date_time, str):
            output_date = datetime.strptime(date_time, '%Y-%m-%d-%H-%M').isoformat()
        else:
            output_date = date_time.strftime("%Y-%m-%d, %H:%M:%S")
            output_date = output_date[:10] + 'T' + output_date[12:]
        return output_date

    async def multiple_data_retrieve(self, i,start,request_volume,max_per_mssg):
      provisional_start = start + timedelta(0, i * (self.granularity * max_per_mssg))
      provisional_start = self._date_cleaner(provisional_start)
      provisional_end = start + timedelta(0, (i + 1) * (self.granularity * max_per_mssg))
      provisional_end = self._date_cleaner(provisional_end)
      if self.verbose:
        print("Provisional Start: {}".format(provisional_start))
        print("Provisional End: {}".format(provisional_end))
      # response = requests.get(
      #     "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
      #         self.ticker,
      #         provisional_start,
      #         provisional_end,
      #         self.granularity))
      # time.sleep(randint(0, 5))
      async with aiohttp.ClientSession() as session:
        hist_url = "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
              self.ticker,
              provisional_start,
              provisional_end,
              self.granularity)
        # time.sleep(randint(1,5))
        async with session.get(hist_url) as resp:
          response = resp
          text2 = await response.text()
        # response = await resp.json()
          # response = resp
        # response = await session.get(hist_url)

          if response.status in [200, 201, 202, 203, 204]:
              if self.verbose:
                  print('Data for chunk {} of {} extracted'.format(i+1,
                                                                    (int(request_volume / max_per_mssg) + 1)))
              # text2 = await response.text()
              dataset = pd.DataFrame(json.loads(text2), columns=["time", "low", "high", "open", "close", "volume"])
              # dataset.columns = ["time", "low", "high", "open", "close", "volume"]
              if not dataset.empty:
                  # data = data.append(dataset)
                  dataset["time"] = pd.to_datetime(dataset["time"], unit='s')
                  return dataset
              else:
                  print("""CoinBase Pro API did not have available data for '{}' beginning at {}.  
                  Trying a later date:'{}'""".format(self.ticker,
                                                      self.start_date,
                                                      provisional_start))
                  # If no data is returned, return zero instead
                  d_inx = pd.date_range(start=provisional_start, end=provisional_end, freq=f"{int(self.granularity/60)}"+"min")
                  dataset = pd.DataFrame(np.array(np.zeros((d_inx.shape[0], 6))), columns=["time", "low", "high", "open", "close", "volume"])
                  dataset["time"] = d_inx
                  # print(dataset)
                  return  dataset
          elif response.status in [400, 401, 404]:
              if self.verbose:
                  print(
                      "Status Code: {}, malformed request to the CoinBase Pro API.".format(response.status))
              # sys.exit()
          elif response.status in [403, 500, 501]:
              if self.verbose:
                  print(
                      "Status Code: {}, could not connect to the CoinBase Pro API.".format(response.status))
              # sys.exit()
          else:
              if self.verbose:
                  print("Status Code: {}, error in connecting to the CoinBase Pro API.".format(
                      response.status))
              # sys.exit()

    async def retrieve_data(self):
        """This function returns the data."""
        if self.verbose:
            print("Formatting Dates.")

        await self._ticker_checker()
        self.start_date_string = self._date_cleaner(self.start_date)
        self.end_date_string = self._date_cleaner(self.end_date)
        start = datetime.strptime(self.start_date, "%Y-%m-%d-%H-%M")
        end = datetime.strptime(self.end_date, "%Y-%m-%d-%H-%M")
        request_volume = abs((end - start).total_seconds()) / self.granularity
        # request_volume = 1000 # This is for test purpose, revert to previous code

        if request_volume <= -1: #(The original value here is 300)
            # response = requests.get(
            #     "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
            #         self.ticker,
            #         self.start_date_string,
            #         self.end_date_string,
            #         self.granularity))
            async with aiohttp.ClientSession() as session:

              hist_url = "https://api.pro.coinbase.com/products/{0}/candles?start={1}&end={2}&granularity={3}".format(
                    self.ticker,
                    self.start_date_string,
                    self.end_date_string,
                    self.granularity)
              
              async with session.get(hist_url) as resp:
                response = resp
                text = await response.text()
                  # response = await resp.json()
                  # response = resp
                  # print(pokemon['name'])
              # response = await session.get(hist_url)
            if response.status in [200, 201, 202, 203, 204]:
                if self.verbose:
                    print('Retrieved Data from Coinbase Pro API.')
                # text = await response.text()
                data = pd.DataFrame(json.loads(text))
                data.columns = ["time", "low", "high", "open", "close", "volume"]
                data["time"] = pd.to_datetime(data["time"], unit='s')
                data = data[data['time'].between(start, end)]
                data.set_index("time", drop=True, inplace=True)
                data.sort_index(ascending=True, inplace=True)
                data.drop_duplicates(subset=None, keep='first', inplace=True)
                if self.verbose:
                    print('Returning data.')
                return data
            elif response.status in [400, 401, 404]:
                if self.verbose:
                    print("Status Code: {}, malformed request to the CoinBase Pro API.".format(response.status))
                # sys.exit()
            elif response.status in [403, 500, 501]:
                if self.verbose:
                    print("Status Code: {}, could not connect to the CoinBase Pro API.".format(response.status))
                # sys.exit()
            else:
                if self.verbose:
                    print("Status Code: {}, error in connecting to the CoinBase Pro API.".format(response.status))
                # sys.exit()
        else:
            # The api limit:
            max_per_mssg = 300
            data = pd.DataFrame()
            chunk_size = 5
            total_idx = int(request_volume / max_per_mssg) + 1
            all_idx = list(range(total_idx))
            for idx in range(0,total_idx, chunk_size):
              loop = True
              while loop:
                results = []
                if self.verbose:print(f"{idx}")
                tasks = [self.multiple_data_retrieve(i,start, request_volume, max_per_mssg) for i in all_idx[idx:idx+chunk_size]]
                results = await asyncio.gather(*tasks)
                for dataset in results:
                  if isinstance(dataset, type(None)):
                    loop = True
                    time.sleep(5)
                    break
                  loop = False

              for dataset in results:
                  if not dataset.empty:
                    data = data.append(dataset)
              time.sleep(randint(0, 2))
            # print(f"data --> {data}")
            data.columns = ["time", "low", "high", "open", "close", "volume"]
            data.drop_duplicates('time',keep='last', inplace=True)

            data["time"] = pd.to_datetime(data["time"], unit='s')
            data = data[data['time'].between(start, end)]
            data.set_index("time", drop=True, inplace=True)
            data.sort_index(ascending=True, inplace=True)
            # data.drop_duplicates(subset=None, keep='first', inplace=True)
            return data
        
## Import required libraries
from datetime import datetime
from torch.utils.data import Dataset
# from Historic_Crypto import HistoricalData
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch as tch
import ta
import asyncio
import time

class DatasetBaseBackend(Dataset):
  def __init__(self, asset: str, resolution: Union[int,float,str], start_date: str, end_date: str, glob_time_step_forwards: int = 7, 
                glob_time_step_backwards: Optional[int] = None, fea_output_per_data_slice: int = 30, fea_data_slice: int = 7, 
                technical_analysis_config: Optional[List[Dict]] = None,**kwarg)-> None:
        """
        Initialize a custom dataset for trading with basic methods required to extend to a new dataset.

        :param asset: The  Exchage  pair to pull e.g BTC-USD (Backend specific parameter, read backend doc to know format)
        :param resolution: Timeframe between data in seconds. E.g 1m == 60, 2m == 120 (Backend specific parameter, read backend doc to know format)
        :param start_date: Date for dataset object to start pulling data from. Expected format '%Y-%m-%d-%H-%M-%S'
        :param end_date: Date for dataset object to stop pulling data. Expected format '%Y-%m-%d-%H-%M-%S'
        :param glob_time_step_forwards: Number of steps forward from start_date. Default is 7.
        :param glob_time_step_backwards: Number of steps backward from start_date. Default is None.
        :param technical_analysis_config: List of dictionaries for each ta function {"ta_func_name":"RSI", ta_func_config:{}}.
        :param fea_data_slice: Number of data slices a single data point should have.
        :param fea_output_per_data_slice: Number of data points in each data slice.

        """

        self.start_date = datetime.strptime(start_date, '%Y-%m-%d-%H-%M-%S')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d-%H-%M-%S') + timedelta(seconds=resolution) 
        self.asset = asset
        self.fea_output_per_data_slice = fea_output_per_data_slice
        self.fea_data_slice = fea_data_slice
        self.resolution = resolution
        self.glob_time_step_forwards = glob_time_step_forwards

        if isinstance(self.resolution, (int, float)):
          self.normal_step = self.resolution
        else:
          # TODO: map string and convert string to respective integer value in minutes
          #        this will be used in the getitem method to get ith index data point.
          pass

        if glob_time_step_backwards is None:
            self.glob_time_step_backwards = (self.backward_step_func(fea_data_slice)*fea_output_per_data_slice)*5
        else:
            self.glob_time_step_backwards = glob_time_step_backwards

        # print(self.glob_time_step_backwards)
        self.set_available_technical_analysis_functions()
        self.set_technical_analysis_for_dataloader(technical_analysis_config)
        self.data_scaled = False
        self._extra_direction_fields = ["direction_-1_conf","direction_0_conf","direction_1_conf"]
        self._required_base_features = ['close','high','low','open','volume']
        self._required_base_target = ['close']
        self.squeeze_forward = kwarg.get('squeeze_forward',False)
        self.init_preload(**kwarg)
  def convert_date_from_backend_format(self,date, format=None):
    """
      This method is used to convert from custom dataset datetime format to 
      backend format, If data is datetime convert to backend string else if 
      data is string leave as it is.

      date: a datetime string
      format: a datetime string format e.g '%Y-%m-%d-%H-%M-%S'
    """
    return datetime.strptime(date, format)

  @staticmethod
  def convert_date_to_backend_format(date, format='%Y-%m-%d-%H-%M-%S'):
    """
      This method is used to convert date from str to the required date format 
      for the dataloader

      date: a datetime object,
      format: the format in which the  date is  in e.g '%Y-%m-%d-%H-%M-%S'
    """
    if isinstance(date, datetime):
      assert isinstance(format, str), "format should be in string format, please check"
      return date.strftime(format)
    else:
      return date

  def add_custom_targets(self):
    raise NotImplementedError("Sub-class Should implement this method")

  def add_all_targets(self,data_in_required_format_with_ta):
    raise NotImplementedError("Sub-class Should implement this method")
    
  def get_backend(self):
    """
      This method returns the current backend used to access the crypto data
    """
    return self.backend

  def get_required_base_index_name(self):
    """
      These are the required index name to be able to use the inner methods.
    """
    return ['time']

  def get_extra_direction_fields(self):
    """
      These are fields that were added during the process of data processing.
    """
    return self._extra_direction_fields

  def get_required_base_features(self):
    """
      These are the required features to be able to use the inner methods.
    """
    return self._required_base_features

  def get_required_base_target(self):
    """
      These are the required target to be able to use the inner methods.
    """
    return self._required_base_target

  def get_supported_backends(self):
    """
      This method returns a list of strings of supported backends used to access 
      data
    """
    return ['historicCryptoBackend']

  # def backend_get_all_raw(self,asset=None, resolution=None, start_date=None, end_date=None):
  #   """
  #     Implement this function from each backend
  #   """
  #   raise NotImplementedError("Please implement this method in your dataset to return the required datapoints")

  def get_all_raw(self, asset=None, resolution=None, start_date=None, end_date=None):
    """
      This method returns the raw dataframe for the current backend in the specified 
      def backend_get_all_raw(self,asset=None, resolution=None, start_date=None, end_date=None)
      timeframe. This function implicitly calls the backend_get_all_raw function to run the 
      backend specific implementation. 

    """
    try:
      return self.backend_get_all_raw(asset=asset, resolution=resolution, start_date=start_date, end_date=end_date)
    except Exception as e:
      raise Exception(f"Error {e} occured while getting data from backend_get_all_raw method")

  def set_backend(self, name):
    """
      This method is used to set the  current backend used to access crypto data
    """
    self.backend = name 

  def __len__(self):
    """
      Used to get the supposed length  of the data set assuming all columns were 
      loaded and market traded every seconds and minutes of the day.
    """
    num_of_sec_diff = self.end_date - self.start_date
    num_of_expected_samps = int(num_of_sec_diff.total_seconds()/60)
    return num_of_expected_samps


  def expand_dset_to_time(self, data: pd.DataFrame, idx: int, outs: int, steps: int, forward: bool, include_current_idx: bool):
      """
      This method slices the dataframe to include the current index and the number of steps specified in 'outs'
      
      Parameters:
          data (pd.DataFrame): The dataframe to slice
          idx (int): The index of the current step in the dataframe
          outs (int): The number of steps to include in the slice
          steps (int): The step size for the slice
          forward (bool): Determines whether to slice the dataframe forward or backward
          include_current_idx (bool): Determines whether to include the current index in the slice
      
      Returns:
          pd.DataFrame: The sliced dataframe
      """
      # basic input validation
      if idx >= len(data):
          raise ValueError("idx should be less than the length of dataframe")
      if steps <= 0:
          raise ValueError("steps should be greater than zero")
      if outs <= 0:
          raise ValueError("outs should be greater than zero")

      if forward:
          start = idx + (steps - include_current_idx)
          stop = idx + (outs * steps) + include_current_idx
      else:
          start = idx - (outs * steps) + steps
          stop = idx + include_current_idx

      new_data = data.iloc[start:stop:steps]
      return new_data.copy()


  def get_available_technical_analysis_functions(self, return_func=True):
    """
      This returns all the names used to access all technical analysis that this 
      dataloaded currently handles.
    """
    assert self.set_available_technical_analysis_functions_flag, "Call the `set_available_technical_analysis_functions` method to initialize the technical functoins"
    if return_func:
      return self.available_technical_analysis_functions
    return self.available_technical_analysis_functions.keys()

  def set_available_technical_analysis_functions(self):
    self.available_technical_analysis_functions = {
                                                   "SMA" : {"name":"SMAIndicator",'func':ta.trend.SMAIndicator,'data_cols':['close']},

                                                   "RSI" : {
                                                            "name":"RSI Momentum Indicator",
                                                            'func':ta.momentum.RSIIndicator,'data_cols':['close']},
                                                   "STC" : {
                                                            "name" : "STCIndicator", 'func':ta.trend.STCIndicator,'data_cols':['close']
                                                   },
                                                   
                                                   "CCI" : {
                                                            "name":"Commodity Channel Index",
                                                            'func':ta.trend.CCIIndicator,'data_cols':['close']},
                                                    
                                                   "AO"  : {
                                                            "name":"Awesome Oscillator",
                                                            'func':ta.momentum.AwesomeOscillatorIndicator,'data_cols':['close']},
                                                   
                                                   "MACD": {
                                                            "name":"Moving Average Convergence Divergence",
                                                            'func':ta.trend.MACD,'data_cols':['close']},
                                                   
                                                   "ATR" : {
                                                            "name":"Average True Range",
                                                            'func':ta.volatility.AverageTrueRange,'data_cols':['close']},
                                                   
                                                   "OBVI" : {
                                                            "name":"On Balance Volume Indicator",
                                                            'func':ta.volume.OnBalanceVolumeIndicator,'data_cols':['close']},
                                                   
                                                   "KAMA" : {
                                                              "name":"KAMA Indicator",
                                                              'func':ta.momentum.KAMAIndicator,'data_cols':['close']},
                                                   
                                                   "ADX"  : {
                                                            "name":"Directional Movement",
                                                            'func':ta.trend.ADXIndicator,'data_cols':['close']},
                                                   
                                                   "STOCH":{"name":"Stochastic Oscillation",
                                                            'func':ta.momentum.StochasticOscillator,'data_cols':['close']},

                                                   "STOCHRSI":{"name":"Stochastic RSI",
                                                               'func':ta.momentum.StochRSIIndicator,'data_cols':['close']},
                                                   
                                                   "WILLP":{"name":"William's %R",
                                                            'func':ta.momentum.WilliamsRIndicator,'data_cols':['close']},
                                                   "ARN":{"name":"AroonIndicator",'func':ta.trend.AroonIndicator,'data_cols':['close']},
                                                   "MASSIDX":{"name":"MassIndex",'func':ta.trend.MassIndex,'data_cols':['close']},
                                                   "PSAR":{"name":"PSARIndicator",'func':ta.trend.PSARIndicator,'data_cols':['close']}
                                                   }
    self.set_available_technical_analysis_functions_flag =  True

  def get_current_technical_analysis_config_for_dataloader(self):
    return self.technical_analysis_for_dataloader

  def set_technical_analysis_for_dataloader(self, config):
    self.technical_analysis_for_dataloader = config

  def extract_final_ta_level_data(self, final_ta_data_obj):
    """
      This method is ta specific, to implement the final data extract that the ta returns
    """
    if isinstance(final_ta_data_obj, ta.trend.SMAIndicator):
      return final_ta_data_obj.sma_indicator()
    if isinstance(final_ta_data_obj, ta.momentum.RSIIndicator):
      return final_ta_data_obj.rsi()
    if isinstance(final_ta_data_obj, ta.trend.STCIndicator):
      return final_ta_data_obj.stc()
    if isinstance(final_ta_data_obj, ta.trend.CCIIndicator):
      return final_ta_data_obj.cci()
    if isinstance(final_ta_data_obj, ta.momentum.AwesomeOscillatorIndicator):
      return final_ta_data_obj.awesome_oscillator()

  def get_all_ta_cofig_output_columns(self):
    cols = set()
    ta_config = self.get_current_technical_analysis_config_for_dataloader()
    for each_ta in ta_config: cols.add(self.ta_to_col_name(each_ta))
    return list(cols)

  def ta_to_col_name(self, each_ta: Dict[str, Union[str, Dict[str, Any]]]) -> str:
      """
      This function takes in the technical config for each ta dict and the hyper 
      parameter and returns the column name to represent that ta in the data frame
      
      Parameters:
          each_ta (Dict[str, Union[str, Dict[str, Any]]]) : A dictionary containing the configuration 
          of the technical analysis, containing 'ta_func_name' and 'ta_func_config' keys
          
      Returns:
          str : A string representing the column name of that ta

      """
      col_hyper = each_ta.get('ta_func_config', {})
      ta_func_name = each_ta.get('ta_func_name')
      return "-".join(["ta_func:"+ta_func_name]+[f"{ta_key}:{ta_value}" for ta_key, ta_value in col_hyper.items()])

  def add_technical_analysis_to_dataset(self,data_in_required_format):
    """
      This method is responsible for taking in the raw data in a required format, 
      loops through the ta config and adds the ta to the dataset, while also storing 
      their column names.
    """
    # Get the required fields
    # Check for errors in features names, target and index name provided
    try:
      # print(f"data_in_required_format: {data_in_required_format}")
      assert isinstance(data_in_required_format, pd.DataFrame), "Please `data_in_required_format` should be of type pd.DataFrame"
      assert set(data_in_required_format.columns) == set(self.get_required_base_features()+ self.get_required_base_target()), "The feature and target provided does not match the required fields, please run self.get_required_base_features() and self.get_required_base_target()"
      assert set([data_in_required_format.index.name]) == set(self.get_required_base_index_name()), "The index name provided does not match the required index name, please run self.get_required_base_index_name() to see the required index name"
    except Exception as e:
      raise ValueError(f"Error {e} happened when trying to check the quality of data provided")

    # parse the technical analysis config provided to add the new columns to dataframe.
    ta_config =  self.get_current_technical_analysis_config_for_dataloader() # {'ta_func_name': 'SMA', 'ta_func_config': {'window': 7, 'fillna': True}}
    system_ta =   self.get_available_technical_analysis_functions() # "SMA" : {"name":"SMAIndicator",'func':ta.trend.SMAIndicator}
    # This is where we dynamically pick the parameters from the config and load into the ta class
    for each_ta in ta_config:
      ta_function = system_ta.get(each_ta.get('ta_func_name'))
      col_param = {each_param : data_in_required_format[each_param] for each_param in ta_function.get('data_cols')} 
      col_hyper = each_ta.get('ta_func_config')
      total_col_param = {**col_param, **col_hyper}
      ta_obj_result = ta_function.get('func')(**total_col_param)

      # Save each new ta extract with the config name and place in the data format to return
      # ta_name = "-".join(["ta_func:"+c]+[str(ta_key)+":"+str(ta_value) for ta_key, ta_value in col_hyper.items()])
      ta_name = self.ta_to_col_name(each_ta)
      data_in_required_format[ta_name] = self.extract_final_ta_level_data(ta_obj_result)
    data_in_required_format = data_in_required_format.fillna(0)
    data_in_required_format = data_in_required_format[set(self.get_required_base_features()+self.get_required_base_target()+self.get_all_ta_cofig_output_columns())]
    # Y = data_in_required_format[self.return_all_output_col()]
    data_in_required_format = self.replace_inf(data_in_required_format)
    return data_in_required_format

  def return_all_output_col(self, get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=True, get_extra_direction_fields=True):
    """
      This method returns all the expected columns in a dataset
    """
    cols = set()
    if get_required_base_features:
      cols = cols.union(self.get_required_base_features())
    if get_required_base_target:
      cols = cols.union(self.get_required_base_target())
    if get_all_ta_cofig_output_columns:
      cols = cols.union(self.get_all_ta_cofig_output_columns())
    if get_extra_direction_fields:
      cols = cols.union(self.get_extra_direction_fields())
    
    return sorted(list(cols))

  def return_all_output_col_as_dict(self, **kwarg):
    return_all_output_col_as_dict = self.return_all_output_col(**kwarg)
    return {key : value for value, key in enumerate(return_all_output_col_as_dict)}
                                                    
  def get_idx_start_back_end_date(self,idx, forward_retry_value=0):
    """
      This method converts int index into timespace
      in the spcified range, where idx zero is the first
      time space in the specified range.
      It returns idx_start_date which specifies where to start
      pulling data from for a single index, It also returns 
      idx_backwards_date which is by how far the data pull should go back in time
      from the start date and lastly it returns idx_end_date which is by how much 
      data should go forward in time from the start date.

      forward_retry_value: This helps prevent cases when api package does not give an output,
      so we can manually expand the end date and rety the pull.
    """
    # Date we are currently interested in (This is the specific day we would like to index out)
    idx_start_date = self.start_date + pd.Timedelta(seconds=idx*self.normal_step)
    # print("Starting date ",idx_start_date,"--",self.start_date,"--", pd.Timedelta(seconds=idx*self.normal_step))

    # print(self.glob_time_step_backwards, self.normal_step, self.glob_time_step_backwards * self.normal_step)
    # Date to start pulling data from (Given the structure of the output of the data and it's partition, this is how far back we would like to go back in time))
    idx_backwards_date = idx_start_date - pd.Timedelta(seconds=(self.glob_time_step_backwards)*self.normal_step)# - pd.Timedelta(seconds=3000*self.normal_step)
    # print("Backward date ","--",self.glob_time_step_backwards,"--",self.normal_step,"--", idx_backwards_date,"--",idx_start_date,"--", pd.Timedelta(seconds=self.glob_time_step_backwards*self.normal_step))
    
    # Date to stop pulling data from (Given the structure of the output of the data and it's partition,  this is how further into the future we would like to go in time)
    idx_end_date = idx_start_date + pd.Timedelta(seconds=forward_retry_value+self.glob_time_step_forwards*(self.normal_step**2))# + pd.Timedelta(seconds=3000*self.normal_step)
    # print("End date ","--",self.glob_time_step_forwards,"--",self.normal_step,"--",idx_end_date,"--", idx_start_date,"--", pd.Timedelta(seconds=self.glob_time_step_forwards*self.normal_step))

    return idx_start_date, idx_backwards_date, idx_end_date

  def backward_step_func(self,i):
    """
      This is the backward slicing step for each slice in a one data index.
      Say index zero give 7 different slices of data as output for train.
      First slice will have backward step of (1+0)**2, second slice will 
      have a backward step of (1+1)**2 and so on till it gets to the final
      slice 6 which will give (1+6)**2.
    """
    step_func = (1+i)**2
    # print(f"step func --> {step_func}")
    return step_func

  def search_time_index(self, data, time):
    """
      This functionality search for the
    """
    return data.index.searchsorted(time)

  def get_multi_slice_backward_output(self, X, current_level_index):
    # print(self.backward_step_func(self.fea_data_slice)*self.fea_output_per_data_slice , X.shape[0])
    assert (self.backward_step_func(self.fea_data_slice)*self.fea_output_per_data_slice) < X.shape[0], f"number of rows {X.shape[0]} less than required which should be at least {self.backward_step_func(self.fea_data_slice)*self.fea_output_per_data_slice}, Please check your glob_time_step_backwards parameter"
    # if to_numpy:
    #   all_inputs = [self.expand_dset_to_time(data=X, idx=current_level_index, outs=self.fea_output_per_data_slice, steps=self.backward_step_func(i),forward=0,include_current_idx=1).to_numpy() for i in range(self.fea_data_slice)]
    #   all_inputs = np.stack(all_inputs, axis=0)
    #   all_inputs = np.expand_dims(all_inputs, axis=0)
    # else:
    all_inputs = [self.expand_dset_to_time(data=X, idx=current_level_index, outs=self.fea_output_per_data_slice, steps=self.backward_step_func(i),forward=0,include_current_idx=1) for i in range(self.fea_data_slice)]

    return all_inputs

  def get_single_slice_forward_output(self, Y, current_level_index):
    single_output = self.expand_dset_to_time(data=Y, idx=current_level_index, outs=1, steps=self.glob_time_step_forwards,forward=1,include_current_idx=0)
    return single_output

  def get_multi_slice_forward_output(self, Y,current_level_index, to_numpy=True):
    # if to_numpy:
    #   single_output = self.expand_dset_to_time(data=Y, idx=current_level_index, outs=1, steps=self.glob_time_step_forwards,forward=1,include_current_idx=0).to_numpy()
    # else:
    single_output = self.expand_dset_to_time(data=Y, idx=current_level_index, outs=self.glob_time_step_forwards, steps=1,forward=1,include_current_idx=1)
    return single_output

  def retry_func(self, curr_val):
    return ((curr_val+self.normal_step)**curr_val)+1

  def async_get_data_by_idx(self, idx, process_data=True, add_ta=True, apply_nat_log=False, fill_na=True, cal_pct_chg=False, to_numpy=True):
    if idx < 0: idx = len(self) + idx
    if idx >= len(self) or idx < 0: raise IndexError("list index out of range")
    # print(f"started task {idx}")
    retry_count = 0
    max_retry = 4

    # while retry_count < max_retry:
    #   # Get all data in past 3000 minutes and future 3000 minutes
    #   try:
    #     print("in get all")
    #     idx_start_date, idx_backwards_date, idx_end_date = self.get_idx_start_back_end_date(idx, forward_retry_value=self.retry_func(retry_count)) #convert the index into actual start, backward and enddate 
    #     data_in_required_format = self.get_all_raw(self.asset, self.resolution, idx_backwards_date, idx_end_date)
    #     break
    #   except Exception as e:
    #     print(e)
    #   retry_count += 1

    idx_start_date, idx_backwards_date, idx_end_date = self.get_idx_start_back_end_date(idx, forward_retry_value=1) #convert the index into actual start, backward and enddate 
    # print(f"idx_start_date: {idx_start_date}")
    try:
      data_in_required_format = self.get_all_raw(self.asset, self.resolution, idx_backwards_date, idx_end_date)
    except Exception as e:
      raise Exception(f" Error {e} occured while getting record for data point with id {idx}")

    data_in_required_format = self.process_data_raw(data_in_required_format,add_ta=add_ta, apply_nat_log=apply_nat_log, fill_na=fill_na, cal_pct_chg=cal_pct_chg)
    data_in_required_format = data_in_required_format[self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=add_ta, get_extra_direction_fields=cal_pct_chg)]
    # # return base_data
    # if process_data:
    #   data_in_required_format = self.process_data_raw(data_in_required_format,add_ta=add_ta, apply_nat_log=apply_nat_log, fill_na=fill_na, cal_pct_chg=cal_pct_chg)
    #   data_in_required_format = data_in_required_format[self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=add_ta, get_extra_direction_fields=False)]
    # else:
    #   data_in_required_format = data_in_required_format[self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=False, get_extra_direction_fields=False)]
    # return [X,Y]

    # Get current level index id
    current_level_index = self.search_time_index(data_in_required_format, idx_start_date)
    
    # Get all input time step 
    all_inputs = self.get_multi_slice_backward_output(data_in_required_format, current_level_index)
    hold_col = self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=add_ta, get_extra_direction_fields=False)
    all_inputs = [all_input[hold_col] for all_input in all_inputs]

    if self.squeeze_forward:
    #   # Get output
      output_data = self.get_single_slice_forward_output(data_in_required_format, current_level_index)
      if not cal_pct_chg: # Add trend
        output_data.loc[output_data.index[0],self.get_extra_direction_fields()] = self.direction_close(all_inputs[-1].copy().tail(1).close.to_numpy()[0], output_data.copy().close.to_numpy()[0])
      output_data = output_data[self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=add_ta, get_extra_direction_fields=True)]

    else:
      output_data = self.get_multi_slice_forward_output(data_in_required_format, current_level_index)
      if not cal_pct_chg:
        import functools
        def direction_close(close_next, close_prev=None):
          return self.direction_close(close_prev=close_prev, close_next=close_next)

        # Create a partial function with a default threshold value
        direction_close = functools.partial(direction_close, close_prev=all_inputs[-1]['close'].iloc[-1])
        
        # Apply the comparison function to column A of the dataframe and create a new column
        output_data = output_data.assign(
            **{name: output_data['close'].apply(direction_close).str[i] for i, name in enumerate(self.get_extra_direction_fields())}
        )    
        output_data = output_data[self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=add_ta, get_extra_direction_fields=True)]


    if to_numpy:
      all_inputs = np.array(all_inputs)
      output_data = np.array(output_data)
    return [all_inputs,output_data]

  def slice_data_get_old(self, idxs, process_data=True,add_ta=True, apply_nat_log=False, fill_na=True, cal_pct_chg=False, to_numpy=True):
    background_tasks = set()
    results = []
    for idx in idxs:
      if not self.preload_data_file:
        time.sleep(3)
      results.append(self.async_get_data_by_idx(idx=idx, process_data=process_data,add_ta=add_ta, apply_nat_log=apply_nat_log, fill_na=fill_na, cal_pct_chg=cal_pct_chg,to_numpy=to_numpy))
      if not self.preload_data_file:
        time.sleep(3)
    return results
    
  def slice_data_get(self, idxs: List[int], process_data: bool = True, add_ta: bool = True, apply_nat_log: bool = False,
                    fill_na: bool = True, cal_pct_chg: bool = False, to_numpy: bool = True) -> List:
      """
      This function retrieves the data from the given indexes.
      
      Parameters:
          idxs (List[int]) : list of indexes to retrieve the data from
          process_data (bool) : Flag for processing data before returning. default is True
          add_ta (bool) : Flag for adding technical analysis to the data. default is True
          apply_nat_log (bool) : Flag for applying natural log to the data. default is False
          fill_na (bool) : Flag for filling the missing data. default is True
          cal_pct_chg (bool) : Flag for calculating the percent change of the data. default is False
          to_numpy (bool) : Flag for converting the data to numpy array. default is True
      
      Returns:
          List : a list containing the data
      """
      # results = []
      # for idx in idxs:
      #     results.append(self.async_get_data_by_idx(idx=idx, process_data=process_data,add_ta=add_ta,
      #                                             apply_nat_log=apply_nat_log, fill_na=fill_na,
      #                                             cal_pct_chg=cal_pct_chg,to_numpy=to_numpy))
      from concurrent.futures import ThreadPoolExecutor, as_completed
      with ThreadPoolExecutor() as executor:
        futures = [executor.submit(self.async_get_data_by_idx, idx, process_data, add_ta, apply_nat_log, fill_na, cal_pct_chg, to_numpy) for idx in idxs]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
      return results

  def __getitem__(self,idxs):
    return self.get_item(idxs)

  def get_item(self, idxs, process_data=True,add_ta=True, apply_nat_log=False, fill_na=True, cal_pct_chg=False, to_numpy=True):
    """
      Assuming all data points starting from the start date to the end date were 
      available, this method selects the nth row from the full data set.
    """
    import nest_asyncio
    nest_asyncio.apply()
    # print(f"Checking slice: {idxs}")
    if isinstance(idxs, list):  
      pass
    elif isinstance(idxs, slice):  
      start, stop, step = idxs.start or 0, idxs.stop or len(self), idxs.step or 1
      # print("start, stop, step",start, stop, step)
      idxs = range(start, stop, step)
    elif isinstance(idxs, int):  
      idxs = [idxs]
    else:
      raise ValueError("Type slice of int required")

    # Get data
    try:
      # loop = asyncio.get_event_loop()
      # background_tasks = self.slice_data_get(idxs)
      # background_tasks = loop.run_until_complete(background_tasks)
      # background_tasks = asyncio.run(self.slice_data_get(idxs))
      background_tasks = self.slice_data_get(idxs, process_data=process_data,add_ta=add_ta,apply_nat_log=apply_nat_log, fill_na=fill_na, cal_pct_chg=cal_pct_chg, to_numpy=to_numpy)
      # loop.close()
    except Exception as e:
      raise Exception(f"Exception {e} occurred while getting result from self.slice_data_get")
    
    # Extract data
    X,Y = [],[]
    for each_task in background_tasks:
      X.append(each_task[0])
      Y.append(each_task[1])
      # Y2.append(each_task[2])

    # Expand data dim
    if to_numpy:
      try:
        X = np.stack(X, axis=0)
        # X = X.squeeze(1)

        Y = np.stack(Y, axis=0)
        # Y = Y.squeeze(1) 

      except Exception as e:
        raise Exception(f"Exception {e} occurred while expanding dimension of output X of shape {X.shape} and Y of shape {Y.shape}")

    # for each in background_tasks: print(each)
    return [X,Y]

  

  def get_real_start_timeframe(self, start=None):
    global_start_time = self.start if not start else start
    walk_forward_time = self.start + pd.Timedelta(minutes=1500)
    base_data = self.qb.History(self.symbol, global_start_time, walk_forward_time).loc[self.symbol].iloc[[0]].index
    return base_data
  
  def get_real_end_timeframe(self, end=None):
    global_end_time = self.end if not end else end
    walk_backward_time = global_end_time - pd.Timedelta(minutes=5000)
    base_data = self.qb.History(self.symbol, walk_backward_time, global_end_time).loc[self.symbol].iloc[[-1]].index
    return base_data

  def replace_inf(self, X):
      # Replace inf with max value for each feature column
      X = X.clip(lower=X.min(), upper=X.max(), axis=1)
      return X

  def cal_pct_chg(self,X,fill_na):
    # Fill na with zeros
    if fill_na: X = X.pct_change().fillna(0)
    return self.replace_inf(X)

  def direction_close(self,close_prev,close_next):
    if close_prev < close_next:
      return [0,0,1]
    elif close_prev > close_next:
      return [1,0,0]
    else:
      return [0,1,0]

  def direction_from_pct_chg(self,x):
    if x < 0:
      return [1,0,0]
    elif x > 0:
      return [0,0,1]
    else:
      return [0,1,0]

  def apply_nat_log(self, X):
    X = X.apply(lambda x:np.log(x))
    X = self.replace_inf(X)
    return X

  def process_data_raw(self, data_in_required_format, add_ta=True, apply_nat_log=False, fill_na=True, cal_pct_chg=False):
    # print("In process_data_raw")
    if add_ta:
      # Add technical analysis
      data_in_required_format = self.add_technical_analysis_to_dataset(data_in_required_format)
    # print("after ta")
    #Apply natural log to all numerical data features
    if apply_nat_log:
      data_in_required_format = self.apply_nat_log(data_in_required_format)
    # print("after log")

    #Convert to pct_change
    if cal_pct_chg:
      # Replace inf with max value for each feature column
      # print("Before cal_pct_chg")
      data_in_required_format = self.cal_pct_chg(data_in_required_format,fill_na=fill_na)
      data_in_required_format.loc[data_in_required_format.index, self.get_extra_direction_fields()] =  data_in_required_format["close"].apply(lambda x: self.direction_from_pct_chg(x)).to_list()
      # print("After cal_pct_chang")
      # print(data_in_required_format['direction'])
      # if 'direction' not in self._extra_feature_target_fields:
      #   self._extra_feature_target_fields.append('direction')
    return data_in_required_format[self.return_all_output_col(get_required_base_features=True, get_required_base_target=True, get_all_ta_cofig_output_columns=add_ta, get_extra_direction_fields=cal_pct_chg)]

  def scale_raw_data(self, X):
    """
      Method used to scale data
    """
    assert self.data_scaled, "Data has not been transformed, please run self.partially_scale_data()"
    x_shape = X.shape
    scaled_data = self.scaler.transform(X.reshape(-1, X.shape[-1]))
    return scaled_data.reshape(x_shape)

  def partially_scale_data(self):
    """
      Due to the volumn of the data if we were to scale in sec and minute, this 
      method is used to loop through the data in chunks and scale from train 
      start_date to train end_date.
    """
    self.data_scaled = False
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    step = 10
    for idx in range(step,len(self), step):
      print(idx-step,idx)
      X = self[idx-step:idx][0]
      scaler.partial_fit(X.reshape(-1, X.shape[-1]))
    self.scaler = scaler
    self.data_scaled = True
    return scaler

  def backend_get_all_raw(self, asset=None, resolution=None, start_date=None, end_date=None, **kwarg):
    # print("ya",start_date,end_date)
    (asset, resolution, start_date, end_date) = (asset if asset else self.asset, resolution if resolution else self.resolution,\
    start_date if start_date else self.start_date,end_date if end_date else self.end_date)
    # print("ya",start_date,end_date)
    start_date = DatasetBaseBackend.convert_date_to_backend_format(start_date, format='%Y-%m-%d-%H-%M')
    end_date = DatasetBaseBackend.convert_date_to_backend_format(end_date, format='%Y-%m-%d-%H-%M'  )

    # print((asset, resolution, start_date, end_date))
    # print(f"self._preload_data: {self._preload_data}")
    # print("before slice")
    if self._preload_data:
      res = self.load_from_cache(asset, resolution, start_date, end_date, **kwarg)
    else:
      res = self.load_from_api(asset, resolution, start_date, end_date, **kwarg)
    # print(f"Data --> {res.shape}")
    # print("After slice")
    return res

  def init_preload(self,**kwarg):
    """
      This method calls the preload_data_api
      method to preload the base data for a specific 
      data range, this data is what is expanded at each 
      call, the advantage of implementing a preload method
      is to overcome the time take between data request and
      when the api endpoint returns a result. With this method
      correctly implented, the data class just now needs to expand
      the base data at each datapoint request. 
    """
    # print(f"Testing init_preload method: kwarg.get('preload_data_api', False): {kwarg.get('preload_data_api', False)}, kwarg.get('preload_data_file', False): {kwarg.get('preload_data_file', False)}")
    self._preload_data = False
    if kwarg.get('preload_data_api', False):
      self.preload_data_api(**kwarg)
    elif kwarg.get('preload_data_file', False):
      self.preload_data_file(**kwarg)

  def get_preload_safe_start_end_date(self):
    _, idx_backwards_date, _ = self.get_idx_start_back_end_date(0)
    _, _, idx_end_date = self.get_idx_start_back_end_date(len(self))
    return idx_backwards_date, idx_end_date
    
  def preload_data_api(self, **kwarg):
    # _, idx_backwards_date, _ = self.get_idx_start_back_end_date(0)
    # _, _, idx_end_date = self.get_idx_start_back_end_date(len(self))
    idx_backwards_date, idx_end_date = self.get_preload_safe_start_end_date()

    self.cache = self.backend_get_all_raw(asset=self.asset, resolution=self.resolution, start_date=idx_backwards_date, end_date=idx_end_date, **kwarg)
    self._preload_data = True   
    self.presave_data()

  def default_data_path(self):
    return f"inputs/{self.get_backend()}_asset={self.asset},resolution={self.resolution},start_date={self.start_date}, end_date={self.end_date}"

  def presave_data(self):
    if self._preload_data:
      self.cache.to_csv(self.default_data_path())
    else:
      raise NotImplementedError("Preload not available, preload method should set self._preload_data to true once load is complete")

  def preload_data_file(self, **kwarg):
    if kwarg.get('preload_data_file'):
      path = kwarg.get('data_file', None) 
      if path == None:
        path = self.default_data_path()
        print(f"Warning, preload is using the dafault data path {path}, inspect if this is not the desired performance")
      try:
        self.cache = pd.read_csv(path)
        self.cache["time"] = pd.to_datetime(self.cache["time"])
        self.cache.set_index("time", drop=True, inplace=True)
        self._preload_data = True
      except Exception as e:
        raise Exception(f"Error {e} occured while trying to load data from file {kwarg.get('data_file', None)}")
      self.presave_data()

    #filter result to increase memory size
    idx_backwards_date, idx_end_date = self.get_preload_safe_start_end_date()
    idx_backwards_date, idx_end_date = DatasetBaseBackend.convert_date_to_backend_format(idx_backwards_date, format='%Y-%m-%d-%H-%M'), DatasetBaseBackend.convert_date_to_backend_format(idx_end_date, format='%Y-%m-%d-%H-%M')
    self.cache = self.load_from_cache(asset=self.asset, resolution=self.resolution, start_date=idx_backwards_date, end_date=idx_end_date)


  def load_from_cache(self, asset, resolution, start_date, end_date,**kwarg):
    # print(f"self.cache: {self.cache}")
    # convert_date_from_backend_format
    start_date = self.convert_date_from_backend_format(start_date, format='%Y-%m-%d-%H-%M')
    end_date = self.convert_date_from_backend_format(end_date, format='%Y-%m-%d-%H-%M')
    try:
      res = self.cache[(self.cache.index >= start_date) & (self.cache.index <= end_date)].copy()
      return res
    except Exception as e:
      raise Exception(f"Error {e} occured while getting data from cache")


  def load_from_api(self, asset, resolution, start_date, end_date,**kwarg):
    """
      Implement this method  in the  sub class, to receive data from a specified
      start and end datetime.
    """
    raise NotImplementedError("Sub-class Should implement this method")

class QuantConnectDataBackend(DatasetBaseBackend):
  def __init__(self):
    raise NotImplementedError("Sub-class Should implement this method")

class FXCMDataBackend(DatasetBaseBackend):
  def __init__(self):
    raise NotImplementedError("Sub-class Should implement this method")

class yFinanceDataBackend(DatasetBaseBackend):
  def __init__(self):
    raise NotImplementedError("Sub-class Should implement this method")

class cryptocmdDataBackend(DatasetBaseBackend):
  def __init__(self):
    raise NotImplementedError("Sub-class Should implement this method")

class historicCryptoBackend(DatasetBaseBackend):
  def __init__(self,**kwarg):
    self.set_backend("historicCryptoBackend")
    super().__init__(**kwarg)

  def load_from_api(self, asset, resolution, start_date, end_date,**kwarg):
    import nest_asyncio
    nest_asyncio.apply()
    raw_data = asyncio.run(HistoricalData(asset, resolution, start_date, end_date,verbose = kwarg.get('verbose', False)).retrieve_data())
    return raw_data
