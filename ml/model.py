import requests, json, time, sys, aiohttp, torch, functools

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

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

np.seterr(divide = 'ignore') 

import functools

def wasiu_input_check(class_level):
    """
        param: classtype: this is the expected classes to check if input aligns 
        with format, Values: ["wasiunet", "wasiuspace","wasiuspacetime"]
    """
    def checker_decorator(func):
        
        @functools.wraps(func)
        def checker_function(*args, **kwargs):
          src = kwargs.get('src',None)
          trg = kwargs.get('trg',None)

          if type(src) == type(None):
            src = args[1]

          # Check input type aligns with required input
          if class_level in ["wasiunet","wasiuspacetime"]:
            if type(trg) == type(None):
              trg = args[2]
            assert (
                (torch.is_tensor(src) or isinstance(src, np.ndarray)) 
                and 
                (torch.is_tensor(trg) or isinstance(trg, np.ndarray)) 
              ), f"src input of type {type(src)} or trg input of type {type(trg)} is not tensor or numpy, supported type for input is Pytorch tensor or numpy array."

          else:
            assert torch.is_tensor(src) or isinstance(src, np.ndarray), f"src input of type {type(src)} supported type for input is Pytorch tensor or numpy array."

          # Convert src and trg to tensor
          # src shape --> batch_size * time_seq_len * space_seq_len * feat_len e.g (2 * 12 * 60 * 7) or 
          # time_seq_len * space_seq_len * feat_len e.g (12 * 60 * 7)
          src = torch.Tensor(src)

          if class_level in ["wasiunet","wasiuspacetime"]:
            # trg shape -->  space_seq_len * batch_size * feat_len e.g (2 * 100 * 7)
            trg = torch.Tensor(trg)
          
          if class_level in ["wasiunet"]:
            # Write assertion to confirm input shapes
            if src.ndim == 4:
              pass
            elif src.ndim == 3:
              src = src.unsqueeze(0) # expand dimension
            else:
              raise ValueError(f"You have a dimension error in your source input of {src.ndim}, expected dimension is {4} or {3}")

            # Write assertion to confirm output shapes
            if trg.ndim == 3:
              pass
            elif trg.ndim == 2:
              trg = trg.unsqueeze(0) # expand dimension
            else:
              raise ValueError(f"You have a dimension error in your source input of {src.ndim}, expected dimension is {4} or {3}")

          # assert src and trg have the same batch size
          if class_level in ["wasiunet","wasiuspacetime"]:
            assert src.shape[0] == trg.shape[0], f"source batch {src.shape[1]} and target batch {trg.shape[0]} do not have the same batch size"
            return func(src=src, trg=trg)
          else:
            return func(src=src)
        
        return checker_function
    return checker_decorator

class FCView(nn.Module):
    def __init__(self,shape=None):
        super(FCView, self).__init__()
        if shape != None:
            self.shape = shape 
        else:
            self.shape = -1

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        n_b = x.data.size(0)
        x = x.view(self.shape) if self.shape != -1 else x.view(n_b, -1)
        return x

class WasiuSpace(nn.Module):
      """
        This class provides the functionality to process assets at a space level, 
        by space level this means for example we are processing a 60 sequences of 
        1 min timeframe asset data or processing a 60 sequences of 5mins timeframe
        asset data. So basically we embed positions on space level and feed to 
        the transformer model.
        
        :param: d_model: a singular Cryptocurrency ticker. (str)
        :param: src_vocab_size: the price data frequency in seconds, one of: 60, 300, 900, 3600, 21600, 86400. (int)
        :param: trg_vocab_size: a date string in the format YYYY-MM-DD-HH-MM. (str)
        :param: num_heads: printing during extraction, Default=True. (bool)
        :param: num_encoder_layers: a Pandas DataFrame which contains requested cryptocurrency data. (pd.DataFrame)
        :param: forward_expansion: a date string in the format YYYY-MM-DD-HH-MM,  Default=Now. (str)
        :param: dropout: printing during extraction, Default=True. (bool)
        :param: max_len: a Pandas DataFrame which contains requested cryptocurrency data. (pd.DataFrame)
        :param: device: a Pandas DataFrame which contains requested cryptocurrency data. (pd.DataFrame)
      """
      def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, 
                dropout, max_len, device,trans_activation,
                 trans_norm_first,trans_batch_first
                ):
        super(WasiuSpace, self).__init__()
        # self.device = device
        self.max_len = max_len

        # space model
        self.wasiuspace_encoder_layer = nn.TransformerEncoderLayer(
                                                        d_model=d_model, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, 
                                                        dropout=dropout,activation=trans_activation,
                                                        batch_first=trans_batch_first,
                                                        norm_first=trans_norm_first
                                                        )
        
        self.wasiuspace_transformer_encoder = nn.TransformerEncoder(
                                                          encoder_layer=self.wasiuspace_encoder_layer, 
                                                        num_layers=num_encoder_layers
                                                        )
      @wasiu_input_check('wasiuspace')
      def wasiuspace_input_check(src):
          return src

      def forward(self, src):
        # src shape: (src_seq_length, batch_size, time_seq_length, embed_size)
        # trg shape: (trg_seq_length, batch_size)
        src = self.wasiuspace_input_check(src)
        return self.wasiuspace_transformer_encoder(src)

class WasiuSpaceTime(nn.Module):
      """
        This class provides the functionality to process assets at a time level, 
        by time level this means for example we are processing a 60 sequences of 
        1 min timeframe asset data on one end and processing a 60 sequences of 5mins timeframe
        asset data on the other end to both return the next lowest timeframe series into
        the future. So basically we embed positions to time level and we feed it to 
        the space transformer model.
      """

      def __init__(self, embedding_dim,inp_feat, out_feat, in_channels, patch_size,space_seq_len, 
                   expand_HW, nhead, num_encoder_layers, num_decoder_layers, 
                  dim_feedforward, dropout, max_len, device,trans_activation,
                 trans_norm_first,trans_batch_first
                  ):
        super(WasiuSpaceTime, self).__init__()

        self.device = device
        self.max_len = max_len
                                                         
        # Encode positions
        self.src_word_embedding = nn.Sequential(OrderedDict([
                                  ('src_inp_norm_layer_1', nn.LayerNorm(inp_feat)), # Source Create the normalization layer
                                  ('expand_HW',  nn.Flatten(start_dim=-2, end_dim=-1)),
                                  ('src_lin_embedding', nn.Linear(space_seq_len*inp_feat, expand_HW*expand_HW)), # Source Create the word embedding layer
                                  ('reshape_CHW', FCView(shape=(-1,in_channels, expand_HW, expand_HW))),
                                  ('src_patcher_layer', nn.Conv2d(
                                          in_channels = in_channels,
                                          out_channels = embedding_dim,
                                          kernel_size=patch_size,
                                          stride = patch_size,
                                          padding = 0
                                          )), # Input patcher
                                  ('flatten', nn.Flatten(start_dim=2, end_dim=3)),
                                ]))



        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Source Create the word position embedding layer
        self.src_position_embeddding = nn.Embedding(max_len, embedding_dim)

        # Target Create the word embedding layer
        self.trg_word_embedding = nn.Sequential(OrderedDict([
                                  ('trg_inp_norm_layer_1', nn.LayerNorm(out_feat)), # Target Create the normalization layer
                                  ('trg_lin_embedding', nn.Linear(out_feat, embedding_dim)),
                                  ('dropout', nn.Dropout(dropout))
                                ]))
        

        # Target Create the word position embedding layer
        self.trg_position_embedding = nn.Embedding(max_len, embedding_dim)

        # space-time model
        self.wasiuspace_encoder = WasiuSpace(d_model=embedding_dim, nhead=nhead, 
                                            num_encoder_layers=num_encoder_layers, 
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout, max_len=max_len, 
                                            device=device,trans_activation=trans_activation,
                                            trans_batch_first=trans_batch_first,
                                            trans_norm_first=trans_norm_first
                                          )
        
        # space-time model
        self.wasiuspacetime_decoder_layer = nn.TransformerDecoderLayer(
                                                        d_model=embedding_dim, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, 
                                                        dropout=dropout,activation=trans_activation,
                                                        batch_first=trans_batch_first,
                                                        norm_first=trans_norm_first
                                                        )
        
        self.wasiuspacetime_transformer_decoder = nn.TransformerDecoder(
                                                          decoder_layer=self.wasiuspacetime_decoder_layer, 
                                                       num_layers=num_decoder_layers)
        
        self.dropout = nn.Dropout(dropout)

      @wasiu_input_check('wasiuspacetime')
      def wasiuspacetime_input_check(src, trg):
          return src, trg

      def forward(self, src, trg):
        """
          :param: src: source asset data in the format 
            time_seq_len * batch_size * space_seq_len * feat_len 
            e.g (2 * 12 * 60 * 7) or (1 * 12 * 60 * 7).
            (Tensor)

          :param: trg: target asset data in the format 
            space_seq_len * batch_size * feat_len e.g (100 * 2 * 7) or (100 * 1 * 7)
            (Tensor)
        """
        src, trg =  self.wasiuspacetime_input_check(src, trg) # batch_size * seq_len * 14 * 14
        src = self.src_word_embedding(src).permute(0,2,1) # embed and reshape --> batch_sise * 14 * 14 * embedding
        assert self.max_len-1 >= src.shape[1], f"max_length specific {self.max_len} is greater than input seq_len {src.shape[1]}"

        # Get src and trg shapes
        batch_size, src_seq_length, _ = src.shape
        _, trg_seq_length, _ = trg.shape
        
        # Create Positions
        src_position = (
            torch.arange(0, src_seq_length).unsqueeze(0).expand(batch_size, src_seq_length)
            .to(src.device)
        )

        trg_position = (
            torch.arange(0, trg_seq_length).unsqueeze(0).expand(batch_size, trg_seq_length)
            .to(trg.device)
        )

    
        # Embed positions into data
        embed_src = self.dropout(
            (src + self.src_position_embeddding(src_position))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_position)) 
        )
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        encoded_src_memory = self.wasiuspace_encoder(embed_src)

        out = self.wasiuspacetime_transformer_decoder(embed_trg,encoded_src_memory,tgt_mask=trg_mask)
        return out

class WasiuNet(nn.Module):
    def __init__(self, embedding_dim, inp_feat, out_feat, in_channels, patch_size, space_seq_len,
                 expand_HW, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout, max_len, device,trans_activation,
                 trans_norm_first,trans_batch_first,feat_map_dict
                  ):
        super(WasiuNet, self).__init__()
        # super(Transformer, self).__init__()
        # self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        # self.src_position_embeddding = nn.Embedding(max_len, embedding_size)
        # self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        # self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.feat_map_dict = feat_map_dict
        self.target_feature = []
        self.direction_feature = []
        for key, value in self.feat_map_dict.items():
          if key in [ 'direction_-1_conf','direction_0_conf','direction_1_conf']:
            self.direction_feature.append(value)
          else:
            self.target_feature.append(value)

        self.wasiuspacetime = WasiuSpaceTime(embedding_dim=embedding_dim, inp_feat=inp_feat, 
                                             out_feat=out_feat, in_channels=in_channels, 
                                             patch_size=patch_size, space_seq_len=space_seq_len, expand_HW=expand_HW,
                                             nhead=nhead, num_encoder_layers=num_encoder_layers, 
                                              num_decoder_layers=num_decoder_layers,
                                              dim_feedforward=dim_feedforward,
                                              dropout=dropout, max_len=max_len, 
                                              device=device,trans_activation=trans_activation,
                                              trans_batch_first=trans_batch_first,
                                              trans_norm_first=trans_norm_first
                                          )
        # Target Create the word embedding layer
        self.fc_out = nn.Sequential(OrderedDict([
                                  ('dropout', nn.Dropout(dropout)),  
                                  ('decoded_out_1', nn.Linear(embedding_dim, out_feat)),
                                  ('relu',nn.ReLU()),
                                  ('decoded_out_2', nn.Linear(out_feat, out_feat))
                                  
                                ]))
        
        self.dropout = nn.Dropout(dropout)

    @wasiu_input_check('wasiunet')
    def check_wasiunet_input(src, trg):
        return src, trg

    def forward(self, src, trg):        
        """
          :param: src: source asset data in the format 
            time_seq_len * batch_size * space_seq_len * feat_len 
            e.g (2 * 12 * 60 * 7).
            or time_seq_len * space_seq_len * feat_len e.g (60 * 12 * 7)
            (Numpy | Tensor)

          :param: trg: target asset data in the format 
            space_seq_len * batch_size * feat_len e.g (100 * 2 * 7)
            or time_seq_len * feat_len e.g (100 * 7)
            (Numpy | Tensor)
        """
        src, trg = self.check_wasiunet_input(src, trg)
        src, trg = src.type(torch.float32), trg.type(torch.float32)
        out = self.wasiuspacetime(src, trg)
        out = self.fc_out(out)
        return out

class WasiuNetTrainer(pl.LightningModule):
    def __init__(self, model, lr=0.003, multi_label_loss_weight = 0.5, regression_loss_weight = 0.5):
        super(WasiuNetTrainer, self).__init__()
        self.model = model
        self.lr = lr

        # Define a weight for the multi-label loss (you can adjust this to balance the two losses)
        self.multi_label_loss_weight = multi_label_loss_weight

        # Define a weight for the regression loss (you can adjust this to balance the two losses)
        self.regression_loss_weight = regression_loss_weight

    def forward(self, src, trg): 
        return self.model(src, trg)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def criterion(self, multi_label_pred, multi_label_target, regression_pred, regression_target):

        # Define the multi-label loss function
        multi_label_loss_fn = nn.CrossEntropyLoss()

        # Define the regression loss function
        # regression_loss_fn = nn.MSELoss()
        regression_loss_fn = nn.L1Loss()

        # Compute the multi-label loss
        multi_label_loss = multi_label_loss_fn(multi_label_pred.float(), multi_label_target.float())
        print(f"multi_label_loss: {multi_label_loss}") 

        # Compute the regression loss
        regression_loss = regression_loss_fn(regression_pred.float(), regression_target.float()) # get loss
        print(f"regression_loss before scale: {regression_loss}") 
        print(f"scale value: {regression_pred.shape[1]}")     
        
        # Create a tensor for the scaling factor with the correct data type
        scaling_factor = torch.tensor(regression_pred.shape[1], dtype=torch.float)

        regression_loss = torch.div(regression_loss, scaling_factor) # scale data
        print(f"regression_loss after scale: {regression_loss}") 

        
        # Combine the two losses with the specified weights
        combined_loss = self.multi_label_loss_weight * multi_label_loss + self.regression_loss_weight * regression_loss
        return combined_loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = batch[0].squeeze(1)
        y = batch[1].squeeze(1)

        output = self.model(x, y[:,:-1,:])
        y = y[:,1:,:]
        
        # multi_label_pred, multi_label_target, regression_pred, regression_target
        # Compute the multi-label loss
        loss = self.criterion(output[:,:,self.model.direction_feature], y[:,:,self.model.direction_feature], output[:,:,self.model.target_feature], y[:,:,self.model.target_feature])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = batch[0].squeeze(1)
        y = batch[1].squeeze(1)

        output = self.model(x, y[:,:-1,:])
        y = y[:,1:,:]
        
        # multi_label_pred, multi_label_target, regression_pred, regression_target
        # Compute the multi-label loss
        loss = self.criterion(output[:,:,self.model.direction_feature], y[:,:,self.model.direction_feature], output[:,:,self.model.target_feature], y[:,:,self.model.target_feature])
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = batch[0].squeeze(1)
        y = batch[1].squeeze(1)

        output = self.model(x, y[:,:-1,:])
        y = y[:,1:,:]
        
        # multi_label_pred, multi_label_target, regression_pred, regression_target
        # Compute the multi-label loss
        loss = self.criterion(output[:,:,self.model.direction_feature], y[:,:,self.model.direction_feature], output[:,:,self.model.target_feature], y[:,:,self.model.target_feature])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0].squeeze(1)
        y = batch[1].squeeze(1)

        output = self.model(x, y[:,:-1,:])
        return output
