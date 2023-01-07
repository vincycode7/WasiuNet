#### TODOS

# Todo: Convert the deep learning session to user the pytorch lighting (Done)
# Process model output to calculate loss using mse and loss using crossentropy loss

 # from datetime import datetime
import torch
from torch.utils.data import Dataset
from torch import nn
from sklearn.model_selection import train_test_split
import joblib
import torch.multiprocessing as mp
import pytorch_lightning as pl

# Your New Python File
class BaseLSTMTrainer:
    pass

class BaseLSTMTester:
    pass

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

class Lin_View(nn.Module):
    def __init__(self,shape=None):
        super(Lin_View, self).__init__()
        if shape != None:
            self.shape = shape 
        else:
            self.shape = -1
    def forward(self, x):
        return x.view(x.size()[0], self.shape) if self.shape != -1 else x.view(x.size()[0], -1)

class EnsembleLstm(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=False, level_1_n_ensemble_lstm=7):
    super(EnsembleLstm, self).__init__()
    self.level_1_n_ensemble_lstm = level_1_n_ensemble_lstm
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional=bidirectional
    self.learner = nn.ModuleList([
        # nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,dropout=self.dropout, bidirectional=self.bidirectional)

        for _ in range(level_1_n_ensemble_lstm)])
        
  def get_output_channel(self):
    if self.bidirectional:
      return self.hidden_size * 2
    else:
      self.hidden_size
      
  def forward(self,x):
    # print(self.input_size, self.hidden_size, self.num_layers)
    # bi_value = 2 if self.bidirectional else 1
    # self.learner_state_h = torch.randn(self.level_1_n_ensemble_lstm, bi_value*self.num_layers,x.size(0),  self.hidden_size, requires_grad=True)
    # self.learner_state_c = torch.randn(self.level_1_n_ensemble_lstm, bi_value*self.num_layers,x.size(0),  self.hidden_size, requires_grad=True)
    level_output = []
    for rank in range(self.level_1_n_ensemble_lstm):
        # output, (learner_state_hn_, learner_state_cn_) = self.learner[rank](x[:,rank,:,:],(self.learner_state_h[rank], self.learner_state_c[rank])) # Training for each model
        output, (learner_state_hn_, learner_state_cn_) = self.learner[rank](x[:,rank,:,:]) # Training for each model

        level_output.append(output)
    return torch.stack(level_output).permute(1,0,2,3)

class MetaCNN(nn.Module):
  def __init__(self, input_size=7, input_feature=30, output_size=512, dropout=0.1):
    super(MetaCNN, self).__init__()
    self.input_size = input_size
    self.input_feature = input_feature
    self.output_size = output_size
    self.dropout = dropout
    self.learner = nn.Sequential(OrderedDict([
          ('batch_norm1', nn.BatchNorm2d(input_size)),
          ('relu1', nn.ReLU()),
          ('dropout1', nn.Dropout(self.dropout)),
          ('conv2d_1', nn.Conv2d(input_size, input_feature*4, (1,3), stride=1, padding=(0,1))),
          
          ('batch_norm2', nn.BatchNorm2d(input_feature*4)),
          ('relu2', nn.ReLU()),
          ('dropout2', nn.Dropout(self.dropout/1.2)),
          ('conv2d_2', nn.Conv2d(input_feature*4, int(input_feature*7), (1,3), stride=1, padding=(0,1))),
          
          ('batch_norm3', nn.BatchNorm2d(int(input_feature*7))),
          ('relu3', nn.ReLU()),
          ('dropout3', nn.Dropout(self.dropout/2)),
          ('conv2d_3', nn.Conv2d(int(input_feature*7),int((input_feature*2)/3), (3, 5), stride=(1, 1), padding=(1, 2))),
          
          ('batch_norm4', nn.BatchNorm2d(int((input_feature*2)/3))),
          ('relu4', nn.ReLU()),
          ('dropout4', nn.Dropout(self.dropout/2.1)),
          ('conv2d_4', nn.Conv2d(int((input_feature*2)/3),int(input_feature/2), (3, 5), stride=(1, 1), padding=(1, 2))),

          ('batch_norm5', nn.BatchNorm2d(int(input_feature/2))),
          ('relu5', nn.ReLU()),  
          ('dropout5', nn.Dropout(self.dropout/2.5)),     
          ('conv2d_5', nn.Conv2d(int(input_feature/2),self.input_size, (3, 5), stride=(1, 1), padding=(1, 2))),

        ]))

  def forward(self, x):
    return self.learner(x)

class MetaLinear(nn.Module):
  def __init__(self, input_size=107520, output_size=10, dropout=0.5):
    super(MetaLinear, self).__init__()
    self.learner = nn.Sequential(nn.Sequential(OrderedDict([
          ('flat_1', nn.Flatten()),
          ('relu1', nn.ReLU()),
          ('drop1',nn.Dropout(dropout)),
          ('lin_1',nn.Linear(input_size,output_size)),
          ('view', FCView(shape=(-1,output_size)))
        ])))
  def forward(self,x):
    return self.learner(x)
    

class DeepEnsembleLstmCnn(pl.LightningModule):
    def __init__(self,feat_map_dict, input_feature_size=5,sequence_length=30,level_1_n_ensemble_lstm=7,output=8):
        super(DeepEnsembleLstmCnn, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(2, level_1_n_ensemble_lstm, sequence_length, input_feature_size)
        # self.N = batch # batch
        self.feat_map_dict = feat_map_dict
        self.target_feature = []
        self.direction_feature = []
        for key, value in self.feat_map_dict.items():
          if key in [ 'direction_-1_conf','direction_0_conf','direction_1_conf']:
            self.direction_feature.append(value)
          else:
            self.target_feature.append(value)
            
        self.L = sequence_length # seq length
        self.H_in = input_feature_size # input size

        # N - 1 * H_in - 7 * L - 5 (1 * 60 * 5)
        self.queue = []
        self.input_size = input_feature_size
        self.sequence_length = sequence_length
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        self.output = output
        self.level_1_n_ensemble_lstm = level_1_n_ensemble_lstm


        # 1(batch_size) * 7(time_frames - view in diff time frames) * 60(time_steps - for diff time step) * 5(features - for these features)
        # First level embedding
        self.ensemble_lstm_base_learner = EnsembleLstm(input_size=self.input_size, hidden_size=self.input_size, num_layers=2, batch_first=True, dropout=0.50, bidirectional=False,level_1_n_ensemble_lstm=self.level_1_n_ensemble_lstm)
        self.ensemble_bi_lstm_meta_learner = EnsembleLstm(input_size=self.input_size, hidden_size=self.input_size*2, num_layers=2, batch_first=True, dropout=0.45, bidirectional=True,level_1_n_ensemble_lstm=self.level_1_n_ensemble_lstm)

        self.ensemble_cnn_meta_learner = MetaCNN(input_size=self.level_1_n_ensemble_lstm, input_feature=self.sequence_length, output_size=self.ensemble_bi_lstm_meta_learner.get_output_channel(), dropout=0.45)
        self.ensemble_bi_lstm_meta_learner_2 = EnsembleLstm(input_size=self.input_size, hidden_size=self.input_size*2, num_layers=3, batch_first=True, dropout=0.35, bidirectional=True,level_1_n_ensemble_lstm=self.level_1_n_ensemble_lstm)

        self.ensemble_deep_lstm_last_learner = EnsembleLstm(input_size=self.ensemble_bi_lstm_meta_learner.get_output_channel(), hidden_size=self.ensemble_bi_lstm_meta_learner.get_output_channel(), num_layers=10, batch_first=True, dropout=0.30, bidirectional=False,level_1_n_ensemble_lstm=self.level_1_n_ensemble_lstm)
        self.final_output_layer = MetaLinear(input_size=self.level_1_n_ensemble_lstm*self.sequence_length*self.ensemble_bi_lstm_meta_learner.get_output_channel(), output_size=self.output, dropout=0.20)

    def forward(self,x):
        if not isinstance(x, torch.Tensor):
          x = torch.tensor(x).float()
        else:
          x = x.float()
        if x.ndim == 3:
          x = x.unsqueeze(0)
        if x.ndim > 4:
          x = x.view([-1]+list(x.shape[-3:]))
          
          
        # Vanilla Lstm
        level_one_output = self.ensemble_lstm_base_learner(x)
        level_one_output = level_one_output + x # Skip connection 1 
        
        # Bi Directional Lstm with skip connection
        level_two_lstm_output  = self.ensemble_bi_lstm_meta_learner(level_one_output)

        # Parse level two with CNN
        level_two_cnn_output = self.ensemble_cnn_meta_learner(level_one_output[:,:,:,:])
        level_two_cnn_output = level_one_output + level_two_cnn_output # Skip connection 2

        # Bi Directional Lstm with skip connection 2
        level_two_lstm_output_2  = self.ensemble_bi_lstm_meta_learner_2(level_two_cnn_output)

        new_level_two_input = level_two_lstm_output + level_two_lstm_output_2 # Skip connection 3
        # del level_two_lstm_output, level_two_cnn_output, level_two_lstm_output_2

        # Deep Lstm with skip connection
        last_level_output = self.ensemble_deep_lstm_last_learner(new_level_two_input)
        # del new_level_two_input
        
        last_level_output = self.final_output_layer(last_level_output)
        return last_level_output

    def configure_optimizers(self):
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=0.1, momentum=0.9)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        if not isinstance(y, torch.Tensor):
          y = torch.tensor(y).float()
        else:
          y = y.float()
        y = y.view([-1]+list(y.shape[-1:]))
        y_hat = self(x)
 
        loss = F.mse_loss(y_hat[:,self.target_feature], y[:,self.target_feature]) + F.cross_entropy(y_hat[:,self.direction_feature]   ,y[:,self.direction_feature])
        pred_class_hat = torch.softmax(y_hat[:,self.direction_feature], dim=1).argmax(dim=1)
        pred_class_true = y[:,self.direction_feature].argmax(dim=1)
        
        acc = multiclass_accuracy(pred_class_hat, pred_class_true, num_classes=3)
        f1_score = multiclass_f1_score(pred_class_hat, pred_class_true, num_classes=3)
        pre = multiclass_precision(pred_class_hat, pred_class_true, num_classes=3)
        recall = multiclass_recall(pred_class_hat, pred_class_true, num_classes=3)
        # print(f"pred_class_hat: {pred_class_hat}")
        # print(f"pred_class_true: {pred_class_true}")
        # self.log("train_loss", loss)
        # datetime object containing current date and time
        # curr_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        # self.log("train_curr_time", curr_time)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_score", f1_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_pre", pre, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss,
                train_acc=acc,
                train_f1_score=f1_score,
                train_pre=pre,
                train_recall=recall
            )
        )
        # return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        if not isinstance(y, torch.Tensor):
          y = torch.tensor(y).float()
        else:
          y = y.float()
        y = y.view([-1]+list(y.shape[-1:]))
        y_hat = self(x)
        loss = F.mse_loss(y_hat[:,self.target_feature], y[:,self.target_feature]) + F.cross_entropy(y_hat[:,self.direction_feature]   ,y[:,self.direction_feature])
        pred_class_hat = torch.softmax(y_hat[:,self.direction_feature], dim=1).argmax(dim=1)
        pred_class_true = y[:,self.direction_feature].argmax(dim=1)

        acc = multiclass_accuracy(pred_class_hat, pred_class_true, num_classes=3)
        f1_score = multiclass_f1_score(pred_class_hat, pred_class_true, num_classes=3)
        pre = multiclass_precision(pred_class_hat, pred_class_true, num_classes=3)
        recall = multiclass_recall(pred_class_hat, pred_class_true, num_classes=3)

        # self.log("train_loss", loss)
        # datetime object containing current date and time
        # curr_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        # self.log("train_curr_time", curr_time)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_score", f1_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_pre", pre, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
            log=dict(
                test_loss=loss,
                test_acc=acc,
                test_f1_score=f1_score,
                test_pre=pre,
                test_recall=recall
            )
        )

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        if not isinstance(y, torch.Tensor):
          y = torch.tensor(y).float()
        else:
          y = y.float()
        y = y.view([-1]+list(y.shape[-1:]))
        y_hat = self(x)
        loss = F.mse_loss(y_hat[:,self.target_feature], y[:,self.target_feature]) + F.cross_entropy(y_hat[:,self.direction_feature]   ,y[:,self.direction_feature])
        pred_class_hat = torch.softmax(y_hat[:,self.direction_feature], dim=1).argmax(dim=1)
        pred_class_true = y[:,self.direction_feature].argmax(dim=1)
        
        acc = multiclass_accuracy(pred_class_hat, pred_class_true, num_classes=3)
        f1_score = multiclass_f1_score(pred_class_hat, pred_class_true, num_classes=3)
        pre = multiclass_precision(pred_class_hat, pred_class_true, num_classes=3)
        recall = multiclass_recall(pred_class_hat, pred_class_true, num_classes=3)
        # print(f"pred_class_hat: {pred_class_hat}")
        # print(f"pred_class_true: {pred_class_true}")
        # self.log("train_loss", loss)
        # datetime object containing current date and time
        # curr_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        # self.log("train_curr_time", curr_time)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_score", f1_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_pre", pre, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(
            loss=loss,
            log=dict(
                val_loss=loss,
                val_acc=acc,
                val_f1_score=f1_score,
                val_pre=pre,
                val_recall=recall
            )
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return y_hat

    