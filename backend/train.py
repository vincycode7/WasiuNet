ta_config = [
              {"ta_func_name":"SMA", 'ta_func_config':{'window':11,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':9,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':7,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':5,'fillna':False}},
              {"ta_func_name":"SMA", 'ta_func_config':{'window':3,'fillna':False}},

              {"ta_func_name":"RSI", 'ta_func_config':{'window':10,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':7,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':5,'fillna':False}},
              {"ta_func_name":"RSI", 'ta_func_config':{'window':3,'fillna':False}},

              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':11,'window_slow':5, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':9,'window_slow':4, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':7,'window_slow':3, 'fillna':False}},
              {"ta_func_name":"STC", 'ta_func_config':{'window_fast':5,'window_slow':2, 'fillna':False}}
             ]
technical_analysis_config=ta_config
asset = "BTC-USD"
resolution = 60
fea_output_per_data_slice = 60
fea_data_slice = 12
glob_time_step_forwards= 10
batch_size = 5
num_worker = 2

# train data engineering
# file_path = "inputs/coinbase_asset=BTC-USD,resolution=60,start_date=2015-10-01-00-00, end_date=2022-11-19-00-00"
file_path = "inputs/historicCryptoBackend_asset=BTC-USD,resolution=60,start_date=2015-01-14 00:00:00, end_date=2022-11-23 00:00:00"
btc_usd_train =  historicCryptoBackend(start_date='2022-01-01-00-00-00', end_date='2022-10-01-00-00-00', asset=asset, resolution=resolution,technical_analysis_config=ta_config,fea_output_per_data_slice=fea_output_per_data_slice, fea_data_slice=fea_data_slice,glob_time_step_forwards=glob_time_step_forwards, preload_data_api=False, preload_data_file=True, data_file=file_path, verbose=True)
train_loader = DataLoader(btc_usd_train,batch_size=batch_size, shuffle=True, num_workers=num_worker)

# val data engineering
btc_usd_val =  historicCryptoBackend(start_date='2022-10-01-00-00-00', end_date='2022-10-31-00-00-00', asset=asset, resolution=resolution,technical_analysis_config=ta_config,fea_output_per_data_slice=fea_output_per_data_slice, fea_data_slice=fea_data_slice,glob_time_step_forwards=glob_time_step_forwards, preload_data_api=False, preload_data_file=True, data_file=file_path, verbose=True)
valid_loader = DataLoader(btc_usd_val,batch_size=batch_size, shuffle=True, num_workers=num_worker)

# test data engineering
btc_usd_test =  historicCryptoBackend(start_date='2022-10-31-00-00-00', end_date='2022-11-23-00-00-00', asset=asset, resolution=resolution,technical_analysis_config=ta_config,fea_output_per_data_slice=fea_output_per_data_slice, fea_data_slice=fea_data_slice,glob_time_step_forwards=glob_time_step_forwards, preload_data_api=False, preload_data_file=True, data_file=file_path, verbose=True)
test_loader = DataLoader(btc_usd_test,batch_size=batch_size, shuffle=True, num_workers=num_worker)

print(f"btc_usd_train:- {len(btc_usd_train)}, btc_usd_test:- {len(btc_usd_test)}, btc_usd_val:- {len(btc_usd_val)}")


# model
feat_map_dict = btc_usd_train.return_all_output_col_as_dict()
X_raw, Y_raw = btc_usd_train.get_item([-1, 0], process_data = True, add_ta = True, apply_nat_log = True, fill_na = True, cal_pct_chg = False, to_numpy=True)
input_feature_size=X_raw.shape[3]
sequence_length=X_raw.shape[2]
level_1_n_ensemble_lstm=X_raw.shape[1]
output=Y_raw.shape[-1]
deepensemblelstmcnn = DeepEnsembleLstmCnn(feat_map_dict=feat_map_dict,input_feature_size=input_feature_size,sequence_length=sequence_length,level_1_n_ensemble_lstm=level_1_n_ensemble_lstm,output=output)
# deepensemblelstmcnn

from pytorch_lightning.loggers import TensorBoardLogger
tb_logs = "outputs/tb_logs"
logger = TensorBoardLogger(tb_logs, name='deepensemblelstmcnn')

path ="/content/drive/MyDrive/deep learning folder/ensembleLSTM-CNN hybrid model/outputs/checkpoint/inp_feat_size:18-seq_len:60-level_1_n_ensm_lstm:12-out:21/train-model-epoch=03-11-28-2022-22-train_acc=0.22-train_loss=102.84-train_f1_score=0.19-train_pre=0.17-train_recall=0.22.ckpt"
deepensemblelstmcnn = DeepEnsembleLstmCnn.load_from_checkpoint(path)

summary = ModelSummary(deepensemblelstmcnn, max_depth=-1)
print(summary)

from pytorch_lightning.callbacks import ModelCheckpoint
curr_time = datetime.now().strftime("%m-%d-%Y-%H")
model_arch = f"inp_feat_size:{input_feature_size}-seq_len:{sequence_length}-level_1_n_ensm_lstm:{level_1_n_ensemble_lstm}-out:{output}"
train_checkpoint_callback = ModelCheckpoint(dirpath="outputs/checkpoint",every_n_train_steps=1, save_top_k=1,filename=model_arch+"/train-model-{epoch:02d}-"+curr_time+"-{train_acc:.2f}-{train_loss:.2f}-{train_f1_score:.2f}-{train_pre:.2f}-{train_recall:.2f}")
val_checkpoint_callback = ModelCheckpoint(dirpath="outputs/checkpoint", mode="min", save_top_k=1, monitor="val_loss",save_on_train_epoch_end=True,filename=model_arch+"/val-model-{epoch:02d}-"+curr_time+"-{val_acc:.2f}-{val_loss:.2f}-{val_f1_score:.2f}-{val_pre:.2f}-{val_recall:.2f}")
# model_arch

# # train model
# trainer = pl.Trainer(enable_model_summary=False,profiler="simple")
# trainer.fit(deepensemblelstmcnn, btc_usd_train, btc_usd_test)
# train with both splits
# callbacks=[train_checkpoint_callback, val_checkpoint_callback]
# val_check_interval=0,
# , limit_train_batches=-1, limit_test_batches=-1, limit_val_batches=-1
trainer = pl.Trainer(limit_train_batches=100, limit_test_batches=10, limit_val_batches=10, logger=logger,callbacks=[train_checkpoint_callback, val_checkpoint_callback], num_sanity_val_steps=0, default_root_dir="outputs/", accelerator='gpu', devices=-1,enable_model_summary=False,profiler=False, check_val_every_n_epoch=1, fast_dev_run=False, log_every_n_steps=1, max_epochs=10, overfit_batches=0)
trainer.fit(deepensemblelstmcnn, train_loader, valid_loader)