import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
matplotlib.use('Agg')
import datetime

from MySAC import config
from MySAC.preprocessors import FeatureEngineer, data_split
from MySAC.models.DRLAgent import DRLAgent
from stable_baselines3.common.vec_env import VecMonitor
from envs.env_stocktrading_hybrid_control import StockTradingEnv as Env
import pdb
import stable_baselines3.common.utils as utils 
from sklearn.preprocessing import StandardScaler

import time
import random
import torch
import numpy as np

import os

import sys

fix_seed = 1999
version = 'N100/'
model_name='StockFormer/'
short_prediction_model_path = 'Transformer/pretrained/n100/Short/checkpoint.pth'
long_prediction_model_path =  'Transformer/pretrained/n100/Long/checkpoint.pth'
mae_model_path = 'Transformer/pretrained/n100/mae/checkpoint.pth'
full_stock_dir = '../data/N100/'
ticker_list = config.use_ticker_dict['N100']
prediction_len = [1,5]


if not os.path.exists(config.TRAINED_MODEL_DIR):
    os.makedirs(config.TRAINED_MODEL_DIR)
if not os.path.exists(config.TENSORBOARD_LOG_DIR):
    os.makedirs(config.TENSORBOARD_LOG_DIR)
if not os.path.exists(config.RESULTS_DIR):
    os.makedirs(config.RESULTS_DIR)


df = pd.DataFrame([], columns=['date','open','close','high','low','volume','dopen','dclose','dhigh','dlow','dvolume','price','tic'])

for ticker in ticker_list:
    temp_df = pd.read_csv(os.path.join(full_stock_dir,ticker+'.csv'), usecols=['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])
    temp_df['date'] = temp_df['date'].apply(lambda x:str(x))
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df['label_short_term'] = temp_df['close'].pct_change(periods=prediction_len[0]).shift(periods=(-1*prediction_len[0]))
    temp_df['label_long_term'] = temp_df['close'].pct_change(periods=prediction_len[1]).shift(periods=(-1*prediction_len[1]))
    temp_df['tic'] = pd.Series([ticker]*len(temp_df))
    # temp_df = temp_df.rename(columns={'Date':'date', 'Open':'open', 'Close':'close', 'High':'high', 'Low':'low', 'Volume':'volume'})
    df = pd.concat((df, temp_df))

df = df.sort_values(by=['date','tic'])
    
fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

print("ensuring data is correct size...")
# Count the number of unique stock tickers for each date
unique_stocks_per_date = df.groupby('date')['tic'].nunique()

# Filter out dates with incomplete data (less than 88 unique tickers)
complete_dates = unique_stocks_per_date[unique_stocks_per_date == 88].index

# Filter the dataframe to keep only data for complete dates
df = df[df['date'].isin(complete_dates)]

# Define the dates to avoid removing
dates_no_remove = ['20110419', '20181228', '20180102', '20201231',  '20190402', '20211231', '20110117', '20180801', '20180508', '20201231',  '20210104', '20220426']

# Define target number of rows
target_rows = 263560

# Check if the dataframe already has the desired length
if len(df) > target_rows:
    # Get unique dates
    unique_dates = df['date'].unique()

    # Initialize removed dates and remaining rows
    removed_dates = set()
    remaining_rows = len(df)

    # Remove dates until target rows are reached
    while remaining_rows > target_rows:
        # Choose a random date
        date_to_remove = random.choice(unique_dates)

        # Check if it's a date to avoid or already removed
        if date_to_remove in dates_no_remove or date_to_remove in removed_dates:
            continue

        # Remove the date and update counters
        removed_dates.add(date_to_remove)
        remaining_rows -= df[df['date'] == date_to_remove].shape[0]

    # Filter the dataframe to remove chosen dates
    df = df[~df['date'].isin(removed_dates)]

print("generate technical indicator...")
df = fe.preprocess_data(df)

# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
    data_lookback = df.loc[i-lookback:i,:]
    price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close') 
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)
    
    covs = return_lookback.cov().values 
    cov_list.append(covs)


df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)
         

scaler = StandardScaler()
df_data = df[config.TECHNICAL_INDICATORS_LIST]
df_data = df_data.replace([np.inf], config.INF)
df_data = df_data.replace([-np.inf], config.INF*(-1))
data = scaler.fit_transform(df_data.values)
df[config.TECHNICAL_INDICATORS_LIST] = data

train = data_split(df, '2011-01-17','2018-12-28')
eval = data_split(df, '2019-01-02', '2021-12-31')
test = data_split(df,'2018-10-09', '2022-04-16')

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

tensorboard_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, 'mysac')


env_kwargs = {
    "hmax": 100, 
    "initial_amount": 100000,  
    "transaction_cost_pct": 0,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "temporal_feature_list": config.TEMPORAL_FEATURE,
    "additional_list": config.ADDITIONAL_FEATURE,
    "action_space": stock_dimension,
    "reward_scaling": 10,
    "figure_path":'results/figures/'+version+model_name,
    "logs_path": 'results/logs/'+version+model_name,
    "csv_path": 'results/csv/'+version+model_name,
    "mode":'train',
    "time_window_start":config.time_window_start,
    "step_len": 1000,
    "temporal_len": 60,
    "hidden_channel":128,     
    "model_name":model_name[:-1],
    "short_prediction_model_path": short_prediction_model_path,
    "long_prediction_model_path": long_prediction_model_path,
}

env_kwargs_test = {
    "hmax": 100, 
    "initial_amount": 100000,  
    "transaction_cost_pct": 0,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "temporal_feature_list": config.TEMPORAL_FEATURE,
    "additional_list": config.ADDITIONAL_FEATURE,
    "action_space": stock_dimension, 
    "reward_scaling": 10,
    "figure_path":'results/figures/'+version+model_name,
    "logs_path": 'results/logs/'+version+model_name,
    "csv_path": 'results/csv/'+version+model_name,
    "mode":'test',
    "time_window_start":config.time_window_start,
    "step_len": 1000,
    "temporal_len": 60,
    "hidden_channel":128,     
    "model_name":model_name[:-1],
    "short_prediction_model_path":short_prediction_model_path,
    "long_prediction_model_path":long_prediction_model_path,

}


# evaluation environment
ck_dir = os.path.join(config.TRAINED_MODEL_DIR, version[:-1], model_name[:-1])
log_dir = os.path.join(config.RESULTS_DIR, version[:-1], model_name[:-1])

os.makedirs(log_dir, exist_ok=True)
os.makedirs(ck_dir, exist_ok=True)

print("Initial Env...")
eval_trade_gym = Env(df = eval, **env_kwargs_test)
env_eval, _ = eval_trade_gym.get_sb_env()
env_eval_sac = VecMonitor(env_eval, log_dir+'_eval')

test_trade_gym = Env(df = eval, **env_kwargs_test)
env_test, _ = test_trade_gym.get_sb_env()
test_eval_sac = VecMonitor(env_test, log_dir+'_test')

test_trade_gym2 = Env(df = train, **env_kwargs_test)
env_test2, _ = test_trade_gym2.get_sb_env()
test_eval_sac2 = VecMonitor(env_test2, log_dir+'_test2')

e_train_gym = Env(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
env_train_sac = VecMonitor(env_train, log_dir+'_train')
agent = DRLAgent(env = env_train_sac)

MAESAC_PARAMS = {
    "batch_size": 32,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
    "enc_in": 96,
    "dec_in": 96,
    "c_out_construction": 96,
    "d_model":128,
    "d_ff":256,
    "n_heads":4,
    "e_layers":2,
    "d_layers":1,
    "dropout":0.05,
    "transformer_path":mae_model_path,

}

model_sac = agent.get_model("maesac",model_kwargs = MAESAC_PARAMS,tensorboard_log=tensorboard_log_dir, seed=fix_seed)
print('Start training...')


start = time.time()
trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name=model_name,
                             check_freq=1000,
                             log_dir=log_dir,
                             ck_dir=ck_dir,
                             eval_env=env_eval_sac,
                             total_timesteps=30000)
end = time.time()
print("Training time: %.3f"%(end-start))



model_path = os.path.join('trained_models/', version, model_name, 'model30000.zip')
start = time.time()
results = DRLAgent.DRL_prediction_load_from_file(model_name='maesac',environment=test_trade_gym, cwd=model_path)
end = time.time()
print("Test time: %.3f"%(end-start))

df_root = 'results/df_print/'+version+model_name
os.makedirs(df_root, exist_ok=True)
assets_his, df_actions = results[1], results[2]
df_actions.to_csv(df_root+'df_actions_test.csv')
assets_his.to_csv(df_root+'df_assets_his_test.csv')



