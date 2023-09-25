from forex_env import ForexEnv
from stable_baselines3 import DQN
import pandas as pd
import ta

def train_model(env, timesteps):
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1,
                 target_update_interval=1000, tensorboard_log="tensorboard/")
    model.learn(total_timesteps=timesteps)
    model.save("path_to_save_model")

if __name__ == "__main__":
    # Initialize your ForexEnv with historical data
    # Load your historical data
    forex_data = pd.read_csv('data/XAUUSD_M30_202001020100_202309211930.csv',sep='\t') 
    forex_data['MA5'] = forex_data['Close'].rolling(window=5).mean()
    forex_data['MA21'] = forex_data['Close'].rolling(window=21).mean()
    forex_data['RSI'] = ta.momentum.RSIIndicator(forex_data['Close']).rsi()
    forex_data.dropna(inplace=True)
    env = ForexEnv(forex_data)
    train_model(env, 10000)
