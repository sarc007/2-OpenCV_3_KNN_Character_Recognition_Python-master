import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import ta
forex_data = pd.read_csv('data/your_data.csv')
forex_data['MA5'] = forex_data['Close'].rolling(window=5).mean()
forex_data['MA21'] = forex_data['Close'].rolling(window=21).mean()
forex_data['RSI'] = ta.momentum.RSIIndicator(forex_data['Close']).rsi()
forex_data.dropna(inplace=True)
