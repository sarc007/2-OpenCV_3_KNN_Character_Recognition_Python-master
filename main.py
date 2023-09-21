import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import ta
import os
forex_data = pd.read_csv('data/XAUUSD_M30_202001020100_202309211930.csv')
print(forex_data.head())
os.sys.exit()

forex_data['MA5'] = forex_data['Close'].rolling(window=5).mean()
forex_data['MA21'] = forex_data['Close'].rolling(window=21).mean()
forex_data['RSI'] = ta.momentum.RSIIndicator(forex_data['Close']).rsi()
forex_data.dropna(inplace=True)

    
class ForexEnv(gym.Env):
    def __init__(self, forex_data, delay_steps=5):
        super(ForexEnv, self).__init__()
        
        self.forex_data = forex_data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(4)  # Long, Short, Close, Hedge
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)  # OHLC, MA5, MA21, RSI, Volume, position
        self.current_step = 0
        self.delay_steps = delay_steps
        self.position_open_price = 0
        self.position = None  # Can be 'long', 'short', or None

    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.forex_data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA21', 'RSI']].values
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step > len(self.forex_data) - 1:
            self.done = True
            
        obs = self._next_observation()
        immediate_reward = 0

        current_price = obs[3]

        if action == 0:  # Long
            self.position = 'long'
            self.position_open_price = current_price
        elif action == 1:  # Short
            self.position = 'short'
            self.position_open_price = current_price
        elif action == 2:  # Close
            if self.position == 'long':
                immediate_reward = current_price - self.position_open_price
            elif self.position == 'short':
                immediate_reward = self.position_open_price - current_price
            self.position = None
        elif action == 3:  # Hedge
            # Implement your hedging logic here. 
            # For simplicity, it's not implemented in this example.
            pass

        delayed_reward = 0
        if self.current_step >= self.delay_steps:
            delayed_obs = self.forex_data.iloc[self.current_step - self.delay_steps][['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA21', 'RSI']].values
            delayed_price = delayed_obs[3]

            if self.position == 'long':
                delayed_reward = delayed_price - self.position_open_price
            elif self.position == 'short':
                delayed_reward = self.position_open_price - delayed_price

        return obs, immediate_reward + delayed_reward, self.done, {}



env = ForexEnv(forex_data)

model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, target_update_interval=1000)
model.learn(total_timesteps=10000)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

