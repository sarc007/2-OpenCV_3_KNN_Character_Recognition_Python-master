import datetime
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DQN
import ta
import os
import logging

from logging.handlers import RotatingFileHandler

# Set up logging
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')

logFile = 'log/trading_bot.log'

# Use RotatingFileHandler to handle log rotation
my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=10*1024*1024,
                                 backupCount=10, encoding=None, delay=0)

my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.INFO)

# Set up logging to console and add the handler to the root logger
logging.getLogger('').addHandler(my_handler)
logging.getLogger('').setLevel(logging.INFO)

forex_data = pd.read_csv('data/XAUUSD_M30_202001020100_202309211930.csv',sep='\t')

forex_data['MA5'] = forex_data['Close'].rolling(window=5).mean()
forex_data['MA21'] = forex_data['Close'].rolling(window=21).mean()
forex_data['RSI'] = ta.momentum.RSIIndicator(forex_data['Close']).rsi()
forex_data.dropna(inplace=True)

print(forex_data.head())
# os.sys.exit()

    
class ForexEnv(gym.Env):
    def __init__(self, forex_data, delay_steps=5):
        super(ForexEnv, self).__init__()
        
        self.forex_data = forex_data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(4)  # Long, Short, Close, Hedge
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # OHLC, MA5, MA21, RSI, Volume, position
        self.current_step = 0
        self.delay_steps = delay_steps
        self.position_open_price = 0
        self.position = None  # Can be 'long', 'short', or None
        self.hedge_multiplier = 1.2
        self.hedge_position = None  # Can be 'long_hedge', 'short_hedge' or None
        self.hedge_position_open_price = 0
        self.returns = []  # To keep track of returns for calculating Sharpe ratio
        # self.max_portfolio_value = 0  # To keep track of the highest portfolio value for calculating drawdown
        self.portfolio_value = 10000  # Initial portfolio value
        self.leverage = 100
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.returns = []
        self.max_portfolio_value = self.portfolio_value  # Start at initial portfolio value

        

    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.position_open_price = 0
        self.position = None  # Can be 'long', 'short', or None
        self.hedge_position = None  # Can be 'long_hedge', 'short_hedge' or None
        self.hedge_position_open_price = 0
        self.returns = []  # To keep track of returns for calculating Sharpe ratio
        # self.max_portfolio_value = 0  # To keep track of the highest portfolio value for calculating drawdown
        self.portfolio_value = 10000  # Initial portfolio value
        self.leverage = 100
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.returns = []
        self.max_portfolio_value = self.portfolio_value  # Start at initial portfolio value
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.forex_data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA21', 'RSI']].values.astype(np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.forex_data):
            self.done = True
            return None, 0, self.done, {}  # Return a default obs, reward, done, info

                    
        obs = self._next_observation()
        immediate_reward = 0

        current_price = obs[3]

        current_loss = 0
        if self.position == 'long':
            current_loss = self.position_open_price - current_price
        elif self.position == 'short':
            current_loss = current_price - self.position_open_price

        if action == 0:  # Long
            self.position = 'long'
            self.position_open_price = current_price
            self.hedge_position = None  # Reset hedge
            self.hedge_position_open_price = 0  # Reset hedge price
        elif action == 1:  # Short
            self.position = 'short'
            self.position_open_price = current_price
            self.hedge_position = None  # Reset hedge
            self.hedge_position_open_price = 0  # Reset hedge price
        elif action == 2:  # Close
            if self.position == 'long':
                immediate_reward = current_price - self.position_open_price
            elif self.position == 'short':
                immediate_reward = self.position_open_price - current_price
            self.position = None
            self.hedge_position = None  # Reset hedge
            self.hedge_position_open_price = 0  # Reset hedge price
        elif action == 3:  # Hedge
            if current_loss > 0:  # Only allow hedging when the position is at a loss
                if self.position == 'long':
                    self.hedge_position = 'short_hedge'
                    self.hedge_position_open_price = current_price
                elif self.position == 'short':
                    self.hedge_position = 'long_hedge'
                    self.hedge_position_open_price = current_price

        delayed_reward = 0
        if self.current_step >= self.delay_steps:
            delayed_obs = self.forex_data.iloc[self.current_step - self.delay_steps][['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA21', 'RSI']].values
            delayed_price = delayed_obs[3]

            if self.position == 'long':
                delayed_reward = delayed_price - self.position_open_price
            elif self.position == 'short':
                delayed_reward = self.position_open_price - delayed_price
        
        delayed_hedge_reward = 0
        if self.current_step >= self.delay_steps:
            delayed_obs = self.forex_data.iloc[self.current_step - self.delay_steps][['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA21', 'RSI']].values
            delayed_price = delayed_obs[3]

            if self.hedge_position == 'long_hedge':
                delayed_hedge_reward = (delayed_price - self.hedge_position_open_price) * -self.hedge_multiplier
            elif self.hedge_position == 'short_hedge':
                delayed_hedge_reward = (self.hedge_position_open_price - delayed_price) * -self.hedge_multiplier
        

               # Calculate the effective position size considering leverage
        leveraged_position_size = self.portfolio_value * self.risk_per_trade * self.leverage
        
        # Calculate leveraged reward
        immediate_reward *= self.leverage
        delayed_reward *= self.leverage
        delayed_hedge_reward *= self.leverage * self.hedge_multiplier

         # Update portfolio value considering leverage (Note: This is simplified and assumes no margin calls)
        self.portfolio_value += immediate_reward + delayed_reward + delayed_hedge_reward  
        
        # Calculate the leveraged return based on leveraged position size
        leveraged_return = (immediate_reward + delayed_reward + delayed_hedge_reward) / leveraged_position_size
        self.returns.append(leveraged_return)
        
        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Calculate Sharpe ratio (assuming a risk-free rate of 0 for simplicity)
        sharpe_ratio = 0.0
        try:
            if np.std(self.returns) != 0:
                sharpe_ratio = np.mean(self.returns) / np.std(self.returns)
            else:
                sharpe_ratio = 0.0
        except Exception as e:
            print(f"An exception occurred: {e}")
            sharpe_ratio = 0.0
        
        # Calculate Drawdown
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # Log or return the custom metrics
        logging.info(f"Step: {self.current_step}, Action: {action}, Reward: {immediate_reward + delayed_reward + delayed_hedge_reward}, Total Portfolio Value: {self.portfolio_value}, Sharpe Ratio: {sharpe_ratio}, Drawdown: {drawdown}")
        if drawdown > 0.6 or self.portfolio_value < 10000 * 0.5:
            self.done = True
        return obs, immediate_reward + delayed_reward + delayed_hedge_reward, self.done,{ "Total Portfolio Value": self.portfolio_value, 
                                                                                         "sharpe_ratio": sharpe_ratio,
                                                                                           "drawdown": drawdown
                                                                                           }

        

env = ForexEnv(forex_data)
# Initialize counter and other variables
achieved_goal_count = 0
final_portfolio_value_goal = 50000  # Replace with your actual goal value
current_episode = 0
max_episodes = 1000  # Maximum number of episodes to run
best_portfolio_value = 0  # Initialize the best_portfolio_value to 0
model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, 
            target_update_interval=1000, tensorboard_log="tensorboard/")
model.learn(total_timesteps=10000,progress_bar=True,)
# -------------------- old code start ------------
# model.learn(total_timesteps=10000,progress_bar=True,)
# model.save(f'model/fx-model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pkl')
# obs = env.reset()
# done = False
# total_reward = 0
# init_portfolio_value = 0
# print(f'init_portfolio_value: {init_portfolio_value}')
# final_portfolio_value = 0
# while not done:
#     action, _ = model.predict(obs)
#     dict_value= {}
#     obs, reward, done, dict_value   = env.step(action)
#     if init_portfolio_value == 0:
#         init_portfolio_value = dict_value['Total Portfolio Value']
    
#     # print(dict_value)
#     total_reward += reward
    

#     if done:
#         logging.info("Episode done!")
#         obs = env.reset()
#         print(f'init_portfolio_value: {init_portfolio_value}')
#         print(f'final_portfolio_value: {final_portfolio_value}')
#     else:
#             # logging.info(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
#         logging.info(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}, Total Portfolio Value:{dict_value['Total Portfolio Value']},  Sharpe Ratio: {dict_value['sharpe_ratio']}, Drawdown: {dict_value['drawdown']}")
#         final_portfolio_value = dict_value['Total Portfolio Value']
# ------------old code end------------
portfolio_runs = []
while achieved_goal_count < 3 and current_episode < max_episodes:
    obs = env.reset()
    done = False
    init_portfolio_value = env.portfolio_value
    print(f'init_portfolio_value: {init_portfolio_value}')
    final_portfolio_value = 0
        # Retrain the model every 10 episodes (or any other number that makes sense for your application)

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, dict_value   = env.step(action)

        
        # Check final portfolio value
        if done:
            print(f'init_portfolio_value: {init_portfolio_value}')
            print(f'final_portfolio_value: {final_portfolio_value}')
            model_name = f'model/fx-model_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pkl'
            model.save(model_name)
            portfolio_runs.append((init_portfolio_value, final_portfolio_value, model_name))
              # Replace with actual code to get final portfolio value
            if final_portfolio_value >= final_portfolio_value_goal:
                achieved_goal_count += 1
                print(f"Goal achieved {achieved_goal_count} times!")
                # Save the model if this is the best portfolio value so far
                if final_portfolio_value > best_portfolio_value:
                    best_portfolio_value = final_portfolio_value
                    # model.save("model/best_model")
                    model.save(f'Best Model')
                    print(f"Best model saved with portfolio value {best_portfolio_value}")
            # else:
            #     # Reset counter if you want to achieve the goal consecutively
            #     # Remove this line if you want to achieve the goal 3 times irrespective of order
            #     achieved_goal_count = 0
        else:
            logging.info(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Total Portfolio Value:{dict_value['Total Portfolio Value']},  Sharpe Ratio: {dict_value['sharpe_ratio']}, Drawdown: {dict_value['drawdown']}")
            final_portfolio_value = dict_value['Total Portfolio Value']

    # Increment episode counter
    current_episode += 1
    
    # Retrain the model every 10 episodes (or any other number that makes sense for your application)
    if current_episode % 10 == 0:
        model.learn(total_timesteps=10000,progress_bar=True,)
        print("Model retrained.")
        
    # Check for max episodes
    if current_episode >= max_episodes:
        print("Maximum episodes reached. Stopping the loop.")

    df_pr = pd.DataFrame(portfolio_runs, columns=['init_portfolio_value', 'final_portfolio_value','model_name'])
    # print(f'portfolio_runs: {portfolio_runs}')
    df_pr.to_csv('portfolio_runs.csv', index=False)
