import logging
import gym
import numpy as np
# ...
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

class ForexEnv(gym.Env):
    def __init__(self, forex_data, delay_steps=5):
        super(ForexEnv, self).__init__()
        
        self.forex_data = forex_data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.Space.Discrete(4)  # Long, Short, Close, Hedge
        self.observation_space = gym.Space.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # OHLC, MA5, MA21, RSI, Volume, position
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
