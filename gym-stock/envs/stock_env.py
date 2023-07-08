import numpy as np
from scipy.stats import norm

import gym
from gym import spaces
from gym.utils import seeding


class DeltaHedge(gym.Env):
  def __init__(self, spot_price: float=100, strike_price: float=100,
               time2maturity: int=3, time_discrete: int=250, stock_share: float=0,
               exp_volatility=.2, exp_return=.05, risk_free=0.03, kappa=.01,
               transaction_cost=1.5, dividend_yield=.0):

    self.action_space = spaces.Box(low=0, high=1, shape=(1,))
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
    self._seed()
                 
    self.s = spot_price
    self.k = strike_price
    self.m = time2maturity / 12
    self.dt = 1/time_discrete
    self.ttm = 21 * self.m * self.dt  # convert month to reaction time unit
    self.hedge = stock_share

    self.mu = exp_return  # expected return
    self.vol = exp_volatility
    self.rf = risk_free  # risk free rate
    self.div = dividend_yield
    self.kappa = kappa
    self.cost = transaction_cost

    self.state = None
    self.initials = (self.s/self.k, self.ttm, self.hedge)

    self.reset()
  
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
    
  def reset(self):
    self.state = self.initials

  def step(self, action):
    state = self.state
    pos = action

    s_1 = self.state[0] * self.strike
    s = s_1 * ((1 + self.mu * self.dt) + (np.randn * self.vol) * np.sqrt(self.dt))

    ttm = max(0, self.state[1] - self.dt)
    done = ttm < self.dt

    lp = (s - s_1) * self.state[2] - abs(pos - self.state[2]) * s * self.kappa -\
      self.bs_price(s, self.k, self.rf, ttm, self.vol) +\
      self.bs_price(s_1, self.k, self.rf, self.state[1], self.vol)

    if done:
      lp = lp - pos * s * self.kappa

    reward = lp - self.cost * lp^2
    next_state = (s/self.k, ttm, pos)
    self.state = next_state

    return state, reward, done, next_state

  def bs_price(self):
    d1 = (np.log(self.state[0]) + self.m * (self.rf - self.div + (self.vol**2) / 2.0)) / (self.vol * np.sqrt(self.m))
    d2 = d1 - self.vol * np.sqrt(self.m)
    self.s * np.exp(-self.m * self.div) * norm.cdf(d1) - self.k * np.exp(-self.rf * self.m) * norm.cdf(d2)

    return call_bs_price
  def greeks(self):
    pass
