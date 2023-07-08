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
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
    self._seed()
                 
    self.s = spot_price
    self.k = strike_price
    self.dt = 1/time_discrete
    self.ttm = 21 * self.m * self.dt  # convert month to reaction time unit
    self.hedge = stock_share

    self.mu = exp_return  # expected return
    self.vol = exp_volatility
    self.rf = risk_free  # risk free rate
    self.div = dividend_yield
    self.kappa = kappa
    self.cost = transaction_cost

    self.states = None
    self.initials = (self.s/self.k, self.ttm, self.hedge)

    self.reset()    
  
  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
    
  def reset(self):
    self.states = self.initials[:-1]
    self.hedge = self.initials[2]
    self.counter = 0

  def step(self, action):
    self.counter += 1
    state = self.states

    s_1 = self.states[0] * self.k
    s = s_1 * ((1 + self.mu * self.dt) + (np.randn * self.vol) * np.sqrt(self.dt))

    self.ttm = max(0, self.states[1] - self.dt)
    done = self.ttm < self.dt

    lp = (s - s_1) * self.hedge - abs(action - self.hedge) * s * self.kappa -\
      self.bs_price(s, self.k, self.rf, self.ttm, self.vol) +\
      self.bs_price(s_1, self.k, self.rf, self.states[1], self.vol)

    if done:
      lp = lp - action * s * self.kappa

    reward = lp - self.cost * lp^2
    next_state = (s/self.k, self.ttm)
    self.hedge = action
    self.states = next_state

    return state, reward, done, next_state

  def bs_price(self, spot_price, time_to_maturity):
    s = spot_price
    ttm = time_to_maturity
    d1 = (np.log(s/self.k) + ttm * (self.rf - self.div + (self.vol**2) / 2.0)) / (self.vol * np.sqrt(ttm))
    d2 = d1 - self.vol * np.sqrt(ttm)
    call_bs_price = s * np.exp(-ttm * self.div) * norm.cdf(d1) - self.k * np.exp(-ttm * self.rf) * norm.cdf(d2)
    put_bs_price = -s * np.exp(-ttm * self.div) * norm.cdf(-d1) + self.k * np.exp(-ttm * self.rf) * norm.cdf(-d2)
    return call_bs_price, put_bs_price
    
  def greeks(self, spot_price, time_to_maturity):
    s = spot_price
    ttm = time_to_maturity
    d1 = (np.log(s/self.k) + ttm * (self.rf - self.div + (self.vol**2) / 2.0)) / (self.vol * np.sqrt(ttm))
    
    call_delta =  norm.cdf(d1) * np.exp(-ttm * self.div)
    put_delta = (1 - norm.cdf(d1)) * np.exp(-ttm * self.div)
    
    nd1p = self.exp(-(d1**2)/2)/np.sqrt(2*np.pi)
    gamma = (np.exp(-ttm * self.div) * nd1p) / (s * self.vol * np.sqrt(ttm))
    return call_delta, put_delta, gamma
