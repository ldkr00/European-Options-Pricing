import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install backtesting

class Brownian_Motion:
  def __init__(self, stepsize):
    self.stepsize = stepsize
    self.current_value = 0
    self.path = [0]

  def step(self):
    self.s = np.random.normal(0, self.stepsize)
    self.current_value += self.s
    self.path.append(self.current_value)

  def get_step(self):
    return self.s

  def get_current_value(self):
    return self.current_value

  def get_path(self):
    return self.path

class European_Option:
    
  ###############################################################################################
  # The object European_Option can be initialized with all relevant information about the option. 
  # Currently, there are two pricing functions implemented, namely the Black-Scholes model and a 
  # stochastic volatility model, which uses Monte-Carlo simulation to calculate the price.
  ###############################################################################################

  def __init__(self, Option_type, stock_price, strike, vol, time_to_maturity, interest_rate):
    # Option_type can either be "Call" or "Put"
    # stock_price, strike, vol and interest_rate need to be nonnegative floats or integers
    # time_to_maturity needs to be an integer and is interpreted in days

    # initialize
    self.Option_type = Option_type
    self.strike = strike
    self.vol = vol
    self.time_to_maturity = time_to_maturity
    self.stock_price = stock_price
    self.interest_rate = interest_rate
    self.BS_price = 0
    self.Stoch_Vol_price = 0
    self.Merton_price = 0

    # assertions
    assert Option_type == "Put" or Option_type == "Call", "Invalid Option Type: Only Call and Put allowed"
    assert type(self.stock_price) == int or type(self.stock_price) == float, "Stock price needs to be an integer or a float"
    assert type(self.strike) == int or type(self.strike) == float, "Strike needs to be an integer or a float"
    assert type(self.vol) == int or type(self.vol) == float, "Volatility needs to be an integer or a float"
    assert type(self.time_to_maturity) == int or type(self.time_to_maturity) == float, "Time to maturity needs to be an integer or a float"
    assert type(self.interest_rate) == int or type(self.interest_rate) == float, "Interest rate needs to be an integer or a float"
    assert self.stock_price >= 0 and self.strike >= 0 and self.vol >= 0 and self.time_to_maturity >= 0 and self.interest_rate >= 0, "Invalid Input, Numerical Inputs need to be nonnegative"
    
  # Pricing functions
  def get_BS_price(self):
    self.BS_price = self.BS_price_calculator()
    return self.BS_price

  def get_stoch_vol_price(self, n = 1000, rho = -0.7, dt = 1/252, vega = 0.9, alpha = 0.5, beta = 2, sigma_var = 0.3):
    self.Stoch_Vol_price = self.Monte_Carlo_Simulation(n, rho, dt, vega, alpha, beta, sigma_var, Stochastic_Vol = True)
    return self.Stoch_Vol_price

  def get_Merton_price(self, n = 1000, rho = -0.7, dt = 1/252, vega = 0.9, alpha = 0.5, beta = 2, sigma_var = 0.3):
    self.Merton_price = self.Monte_Carlo_Simulation(n, rho, dt, vega, alpha, beta, sigma_var, Jumps = True)
    return self.Merton_price

  def BS_price_calculator(self, delta = False):
    d1 = ((np.log(self.stock_price/self.strike) + (self.interest_rate + (self.vol**2)/2)*self.time_to_maturity)/(self.vol * np.sqrt(self.time_to_maturity)))
    d2 = d1 - self.vol * np.sqrt(self.time_to_maturity)
    if self.Option_type == "Call":
      price = (self.stock_price * norm.cdf(d1)) - (self.strike * np.exp(-self.interest_rate*self.time_to_maturity) * norm.cdf(d2))
    if self.Option_type == "Put":
      price =  (self.strike * np.exp(-self.interest_rate*self.time_to_maturity) * norm.cdf(d2)) - (self.stock_price * norm.cdf(d1))
    if delta:
      delta_value = norm.cdf(d1)
      return price, delta_value
    else:
      return price

  def Monte_Carlo_Simulation(self, n = 1000, rho = -0.2, dt = 1/252, vega = 0.9, alpha = 0.5, beta = 2, sigma_var = 0.3, Jumps = False, Stochastic_Vol = False):
    payoffs = []
    final_prices = []
    returns = []
    # initialize brownian motions
    BM_1 = Brownian_Motion(1)
    BM_2 = Brownian_Motion(1)
    for i in range(0,n):
      BM_1.step()
      BM_2.step()
      S_t = self.stock_price
      current_variance = vega
      previous_variance = current_variance
      step = 0
      # Simulation of a path
      while (step < self.time_to_maturity):      
        # Stochastic Volatility
        if Stochastic_Vol: 
          # Calculate increments of two correlated brownian motions (See Documentation)
          # W1 = np.random.randn()
          W1 = BM_1.get_step()
          # W2 = W1 * rho + np.sqrt(1-(rho**2)) * np.random.randn()
          W2 = W1 * rho + np.sqrt(1-(rho**2)) * BM_2.get_step()
          # Calculate stock price at new time point (See Documentation) 
          S_t_old = S_t
          S_t = S_t + self.interest_rate * S_t * dt + S_t * np.sqrt(current_variance * dt) * W1
          ret = (S_t - S_t_old)/S_t_old
          returns.append(ret)
          # Update Variance
          previous_variance = current_variance
          current_variance = previous_variance + (alpha - beta * previous_variance) * dt + sigma_var * np.sqrt(previous_variance * dt) * W2
          print(current_variance)
          #Variance should not get too low (convergence)
          if current_variance > 0.0000001:
            pass
          else:
            current_variance = 0.0000001

        # Merton Model TODO
        elif Jumps:
          pass

        step += dt
      
      # Calculate payoff of simulated path
      if self.Option_type == "Call":
        if S_t > self.strike: 
          payoff = np.exp(self.interest_rate * self.time_to_maturity) * (S_t - self.strike)
        else:
          payoff = 0
      else:
        if S_t < self.strike: 
          payoff = np.exp(self.interest_rate * self.time_to_maturity) * (self.strike - S_t)
        else:
          payoff = 0
      payoffs.append(payoff)
      final_prices.append(S_t)
    # plotting returns or payoffs TODO
    # plt.hist(returns, bins = 100)
    # plt.show()
    # sns.set_style('whitegrid')
    # sns.kdeplot(np.array(returns))
    return np.average(payoffs)

