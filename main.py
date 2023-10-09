import numpy as np
import yfinance as yf
import pandas as pd
from pandas_datareader import data as wb
import time
import matplotlib.pyplot as plt


class DataManager():
    def __init__(self, user_ticker, user_sdate, user_edate):
         self.user_ticker = user_ticker
         self.user_sdate = user_sdate
         self.user_edate = user_edate
         self.df = self.load_data()
    
    def load_data(self):
        yf.pdr_override()                                                                   #zastąpienie domyślnego pandasa przez yfinance
        ticker_info = wb.DataReader(self.user_ticker, start=self.user_sdate, end=self.user_edate)  
        df = pd.DataFrame(ticker_info)
        if "Date" in df.index.names:
            df.reset_index(inplace=True)
        df.dropna(inplace=True)      
        return df             
        
    def get_simple_returns(self):
        self.df["Daily Returns"] = ((self.df["Close"]-self.df["Open"])/self.df["Open"])
        return self.df["Daily Returns"]
    
    def get_log_returns(self):
        self.df["Log Returns"] = np.log(self.df["Close"] / self.df["Close"].shift(1))  # Calculate log returns
        return self.df["Log Returns"]
    
    def get_std(self):
        standard_dev =  self.df["Log Returns"].std()*100
        print("Standard deviation: "+str(standard_dev)+"%")
        return standard_dev
    
    def plot_price(self):
        plt.figure(figsize=(3,3))
        plt.xlabel("Date")
        plt.ylabel(self.user_ticker+" price")
        plt.title(self.user_ticker+" price")
        plt.plot(self.df["Date"], self.df["Close"])
        fig = plt.gcf()
        fig_manager = fig.canvas.manager
        fig_manager.window.state('zoomed')  
        plt.subplots_adjust(left=0.1, right=0.7, bottom=0.4)
        plt.show()

        
class MonteCarlo():
    def __init__(self, df, confidence_level, time_horizon):
         self.df = df
         self.confidence_level = confidence_level
         self.time_horizon = time_horizon
         self.value_at_risk = None


    def monte_carlo_historical(self):
        simulations_number = 10000
        simulation_results = []
        for i in range(simulations_number):
            random_returns = np.random.normal(
                self.df.mean(), self.df.std(), self.time_horizon
            )
            random_sum = np.sum(random_returns)
            simulation_results.append(random_sum)
        simulation_results.sort()
        index_var = int((1-self.confidence_level)*simulations_number)
        self.value_at_risk = -simulation_results[index_var]
        return f"There is {int((1-self.confidence_level)*100)}% probability that instrument's price will fall by more than {round(self.value_at_risk*100,8)} % over {self.time_horizon} day period"

    def plot_histogram(self):
        plt.figure(figsize=(8,8))
        plt.subplot(2, 1, 1)
        plt.hist(self.df.dropna(), bins= 30, density=False)
        plt.xlabel(f"{self.time_horizon}-day Security's Return")
        plt.ylabel('Frequency')
        plt.title(f"Distribution of security's {self.time_horizon}-day returns")
        plt.axvline(-self.value_at_risk, color='r', linestyle='dashed', linewidth=2,label=f'Var at {self.confidence_level}')
        plt.legend()
        fig = plt.gcf()
        fig_manager = fig.canvas.manager
        fig_manager.window.state('zoomed')  
        plt.subplots_adjust(left=0.1, right=0.7)
        plt.figtext(0.4, 0.3, f"There is {int((1-self.confidence_level)*100)}% probability that instrument's price will fall by more than {round(self.value_at_risk*100,8)} % over {self.time_horizon} day period", fontsize=12, color='black', ha="center")
        plt.show()

seed = int(time.time())
np.random.seed(seed)

user_ticker = input("Ticker: ")
user_sdate = input("Starting Date: ")
user_edate = input("Ending Date: ")
df = DataManager(user_ticker,user_sdate,user_edate)
while True:
    print(user_ticker+" DATA")
    print("*************")
    print("1. Price chart")
    print("2. VaR histogram")
    print("3. Close")
    choice = input("Choose number: ")
    if choice == "1":
        df.plot_price()
    elif choice =="2":
        print("1- log returns\n2- simple returns")
        user_returns = input("Select returns type: ")
        if user_returns == "1":
            print("log")
            returns = df.get_log_returns()
            print(returns)
        elif user_returns =="2":
            returns = df.get_simple_returns()
        else:
            print("Wrong option")
        simulation1 = MonteCarlo(returns,0.95, 5)
        simulation1.monte_carlo_historical()
        simulation1.plot_histogram()
    elif choice == "3":
        break


