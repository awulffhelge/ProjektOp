""" Plots """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions


def plot_mentions_and_stocks_together(data_base):

    for i, stock in enumerate(data_base.keys()):
        # Sum the mentions of stock
        summed_mentions = data_base[stock].sum(axis=0)
        # Get stock prices
        ticker_df = functions.get_stock_prices(stock)

        # Plot mentions and stock together
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.plot(ticker_df.index, ticker_df.values, color="gold", label="Stock price")
        ax1.plot(pd.to_datetime(summed_mentions.index), summed_mentions.values, color="teal", label="Mentions")
        ax0.legend(loc=2)
        ax1.legend(loc=6)
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Stock Price [$]")
        ax1.set_ylabel("Mentions [n]")
        fig.suptitle(stock)


