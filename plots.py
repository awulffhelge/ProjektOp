""" Plots """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions


def plot_mentions_and_stocks_together(df_mentions, stock, start_date, df_ticker):
    try:
        # Get stock prices in percent and raw
        percentage_value = functions.compute_stock_percentage(df_ticker)
        # Sum the mentions of stock
        df_mentions_over_time = functions.get_mentions_over_time(df_mentions, start_date)

        # Plot mentions and stock together
        fig, ax0 = plt.subplots()
        ax1 = ax0.twinx()
        ax0.plot(df_ticker.index, percentage_value, color="gold", label="Stock price", zorder=1)
        ax1.bar(df_mentions_over_time.index, df_mentions_over_time.values, color="teal", label="Mentions", zorder=100)
        ax0.legend(loc=2)
        ax1.legend(loc=6)
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Stock Price [%]")
        ax1.set_ylabel("Mentions [n]")
        ax1.set_ylim(0, 8)
        fig.suptitle(stock + " - start price: " + str(np.round(df_ticker[0], 2).item()) + " $")
        ax0.set_zorder(ax1.get_zorder() + 1)
        ax0.patch.set_visible(False)
        plt.show()
    except IndexError as e:
        print(e)




