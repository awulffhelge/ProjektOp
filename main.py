""" Projekt Op main script """


import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plots
import sqlite3


# Set some parameters
make_db = False  # If True, make data base
most_common_words = list(joblib.load("most_common_words1000.joblib"))  # Read file with most common words
# Make a list of words to manually add
not_stocks = ["fund", "per", "index", "tech", "ways", "corp", "uk", "tips", "dash", "im", "dow", "earn", "wins",
              "pros", "jobs", "away", "mini", "plus", "pure", "vice", "ai", "sub", "apps", "ups", "usd", "away",
              "giga", "edit", "media"]
# Maybe list: joe, spaq, gene, cap, ma, ba, dal, riot, nflx, iq, rare, snap, net, wrap, bio, att
# Problems with some stocks (PIC)
# Add words if needed
#joblib.dump(np.asarray([str(word) for word in most_common_words]), "most_common_words1000.joblib")

# Construct data base, processing Adams sqlite data base and making a new called youtube_stock.db
if make_db:
    functions.construct_data_base(most_common_words)

# Load data sets
con = sqlite3.connect("data/youtube_stocks.db")
df_youtubeVideos = pd.read_sql_query("SELECT * from stockMentions", con)
df_youtubeSources = pd.read_sql_query("SELECT * from youtubeSources", con)
con.close()
# Remove all not_stocks from data set
for name in not_stocks:
    df_youtubeVideos = df_youtubeVideos.loc[df_youtubeVideos["stock"].values != name.upper()]
# Convert dates to Timestamp
df_youtubeVideos["date"] = df_youtubeVideos["date"].apply(lambda x: pd.Timestamp(x))
# Get names of youtube channels
youtube_channels = df_youtubeSources["name"]
# Get the start date. For all data use: df_youtubeVideos["date"].min()
start_date = pd.Timestamp(2020, 7, 1)
# Remove all data before start_date
df_youtubeVideos = df_youtubeVideos[df_youtubeVideos["date"] > start_date]
# Get array containing names of all stocks mentioned more than once and array of stock discarded
stock_names = df_youtubeVideos["stock"].value_counts().loc[lambda x: x > 1].index
stock_discarded = df_youtubeVideos["stock"].value_counts().loc[lambda x: x < 2].index
# Remove all stocks not in stock_names
for stock in stock_discarded:
    df_youtubeVideos = df_youtubeVideos.loc[df_youtubeVideos["stock"].values != stock]


# Only visualize a single channel
visualize_channel = "RexFinance"
if visualize_channel:
    youtuber_id = df_youtubeSources["id"][df_youtubeSources["name"] == visualize_channel].item()
    df_youtubeVideos = df_youtubeVideos[df_youtubeVideos["source"] == youtuber_id]
    # Get array containing names of all stocks
    stock_names = df_youtubeVideos["stock"].unique()

# Visualize stocks and mentions over time
for stock in stock_names:
    # Only take the relevant mentions
    df_mentions = df_youtubeVideos[df_youtubeVideos["stock"] == stock]
    # Plot a stock
    plots.plot_mentions_and_stocks_together(df_mentions, stock, start_date)


# Define list to keep track of stats
avg_stats = list()
# Get stats from each youtuber
for youtuber_id in df_youtubeSources["id"]:
    # Make DataFrame with all rows related to youtuber_id
    df_youtuber = df_youtubeVideos[df_youtubeVideos["source"] == youtuber_id]
    # Get all the unique stocks mentioned in the channel
    stocks_in_channel = df_youtuber["stock"].unique()
    # For each stock:
    stat_list = list()
    for i, stock in enumerate(stocks_in_channel):
        # Get the date of the first mention of the stock
        first_mention = df_youtuber["date"][df_youtuber["stock"] == stock].min().date()
        # Get the stock prices from the date an onwards
        try:
            percentage_values, df_ticker = functions.get_stock_percentage(stock, first_mention)
        except IndexError as e:
            print(e)
            continue
        # Get the peak increase and decrease during the next month
        peak_increase = percentage_values[:31].max()
        peak_decrease = percentage_values[:31].min()
        #print(stock, peak_increase, peak_decrease)
        stat_list.append([peak_increase, peak_decrease])
    avg_stats.append(np.mean(stat_list, axis=0))
avg_stats = np.asarray(avg_stats)
# Get top performance and bottom performance
top_performers = np.argpartition(avg_stats[:, 0], -3)[-3:]
bottom_performers = np.argpartition(avg_stats[:, 0] * -1, -3)[-3:]
# Print it
print("-----------------------")
print(f"The three top youtube channels were ")
print(f"{df_youtubeSources.name.iloc[top_performers]}. Earning between"
      f" {avg_stats[top_performers, 0]} and {avg_stats[top_performers, 1]} %")
print(f"The three bottom youtube channels were ")
print(f"{df_youtubeSources.name.iloc[bottom_performers]}. Earning between"
      f" {avg_stats[bottom_performers, 0]} and {avg_stats[bottom_performers, 1]} %")
print("-----------------------")
