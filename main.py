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
words_to_add = ["fund", "per", "index", "tech", "ways", "corp", "uk", "tips", "dash", "im", "dow", "earn", "wins",
                "pros", "jobs", "away", "mini", "plus", "pure", "vice", "ai", "sub", "apps", "ups", "usd", "away",
                "giga", "edit"]
# Maybe list: joe, spaq, gene, cap, ma, ba, dal, riot, nflx, iq, rare, snap, net, wrap, bio,
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
# Convert dates to Timestamp
df_youtubeVideos["date"] = df_youtubeVideos["date"].apply(lambda x: pd.Timestamp(x))
# Get names of youtube channels
youtube_channels = df_youtubeSources["name"]
# Get the start date. For all data use: df_youtubeVideos["date"].min()
start_date = pd.Timestamp(2020, 7, 1)
# Remove all data before start_date
df_youtubeVideos = df_youtubeVideos[df_youtubeVideos["date"] > start_date]
# Get array containing names of all stocks
stock_names = df_youtubeVideos["stock"].value_counts().loc[lambda x: x > 1].index

for stock in stock_names[280:]:
    # Visualize
    df_mentions = df_youtubeVideos[df_youtubeVideos["stock"] == stock]
    # Plot a stock
    plots.plot_mentions_and_stocks_together(df_mentions, stock, start_date)


