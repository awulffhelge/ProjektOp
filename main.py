""" Projekt Op main script"""


import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plots
import sqlite3
import classes


# Set some parameters
make_db = False  # If True, make data base
most_common_words = list(joblib.load("most_common_words1000.joblib"))  # Read file with most common words
# Make a list of words to manually add
not_stocks = ["fund", "per", "index", "tech", "ways", "corp", "uk", "tips", "dash", "im", "dow", "earn", "wins",
              "pros", "jobs", "away", "mini", "plus", "pure", "vice", "ai", "sub", "apps", "ups", "usd", "away",
              "giga", "edit", "media", "gmc", "loop", "mp", "dphc", "pic", "gaxy", "gpvrf", "loan", "cap", "loan"]
# Maybe list: joe, spaq, gene, ma, ba, dal, riot, nflx, iq, rare, snap, net, wrap, bio, att
# Add words if needed
#joblib.dump(np.asarray([str(word) for word in most_common_words]), "most_common_words1000.joblib")


# Construct data base, processing Adams sqlite data base and making a new called youtube_stock.db
if make_db:
    functions.construct_data_base(most_common_words, start_date=pd.Timestamp(2020, 7, 1))


# Load data sets
con = sqlite3.connect("data/youtube_stocks.db")
df_stockMentions = pd.read_sql_query("SELECT * from stockMentions", con)
df_youtubeSources = pd.read_sql_query("SELECT * from youtubeSources", con)
df_stockPrices = pd.read_sql_query("SELECT * from stockPrices", con)
df_trainTest = pd.read_sql_query("SELECT * from stockTrainOrTestSet", con)
con.close()
# Make Date column in stockPrices into Timestamps and set it as index
df_stockPrices["index"] = df_stockPrices["index"].apply(lambda x: pd.Timestamp(x))
df_stockPrices = df_stockPrices.set_index("index")
# Remove all not_stocks from data set
for name in not_stocks:
    df_stockMentions = df_stockMentions.loc[df_stockMentions["stock"].values != name.upper()]
# Convert dates to Timestamp
df_stockMentions["date"] = df_stockMentions["date"].apply(lambda x: pd.Timestamp(x))
# Get names of youtube channels
youtube_channels = df_youtubeSources["name"]
# Get the start date. For all data use: df_stockMentions["date"].min()
start_date = pd.Timestamp(2020, 7, 1)
# Remove all data before start_date
df_stockMentions = df_stockMentions[df_stockMentions["date"] > start_date]
# Get array containing names of all stocks mentioned more than once and array of stock discarded
stock_names = df_stockMentions["stock"].value_counts().index
#stock_discarded = df_stockMentions["stock"].value_counts().loc[lambda x: x < 2].index
# Remove all stocks not in stock_names
#for stock in stock_discarded:
#    df_stockMentions = df_stockMentions.loc[df_stockMentions["stock"].values != stock]


"""# Only visualize a single channel
visualize_channel = "RexFinance"
if visualize_channel:
    youtuber_id = df_youtubeSources["id"][df_youtubeSources["name"] == visualize_channel].item()
    df_stockMentions = df_stockMentions[df_stockMentions["source"] == youtuber_id]
    # Get array containing names of all stocks
    stock_names = df_stockMentions["stock"].unique()

# Visualize stocks and mentions over time
for stock in stock_names[60:120]:
    # Only take the relevant mentions
    df_mentions = df_stockMentions[df_stockMentions["stock"] == stock]
    # Plot a stock
    plots.plot_mentions_and_stocks_together(df_mentions, stock, start_date, df_stockPrices[stock])


# Define list to keep track of stats
avg_stats = list()
# Get stats from each youtuber
for youtuber_id in df_youtubeSources["id"]:
    # Make DataFrame with all rows related to youtuber_id
    df_youtuber = df_stockMentions[df_stockMentions["source"] == youtuber_id]
    # Get all the unique stocks mentioned in the channel
    stocks_in_channel = df_youtuber["stock"].unique()
    # For each stock:
    stat_list = list()
    for i, stock in enumerate(stocks_in_channel):
        # Get the date of the first mention of the stock
        first_mention = df_youtuber["date"][df_youtuber["stock"] == stock].min().date()
        # Get the stock prices from the date an onwards
        try:
            df_ticker = df_stockPrices[stock][df_stockPrices.index > pd.Timestamp(first_mention)].iloc[:31]
            if np.isnan(df_ticker.values).all():
                print(f"{stock, first_mention} prices contains only NANs within first 31 days the stocks are open after first mention")
                continue
            percentage_values = functions.compute_stock_percentage(df_ticker)
        except KeyError as e:
            print(f"{stock} not available")
            continue
        # Get the peak increase and decrease during the next month
        peak_increase = np.nanmax(percentage_values)
        peak_decrease = np.nanmin(percentage_values)
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
sort = np.flip(np.argsort(avg_stats[:, 0]))
df_followers = pd.DataFrame({"Name": df_youtubeSources["name"].values[sort], "Followers": df_youtubeSources["subscribers"].values[sort],
                             "Best performance": avg_stats[sort][:, 0], "Worst performance": avg_stats[sort][:, 1]})
print(f"Performance relative to followers: {df_followers.Followers}")
print(df_followers)
"""

# Train model tree
ml_model_forest = classes.MLModelForest(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)
ml_model_forest.get_parameters_and_labels("train")
ml_model_forest.cv_score(lay=4, leaf=5, bootstrap=True)
ml_model_forest.cv_predict_similarity()
ml_model_forest.money_out_train(plot_it=False)
ml_model_forest.fit()

"""for lay in [3, 4, 5, 6]:
    for leaf in [4, 5, 6, 7, 8, 9]:
        print("")
        print(f"Layers = {lay} and leafs = {leaf}")
        ml_model_forest.cv_score(lay=lay, leaf=leaf, bootstrap=True)
        ml_model_forest.cv_predict_similarity()
        ml_model_forest.money_out_train(plot_it=False)"""


# Test model tree
#ml_model_forest.get_parameters_and_labels("test")
#ml_model_forest.get_test_results(plot_it=False)

