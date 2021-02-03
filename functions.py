""" Functions """


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import os
import joblib
import random
import sqlite3
import finnhub
import time


def compute_stock_percentage(df_ticker):
    return (df_ticker.values / df_ticker.dropna().values[0]) * 100


def get_stock_percentage(ticker_symbol, start_date, finnhub_client):
    # Get stock prices
    df_ticker = get_stock_prices(ticker_symbol, start_date, finnhub_client)
    # Get stock prices in percentage
    percentage_value = (df_ticker.values / df_ticker.values[0]) * 100
    return percentage_value, df_ticker


def get_stock_prices(ticker_symbol, start_date, finnhub_client):
    """ Get the closing stock price for the last year
    Parameters
    ----------
        ticker_symbol: str
            Stock ticker symbol
    Return
    -------
        ticker_df["Close"]: pandas dataframe
            Dataframe containing all closing values in the only column and corresponding dates in the index. Only dates
            where the stockmarket is open. """
    end_date = pd.Timestamp(pd.Timestamp.today().date())
    end_unix = get_unix_time(end_date)
    start_unix = get_unix_time(start_date)

    # Pause shortly
    time.sleep(1)

    # Stock candles
    res = finnhub_client.stock_candles(ticker_symbol, 'D', start_unix, end_unix)
    if res["s"] == "no_data":
        return pd.DataFrame()
    # Convert to Pandas Dataframe
    df_finnhub = pd.DataFrame(res)
    timestamp_index = df_finnhub["t"].apply(lambda x: pd.Timestamp(pd.to_datetime(x, unit='s', origin='unix').date()))
    df_ticker = pd.DataFrame(df_finnhub["c"].values, index=timestamp_index.values)
    return df_ticker


def get_stocks_mentioned(text, most_common_words, finnhub_client, start_date, print_stats=False):
    not_stocks = list()
    actual_stocks = list()
    text = clean_text(text)
    val_counts = pd.Series(text).value_counts()
    word_count = len(pd.Series(text))
    if print_stats:
        print(word_count)
    for i, word in enumerate(val_counts.index[val_counts > round(word_count / 200)]):
        if word.lower() in most_common_words:
            continue
        if print_stats:
            print(word, val_counts[i])
        stock_prices = get_stock_prices(word, start_date, finnhub_client)  # The second input is random. See if stock exist
        if len(stock_prices) == 0:
            not_stocks.append(word.lower())
        else:
            actual_stocks.append(word)
    return actual_stocks, not_stocks


def clean_text(text):
    return (text.replace(",", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            .replace(".", "").replace("$", "").replace("#", "").replace("@", "").replace(":", "")
            .replace(";", "").replace("/", "").replace("!", "").replace("?", "").replace("-", "")
            .replace("_", "").replace("&", "").replace("*", "").upper().split())


def days_from_start_to_today(start_date):
    list_of_days = [(start_date + pd.Timedelta(days=days)).date() for days in
                    range((pd.Timestamp.today() - start_date).days + 1)]
    return list_of_days


def get_mentions_over_time(df_mentions, start_date):
    date_index = days_from_start_to_today(start_date)
    df_mentions_over_time = pd.Series(data=np.zeros(len(date_index)), index=date_index)
    for mention in df_mentions["date"]:
        df_mentions_over_time.loc[mention.date()] += 1
    return df_mentions_over_time


def construct_data_base(most_common_words, start_date, end_date, last_check=pd.Timestamp(2020, 7, 1)):
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect("get_data/aktiespekulanterne/data/foo.db")
    # Load files
    """ foo.db sql data base build
    TABLE `youtubeSources` (
        `id`            INTEGER PRIMARY KEY,
        `name`          TEXT,
        `channelId`     TEXT,
        `playlistId`    TEXT,
        `subscribers`   INTEGER
    );
    TABLE `youtubeVideos` (
        `id`            INTEGER PRIMARY KEY,
        `date`          INTEGER,
        `title`         TEXT,
        `description`   TEXT,
        `tags`          TEXT,
        `videoId`       TEXT,
        `source`        INTEGER
    );"""
    df_youtubeVideos = pd.read_sql_query("SELECT * from youtubeVideos", con)
    df_youtubeSources = pd.read_sql_query("SELECT * from youtubeSources", con)
    df_youtubeMentions = pd.read_sql_query("SELECT * from youtubeMentions", con)
    df_tickers = pd.read_sql_query("SELECT * from tickers", con)
    # Close connection
    con.close()

    # First way of making data base
    """
    # Set up finnhub client
    finnhub_client = finnhub.Client(api_key="bvvj5q748v6qmipnulbg")

    # Make one big text field of title, description and tags
    df_youtubeVideos["added"] = (df_youtubeVideos["title"] + " " + df_youtubeVideos["description"] + " " +
                                 df_youtubeVideos["tags"].apply(lambda x: x.replace(";", " ")))
    # Get the stocks mentioned in the text field
    df_youtubeVideos["output"] = df_youtubeVideos["added"].apply(lambda x:
                                                                 get_stocks_mentioned(x, most_common_words,
                                                                                      finnhub_client, start_date))
    # Split the output into a stocks mentioned column and non-stock words column
    df_youtubeVideos[["stock", "not_stocks"]] = pd.DataFrame(df_youtubeVideos["output"].tolist(),
                                                             index=df_youtubeVideos.index)
    # Drop unnecessary columns
    df_youtubeVideos = df_youtubeVideos.drop(columns=["added", "output", "not_stocks"])

    # Convert unix time to datetime
    df_youtubeVideos["date"] = pd.to_datetime(df_youtubeVideos["date"], unit="s")

    # Split list of stock mentioned in one video (i.e. row) into multiple rows
    df_youtubeVideos = df_youtubeVideos.explode("stock").dropna()"""

    # Remove stocks that exist in most_common_words
    not_ticker_bool = df_tickers["symbol"].apply(lambda x: x.lower() in most_common_words)
    df_tickers_id = df_tickers[~not_ticker_bool]["id"].values
    mentions_bool = df_youtubeMentions["ticker"].apply(lambda x: x in df_tickers_id)
    df_youtubeMentions = df_youtubeMentions[mentions_bool]

    # Put the stocks mentioned into df_youtubeVideos
    df_youtubeMentions.ticker = df_youtubeMentions.ticker.apply(lambda x: df_tickers["symbol"][df_tickers["id"] == x].values[0])

    # Drop and rename columns
    df_youtubeMentions = df_youtubeMentions.rename(columns={"ticker": "stock"})
    df_youtubeVideos = df_youtubeVideos.rename(columns={"videoId": "url"})
    df_youtubeMentions = df_youtubeMentions.drop(columns=["id", "source", "date"])

    # Associate ticker symbols with the videos
    df_youtubeVideos = df_youtubeMentions.merge(df_youtubeVideos, left_on="videoId", right_on="id")
    df_youtubeVideos = df_youtubeVideos.drop(columns=["videoId"])

    # Convert unix time to datetime
    df_youtubeVideos["date"] = pd.to_datetime(df_youtubeVideos["date"], unit="s")

    # Remove videos from youtubers with more than 2 videos per day
    bool_mask = df_youtubeVideos["date"] > pd.Timestamp.today() - pd.Timedelta(days=120)
    video_value_counts = df_youtubeVideos[bool_mask].source.value_counts()
    df_value_counts = pd.DataFrame({"a": video_value_counts.index[video_value_counts < int(3 * 120)]})
    df_youtubeVideos = df_youtubeVideos.merge(df_value_counts, left_on="source", right_on="a").drop(columns="a")

    # Extract stocks used for train and test set by removing all mentions that are before start_date and after end_date
    bool_mask = np.logical_and(df_youtubeVideos["date"] > start_date, df_youtubeVideos["date"] < end_date)
    stock_names_trandte = df_youtubeVideos["stock"][bool_mask].value_counts()
    stock_names_trandte = stock_names_trandte[stock_names_trandte > 2].index.values
    # Extract new_stocks
    if last_check < pd.Timestamp(pd.Timestamp.today().date()) - pd.Timedelta(days=1):
        bool_mask = df_youtubeVideos["date"] > pd.Timestamp(pd.Timestamp.today().date()) - pd.Timedelta(days=1)
    else:
        bool_mask = df_youtubeVideos["date"] > last_check
    stock_names_new = df_youtubeVideos["stock"][bool_mask].value_counts().index.values

    # Split into test and training, and append new stocks
    random.Random(4).shuffle(stock_names_trandte)
    train_stocks = stock_names_trandte[:int(len(stock_names_trandte) * 0.7)]
    test_stocks = stock_names_trandte[int(len(stock_names_trandte) * 0.7):]
    traintest_stock = np.concatenate((train_stocks, test_stocks, stock_names_new))
    traintest_param = ["train"] * len(train_stocks) + ["test"] * len(test_stocks) + ["since_last"] * len(stock_names_new)
    df_traintest = pd.DataFrame({"data_set": traintest_param}, index=traintest_stock)

    # Get stock data for stocks mentioned more than twice
    if last_check == pd.Timestamp(2020, 7, 1):
        df_stockPrices = get_stock_data(traintest_stock, start_date)
    else:
        df_stockPrices = get_stock_data(stock_names_new, start_date)

    if last_check == pd.Timestamp(2020, 7, 1):
        # Get performance of youtubers
        df_performance = get_best_youtubers(df_youtubeSources, df_youtubeVideos[np.logical_and(
            df_youtubeVideos["date"] > start_date, df_youtubeVideos["date"] < end_date)], df_traintest, df_stockPrices.transpose())
    else:
        # Get access to sqlite data base
        con = sqlite3.connect("get_data/aktiespekulanterne/data/youtube_stocks.db")
        # Get Dataframe
        df_performance = pd.read_sql_query("SELECT * from df_performance", con)
        # Close connection
        con.close()

    # Keep only the 30 best youtubers in df_youtubeVideos and df_youtubeSources
    keep_youtuber = df_performance.Name[:30]
    keep_youtuber = keep_youtuber.apply(lambda x: df_youtubeSources["id"][df_youtubeSources.name == x].values[0]).values
    df_youtubeVideos = df_youtubeVideos.merge(pd.DataFrame({"keep": keep_youtuber}), left_on="source", right_on="keep")
    df_youtubeVideos = df_youtubeVideos.drop(columns="keep")

    # Get access to sqlite data base
    con = sqlite3.connect("get_data/aktiespekulanterne/data/youtube_stocks.db")
    # Write to sqlite data base
    if last_check == pd.Timestamp(2020, 7, 1):
        df_youtubeVideos.to_sql("stockMentions", con, if_exists="replace")
        df_youtubeSources.to_sql("youtubeSources", con, if_exists="replace")
        df_traintest.to_sql("stockTrainOrTestSet", con, if_exists="replace")
        df_performance.to_sql("df_performance", con, if_exists="replace")
    df_stockPrices.to_sql("stockPrices", con, if_exists="replace")
    # Close connection
    con.close()


def get_stock_data(stock_names, start_date):
    # Set end date
    end_date = pd.Timestamp(pd.Timestamp.today().date())
    df = pd.DataFrame(index=pd.date_range(start_date, end_date))

    # Setup client
    finnhub_client = finnhub.Client(api_key="bvvj5q748v6qmipnulbg")
    for stock in stock_names:
        # Get prices of stock and save them in a pd.DataFrame
        try:
            _, df_ticker = get_stock_percentage(stock, start_date, finnhub_client)
        except (IndexError, ValueError) as e:
            print(f"{stock} not loaded, error message: {e}")
            continue
        df[stock] = df_ticker
    return df.transpose()


def get_unix_time(timestamp):
    return int((timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))


def get_best_youtubers(df_youtubeSources, df_stockMentions, df_traintest, df_stockPrices):
    # Define list to keep track of stats
    train_stocks = df_traintest.index[df_traintest["data_set"] == "train"].values
    avg_stats = list()
    # Get stats from each youtuber
    for youtuber_id in df_youtubeSources["id"]:
        # Make DataFrame with all rows related to youtuber_id
        df_youtuber = df_stockMentions[df_stockMentions["source"] == youtuber_id]
        # Get all the unique stocks mentioned in the channel
        stocks_in_channel = df_youtuber["stock"].unique()
        # For each stock:
        stat_list = list()
        for stock in stocks_in_channel:
            if stock not in train_stocks:
                continue
            # Get the date of the first mention of the stock
            first_mention = df_youtuber["date"][df_youtuber["stock"] == stock].min().date()
            # Get the stock prices from the date an onwards
            try:
                df_ticker = df_stockPrices[stock][df_stockPrices.index > pd.Timestamp(first_mention)].iloc[:45]
                if np.isnan(df_ticker.values.astype(np.float64)).all():
                    #print(
                    #    f"{stock, first_mention} prices contains only NANs within first 31 days the stocks are open after first mention")
                    continue
                percentage_values = (df_ticker.values.astype(np.float64) / df_ticker.dropna().values.astype(np.float64)[
                    0]) * 100
            except KeyError as e:
                #print(f"{stock} not available")
                continue
            # Get the peak increase and decrease during the next month
            peak_increase = np.nanmax(percentage_values)
            peak_decrease = np.nanmin(percentage_values)
            # print(stock, peak_increase, peak_decrease)
            stat_list.append([peak_increase, peak_decrease])
        if len(stat_list) == 0:
            avg_stats.append(np.asarray([0, 0]))
        else:
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
    df_followers = pd.DataFrame(
        {"Name": df_youtubeSources["name"].values[sort], "Followers": df_youtubeSources["subscribers"].values[sort],
         "Best performance": avg_stats[sort][:, 0], "Worst performance": avg_stats[sort][:, 1]})
    return df_followers
