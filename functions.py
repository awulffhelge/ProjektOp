""" Functions """


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import os
import joblib
import sqlite3


def get_stock_percentage(df_ticker):
    # Get stock prices in percentage
    percentage_value = (df_ticker.values / df_ticker.values[0]) * 100
    return percentage_value


def get_stock_prices(ticker_symbol, start_date):
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
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = datetime.today().date().strftime("%Y-%m-%d")

    #get data on this ticker
    ticker_data = yf.Ticker(ticker_symbol)

    #get the historical prices for this ticker
    df_ticker = ticker_data.history(period='1d', start=start_date, end=end_date)

    return df_ticker["Close"]


def get_stocks_mentioned(text, most_common_words, print_stats=False):
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
        stock_prices = get_stock_prices(word, [2019, 7, 1])  # The second input is random. See if stock exist
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


def construct_data_base(most_common_words):
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect("data/foo.db")
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
    # Close connection
    con.close()

    # Make one big text field of title, description and tags
    df_youtubeVideos["added"] = (df_youtubeVideos["title"] + " " + df_youtubeVideos["description"] + " " +
                                 df_youtubeVideos["tags"].apply(lambda x: x.replace(";", " ")))
    # Get the stocks mentioned in the text field
    df_youtubeVideos["output"] = df_youtubeVideos["added"].apply(lambda x:
                                                                 get_stocks_mentioned(x, most_common_words))
    # Split the output into a stocks mentioned column and non-stock words column
    df_youtubeVideos[["stock", "not_stocks"]] = pd.DataFrame(df_youtubeVideos["output"].tolist(),
                                                             index=df_youtubeVideos.index)
    # Drop unnecessary columns
    df_youtubeVideos = df_youtubeVideos.drop(columns=["added", "output", "not_stocks"])
    # Convert unix time to datetime
    df_youtubeVideos["date"] = pd.to_datetime(df_youtubeVideos["date"], unit="s")

    # Split list of stock mentioned in one video (i.e. row) into multiple rows
    df_youtubeVideos = df_youtubeVideos.explode("stock").dropna()

    # Get access to sqlite data base
    con = sqlite3.connect("data/youtube_stocks.db")
    # Write to sqlite data base
    df_youtubeVideos.to_sql("stockMentions", con, if_exists="replace")
    df_youtubeSources.to_sql("youtubeSources", con, if_exists="replace")
    # Close connection
    con.close()


def get_mentions_over_time(df_mentions, start_date):
    date_index = days_from_start_to_today(start_date)
    df_mentions_over_time = pd.Series(data=np.zeros(len(date_index)), index=date_index)
    for mention in df_mentions["date"]:
        df_mentions_over_time.loc[mention.date()] += 1
    return df_mentions_over_time
