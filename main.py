""" Projekt Op main script"""


import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plots
import sqlite3
import classes
import os

# Load most common words
most_common_words = list(joblib.load("most_common_words1000.joblib"))  # Read file with most common words

if not os.path.isfile("moneyModel.joblib"):
    make_db = False  # If True, make data base

    # Make a list of words to manually add
    not_stocks = []
    # Maybe list: joe, spaq, gene, ma, ba, dal, riot, nflx, iq, rare, snap, net, wrap, bio, att
    # Add words if needed
    #most_common_words = most_common_words + not_stocks
    #joblib.dump(np.asarray([str(word) for word in most_common_words]), "most_common_words1000.joblib")

    # Construct data base, processing Adams sqlite data base and making a new called youtube_stock.db
    # Get the start date and end date of the training set.
    start_date = pd.Timestamp(2020, 7, 1)
    end_date = pd.Timestamp(2021, 1, 1)
    if make_db:
        functions.construct_data_base(most_common_words, start_date=start_date,
                                      end_date=end_date)

    # Load data sets
    con = sqlite3.connect("get_data/aktiespekulanterne/data/youtube_stocks_training.db")
    df_stockMentions = pd.read_sql_query("SELECT * from stockMentions", con)
    df_youtubeSources = pd.read_sql_query("SELECT * from youtubeSources", con)
    df_stockPrices = pd.read_sql_query("SELECT * from stockPrices", con)
    df_trainTest = pd.read_sql_query("SELECT * from stockTrainOrTestSet", con)
    con.close()

    # Transpose Dataframe and str index in stockPrices into Timestamps
    df_stockPrices = df_stockPrices.set_index("index").transpose()
    timestamps = list()
    for i in range(len(df_stockPrices)):
        timestamps.append(pd.Timestamp(df_stockPrices.index[i]))
    df_stockPrices.index = timestamps
    # Convert dates to Timestamp
    df_stockMentions["date"] = df_stockMentions["date"].apply(lambda x: pd.Timestamp(x))

    # Get names of youtube channels
    youtube_channels = df_youtubeSources["name"]
    # Get array containing names of all stocks mentioned more than once and array of stock discarded
    stock_names = df_stockMentions["stock"].value_counts().index

    # Train model tree
    ml_model_forest = classes.MLModelForest(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)
    ml_model_forest.get_parameters_and_labels("train", end_date=end_date)
    ml_model_forest.cv_score(lay=3, leaf=5, bootstrap=True)
    ml_model_forest.cv_predict_similarity()
    ml_model_forest.money_out_train(plot_it=False, thrs=1000)
    ml_model_forest.fit()

    print(ml_model_forest.feature_names)
    print(ml_model_forest.forest.feature_importances_)

    """for lay in [2, 3, 4, 5, 6]:
        for leaf in [3, 5, 7, 9, 11]:
            print("")
            print(f"Layers = {lay} and leafs = {leaf}")
            ml_model_forest.cv_score(lay=lay, leaf=leaf, bootstrap=True)
            ml_model_forest.cv_predict_similarity()
            ml_model_forest.money_out_train(plot_it=False, thrs=1000)"""

    # Test model tree
    ml_model_forest.get_parameters_and_labels("test", end_date=end_date)
    ml_model_forest.get_test_results(plot_it=False)

    joblib.dump(ml_model_forest, "moneyModel.joblib")

else:
    # Get time for last check
    if os.path.isfile("algorithm_predictions.csv"):
        df = pd.read_csv("algorithm_predictions.csv").drop(columns="Unnamed: 0")
        last_check = pd.Timestamp(df["time_of_prediction"].iloc[-1])
    else:
        last_check = pd.Timestamp(2021, 1, 1)
    # Load model
    ml_model_forest = joblib.load("moneyModel.joblib")
    # Update data base
    ml_model_forest.update_database(most_common_words, last_check)
    # Predict which stocks to buy
    ml_model_forest.get_new_buys(last_check)
