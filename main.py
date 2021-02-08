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
    con = sqlite3.connect("get_data/aktiespekulanterne/data/youtube_stocks.db")
    df_stockMentions = pd.read_sql_query("SELECT * from stockMentions", con)
    df_youtubeSources = pd.read_sql_query("SELECT * from youtubeSources", con)
    df_stockPrices = pd.read_sql_query("SELECT * from stockPrices", con)
    df_trainTest = pd.read_sql_query("SELECT * from stockTrainOrTestSet", con)
    con.close()

    # Make Date column in stockPrices into Timestamps and set it as index
    #df_stockPrices["index"] = df_stockPrices["index"].apply(lambda x: pd.Timestamp(x))
    #df_stockPrices = df_stockPrices.set_index("index")

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
    ml_model_forest.cv_score(lay=3, leaf=9, bootstrap=True)
    ml_model_forest.cv_predict_similarity()
    ml_model_forest.money_out_train(plot_it=False, thrs=1000)
    ml_model_forest.fit()

    print(ml_model_forest.feature_names)
    print(ml_model_forest.forest.feature_importances_)

    """for lay in [50, 55, 60, 65, 70, 75]:
        print("")
        print(f"Layers = {lay} and leafs = {9}")
        ml_model_forest.cv_score(lay=3, leaf=9, bootstrap=True)
        ml_model_forest.cv_predict_similarity()
        ml_model_forest.money_out_train(plot_it=False, thrs=lay)"""

    # Test model tree
    ml_model_forest.get_parameters_and_labels("test", end_date=end_date)
    ml_model_forest.get_test_results(plot_it=False)

    joblib.dump([ml_model_forest, end_date], "moneyModel.joblib")

else:
    # Load model and last check time
    ml_model_forest, last_check = joblib.load("moneyModel.joblib")
    # Change last_check if it is too long ago
    if last_check < pd.Timestamp(pd.Timestamp.today().date()) - pd.Timedelta(days=1):
        last_check = pd.Timestamp(pd.Timestamp.today().date()) - pd.Timedelta(days=1)
    # Update data base
    ml_model_forest.update_database(most_common_words, last_check)
    input("Press Enter to predict and save...")
    # Predict which stocks to buy
    ml_model_forest.get_new_buys(last_check)
    # Save model and last check time
    print("Saving...")
    joblib.dump([ml_model_forest, pd.Timestamp.today()], "moneyModel.joblib")
