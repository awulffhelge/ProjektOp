""" Classes """


import numpy as np
import pandas as pd
import random
import sqlite3
from sklearn.model_selection import cross_val_predict, PredefinedSplit, cross_validate
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import functions
import os
np.seterr("raise")


class MLModel:
    """ Ideas for features:
     """
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        self.df_stockMentions = df_stockMentions
        self.df_stockPrices = df_stockPrices
        self.df_youtubeSources = df_youtubeSources

        self.train_set_params = np.empty((0, 9))
        self.test_set_params = np.empty((0, 9))
        self.since_last_set_params = np.empty((0, 9))

        self.train_set_labels = np.empty(0)
        self.test_set_labels = np.empty(0)

        self.train_set_money = np.empty((0, 61))
        self.test_set_money = np.empty((0, 61))

        self.feature_names = ["price_today", "price_avg_last_3_days", "price_avg_last_7_days", "days_since_first",
                              "followers", "price_avg_last_30_days", "num_mentions_before", "diff_num_youtuber",
                              "mentions_last_week"]

        self.stock_names = df_stockMentions["stock"].value_counts().index.values

        self.train_stocks = df_trainTest["index"][df_trainTest["data_set"] == "train"].values
        self.test_stocks = df_trainTest["index"][df_trainTest["data_set"] == "test"].values
        self.new_stocks = df_trainTest["index"][df_trainTest["data_set"] == "since_last"].values

        self.amount_of_stock_input = list()
        self.stock_kfold_idxs = list()  # Pre determined folds to ensure that stocks are split between train and val

        self.stock_list_train = list()
        self.stock_list_test = list()
        self.stock_list_since_last = list()

        self.mention_dates_since_last = np.empty(0)

        self.mention_futures_train = list()
        self.mention_futures_test = list()

    def _get_mention_parameters(self, df_stockMentions_small, date_of_mention):
        # Count mentions of stock before date_of_mention
        count_mentions_before = len(df_stockMentions_small["date"][df_stockMentions_small["date"] < date_of_mention])
        # Count mentions of stock before date_of_mention by different people
        diff_mentions_before = len(df_stockMentions_small["source"][df_stockMentions_small["date"] <= date_of_mention].unique())
        # Days since first mention
        days_since_first = (date_of_mention - df_stockMentions_small["date"].min()).days
        # Mentions the last week
        mention_diff = (date_of_mention - df_stockMentions_small["date"]).apply(lambda x: x.days).values
        mentions_last_week = len(mention_diff[np.logical_and(0 < mention_diff, mention_diff < 7)])
        # Sum of followers on mentioning channels
        mentions_today = np.unique(df_stockMentions_small["source"].iloc[np.nonzero(mention_diff == 0)]) - 1
        followers = self.df_youtubeSources.loc[mentions_today, "subscribers"].sum()
        # Make parameters for visualizing later mentions
        mentions_0t2weeks = len(df_stockMentions_small["date"][np.logical_and(date_of_mention < df_stockMentions_small["date"], df_stockMentions_small["date"] <
                                                                date_of_mention + pd.Timedelta(days=14))])
        mentions_2t4weeks = len(df_stockMentions_small["date"][
                                     np.logical_and(date_of_mention + pd.Timedelta(days=14) < df_stockMentions_small["date"],
                                                    df_stockMentions_small["date"] < date_of_mention + pd.Timedelta(days=28))])
        mentions_4t6weeks = len(df_stockMentions_small["date"][
                                      np.logical_and(
                                          date_of_mention + pd.Timedelta(days=28) < df_stockMentions_small["date"],
                                          df_stockMentions_small["date"] < date_of_mention + pd.Timedelta(days=42))])
        return count_mentions_before, days_since_first, followers, diff_mentions_before, mentions_0t2weeks, mentions_2t4weeks, mentions_4t6weeks, mentions_last_week

    def _get_price_parameters_and_labels(self, df_stockPrices_small, stock, date_of_mention, data_set):
        # Add time to date to ensure that the mention is related to the correct opening price
        if data_set == "since_last":
            date_of_mention = date_of_mention + pd.Timedelta(hours=2, minutes=30)
        else:
            date_of_mention = date_of_mention + pd.Timedelta(hours=8, minutes=25)
        # See if the necessary stock price data is there
        try:
            # Find the difference between the date_of_mention and all days with stock prices
            day_diff = df_stockPrices_small.dropna().index - pd.Timestamp(date_of_mention.date())
            # Find the closing days, even if weekend
            closing_day = df_stockPrices_small.dropna()[day_diff > - pd.Timedelta(days=1)].index.values.min()
            # Check that the minimum distance to a day with prices is 1 day
            days_to_closest = abs(df_stockPrices_small.dropna().index - pd.Timestamp(date_of_mention.date())).min()
            if days_to_closest > pd.Timedelta(days=2):
                raise ValueError(f"{stock} price for today is not available. Closest day {days_to_closest}")
            price_today = df_stockPrices_small.loc[closing_day]
            # Compute the percentage of the stock price where price today is 100 %
            percentage_price = (df_stockPrices_small / price_today) * 100
            # Percentage price before closing day and after closing day
            perc_before_close = percentage_price.dropna().loc[percentage_price.dropna().index < closing_day]
            perc_after_close = percentage_price.loc[percentage_price.index > closing_day]

            # Compute avg price change last 3 days where the stocks were open
            if len(perc_before_close) <= 7:
                price_avg_last_3_days = 100  # 100 percent
                price_avg_last_7_days = 100  # 100 percent
                price_avg_last_30_days = 100  # 100 percent
            else:
                # Compute avg price change last days where the stocks were open
                price_avg_last_3_days = np.mean(perc_before_close.values[-3:])
                price_avg_last_7_days = np.mean(perc_before_close.values[-7:])
                price_avg_last_30_days = np.mean(perc_before_close.values[-30:])
            if data_set == "since_last":
                return (price_today, price_avg_last_3_days, price_avg_last_7_days,
                        date_of_mention - pd.Timedelta(hours=2, minutes=30), np.asarray([0]), price_avg_last_30_days)
            if perc_after_close.isnull().all():
                raise ValueError("True label became NAN")
            # Compute true label. True label is defined as the maximum stock increase within next 31 days
            # converted to a three class problem of don't buy, buy, and buy many.
            max_increase = np.nanmax(perc_after_close.values[:45])
            if price_today < 10 and max_increase > 165:
                true_label = 1
            elif 10 < price_today < 15 and max_increase > 165:
                true_label = 1
            #elif 20 < price_today < 40 and max_increase > 165:
            #    true_label = 0
            #elif 40 < price_today and max_increase > 165:
            #    true_label = 1
            elif np.isnan(max_increase) or price_today > 15:
                raise ValueError("True label became NAN")
            else:
                true_label = 0
            return price_today, price_avg_last_3_days, price_avg_last_7_days, true_label, perc_after_close.values[:46], price_avg_last_30_days
        except (IndexError, ValueError) as e:
            #print(f"Error message: {e}")
            #print(f"If Error message started with zero-value, then {stock} was mentioned before available stock prices: {np.min(day_diff)}")
            #print(f"Error message was True label became NAN, then it was because no stock data existed after mention")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    def _merge_parameters_and_labels(self, df_stockMentions_small, df_stockPrices_small, stock, data_set, last_check):
        parameters_stock, label_stock, perc_money, stock_list, mention_fut = list(), list(), list(), list(), list()
        # For each mention in stock mentions compute parameters
        for i, date_of_mention in enumerate(df_stockMentions_small["date"].sort_values()):
            # Get number of mentions last two months
            mentions_last_two_months = np.count_nonzero(np.logical_and(date_of_mention > df_stockMentions_small["date"],
                                            date_of_mention - pd.Timedelta(days=61) < df_stockMentions_small["date"]))

            if mentions_last_two_months == 0 or data_set == "since_last" and date_of_mention < last_check:
                continue
            elif i > 14:
                break

            price_today, price_avg_last_3_days, price_avg_last_7_days, true_label, perc_after_close, price_avg_last_30_days = self._get_price_parameters_and_labels(
                df_stockPrices_small, stock, date_of_mention, data_set)

            cmb, days_since_first, followers, dmb, mentions_0t2weeks, mentions_2t4weeks, mentions_4t6weeks, mlw = self._get_mention_parameters(
                df_stockMentions_small, date_of_mention)

            # Save params in array
            params = np.asarray([price_today, price_avg_last_3_days, price_avg_last_7_days,
                                 days_since_first, followers, price_avg_last_30_days, cmb, dmb, mlw])

            # Check for NANs and discard values if Nans
            if np.isnan(params).any():
                continue

            # Save percentage money output in array
            while len(perc_after_close) != 61:
                perc_after_close = np.concatenate((perc_after_close, np.expand_dims(np.asarray(np.nan), axis=0)))
            percentage_money = np.asarray(perc_after_close)

            # Append parameters and true labels
            parameters_stock.append(params)
            label_stock.append(true_label)
            perc_money.append(percentage_money)
            stock_list.append([stock])
            mention_fut.append([mentions_0t2weeks, mentions_2t4weeks, mentions_4t6weeks])

        return np.asarray(parameters_stock), np.asarray(label_stock), np.asarray(perc_money), stock_list, mention_fut

    def get_parameters_and_labels(self, data_set, end_date, last_check=pd.Timestamp(2020,1,1)):
        if data_set == "train":
            stock_names = self.train_stocks
        elif data_set == "test":
            stock_names = self.test_stocks
        elif data_set == "since_last":
            stock_names = self.new_stocks
        else:
            raise Exception(f"data_set should be test, train or since_last, not {data_set}")
        # Shuffle list
        random.Random(2).shuffle(stock_names)
        # For each stock
        for stock in stock_names:
            # If stock is not present, skip it
            if stock not in self.df_stockMentions["stock"].values:
                print(f"Skipped {stock} not in df_stockMentions")
                continue
            # Isolate all mentions of the stock and the stock prices
            try:
                if data_set == "train" or data_set == "test":
                    bool_mask = np.logical_and(self.df_stockMentions["stock"].values == stock,
                                               self.df_stockMentions["date"] < end_date)
                else:
                    bool_mask = self.df_stockMentions["stock"].values == stock
                df_stockMentions_small = self.df_stockMentions[bool_mask]
                df_stockPrices_small = self.df_stockPrices.loc[:, stock]
            except KeyError as e:
                print(f"Keyerror {e} not in df_stockMentions or df_stockPrices")
                continue

            # Get parameters and labels for the stock
            parameters_stock, label_stock, percentage_money, stock_list, mention_fut = self._merge_parameters_and_labels(df_stockMentions_small,
                                                                                                df_stockPrices_small,
                                                                                                stock, data_set, last_check)

            # Check that there was an actual output from the stock and expand dim if the output was 1D
            if len(label_stock) == 0:
                continue
            elif parameters_stock.ndim == 1:
                parameters_stock = np.expand_dims(parameters_stock, 0)
                percentage_money = np.expand_dims(percentage_money, 0)

            # Save length for later splitting of test and training
            self.amount_of_stock_input.append(len(label_stock))

            # Append parameters and labels
            if data_set == "train":
                self.train_set_params = np.concatenate((self.train_set_params, parameters_stock), axis=0)
                self.train_set_labels = np.concatenate((self.train_set_labels, label_stock))
                self.train_set_money = np.concatenate((self.train_set_money, percentage_money), axis=0).astype(np.float64)
                self.stock_list_train += stock_list
                self.mention_futures_train += mention_fut
            elif data_set == "test":
                self.test_set_params = np.concatenate((self.test_set_params, parameters_stock), axis=0)
                self.test_set_labels = np.concatenate((self.test_set_labels, label_stock))
                self.test_set_money = np.concatenate((self.test_set_money, percentage_money), axis=0).astype(np.float64)
                self.stock_list_test += stock_list
                self.mention_futures_test += mention_fut
            else:
                self.since_last_set_params = np.concatenate((self.since_last_set_params, parameters_stock), axis=0)
                self.stock_list_since_last += stock_list
                self.mention_dates_since_last = np.concatenate((self.mention_dates_since_last, label_stock))

        # For making the list for predefined splits to keep stocks in separated groups when doing CV
        splits_idxs = np.array_split(np.arange(len(self.amount_of_stock_input)), 5)
        for i, idxs in enumerate(splits_idxs):
            self.stock_kfold_idxs += np.asarray(self.amount_of_stock_input)[idxs].sum() * [i]


class MLModelForest(MLModel):
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        super().__init__(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)
        self.forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_depth=4, class_weight="balanced", bootstrap=False)

    def cv_score(self, leaf, lay, bootstrap):
        balance = {0: 1, 1: 1.4}
        self.forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, max_depth=lay, class_weight=balance, bootstrap=bootstrap)

        cv_score = cross_validate(self.forest, self.train_set_params, self.train_set_labels,
                                  cv=PredefinedSplit(self.stock_kfold_idxs), return_train_score=True)
        train_score, test_score = cv_score["train_score"], cv_score["test_score"]
        print(f"The training score was {np.mean(train_score)} and the validation score {np.mean(test_score)}")

    def cv_predict_similarity(self):
        self.cv_predictions = cross_val_predict(self.forest, self.train_set_params, self.train_set_labels)
        print(f"The predictions contained {len(np.nonzero(self.cv_predictions)[0])} buy commands and the actual buy commands"
              f" were {len(np.nonzero(self.train_set_labels)[0])}")
        print("---------------")
        perc_2_both = (len(np.unique(np.asarray(self.stock_list_train)[np.logical_and(self.cv_predictions == 1, self.train_set_labels == 1)])) /
         len(np.unique(np.asarray(self.stock_list_train)[self.train_set_labels == 1]))) * 100
        print(f"The predictions contained buy commands on {perc_2_both} percentage of the places where "
              f"the growth exceeded 50 % ({len(np.unique(np.asarray(self.stock_list_train)[self.train_set_labels == 1]))})")

    def money_out_train(self, plot_it, thrs):
        # If we bought all stocks the algorithm recommended
        mask_all = (self.cv_predictions != 0)
        money_mask = self.train_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list_train)[mask_all]
        price_today = np.asarray(self.train_set_params[:, 0])[mask_all]
        sales_value = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            sales_value.append(self._get_sales_value(money_mask[idx], price_today[idx], stock, plot_it, thrs))

        # Compute the earnings by remembering to multiply them over time (renters rente)
        average_time_to_sell = np.mean(np.asarray(sales_value)[:, 5])
        rounds_per_six_month = int(np.floor(180 / average_time_to_sell))
        remove_it = len(np.unique(stock_mask)) % rounds_per_six_month
        if remove_it == 0:
            first_part = np.mean(np.reshape(np.asarray(sales_value)[:, 4], (rounds_per_six_month, -1)), axis=1)
        else:
            first_part = np.mean(np.reshape(np.asarray(sales_value)[:-remove_it, 4], (rounds_per_six_month, -1)), axis=1)
        final_earn = np.prod(first_part / 100)

        print(f"If we buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(sales_value, axis=0)}, prod {final_earn}")

    def fit(self):
        self.forest.fit(self.train_set_params, self.train_set_labels)

    def get_test_results(self, plot_it):
        print(f"The test score was {self.forest.score(self.test_set_params, self.test_set_labels)}")
        predictions = self.forest.predict(self.test_set_params)

        # Find the percentage of correctly bought stocks
        perc_2_both = (len(np.unique(
            np.asarray(self.stock_list_test)[np.logical_and(predictions == 1, self.test_set_labels == 1)])) /
                       len(np.unique(np.asarray(self.stock_list_test)[self.test_set_labels == 1]))) * 100
        print(f"The predictions contained buy commands on {perc_2_both} percentage of the places where "
              f"the growth exceeded 50 % ({len(np.unique(np.asarray(self.stock_list_test)[self.test_set_labels == 1]))})")

        # See how much money comes out
        self._money_out_test(predictions, plot_it)

    def _money_out_test(self, predictions, plot_it):
        # If we bought all stocks the algorithm recommended
        mask_all = (predictions != 0)
        money_mask = self.test_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list_test)[mask_all]
        price_today = np.asarray(self.test_set_params[:, 0])[mask_all]
        sales_value = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            sales_value.append(self._get_sales_value(money_mask[idx], price_today[idx], stock, plot_it))

        # Compute the earnings by remembering to multiply
        average_time_to_sell = np.mean(np.asarray(sales_value)[:, 5])
        rounds_per_six_month = int(np.floor(180 / average_time_to_sell))
        remove_it = len(np.unique(stock_mask)) % rounds_per_six_month
        try:
            if remove_it == 0:
                first_part = np.mean(np.reshape(np.asarray(sales_value)[:, 4], (rounds_per_six_month, -1)), axis=1)
            else:
                first_part = np.mean(np.reshape(np.asarray(sales_value)[:-remove_it, 4], (rounds_per_six_month, -1)),
                                     axis=1)
            # second_part = np.mean(sales_value[-remove_it:, 4])
            final_earn = np.prod(first_part / 100)
        except FloatingPointError as e:
            final_earn = e

        print(f"If we buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(sales_value, axis=0)}, prod {final_earn}")

    def _get_sales_value(self, nan_val, price_today, stock, plot_it=False, thrs=1000):
        # Remove nan values
        non_nan_val = nan_val[~np.isnan(nan_val)]

        # Define thresholds
        sales_incr = thrs
        # Set parameters
        low = 50  # How low the stock can drop before sale
        max_keep = 45  # The latest sell (remember to account for weekends) (i.e. x weekdays after buy)
        nan_max_keep = int(max_keep - max_keep // 3.5)

        # Compute moving average window of 9
        n = 9
        mov_avg = np.convolve(np.concatenate(([100] * 8, non_nan_val)), np.ones(n) / n, mode='valid')

        # Compute exponential moving avg of size 20
        n = 20
        multiplier = 2 / (n + 1)
        ema = [(non_nan_val[0] - 100) * multiplier + 100]
        for i in range(len(non_nan_val) - 1):
            ema.append((non_nan_val[i + 1] - ema[i]) * multiplier + ema[i])

        if plot_it:
            plt.ion()
            plt.figure()
            plt.plot(non_nan_val, label="Raw")
            plt.plot(mov_avg, label="Mov_avg")
            plt.plot(ema, label="EMA")
            plt.title(stock)
            plt.legend()
            plt.draw()

        # Compute cumulative value 4 values back
        cumul_val = np.cumsum(np.diff(np.concatenate((np.asarray([100]), non_nan_val))))
        cumul_val[4:] = cumul_val[4:] - cumul_val[:-4]
        cumul_val[:4] = cumul_val[:4]

        # If there is a place where the 4 day cumulative value goes above sales_incr then set that as idx
        idx_cum = np.nonzero(cumul_val > sales_incr)[0]
        if len(idx_cum) != 0:
            idx_cum = idx_cum[0]
            idx_cum = np.min(np.nonzero(nan_val == non_nan_val[idx_cum])[0])
        else:
            idx_cums = np.nonzero(non_nan_val[len(non_nan_val[:nan_max_keep]) - 1] == nan_val)[0]
            idx_cum = idx_cums[np.argmin(abs(idx_cums - max_keep))]

        # If there is a place where the value goes below low then set that as idx
        idx_low = np.nonzero(non_nan_val < low)[0]
        if len(idx_low) != 0:
            idx_low = idx_low[0]
            idx_low = np.min(np.nonzero(nan_val == non_nan_val[idx_low])[0])
        else:
            idx_lows = np.nonzero(non_nan_val[len(non_nan_val[:nan_max_keep]) - 1] == nan_val)[0]
            idx_low = idx_lows[np.argmin(abs(idx_lows - max_keep))]

        # Set the indices by choosing the first coming of the two indices
        idx = np.min([idx_cum, idx_low])
        our_sell1 = idx.item()

        top_val1 = np.nanargmax(nan_val).item()
        last_val1 = np.nonzero(non_nan_val[len(non_nan_val) - 1] == nan_val)[0][-1]

        #
        diff = np.nonzero(np.diff(np.sign(mov_avg - ema)) < 0)[0]
        if len(diff) > 0:
            diff = diff[0]
        else:
            diff = len(non_nan_val[:nan_max_keep]) - 1
        if plot_it:
            plt.vlines(top_val1, 99, 110, color="g")
            plt.vlines(last_val1, 98, 110, color="k")
            plt.vlines(our_sell1, 97, 110, color="r")
            plt.vlines(diff, 96, 100, color="orange")

        return nan_val[top_val1], top_val1, nan_val[last_val1], last_val1, nan_val[our_sell1], our_sell1

    def update_database(self, most_common_words, last_check):
        # Update data base
        #functions.construct_data_base(most_common_words, start_date=pd.Timestamp(2020, 7, 1),
        #                              end_date=pd.Timestamp(2021, 1, 1), last_check=last_check)

        con = sqlite3.connect("get_data/aktiespekulanterne/data/youtube_stocks.db")
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
        # Keep only the stocks mentioned during the last 6 months
        df_stockMentions = df_stockMentions[df_stockMentions["date"] > pd.Timestamp.today() - pd.Timedelta(weeks=26)]

        self.df_stockMentions = df_stockMentions
        self.df_stockPrices = df_stockPrices
        self.df_youtubeSources = df_youtubeSources
        self.new_stocks = df_trainTest["index"][df_trainTest["data_set"] == "since_last"].values

    def get_new_buys(self, last_check):
        # Remove previous computed parameters and stocks
        self.since_last_set_params = np.empty((0, 9))
        self.stock_list_since_last = list()

        # Compute input parameters for newest mentions
        self.get_parameters_and_labels(data_set="since_last", end_date=None, last_check=last_check)

        if len(self.stock_list_since_last) == 0:
            print("----------------------------------------")
            print("No stocks were considered since last run")
            print("----------------------------------------")
        else:
            # Use model to predict
            predictions = self.forest.predict(self.since_last_set_params).astype(np.int64)

            # Make .csv for predictions and parameters
            df_csv = pd.DataFrame(self.since_last_set_params, columns=self.feature_names)
            df_csv["ticker"] = self.stock_list_since_last
            sources = list()
            mention_num = list()
            idxs = np.unique(self.stock_list_since_last, return_index=True)[1]
            stock_list = [self.stock_list_since_last[index] for index in sorted(idxs)]
            sto, cou = np.unique(self.stock_list_since_last, return_counts=True)
            for stock in stock_list:
                # Get number of mentions of stock in self.stock_list_since_last
                sto_cou = cou[sto == stock]
                # Extract stock mentions and sort according to date
                df_helper = self.df_stockMentions[self.df_stockMentions["stock"] == stock[0]]
                df_helper = df_helper.sort_values("date", ascending=True)
                # Get whole len
                whole_len = len(df_helper)
                # Get the source name(s)
                men_num = 0
                for i in range(len(df_helper)):
                    mentions_last_two_months = np.count_nonzero(
                        np.logical_and(df_helper.iloc[i]["date"] > df_helper["date"],
                                       df_helper.iloc[i]["date"] - pd.Timedelta(days=61) < df_helper["date"]))
                    if mentions_last_two_months == 0 or df_helper.iloc[i]["date"] < last_check or i > 14:
                        continue
                    men_num += 1
                    if men_num > sto_cou:
                        continue
                    sources.append(self.df_youtubeSources["name"][self.df_youtubeSources["id"] == df_helper.iloc[i]["source"]].item())
                    mention_num.append(whole_len - (len(df_helper.values) - i - 1))
            df_csv["source"] = sources
            df_csv["num_of_mention"] = mention_num
            df_csv["date_of_mention"] = self.mention_dates_since_last
            df_csv["date_of_prediction"] = [pd.Timestamp.today().date()] * len(predictions)
            df_csv["time_of_prediction"] = [pd.Timestamp.today()] * len(predictions)
            df_csv["predictions"] = predictions
            if os.path.isfile("algorithm_predictions.csv"):
                df = pd.read_csv("algorithm_predictions.csv").drop(columns="Unnamed: 0")
                df_csv = df.append(df_csv)
            df_csv = df_csv.sort_values("date_of_mention", ascending=True)
            df_csv.to_csv("algorithm_predictions.csv")

            # Print results
            print("----------------------------------------")
            print(f"Considering: {self.stock_list_since_last}")
            if len(np.asarray(self.stock_list_since_last)[np.nonzero(predictions)]) == 0:
                print("Nothing here of any value.")
            else:
                print(f"Buy it! BUY IT, GODDAMMIT!!!: {np.asarray(self.stock_list_since_last)[np.nonzero(predictions)]}")
            print("----------------------------------------")
