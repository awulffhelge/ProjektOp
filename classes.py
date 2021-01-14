""" Classes
Comments and to-do's: Change stock from absolute to percentage """


import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
np.seterr("raise")


class MLModel:
    """ Ideas for features:
     """
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        self.df_stockMentions = df_stockMentions
        self.df_stockPrices = df_stockPrices
        self.df_youtubeSources = df_youtubeSources

        self.train_set_params = np.empty((0, 8))
        self.test_set_params = np.empty((0, 8))
        self.train_set_labels = np.empty(0)
        self.test_set_labels = np.empty(0)
        self.train_set_money = np.empty((0, 2))
        self.test_set_money = np.empty((0, 2))

        self.stock_names = df_stockMentions["stock"].value_counts().index.values

        self.train_stocks = df_trainTest["index"][df_trainTest["data_set"] == "train"].values
        self.test_stocks = df_trainTest["index"][df_trainTest["data_set"] == "test"].values

        self.model = SVC()

        self.amount_of_stock_input = list()

        self.stock_list = list()

    def _get_mention_parameters(self, df_stockMentions_small, date_of_mention):
        # Count mentions of stock before date_of_mention
        count_mentions_before = len(df_stockMentions_small["date"][df_stockMentions_small["date"] < date_of_mention])
        # Days since first mention
        days_since_first = (date_of_mention - df_stockMentions_small["date"].min()).days
        # Mentions the last week
        mention_diff = (date_of_mention - df_stockMentions_small["date"]).apply(lambda x: x.days).values
        mentions_last_week = len(mention_diff[np.logical_and(0 < mention_diff, mention_diff < 7)])
        # Sum of followers on mentioning channels
        mentions_today = np.unique(df_stockMentions_small["source"].iloc[np.nonzero(mention_diff == 0)]) - 1
        followers = self.df_youtubeSources.loc[mentions_today, "subscribers"].sum()
        return count_mentions_before, days_since_first, mentions_last_week, len(mentions_today), followers

    def _get_price_parameters_and_labels(self, df_stockPrices_small, stock, date_of_mention):
        # See if the necessary stock price data is there
        try:
            # Find the difference between the date_of_mention and all days with stock prices
            day_diff = df_stockPrices_small.dropna().index - pd.Timestamp(date_of_mention.date())
            # Find the closing days, even if weekend
            closing_day = df_stockPrices_small.dropna()[day_diff < pd.Timedelta(days=1)].index.values.max()
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
            if len(perc_before_close) == 0:
                price_avg_last_3_days = 100  # 100 percent
                price_avg_last_7_days = 100  # 100 percent
            else:
                price_avg_last_3_days = np.mean(perc_before_close.values[-3:])
                # Compute avg price change last 7 days where the stocks were open
                price_avg_last_7_days = np.mean(perc_before_close.values[-7:])
            # Compute true label. True label is defined as the maximum stock increase within next 31 days
            # converted to a three class problem of don't buy, buy, and buy many.
            max_increase = np.nanmax(perc_after_close.values[:31])
            max_decrease = np.nanmin(perc_after_close.values[:31])
            if max_increase > 150:
                true_label = 2
            elif max_increase > 120:
                true_label = 1
            elif np.isnan(max_increase):
                raise ValueError("True label became NAN")
            else:
                true_label = 0
            return price_today, price_avg_last_3_days, price_avg_last_7_days, true_label, max_increase, max_decrease
        except (IndexError, ValueError) as e:
            #print(f"Error message: {e}")
            #print(f"If Error message started with zero-value, then {stock} was mentioned before available stock prices: {np.min(day_diff)}")
            #print(f"Error message was True label became NAN, then it was because no stock data existed after mention")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    def _merge_parameters_and_labels(self, df_stockMentions_small, df_stockPrices_small, stock):
        parameters_stock, label_stock, perc_money, stock_list = list(), list(), list(), list()
        # For each mention in stock mentions compute parameters
        for i, date_of_mention in enumerate(df_stockMentions_small["date"].sort_values()):

            price_today, price_avg_last_3_days, price_avg_last_7_days, true_label, max_incr, max_decr = self._get_price_parameters_and_labels(
                df_stockPrices_small, stock, date_of_mention)

            count_mentions_before, days_since_first, mentions_last_week, mentions_today, followers = self._get_mention_parameters(
                df_stockMentions_small, date_of_mention)

            # Save params in array
            params = np.asarray([price_today, price_avg_last_3_days, price_avg_last_7_days, count_mentions_before,
                                 days_since_first, mentions_last_week, mentions_today, followers])
            # Save percentage money output in array
            percentage_money = np.asarray([max_incr, max_decr])

            # Check for NANs and discard values if Nans
            if np.isnan(params).any():
                continue

            # Append parameters and true labels
            parameters_stock.append(params)
            label_stock.append(true_label)
            perc_money.append(percentage_money)
            stock_list.append([stock])

            if i == 9:
                break
        return np.asarray(parameters_stock), np.asarray(label_stock), np.asarray(perc_money), stock_list

    def _z_score_norm(self, data_set):
        if data_set == "train":
            self.train_set_params = (self.train_set_params - np.mean(self.train_set_params, axis=0)) / np.std(
                self.train_set_params, axis=0)
        elif data_set == "test":
            self.test_set_params = (self.test_set_params - np.mean(self.test_set_params, axis=0)) / np.std(
                self.test_set_params, axis=0)

    def get_parameters_and_labels(self, data_set):
        if data_set == "train":
            stock_names = self.train_stocks
        elif data_set == "test":
            stock_names = self.test_stocks
        else:
            raise Exception(f"set should be test or train, not {data_set}")
        # For each stock
        for stock in stock_names:
            # If stock is not present, skip it
            if stock not in self.df_stockMentions["stock"].values:
                print(f"Skipped {stock} not in df_stockMentions")
                continue
            # Isolate all mentions of the stock and the stock prices
            try:
                df_stockMentions_small = self.df_stockMentions[self.df_stockMentions["stock"].values == stock]
                df_stockPrices_small = self.df_stockPrices.loc[:, stock]
            except KeyError as e:
                print(e)
                continue

            # Get parameters and labels for the stock
            parameters_stock, label_stock, percentage_money, stock_list = self._merge_parameters_and_labels(df_stockMentions_small,
                                                                                                df_stockPrices_small,
                                                                                                stock)

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
                self.train_set_money = np.concatenate((self.train_set_money, percentage_money), axis=0)
                self.stock_list += stock_list
            else:
                self.test_set_params = np.concatenate((self.test_set_params, parameters_stock), axis=0)
                self.test_set_labels = np.concatenate((self.test_set_labels, label_stock))
                self.test_set_money = np.concatenate((self.test_set_money, percentage_money), axis=0)

        # Apply z_score normalization
        self._z_score_norm(data_set)


class MLModelV1(MLModel):
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        super().__init__(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)

        self.svm = SVC(class_weight="balanced")

    def cv_score(self):
        print(cross_val_score(SVC(class_weight="balanced"), self.train_set_params, self.train_set_labels))

    def cv_predict_similarity(self):
        self.cv_predictions = cross_val_predict(SVC(class_weight="balanced"), self.train_set_params, self.train_set_labels)
        print(f"The predictions contained {len(np.nonzero(self.cv_predictions)[0])} buy commands and the actual buy commands"
              f" were {len(np.nonzero(self.train_set_labels)[0])}")
        print("---------------")
        perc_2 = len(np.nonzero(np.logical_and(self.cv_predictions != 0, self.train_set_labels == 2))[0]) / len(
            np.nonzero(self.train_set_labels == 2)[0])
        print(f"The predictions contained contained buy commands on {perc_2} percentage of the places where the growth "
              f"exceeded 50 %")
        print("---------------")
        perc_2_both = len(np.nonzero(np.logical_and(self.cv_predictions == 2, self.train_set_labels == 2))[0]) / len(
            np.nonzero(self.train_set_labels == 2)[0])
        print(f"The predictions contained contained high buy commands on {perc_2_both} percentage of the places where "
              f"the growth exceeded 50 %")

    def money_out(self):
        # If we only buy the ones where there are a big buy incentive
        mask_2 = (self.cv_predictions == 2)
        stock_mask = np.asarray(self.stock_list)[mask_2]
        money_mask = self.train_set_money[mask_2, :]
        earnings_2 = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            earnings_2.append(money_mask[idx])
        print(f"If we only buy the stock with a high buy incentive. Then the earnings/loss over a month would be: {np.mean(earnings_2, axis=0)}")
        # If we bought all stocks the algorithm recommended
        mask_all = (self.cv_predictions != 0)
        money_mask = self.train_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list)[mask_all]
        earnings_all = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            earnings_all.append(money_mask[idx])
        print(f"If buy all the stocks recommended by the algorithm. Then the earnings/loss over a month would be: {np.mean(earnings_all, axis=0)}")
