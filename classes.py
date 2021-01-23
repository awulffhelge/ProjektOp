""" Classes
Comments and to-do's: How many different youtuberes mentioned the stock """


import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score, cross_val_predict, PredefinedSplit, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
np.seterr("raise")


class MLModel:
    """ Ideas for features:
     """
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        self.df_stockMentions = df_stockMentions
        self.df_stockPrices = df_stockPrices
        self.df_youtubeSources = df_youtubeSources

        self.train_set_params = np.empty((0, 7))
        self.test_set_params = np.empty((0, 7))
        self.train_set_labels = np.empty(0)
        self.test_set_labels = np.empty(0)
        self.train_set_money = np.empty((0, 61))
        self.test_set_money = np.empty((0, 61))

        self.feature_names = ["price_today", "price_avg_last_3_days", "price_avg_last_7_days", "count_mentions_before",
                              "days_since_first", "followers", "diff_mentions_before"]

        self.stock_names = df_stockMentions["stock"].value_counts().index.values

        self.train_stocks = df_trainTest["index"][df_trainTest["data_set"] == "train"].values
        self.test_stocks = df_trainTest["index"][df_trainTest["data_set"] == "test"].values

        self.amount_of_stock_input = list()
        self.stock_kfold_idxs = list()  # Pre determined folds to ensure that stocks are split between train and val

        self.stock_list_train = list()
        self.stock_list_test = list()
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
        return count_mentions_before, days_since_first, followers, diff_mentions_before, mentions_0t2weeks, mentions_2t4weeks, mentions_4t6weeks

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
            max_increase = np.nanmax(perc_after_close.values[:46])
            max_decrease = np.nanmin(perc_after_close.values[:31])
            if max_increase > 165:
                true_label = 1
            elif np.isnan(max_increase):
                raise ValueError("True label became NAN")
            else:
                true_label = 0
            return price_today, price_avg_last_3_days, price_avg_last_7_days, true_label, perc_after_close.values[:46]
        except (IndexError, ValueError) as e:
            #print(f"Error message: {e}")
            #print(f"If Error message started with zero-value, then {stock} was mentioned before available stock prices: {np.min(day_diff)}")
            #print(f"Error message was True label became NAN, then it was because no stock data existed after mention")
            return np.nan, np.nan, np.nan, np.nan, np.nan

    def _merge_parameters_and_labels(self, df_stockMentions_small, df_stockPrices_small, stock):
        parameters_stock, label_stock, perc_money, stock_list, mention_fut = list(), list(), list(), list(), list()
        # For each mention in stock mentions compute parameters
        for i, date_of_mention in enumerate(df_stockMentions_small["date"].sort_values()):

            if i < 2:
                continue

            price_today, price_avg_last_3_days, price_avg_last_7_days, true_label, perc_after_close = self._get_price_parameters_and_labels(
                df_stockPrices_small, stock, date_of_mention)

            count_mentions_before, days_since_first, followers, diff_mentions_before, mentions_0t2weeks, mentions_2t4weeks, mentions_4t6weeks = self._get_mention_parameters(
                df_stockMentions_small, date_of_mention)

            # Save params in array
            params = np.asarray([price_today, price_avg_last_3_days, price_avg_last_7_days, count_mentions_before,
                                 days_since_first, followers, diff_mentions_before])

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

            if i == 6:
                break
        return np.asarray(parameters_stock), np.asarray(label_stock), np.asarray(perc_money), stock_list, mention_fut

    def get_parameters_and_labels(self, data_set):
        if data_set == "train":
            stock_names = self.train_stocks
        elif data_set == "test":
            stock_names = self.test_stocks
        else:
            raise Exception(f"set should be test or train, not {data_set}")
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
                df_stockMentions_small = self.df_stockMentions[self.df_stockMentions["stock"].values == stock]
                df_stockPrices_small = self.df_stockPrices.loc[:, stock]
            except KeyError as e:
                print(e)
                continue

            # Get parameters and labels for the stock
            parameters_stock, label_stock, percentage_money, stock_list, mention_fut = self._merge_parameters_and_labels(df_stockMentions_small,
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
                self.stock_list_train += stock_list
                self.mention_futures_train += mention_fut
            else:
                self.test_set_params = np.concatenate((self.test_set_params, parameters_stock), axis=0)
                self.test_set_labels = np.concatenate((self.test_set_labels, label_stock))
                self.test_set_money = np.concatenate((self.test_set_money, percentage_money), axis=0)
                self.stock_list_test += stock_list
                self.mention_futures_test += mention_fut

        # For making the list for predefined splits to keep stocks in separated groups when doing CV
        splits_idxs = np.array_split(np.arange(len(self.amount_of_stock_input)), 5)
        for i, idxs in enumerate(splits_idxs):
            self.stock_kfold_idxs += np.asarray(self.amount_of_stock_input)[idxs].sum() * [i]


class MLModelV1(MLModel):
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        super().__init__(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)

        self.svm = SVC(class_weight="balanced")

    def z_score_norm(self, data_set):
        if data_set == "train":
            self.train_set_params = (self.train_set_params - np.mean(self.train_set_params, axis=0)) / np.std(
                self.train_set_params, axis=0)
        elif data_set == "test":
            self.test_set_params = (self.test_set_params - np.mean(self.test_set_params, axis=0)) / np.std(
                self.test_set_params, axis=0)
        else:
            raise Exception(f"set should be test or train, not {data_set}")

    def cv_score(self):
        cv_score = cross_validate(self.svm, self.train_set_params, self.train_set_labels,
                                  cv=PredefinedSplit(self.stock_kfold_idxs), return_train_score=True)
        train_score, test_score = cv_score["train_score"], cv_score["test_score"]
        print(f"The training score was {train_score} and the validation score {test_score}")

    def cv_predict_similarity(self):
        self.cv_predictions = cross_val_predict(self.svm, self.train_set_params, self.train_set_labels)
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
        print(f"If we only buy the stocks with a high buy incentive ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(earnings_2, axis=0)}")
        # If we bought all stocks the algorithm recommended
        mask_all = (self.cv_predictions != 0)
        money_mask = self.train_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list)[mask_all]
        earnings_all = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            earnings_all.append(money_mask[idx])
        print(f"If buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(earnings_all, axis=0)}")


class MLModelTree(MLModel):
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        super().__init__(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)
        self.class_balance = {0: 0.2, 1: 0.8}
        self.tree = DecisionTreeClassifier(min_samples_leaf=9, max_depth=4, class_weight=self.class_balance)

    def cv_score(self, leaf, lay):
        self.tree = DecisionTreeClassifier(min_samples_leaf=leaf, max_depth=lay, class_weight=self.class_balance)

        cv_score = cross_validate(self.tree, self.train_set_params, self.train_set_labels,
                                  cv=PredefinedSplit(self.stock_kfold_idxs), return_train_score=True)
        train_score, test_score = cv_score["train_score"], cv_score["test_score"]
        print(f"The training score was {np.mean(train_score)} and the validation score {np.mean(test_score)}")

    def cv_predict_similarity(self):
        self.cv_predictions = cross_val_predict(self.tree, self.train_set_params, self.train_set_labels)
        print(f"The predictions contained {len(np.nonzero(self.cv_predictions)[0])} buy commands and the actual buy commands"
              f" were {len(np.nonzero(self.train_set_labels)[0])}")
        print("---------------")
        perc_2_both = (len(np.unique(np.asarray(self.stock_list_train)[np.logical_and(self.cv_predictions == 1, self.train_set_labels == 1)])) /
         len(np.unique(np.asarray(self.stock_list_train)[self.train_set_labels == 1])))
        print(f"The predictions contained buy commands on {perc_2_both} percentage of the places where "
              f"the growth exceeded 50 %")

    def money_out_train(self, plot_it):
        # If we bought all stocks the algorithm recommended
        mask_all = (self.cv_predictions != 0)
        money_mask = self.train_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list_train)[mask_all]
        mentions_mask = np.asarray(self.mention_futures_train)[mask_all]
        sales_value = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            non_nan_money = money_mask[idx][~np.isnan(money_mask[idx])]
            sales_value.append(self._get_sales_value(non_nan_money, mentions_mask[idx], stock, plot_it))
        print(f"If we buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(sales_value, axis=0)}")

    def fit(self):
        self.tree.fit(self.train_set_params, self.train_set_labels)

    def get_test_results(self, plot_it):
        print(f"The test score was {self.tree.score(self.test_set_params, self.test_set_labels)}")
        predictions = self.tree.predict(self.test_set_params)

        # Find the percentage of correctly bought stocks
        perc_2_both = (len(np.unique(
            np.asarray(self.stock_list_test)[np.logical_and(predictions == 1, self.test_set_labels == 1)])) /
                       len(np.unique(np.asarray(self.stock_list_test)[self.test_set_labels == 1])))
        print(f"The predictions contained buy commands on {perc_2_both} percentage of the places where "
              f"the growth exceeded 50 %")

        # See how much money comes out
        self._money_out_test(predictions, plot_it)

    def _money_out_test(self, predictions, plot_it):
        # If we bought all stocks the algorithm recommended
        mask_all = (predictions != 0)
        money_mask = self.test_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list_test)[mask_all]
        mentions_mask = np.asarray(self.mention_futures_test)[mask_all]
        sales_value = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            non_nan_money = money_mask[idx][~np.isnan(money_mask[idx])]
            sales_value.append(self._get_sales_value(non_nan_money, mentions_mask[idx], stock, plot_it))
        print(f"If we buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(sales_value, axis=0)}")

    def _get_sales_value_mentions(self, val, mentions_fut, stock, plot_it=False):
        sales_incr = 65
        if plot_it:
            plt.ion()
            plt.figure()
            plt.plot(val)
            plt.title(stock + str(mentions_fut))
            plt.draw()
        # Compute cumulative value 4 values back
        cumul_val = np.cumsum(np.diff(np.concatenate((np.asarray([100]), val))))
        cumul_val[4:] = cumul_val[4:] - cumul_val[:-4]
        cumul_val[:4] = cumul_val[:4]
        if len(val) < 10:
            if (cumul_val[:10] > sales_incr).any():
                idx = np.nonzero(cumul_val[:10] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        elif mentions_fut[0] == 0:
            if (cumul_val[:10] > sales_incr).any():
                idx = np.nonzero(cumul_val[:10] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = 9
        elif len(val) < 20:
            if (cumul_val[:20] > sales_incr).any():
                idx = np.nonzero(cumul_val[:20] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        elif mentions_fut[1] < 2:
            if (cumul_val[:20] > sales_incr).any():
                idx = np.nonzero(cumul_val[:20] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = 19
        elif len(val) < 30:
            if (cumul_val[:30] > sales_incr).any():
                idx = np.nonzero(cumul_val[:30] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        elif mentions_fut[2] < 2:
            if (cumul_val[:30] > sales_incr).any():
                idx = np.nonzero(cumul_val[:30] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = 29
        elif len(val) < 40:
            if (cumul_val[:40] > sales_incr).any():
                idx = np.nonzero(cumul_val[:40] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        else:
            our_sell1 = 39
        top_val1 = np.argmax(val).item()
        last_val1 = len(val) - 1
        if plot_it:
            plt.vlines(top_val1, 99, 110, color="g")
            plt.vlines(last_val1, 98, 110, color="k")
            plt.vlines(our_sell1, 97, 110, color="r")
        return val[top_val1], top_val1, val[last_val1], last_val1, val[our_sell1], our_sell1

    def _get_sales_value(self, val, mentions_fut, stock, plot_it=False):
        # Set parameters
        sales_incr = 65
        low = 80

        # Compute moving average window of 9
        n = 9
        mov_avg = np.convolve(np.concatenate(([100] * 8, val)), np.ones(n) / n, mode='valid')

        # Compute exponential moving avg of size 20
        n = 20
        multiplier = 2 / (n + 1)
        ema = [(val[0] - 100) * multiplier + 100]
        for i in range(len(val) - 1):
            ema.append((val[i + 1] - ema[i]) * multiplier + ema[i])

        if plot_it:
            plt.ion()
            plt.figure()
            plt.plot(val, label="Raw")
            plt.plot(mov_avg, label="Mov_avg")
            plt.plot(ema, label="EMA")
            plt.title(stock + str(mentions_fut))
            plt.legend()
            plt.draw()
        # Compute cumulative value 4 values back
        cumul_val = np.cumsum(np.diff(np.concatenate((np.asarray([100]), val))))
        cumul_val[4:] = cumul_val[4:] - cumul_val[:-4]
        cumul_val[:4] = cumul_val[:4]

        # If there is a place where the 4 day cumulative value goes above sales_incr then set that as idx
        idx_cum = np.nonzero(cumul_val > sales_incr)[0]
        if len(idx_cum) != 0:
            idx_cum = idx_cum[0]
        else:
            idx_cum = len(val[:40]) - 1

        # If there is a place where the value goes below low then set that as idx
        idx_low = np.nonzero(val < low)[0]
        if len(idx_low) != 0:
            idx_low = idx_low[0]
        else:
            idx_low = len(val[:40]) - 1

        idx = np.min([idx_cum, idx_low])
        our_sell1 = idx.item()

        top_val1 = np.argmax(val).item()
        last_val1 = len(val) - 1

        #
        diff = np.nonzero(np.diff(np.sign(mov_avg - ema)) < 0)[0]
        if len(diff) > 0:
            diff = diff[0]
        else:
            diff = len(val[:40]) - 1
        if plot_it:
            plt.vlines(top_val1, 99, 110, color="g")
            plt.vlines(last_val1, 98, 110, color="k")
            plt.vlines(our_sell1, 97, 110, color="r")
            plt.vlines(diff, 96, 100, color="orange")

        return val[top_val1], top_val1, val[last_val1], last_val1, val[our_sell1], our_sell1


class MLModelForest(MLModel):
    def __init__(self, df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest):
        super().__init__(df_stockMentions, df_stockPrices, df_youtubeSources, df_trainTest)
        self.class_balance = {0: 0.2, 1: 0.8}
        self.forest = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, max_depth=4, class_weight="balanced", bootstrap=False)

    def cv_score(self, leaf, lay, bootstrap):
        self.forest = RandomForestClassifier(n_estimators=50, min_samples_leaf=leaf, max_depth=lay, class_weight="balanced", bootstrap=bootstrap)

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

    def money_out_train(self, plot_it):
        # If we bought all stocks the algorithm recommended
        mask_all = (self.cv_predictions != 0)
        money_mask = self.train_set_money[mask_all, :]
        stock_mask = np.asarray(self.stock_list_train)[mask_all]
        mentions_mask = np.asarray(self.mention_futures_train)[mask_all]
        sales_value = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            non_nan_money = money_mask[idx][~np.isnan(money_mask[idx])]
            sales_value.append(self._get_sales_value(non_nan_money, mentions_mask[idx], stock, plot_it))
        print(f"If we buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(sales_value, axis=0)}")

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
        mentions_mask = np.asarray(self.mention_futures_test)[mask_all]
        sales_value = list()
        for stock in np.unique(stock_mask):
            idx = np.nonzero(stock_mask == stock)[0][0]
            non_nan_money = money_mask[idx][~np.isnan(money_mask[idx])]
            sales_value.append(self._get_sales_value(non_nan_money, mentions_mask[idx], stock, plot_it))
        print(f"If we buy all the stocks recommended by the algorithm ({len(np.unique(stock_mask))} in total). "
              f"Then the earnings/loss over a month would be: {np.mean(sales_value, axis=0)}")

    def _get_sales_value_mentions(self, val, mentions_fut, stock, plot_it=False):
        sales_incr = 65
        if plot_it:
            plt.ion()
            plt.figure()
            plt.plot(val)
            plt.title(stock + str(mentions_fut))
            plt.draw()
        # Compute cumulative value 4 values back
        cumul_val = np.cumsum(np.diff(np.concatenate((np.asarray([100]), val))))
        cumul_val[4:] = cumul_val[4:] - cumul_val[:-4]
        cumul_val[:4] = cumul_val[:4]
        if len(val) < 10:
            if (cumul_val[:10] > sales_incr).any():
                idx = np.nonzero(cumul_val[:10] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        elif mentions_fut[0] == 0:
            if (cumul_val[:10] > sales_incr).any():
                idx = np.nonzero(cumul_val[:10] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = 9
        elif len(val) < 20:
            if (cumul_val[:20] > sales_incr).any():
                idx = np.nonzero(cumul_val[:20] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        elif mentions_fut[1] < 2:
            if (cumul_val[:20] > sales_incr).any():
                idx = np.nonzero(cumul_val[:20] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = 19
        elif len(val) < 30:
            if (cumul_val[:30] > sales_incr).any():
                idx = np.nonzero(cumul_val[:30] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        elif mentions_fut[2] < 2:
            if (cumul_val[:30] > sales_incr).any():
                idx = np.nonzero(cumul_val[:30] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = 29
        elif len(val) < 40:
            if (cumul_val[:40] > sales_incr).any():
                idx = np.nonzero(cumul_val[:40] > sales_incr)[0][0]
                our_sell1 = idx.item()
            else:
                our_sell1 = len(val) - 1
        else:
            our_sell1 = 39
        top_val1 = np.argmax(val).item()
        last_val1 = len(val) - 1
        if plot_it:
            plt.vlines(top_val1, 99, 110, color="g")
            plt.vlines(last_val1, 98, 110, color="k")
            plt.vlines(our_sell1, 97, 110, color="r")
        return val[top_val1], top_val1, val[last_val1], last_val1, val[our_sell1], our_sell1

    def _get_sales_value(self, val, mentions_fut, stock, plot_it=False):
        # Set parameters
        sales_incr = 65
        low = 80

        # Compute moving average window of 9
        n = 9
        mov_avg = np.convolve(np.concatenate(([100] * 8, val)), np.ones(n) / n, mode='valid')

        # Compute exponential moving avg of size 20
        n = 20
        multiplier = 2 / (n + 1)
        ema = [(val[0] - 100) * multiplier + 100]
        for i in range(len(val) - 1):
            ema.append((val[i + 1] - ema[i]) * multiplier + ema[i])

        if plot_it:
            plt.ion()
            plt.figure()
            plt.plot(val, label="Raw")
            plt.plot(mov_avg, label="Mov_avg")
            plt.plot(ema, label="EMA")
            plt.title(stock + str(mentions_fut))
            plt.legend()
            plt.draw()
        # Compute cumulative value 4 values back
        cumul_val = np.cumsum(np.diff(np.concatenate((np.asarray([100]), val))))
        cumul_val[4:] = cumul_val[4:] - cumul_val[:-4]
        cumul_val[:4] = cumul_val[:4]

        # If there is a place where the 4 day cumulative value goes above sales_incr then set that as idx
        idx_cum = np.nonzero(cumul_val > sales_incr)[0]
        if len(idx_cum) != 0:
            idx_cum = idx_cum[0]
        else:
            idx_cum = len(val[:40]) - 1

        # If there is a place where the value goes below low then set that as idx
        idx_low = np.nonzero(val < low)[0]
        if len(idx_low) != 0:
            idx_low = idx_low[0]
        else:
            idx_low = len(val[:40]) - 1

        idx = np.min([idx_cum, idx_low])
        our_sell1 = idx.item()

        top_val1 = np.argmax(val).item()
        last_val1 = len(val) - 1

        #
        diff = np.nonzero(np.diff(np.sign(mov_avg - ema)) < 0)[0]
        if len(diff) > 0:
            diff = diff[0]
        else:
            diff = len(val[:40]) - 1
        if plot_it:
            plt.vlines(top_val1, 99, 110, color="g")
            plt.vlines(last_val1, 98, 110, color="k")
            plt.vlines(our_sell1, 97, 110, color="r")
            plt.vlines(diff, 96, 100, color="orange")

        return val[top_val1], top_val1, val[last_val1], last_val1, val[our_sell1], our_sell1