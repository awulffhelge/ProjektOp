""" Projekt Op main script """


import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


youtube_channels = ["https://www.youtube.com/c/MeetKevin/videos", "https://www.youtube.com/c/BestofUSHomesKWRealty/videos",
                    "https://www.youtube.com/c/jayconomics/videos", "https://www.youtube.com/c/Mcash-Channel",
                    "https://www.youtube.com/c/MyFinancialFriend/videos", "https://www.youtube.com/channel/UC30QJvWq_Nh3UOW7PoU1FAQ/videos",
                    "https://www.youtube.com/c/LetsTalkMoneywithJosephHogueCFA/videos", "https://www.youtube.com/channel/UCf3g0uNBn6wPpgrTK_AbJrQ/videos",
                    "https://www.youtube.com/c/ZipTrader/videos", "https://www.youtube.com/c/StealthWealthInvesting/videos",
                    "https://www.youtube.com/channel/UCvGKSQPQR2KiIyS8HHYLbxQ/videos", "https://www.youtube.com/channel/UCie-TaHu03Ka04kkaUiNDkg",
                    "https://www.youtube.com/channel/UCrTFPf6rq5OUSWb7ILW9trg/videos", "https://www.youtube.com/channel/UCNeBCizpA1NbX7A9_ia6jYg/videos",
                    "https://www.youtube.com/c/StockMoe/videos", "https://www.youtube.com/c/JMacInvesting/videos",
                    "https://www.youtube.com/c/NicholasJParis/videos", "https://www.youtube.com/c/JackSpencerInvesting/videos",
                    "https://www.youtube.com/c/Deadnsyde/videos", "https://www.youtube.com/c/BelmontCapital/videos",
                    "https://www.youtube.com/channel/UC0h64epNmMIPYuqSGKCrj9Q/videos", "https://www.youtube.com/c/DaveLeeonInvesting/videos",
                    "https://www.youtube.com/c/FinancialEducation/videos", "https://www.youtube.com/c/RexFinance/videos",
                    "https://www.youtube.com/channel/UC9ekG7-ehEtitg4-_2yYwpQ/videos", "https://www.youtube.com/c/CourtsideFinancial/videos",
                    "https://www.youtube.com/c/DONGXiiTheChinaOpportunity/videos", "https://www.youtube.com/channel/UC_EnNG4bwy4FIUKzPTG8UCg/videos",
                    "https://www.youtube.com/channel/UC0OnreqP55xLpA6W5nzxb5Q/videos", "https://www.youtube.com/channel/UC2Zg12phLyX1KTnH6D47d9g/videos",
                    "https://www.youtube.com/channel/UC-CfltxhlMMCT5pJNIfQhlQ/videos", "https://www.youtube.com/c/Vincentc/videos",
                    "https://www.youtube.com/c/JayDeemInvesting/videos"
                    ]

# Read file with most common words
most_common_words = list(joblib.load("most_common_words1000.joblib"))
#most_common_words += words_to_add
#joblib.dump(np.asarray([str(abba) for abba in most_common_words]), "most_common_words1000.joblib")
words_to_add = []


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Get some stock prices
    #ticker_df = functions.get_stock_prices("NOW?")
    #print(ticker_df)

    # Scrape videos from last 24 hours
    new_stocks, words_to_add = functions.scrape_videos_24h(youtube_channels, most_common_words)


# MOMENTARY - Update database with new stock recommendations from the last 24h
data_sets = dict()
#data_sets = joblib.load("data_sets.joblib")
for stock in new_stocks.keys():
    if stock in [*data_sets]:
        stock_df = data_sets[stock]
        data_sets.update({stock: functions.add_todays_stock_mentions(stock_df, new_stocks[stock])})
    else:
        stock_df = pd.DataFrame(index=youtube_channels)
        list_of_timestamps = functions.days_from_start_to_today()
        for timestamp in list_of_timestamps:
            stock_df[timestamp] = 0
        data_sets.update({stock: functions.add_todays_stock_mentions(stock_df, new_stocks[stock])})
#joblib.dump(data_sets, "data_sets.joblib")


# Construct data base by scraping all videos from all channels
all_data = joblib.load("all_data_during_processing.joblib")  # Load already scraped channels
start_date = [2019, 7, 1]
data_base, most_common_words = functions.construct_date_base(youtube_channels, most_common_words, start_date, all_data)


# Visualize