""" Projekt Op main script """


import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_text(text):
    return (text.replace(",", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            .replace(".", "").replace("$", "").replace("#", "").replace("@", "").replace(":", "")
            .replace(";", "").replace("/", "").replace("!", "").replace("?", "").replace("-", "")
            .replace("_", "").replace("&", "").replace("*", "").upper().split())


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
import joblib
most_common_words = list(joblib.load("most_common_words1000.joblib"))
#most_common_words += words_to_add
#joblib.dump(np.asarray(most_common_words), "most_common_words1000.joblib")
words_to_add = []

first_stocks = dict()
words_to_add = list()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Get some stock prices
    #ticker_df = functions.get_stock_prices("NOW?")
    #print(ticker_df)

    scraped_data = dict()
    for youtube_channel in youtube_channels:
        url_list = functions.get_new_videos(youtube_channel)
        for url in url_list:
            results = functions.scrape_video_url(url)
            scraped_data.update({youtube_channel: results})

            # Try to look only in video tags for messages
            try:
                print("--------------------------------------")
                print(results["title"])
                title = results["title"]
                #print(results["description"])
                desc = results["description"]
                print(results["tags"])
                tags = results["tags"]
                split_title = clean_text(title)
                split_desc = clean_text(desc)
                split_tags = clean_text(tags)
                val_counts = pd.Series(split_tags + split_title + split_desc).value_counts()
                word_count = len(pd.Series(split_tags + split_title + split_desc))
                print(word_count)
                #print(val_counts)
                for i, word in enumerate(val_counts.index[val_counts > round(word_count / 200)]):
                    if word.lower() in most_common_words:
                        continue
                    print(word, val_counts[i])
                    a = functions.get_stock_prices(word)
                    if len(a) == 0:
                        words_to_add.append(word.lower())
                    else:
                        if word in first_stocks.keys():
                            first_stocks[word].append(youtube_channel)
                        else:
                            first_stocks.update({word: [youtube_channel]})
            except KeyError as e:
                print(f"{youtube_channel} did not post any videos today")


def add_todays_stock_mentions(stock_df, stock_mentions):
    stock_df[pd.Timestamp.today().strftime("%Y-%m-%d")] = 0
    for mention in stock_mentions:
        stock_df.loc[mention, pd.Timestamp.today().strftime("%Y-%m-%d")] = 1
    return stock_df

data_sets = dict()
for stock in first_stocks.keys():
    if stock in data_sets.keys():
        stock_df = data_sets[stock]
        data_sets.update({stock: add_todays_stock_mentions(stock_df, first_stocks)})
    else:
        stock_df = pd.DataFrame(index=youtube_channels)
        data_sets.update({stock: add_todays_stock_mentions(stock_df, first_stocks[stock])})

joblib.dump(data_sets, "data_sets.joblib")


plt.figure()
ticker_df.plot(label="Raw")
ticker_df.diff().plot(label="Diff")

# Smooth it
win_size = 3
smoothed = ticker_df.rolling(win_size, closed="right").mean()
smoothed.dropna().plot(label="Smoothed with window size " + str(win_size))

plt.legend()


# Get all videos from the channels
for youtube_channel in youtube_channels:
    url_list = functions.get_all_videos(youtube_channel)
    print(url_list)
    print(len(url_list))


