""" Functions """


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import os
import joblib


def get_stock_prices(ticker_symbol):
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
    now = datetime.now()  # Get current time
    a_year_ago = now - timedelta(days=365)  # Get current time 365 days ago
    yesterday = now.strftime("%Y-%m-") + str(int(now.strftime("%d")) - 1)  # Make it into a str
    yesterday_a_year_ago = a_year_ago.strftime("%Y-%m-") + str(int(a_year_ago.strftime("%d")) - 1)  # Make it into a str

    #get data on this ticker
    ticker_data = yf.Ticker(ticker_symbol)

    #get the historical prices for this ticker
    ticker_df = ticker_data.history(period='1d', start=yesterday_a_year_ago, end=yesterday)

    #see your data
    return ticker_df["Close"]


def get_soup(url):
    """ Return a beautiful soup object given a url
    Parameters
    ----------
        url: str
            Youtube URL
    Return
    -------
        bs(response.html.html, "html.parser"): beautiful soup object
            beautiful soup object containing all html code for the webpage """
    # init session
    session = HTMLSession()
    # download HTML code
    response = session.get(url)
    # execute Javascript
    response.html.render(sleep=3, timeout=60, keep_page=True)
    # create beautiful soup object to parse HTML
    soup = bs(response.html.html, "html.parser")
    # Close the HTMLSession
    session.close()
    return soup


def get_all_videos(channel_url):
    """ Get the urls for the new videos
    Parameters
    ----------
        channel_url: str
            Youtube URL
    Return
    -------
        video_url_list: list
            list containing the urls for all videos posted """
    path = os.getcwd()  # Get current dir
    os.chdir(path + "/get_data/")  # Go to dir with youtube scraping script
    os.system("cmd /c youtube-dl --get-id " + channel_url + " > url_list.txt")  # Run scraping script
    # Read written file for all the urls and put them into a list
    with open(path + "/get_data/url_list.txt") as f:
        lines = f.readlines()
    video_url_list = ["https://www.youtube.com/watch?v=" + line[:-1] for line in lines]  # Make into functional urls
    os.chdir(path)  # Go back to original dir
    return video_url_list


def get_new_videos(url_str):
    """ Get the urls for the new videos
    Parameters
    ----------
        url_str: str
            Youtube URL
    Return
    -------
        video_url_list: list
            list containing the urls for videos posted within the last 24 hours"""
    soup = get_soup(url_str)  # create beautiful soup object to parse HTML
    # Make a list of times since now of video post for all videos and number of viewings
    mixed_list = soup.find_all("span", attrs={"class": "style-scope ytd-grid-video-renderer"})
    # Get indices of videos which are made withing the last 24 hours in a list
    idxs = list()
    for i, t in enumerate(mixed_list):
        if (t.text[-11:-6] == "timer" or t.text[-10:-6] == "time" or t.text[-14:-6] == "minutter" or
                t.text[-11:-6] == "minut" or t.text[-14:-6] == "sekunder" or t.text[-12:-6] == "sekund"):
            idxs.append(int((i - 1) / 2))
    # Make list of all urls and select only the relevant ones according to list
    url_list = soup.find_all("h3", attrs={"class": "style-scope"})
    video_url_list = ["https://www.youtube.com/" + url_list[idx].find("a")["href"] for idx in idxs]
    return video_url_list


def scrape_video_url(url_str):
    """ Scrape data of a website
        Parameters
        ----------
            url_str: str
                Youtube URL
        Return
        -------
            result: dict
                dictionary containing all the many scraped data """
    # Ensure that there will be made multiple attempt to scrape the video
    error_count = 0
    soup = get_soup(url_str)  # create beautiful soup object to parse HTML
    # initialize the result
    result = dict()
    try:
        # video title
        result["title"] = soup.find("h1").text.strip()
        # video views (converted to integer)
        #result["views"] = int(''.join([c for c in soup.find("span", attrs={"class": "view-count"}).text if c.isdigit()]))
        # video description
        result["description"] = soup.find("yt-formatted-string", {"class": "content"}).text
        # date published
        result["date_published"] = soup.find("div", {"id": "date"}).text[1:]
        # get the video tags
        result["tags"] = ', '.join([meta.attrs.get("content") for meta in soup.find_all("meta", {"property": "og:video:tag"})])
        # number of likes
        #text_yt_formatted_strings = soup.find_all("yt-formatted-string", {"id": "text", "class": "ytd-toggle-button-renderer"})
        #result["likes"] = text_yt_formatted_strings[0].text
        # number of dislikes
        #result["dislikes"] = text_yt_formatted_strings[1].text
        # channel details
        channel_tag = soup.find("yt-formatted-string", {"class": "ytd-channel-name"}).find("a")
        # channel name
        channel_name = channel_tag.text
        # channel URL
        channel_url = f"https://www.youtube.com{channel_tag['href']}"
        # number of subscribers as str
        channel_subscribers = soup.find("yt-formatted-string", {"id": "owner-sub-count"}).text.strip()
        result['channel'] = {'name': channel_name, 'url': channel_url, 'subscribers': channel_subscribers}
    except AttributeError as e:
        print("########################")
        print(f"AttributeError occurred trying to scrape {url_str} for the {error_count} time")
        print("########################")
    return result


def print_scraped_data(url):
    # get the data
    data = scrape_video_url(url)
    # print in nice format
    print(f"Title: {data['title']}")
    print(f"Views: {data['views']}")
    print(f"Published at: {data['date_published']}")
    print(f"Video tags: {data['tags']}")
    print(f"Likes: {data['likes']}")
    print(f"Dislikes: {data['dislikes']}")
    print(f"\nDescription: {data['description']}\n")
    print(f"\nChannel Name: {data['channel']['name']}")
    print(f"Channel URL: {data['channel']['url']}")
    print(f"Channel Subscribers: {data['channel']['subscribers']}")


def get_stocks_mentioned(title, desc, tags, most_common_words, print_stats=False):
    not_stocks = list()
    actual_stocks = list()
    split_title = clean_text(title)
    split_desc = clean_text(desc)
    split_tags = clean_text(tags)
    val_counts = pd.Series(split_tags + split_title + split_desc).value_counts()
    word_count = len(pd.Series(split_tags + split_title + split_desc))
    if print_stats:
        print(word_count)
    for i, word in enumerate(val_counts.index[val_counts > round(word_count / 200)]):
        if word.lower() in most_common_words:
            continue
        if print_stats:
            print(word, val_counts[i])
        stock_prices = get_stock_prices(word)
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


def construct_date_base(youtube_channels, most_common_words, start_date, all_data=None):
    # If first run, create new all_data dict
    if all_data is None:
        all_data = dict()
    else:
        # If some channels have been scraped, make sure these channels are not scraped again
        remaining_channels = np.setdiff1d(youtube_channels, [*all_data])
    # Scrape remaining channels
    for youtube_channel in remaining_channels:
        url_list = get_all_videos(youtube_channel)
        print(f"There are {len(url_list)} videos in the channel {youtube_channel}")

        scraped_data = dict()
        for i, url in enumerate(url_list):
            results = scrape_video_url(url)
            actual_stocks, not_stocks = get_stocks_mentioned(results["title"], results["description"],
                                                             results["tags"], most_common_words, print_stats=False)
            scraped_data.update({results["date_published"]: actual_stocks})
            most_common_words += not_stocks

            if i % 99 == 0 and i > 0:
                print(f"{i + 1} videos scraped out of {len(url_list)}")
                joblib.dump([scraped_data, [str(w) for w in most_common_words]], "scraped_data_during_processing.joblib")

        all_data.update({youtube_channel: scraped_data})
        joblib.dump(all_data, "all_data_during_processing.joblib")
    # Rewrite scraped data to data_base
    data_base = from_scraped_channels_to_data_base(all_data, start_date, youtube_channels)
    return data_base, most_common_words


def from_scraped_channels_to_data_base(all_data, start_date, youtube_channels):
    data_base = dict()
    for youtube_channel in all_data.keys():
        # Get list of pub. dates
        keys_in_list = [*all_data[youtube_channel]]
        # Split strings in list
        split_keys_in_list = [date.replace(".", "").split() for date in keys_in_list]
        # Make an list of month to index for month abbr.
        months = ["", "jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]
        # Write all publication dates as pandas Timestamp objects
        all_dates_as_timestamps = [pd.Timestamp(int(date[-1]), int(months.index(date[-2])), int(date[-3])) for date in
                                   split_keys_in_list]

        # Sort keys in order from oldest to newest
        sort_order = np.argsort(all_dates_as_timestamps)
        sorted_all_dates_as_timestamps = np.asarray(all_dates_as_timestamps)[sort_order]
        sorted_keys_in_list = np.asarray(keys_in_list)[sort_order]

        # Make a mask to only get timestamps after start_date
        mask = sorted_all_dates_as_timestamps > pd.Timestamp(start_date[0], start_date[1], start_date[2])
        # Extract relevant dates and keys
        relevant_dates_as_timestamps = sorted_all_dates_as_timestamps[mask]
        relevant_keys_in_list = sorted_keys_in_list[mask]
        # Make a list with keys to pop from dictionary
        pop_list = sorted_keys_in_list[~mask]
        # Pop/remove elements in pop_list from dictionary
        for pop_element in pop_list:
            all_data[youtube_channel].pop(pop_element)

        # Assert if there are indeed any videos on the channel
        assert len(relevant_keys_in_list) > 0

        # Get list of all stocks mentioned on this channel
        all_stocks_mentioned = np.unique([item for sublist in [*all_data[youtube_channel].values()] for item in sublist])

        # Find all stocks not already in the data_base
        new_stocks = np.setdiff1d(all_stocks_mentioned, [*data_base])
        # Make a data_base entry for each new_stock and fill it with zeros from start up until today
        for stock in new_stocks:
            stock_df = pd.DataFrame(index=youtube_channels)
            list_of_timestamps = days_from_start_to_today()
            for timestamp in list_of_timestamps:
                stock_df[timestamp.strftime("%Y-%m-%d")] = 0
            data_base.update({stock: stock_df})

        # Go over all dates with videos in youtube_channel
        for i, key in enumerate(relevant_keys_in_list):
            # Get the stock mentioned on the day of key
            stocks_the_keyth_day = all_data[youtube_channel][key]
            # If no stocks are mentioned, skip the video date
            if len(stocks_the_keyth_day) == 0:
                continue
            else:
                # For all stocks mentioned on the day, take a note that they were mentioned by youtube_channel
                for stock in stocks_the_keyth_day:
                    data_base[stock].loc[youtube_channel, relevant_dates_as_timestamps[i].strftime("%Y-%m-%d")] = 1
    return data_base


def add_todays_stock_mentions(stock_df, stock_mentions):
    stock_df[pd.Timestamp.today().strftime("%Y-%m-%d")] = 0
    for mention in stock_mentions:
        stock_df.loc[mention, pd.Timestamp.today().strftime("%Y-%m-%d")] = 1
    return stock_df


def days_from_start_to_today():
    start_date = pd.Timestamp(2020, 7, 1)
    today = pd.Timestamp.today()
    list_of_days = [start_date + pd.Timedelta(days=days) for days in range((today - start_date).days + 1)]
    return list_of_days


def scrape_videos_24h(youtube_channels, most_common_words):
    new_stocks = dict()
    words_to_add = list()
    scraped_data = dict()
    for youtube_channel in youtube_channels:
        url_list = get_new_videos(youtube_channel)
        for url in url_list:
            results = scrape_video_url(url)
            scraped_data.update({youtube_channel: results})

            # Try to look only in video tags for messages
            try:
                print("--------------------------------------")
                print(results["title"])
                #print(results["description"])
                print(results["tags"])

                actual_stocks, not_stocks = get_stocks_mentioned(results["title"], results["description"],
                                                                 results["tags"], most_common_words, print_stats=True)
                print(not_stocks, actual_stocks)
                words_to_add += not_stocks
                for stock in actual_stocks:
                    if stock in new_stocks.keys():
                        new_stocks[stock].append(youtube_channel)
                    else:
                        new_stocks.update({stock: [youtube_channel]})
            except KeyError as e:
                print(f"{youtube_channel} did not post any videos today")
    return new_stocks, words_to_add
