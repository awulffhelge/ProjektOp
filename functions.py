""" Functions """


import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import os


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
    response.html.render(sleep=2)
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
