"""
Scrapes J1 and J2 league data from
"https://data.j-league.or.jp/SFTP01/"

The train data given by Signate only covers J1 league data with the
timeframe of 2006 ~ 2016.

We found out that we needed more data to train our model. Therefore we
decided to collect more data.

The code below scrapes J1 league data from 1993, J2 league data from
1999 since those years are the official commencement of the
J1 and J2 league.
"""

import os
import requests

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scrape_utils import random_waiter, check_and_find_html_elements, connection_checker


def process_pandas_table_for_jleague(table_pd):
        # add year and date, turn it into pd datetime object!
    table_pd['match_date'] = pd.to_datetime(
        table_pd['年度'].astype(np.str)
        + '/'
        + table_pd['試合日'].str.replace(r"\(.*\)", "")
    )

    # rename japanese columns into english, and drop date columns
    table_pd = table_pd.rename({
        '大会': 'division', '節': 'section', 'K/O時刻': 'kick_off_time',
        'ホーム': 'home_team', 'スコア': 'score', 'アウェイ': 'away_team',
        'スタジアム': 'stadium', '入場者数': 'attendance',
        'インターネット中継・TV放送': 'broadcasters'
    }, axis='columns')

    table_pd = table_pd.drop(['年度', '試合日'], axis=1)

    """
    split the section column into to columns: round and sections
    to match the column names and values of the original_data given by
    signate
    """
    table_pd['round'] = table_pd['section'].str.extract(r'(第\d+日)')
    table_pd['section'] = table_pd['section'].str.extract(r'(第\d+節)')


def scrape_jleague_data(session, year, division):
    base_url = "https://data.j-league.or.jp/SFMS01/search"
    search_input = {'competition_years': year,
                    'competition_frame_ids': division}

    r = session.get(base_url, params=search_input)

    # check if we recieved a success http_status_code
    connection_checker(r)

    # check if we have the following html tag and attribute
    # then return it!
    soup = check_and_find_html_elements(
        r.content,
        ('table', {'class': 'table-base00 search-table'}),
    )

    # now create a dataframe to gather and process data!
    table_pd = pd.read_html(str(soup))[0]
    processed_table = process_pandas_table_for_jleague(table_pd)

    return processed_table


if __name__ == '__main__':
    scrape_options = {1: (1994, 2019), 2: (2000, 2019)}

    """
    extra_pd will be the final dataframe of the scraping module!
    it will hold all match data of J1, J2 league starting from
    1993, 1999 respectively
    """
    extra_pd = pd.DataFrame(
        columns=[
            'attendance', 'division', 'section', 'round', 'home_team',
            'away_team', 'stadium', 'score', 'broadcasters'])

    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla / 5.0 (Windows NT 10.0; Win64; x64)\
                           AppleWebKit / 537.36 (KHTML, like Gecko)\
                           Chrome / 70.0.3538.77 Safari / 537.36'
    }
    )

    for opt in (1, 2):
        time_frame = scrape_options.pop(opt)
        for year in range(time_frame[0], time_frame[1]):
            random_waiter(20, 50)

            extra_pd = pd.concat([extra_pd,
                                  scrape_jleague_data(s, year, opt)])

    extra_pd.to_csv('test')
