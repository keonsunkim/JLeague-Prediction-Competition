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
import re

import requests

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scrape_utils import (
    random_waiter, check_and_find_html_elements,
    clean_string, connection_checker
)


def process_table_for_jleague_data1(soup):

    # the column name and its location in the website
    column_and_table_order = (
        ('attendance', 9), ('division', 1), ('section_with_round', 2),
        ('home_team', 5), ('away_team', 7), ('broadcasters', 10),
        ('score', 6), ('match_year', 0), ('month_day', 3), ('match_id', 6)
    )

    # table_pd will end up being the returned table!
    table_pd = pd.DataFrame()

    data = list()

    # 'tr' is the html tag for table row,
    # 'td' is the element of the row
    for tr in soup.find_all('tr'):
        td = tr.find_all('td')
        td_list = [td[opt[1]].text.strip() if opt[0] != 'match_id' else td[opt[1]]
                   for opt in column_and_table_order]
        data.append(td_list)

    table_pd = pd.DataFrame(
        columns=[opt[0] for opt in column_and_table_order],
        data=data,
    )

    # add year and date, turn it into pd datetime object!
    table_pd['match_date'] = pd.to_datetime(
        table_pd['match_year'].astype(np.str)
        + '/'
        + table_pd['month_day'].str.replace(r"\(.*\)", "")
    )

    # Split the section column into to columns: round and sections
    # to match the column names and values of the original_data given by
    # signate

    # split section_with_round, and get section and round
    table_pd['round'] = table_pd[
        'section_with_round'].str.extract(r'(第\d+日)')
    table_pd['section'] = table_pd[
        'section_with_round'].str.extract(r'(第\d+節)')

    table_pd['match_id'] = table_pd['match_id'].astype(
        str).str.extract(r'/SFMS02/\?match_card_id=(\d+)')
    table_pd['home_team_score'] = table_pd['score'].astype(
        str).str.extract(r'(\d+)-')
    table_pd['away_team_score'] = table_pd['score'].astype(
        str).str.extract(r'-(\d+)')

    table_pd = table_pd.drop(
        ['match_year', 'month_day', 'section_with_round', 'score'], axis=1
    )

    return table_pd


def process_table_for_jleague_data2(
        match_id, home_team_soup, away_team_soup, stadium_soup):
    """ Returns row data of a certain match_id
    The order of the list is
    [match_id, home_team_player 1 ~ 11, away_team_player 1~ 11,
    stadium_name, weather, temperature, humidity]
    """

    # initiate a row starting with match_id
    data_list = [match_id, ]

    # add home team first then add away team players
    for soup in [soup_contain_home_player_data, soup_contain_away_player_data]:
        for tr in soup.find_all('tr'):
            td = tr.find_all('td')
            data_list.append(clean_string(
                ' '.join([td[0].text, td[2].text, td[1].text])
            ))

    for tr in soup_contain_stadium_data.find_all('tr'):
        td = tr.find_all('td')
        data_list.extend(
            map(clean_string, [td[2].text, td[4].text, td[5].text, td[6].text])
        )


def scrape_jleague_match_data1(session, year, division):
    """ Scrapes jleague data from table result page.
    Table page = "https://data.j-league.or.jp/SFMS01/search?"
    We still need data on weather, stadium name, temperature, humidity.
    That needs to be scraped from the detail page, which will be
    scraped in function "scrape_jleague_match_data2"
    """
    base_url = "https://data.j-league.or.jp/SFMS01/search"
    search_input = {'competition_years': year,
                    'competition_frame_ids': division}

    r = session.get(base_url, params=search_input)

    # check if we recieved a success http_status_code
    connection_checker(r)

    # check if we have the following html tag and attribute
    # then return it!
    soup_contain_result_table = check_and_find_html_elements(
        r.content,
        ('table', {'class': 'table-base00 search-table'}),
        ('tbody', None), nested=True
    )
    # now create a dataframe to gather and process data!

    processed_table = process_table_for_jleague_data1(
        soup_contain_result_table)

    return processed_table


def scrape_jleague_match_data2(session, match_id):
    """ Scrapes jleague data from match detail page.
    Detail Page example =
    "https://data.j-league.or.jp/SFMS02/?match_card_id=19075"
    """

    base_url = 'https://data.j-league.or.jp/SFMS02/?match_card_id='
    r = sesstion.get(match_data + str(match_id))

    # check if we recieved a success http_status_code
    connection_checker(r)

    # get the table on home players
    soup_contain_home_player_data = check_and_find_html_elements(
        r.content,
        ('div', {'class': 'two-column-table-area two-column-table-area840'}),
        ('div', {'class': 'two-column-table-box-l'}),
        ('div', {'class': 'two-column-table-base'}),
        ('tbody', None),
        nested=True
    )

    # get the table on away players
    soup_contain_away_player_data = check_and_find_html_elements(
        r.content,
        ('div', {'class': 'two-column-table-area two-column-table-area840'}),
        ('div', {'class': 'two-column-table-box-r'}),
        ('div', {'class': 'two-column-table-base'}),
        ('tbody', None),
        nested=True
    )

    soup_contain_stadium_data = check_and_find_html_elements(
        r.content,
        ('div', {'class': 'two-column-table-bottom'}),
        ('tbody', None),
        nested=True
    )

    processed_list = process_table_for_jleague_data2(
        match_id,
        soup_contain_home_player_data,
        soup_contain_away_player_data,
        soup_contain_stadium_data)

    return pocessed_list


if __name__ == '__main__':
    scrape_options = {1: (1994, 2019), 2: (2000, 2019)}

    # extra_pd will be the final dataframe of the scraping module!
    # it will hold all match data of J1, J2 league starting from
    # 1993, 1999 respectively
    extra_data1 = pd.DataFrame(
        columns=[
            'attendance', 'division', 'section', 'round', 'home_team',
            'away_team', 'stadium', 'score', 'broadcasters', 'match_id'])

    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla / 5.0 (Windows NT 10.0; Win64; x64)\
                           AppleWebKit / 537.36 (KHTML, like Gecko)\
                           Chrome / 70.0.3538.77 Safari / 537.36'
    }
    )

    for opt in (1, 2):
        print(opt)
        time_frame = scrape_options.pop(opt)
        for year in range(time_frame[0], time_frame[1]):
            print(year)
            random_waiter(20, 50)
            extra_data1 = pd.concat([extra_data1,
                                     scrape_jleague_match_data1(s, year, opt)
                                     ])

    extra_data1.to_csv('extra_data1')

    # extra_data2 = pd.DataFrame(
    #     columns=[
    #         'match_id', 'home_team_player_1', 'home_team_player_2',
    #         'home_team_player_3', 'home_team_player_4', 'home_team_player_5',
    #         'home_team_player_6', 'home_team_player_7', 'home_team_player_8',
    #         'home_team_player_9', 'home_team_player_10', 'home_team_player_11',
    #         'away_team_player_1', 'away_team_player_2',  'away_team_player_3',
    #         'away_team_player_4', 'away_team_player_5', 'away_team_player_6',
    #         'away_team_player_7', 'away_team_player_8', 'away_team_player_9',
    #         'away_team_player_10', 'away_team_player_11', 'stadium_name',
    #         'weather', 'temperature', 'humidity'
    #     ]
    # )

    # data_list_for_extra_list = list()
    # for match_id in list(extra_data1['match_id']):
    # data_list_for_extra_list.append(scrape_jleague_match_data2(s, match_id)
