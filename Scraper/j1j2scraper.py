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
import sys
import re
from itertools import product
from multiprocessing import Pool, cpu_count

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from Utils.scrape_utils import (
    random_waiter, check_and_find_html_elements,
    clean_string, connection_checker
)


def process_table_for_jleague_data1(soup):

    # the column name and its location in the website
    column_and_table_order = (
        ('attendance', 9), ('division', 1), ('section_with_round', 2),
        ('home_team', 5), ('away_team', 7), ('broadcasters', 10),
        ('score', 6), ('match_year', 0), ('month_day', 3), ('id', 6)
    )

    # table_pd will end up being the returned table!
    table_pd = pd.DataFrame()

    data = list()

    # 'tr' is the html tag for table row,
    # 'td' is the element of the row
    for tr in soup.find_all('tr'):
        td = tr.find_all('td')
        td_list = [td[opt[1]].text.strip() if opt[0] != 'id' else td[opt[1]]
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

    table_pd['id'] = table_pd['id'].astype(
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
        id, home_team_soup, away_team_soup, stadium_soup):
    """ Returns row data of a certain id
    The order of the list is
    [id, home_team_player 1 ~ 11, away_team_player 1~ 11,
    stadium_name, weather, temperature, humidity]
    """

    # initiate a row starting with id
    data_list = [id, ]

    # add home team first then add away team players
    for soup in [home_team_soup, away_team_soup]:
        for tr in soup.find_all('tr'):
            td = tr.find_all('td')
            data_list.append(clean_string(
                ' '.join([td[0].text, td[2].text, td[1].text])
            ))

    for tr in stadium_soup.find_all('tr'):
        td = tr.find_all('td')
        data_list.extend(
            map(clean_string, [td[2].text, td[4].text, td[5].text, td[6].text])
        )

    return data_list


def scrape_jleague_match_data1(session, year, division, **kwargs):
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


def scrape_jleague_match_data2(session, id, **kwargs):
    """ Scrapes jleague data from match detail page.
    Detail Page example below
    "https://data.j-league.or.jp/SFMS02/?match_card_id=19075"
    """

    base_url = 'https://data.j-league.or.jp/SFMS02/?match_card_id='
    r = session.get(base_url + str(id))

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
        id,
        soup_contain_home_player_data,
        soup_contain_away_player_data,
        soup_contain_stadium_data)

    return processed_list


def ask_scrape_loc_and_process_options():

    scrape_local_or_website_msg = (
        'If you want to run scraping script with local files, enter "local"'
        ', if you want to scrape directly from the website enter "website" : '
    )

    scrape_serial_or_multi_msg = (
        'If you want to run your scraping script with multiprocessing enter '
        '"multi", if you want to run it with single core, enter "serial" : '
    )

    scrape_option_dict = {
        'scrape_local': {
            'msg': scrape_local_or_website_msg,
            'local': True,
            'website': False
        },
        'scrape_multi': {
            'msg': scrape_serial_or_multi_msg,
            'multi': True,
            'serial': False
        }
    }

    return_dict = dict()

    for key, values in scrape_option_dict.items():
        while True:
            answer = input(values['msg']).lower()
            check_list = list(values.keys())
            check_list.remove('msg')
            if answer in check_list:
                print(values)
                return_dict[key] = values[answer]
                break
    return return_dict


def main_scraper():
    scrape_timeframe = {1: (1994, 2019), 2: (2000, 2019)}

    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla / 5.0 (Windows NT 10.0; Win64; x64)\
                           AppleWebKit / 537.36 (KHTML, like Gecko)\
                           Chrome / 70.0.3538.77 Safari / 537.36'
    })

    scrape_options = ask_scrape_loc_and_process_options()

    # extra_pd will be the final dataframe of the scraping module!
    # it will hold all match data of J1, J2 league starting from
    # 1993, 1999 respectively
    extra_data1 = pd.DataFrame(
        columns=[
            'attendance', 'division', 'section', 'round', 'home_team',
            'away_team', 'broadcasters', 'id'])

    for opt in (1, ):
        print(opt)
        time_frame = scrape_timeframe.pop(opt)
        for year in range(time_frame[0], time_frame[1]):
            print(year)
            random_waiter(20, 50)
            extra_data1 = pd.concat([
                extra_data1,
                scrape_jleague_match_data1(s, year, opt),

            ])

    extra_data1.applymap(clean_string)

    extra_data1.to_csv('extra_data1.csv', index=False)

    data_list_for_extra_list = list()

    core_to_use = cpu_count() if scrape_options['scrape_multi'] == True else 1

    p = Pool(core_to_use)
    print("processing with {} cores".format(core_to_use))

    data_list_for_extra_list = p.starmap(
        scrape_jleague_match_data2,
        product((s, ), list(extra_data1['id']))
    )

    extra_data2 = pd.DataFrame(
        columns=[
            'id', 'home_team_player1', 'home_team_player2',
            'home_team_player3', 'home_team_player4', 'home_team_player5',
            'home_team_player6', 'home_team_player7', 'home_team_player8',
            'home_team_player9', 'home_team_player10', 'home_team_player11',
            'away_team_player1', 'away_team_player2',  'away_team_player3',
            'away_team_player4', 'away_team_player5', 'away_team_player6',
            'away_team_player7', 'away_team_player8', 'away_team_player9',
            'away_team_player10', 'away_team_player11', 'stadium_name',
            'weather', 'temperature', 'humidity'
        ],
        data=data_list_for_extra_list)

    extra_data2.applymap(clean_string)

    extra_data2.to_csv('extra_data2.csv', index=False)

    # now join both dataframe to create a new one, then
    # separate the pd to make it match the format of the
    # data given by signate

    combined_data = extra_data1.join(extra_data2.set_index('id'),  on='id')

    del extra_data1
    del extra_data2

    ex_match_reports = combined_data[[CONSTANTS.MATCH_REPORTS_COLUMNS]]
    ex_train_test = combined_data[[CONSTANTS.MATCH_COLUMNS]]

    ex_match_reports.to_csv('ex_match_reports.csv')
    ex_train_test.to_csv('ex_total.csv')
