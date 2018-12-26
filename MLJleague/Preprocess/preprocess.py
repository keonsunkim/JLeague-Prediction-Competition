"""

#######################################################################
#######################################################################
############################Preprocess.py##############################
#######################################################################
#######################################################################

Warning:
1. The functions are not in alphabetical order. They are in order which
they will be run to create the training data.
2. The file itself is long. I also do believe that it is unsensical to
crowd in all the functions in one .py file, but leave this at it is for
now.
3. It is a PEP8 Nightmare here. The code were originally created in a
.ipynb file which seems to be more lenient with following PEP8. I did
mybest to format the code to conform to PEP8, but it is still a
nightmare.
"""
#######################################################################
###########STEP 0 Imports and Introducing Global Functions#############
#######################################################################

import gc
import math
import operator
import os
import random
import re
import sys
import warnings

from collections import Counter
from glob import glob
from itertools import product

from math import sqrt
# Python standard library

import googlemaps
import requests
from bs4 import BeautifulSoup as bs
# 3rd Party Packages for Web Scraping


import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
# Numpy and Pandas


from meteocalc import Temp, dew_point, heat_index
# 3rd Party Packages for Processing and Analysis


key = 'AIzaSyAVCE-eyUNCAvrcDHiWfRgSDorrz41jA5w'
gmaps = googlemaps.Client(key=key)
# Google map api and keys to run the google map api

warnings.filterwarnings('ignore')
# Set plt options

path = os.getcwd()
os.chdir(path)
# This is extremely important!
# All the functions are created on the premise that the csv files
# will be created inside the folder this file is located!!!
# Make sure this file is and its csv files are contained in a folder!

#######################################################################
#######################################################################
############# STEP 0 General Functions and Variables ##################
#######################################################################
#######################################################################
# Below are General Functions that written in util.property
# which will be continuosly called by other
# functions In alphabetical order
from MLJleague.Utils.preprocess_utils import *


"""
Below are functions related to the names
Stadiums, and Teams
"""

# Function to unificate venue name and team name for better prediction.


def process_stadium_names(df):
    """

    """
    df['stadium'] = df['stadium'].replace(to_replace=same_stadium_dict)


def process_team_names(df):
    # A football team may have the multiple names due to historical
    # reasons and due to denotation reasons. The Japanese 'G' and
    # the English 'G' are the same G for humans but not for the
    # computer.
    # Teams like Ｖ川崎 = Kawasaky Verdy, 東京Ｖ = Tokyo Verdy are the same
    # team just has different names in a different point of time in history.
    df['home_team'] = df['home_team'].replace(to_replace=same_team_dict)
    df['away_team'] = df['away_team'].replace(to_replace=same_team_dict)


# In case it is required to change name of venue at columns what's name
# is not 'venue', which is designated in  tidying_venue_name(df)
# function, we created dictionary to convert venues' name.

# STEP 0

#######################################################################
#######################################################################
######## STEP 1 Loading Datasets Given By Competition Host ############
#######################################################################
#######################################################################
train = pd.read_csv('../original_data/train.csv')
test = pd.read_csv('../original_data/test.csv')
capacity = pd.read_csv('stadium_capacity_mapping.csv')

same_stadium_dict = np.load('dictionary_folder/same_stadium_dict.npy').item()
same_team_dict = np.load('dictionary_folder/same_team_dict.npy').item()


def load_and_merge_original_data():

    # Change the name of the column "venue" to "stadium"
    # both in train and test set.
    for df in [train, test]:
        df.rename(columns={'venue': 'stadium'}, inplace=True)

    # Concatonate train and test into combine
    combine = pd.concat([train, test], ignore_index=True)

    # Change the order of the columns
    combine = combine[[
        'id', 'attendance', 'away_team', 'broadcasters', 'home_team',
        'humidity', 'kick_off_time', 'match_date', 'round', 'section',
        'temperature', 'stadium', 'weather'
    ]]

    # Set the team and stadium names straight
    # on two datasets(combine, capacity)
    process_stadium_names(combine)
    process_team_names(combine)
    process_stadium_names(capacity)

    # Drop all duplicates in capacity
    capacity.drop_duplicates(inplace=True)

    # Merge the stadium data back in!
    combine = combine.merge(capacity, how='left', on='stadium')

    # Change the type of the datatype of column name : 'match_date'
    # from string type to datetime type
    combine['match_date'] = pd.to_datetime(combine['match_date'])
    return combine

combine = load_and_merge_initial_data()
#######################################################################
#######################################################################
############ STEP 2 Set Extra Data J1, J2 League From 1993 ############
#######################################################################
#######################################################################


def set_extra_j1_j2_data(df):
    # must add extra data creating scripts
    """
    This function merges the crawled data of
    match data, and stadium capacities, etc into the main df.
    Since the dataset offered by Signate only includes data of J1
    league and from 2004, we must merge extra
    (j2 leage data, and data from 1993)
    into the main df to use them as training dataset.
    """
    ex = pd.read_csv('ex_total.csv')
    ex.id = pd.to_numeric(ex.id, errors='coerce')
    tidying_venue_name(ex)
    tidying_team_name(ex)
    # load the extra data!
    # ex_total.csv consists of data which its form is equivalent to the
    # data given by Signate.

    games_in_comp_dataset = df.id.unique().tolist()
    games_not_in_set = ex[~(ex.id.isin(games_in_comp_dataset))]

    cap_not_done = pd.concat([df, games_not_in_set], ignore_index=True)

    ex_cap = pd.read_csv('ex_stadium_capacity_mapping.csv')
    ex_cap.rename(columns={'stadium': 'venue'}, inplace=True)
    # load the stadium and its capacity data for extra data!
    # ex) j1, j2 leagues and data from 1993...

    capacity = pd.read_csv('stadium_capacity_mapping.csv')
    capacity.rename(columns={'stadium': 'venue'}, inplace=True)
    # load the stadium and capacity data for original dataset

    venue_same = []
    for x in ex_cap.venue.unique():
        if x in capacity.venue.unique():
            venue_same.append(x)
    # get the overlapping stadium data from original data and extra
    # capacity data

    venue_same_null = ex_cap.loc[(ex_cap.venue.isin(venue_same))
                                 & (ex_cap.capacity.isnull())].venue.unique()
    """
    find the stadiums with null capacity values in the extra_capacity
    data and the data inside the original capacity data. By this way
    we can fill the missing capacity data for extra_capacity through
    the already existing original data
    """

    stad_to_fill = ex_cap.loc[ex_cap.venue.isin(
        venue_same_null)].sort_values(by='venue')

    values_to_fill = capacity.loc[capacity.venue.isin(
        venue_same_null)].sort_values(by='venue')['capacity'].tolist()
    # find the data in the original capacity data to fill in the extra
    # capacity data

    stad_to_fill['capacity'] = values_to_fill
    # fill in the data!

    ex_cap.drop(index=stad_to_fill.index, inplace=True)
    ex_cap_done = pd.concat([ex_cap, stad_to_fill], ignore_index=True)
    # merge the filled in data into the extra_capacity data
    # however, we still have missing values!

    capacity_null = [15454, 35000, 28000, 27495, 14051, 15353, 12000, 9245,
                     21053, 10081, 47000, 20125, 47000, 19694, 30000, 20010,
                     28000, 49970, 30132, 24130, 5000, 17200, 5500, 11105,
                     17200, 15000, 10000, 7000, 5000, 7500, 20000, 47000,
                     26109, 4300]

    ex_cap_done.loc[ex_cap_done.capacity.isnull(), 'capacity'] = capacity_null
    # fill in the venue of no capacity data with searched data

    tidying_venue_name(ex_cap_done)
    ex_cap_done = ex_cap_done.drop_duplicates()
    # drop duplicate data just to make sure!

    ex_cap_done.loc[ex_cap_done.venue == '宮城陸上競技場', 'capacity'] = 30000
    # change capacity for a particular stadium

    ex_cap_dict = dict(zip(ex_cap_done.venue.values,
                           ex_cap_done.capacity.values))
    # create a dictionary with the stadium: capacity values to fill in
    # our original cap_not_done df!

    cap_not_done.loc[
        (cap_not_done.capacity.isnull())
        & (cap_not_done.venue.notnull()),
        'capacity'] = \
        cap_not_done.loc[
            (cap_not_done.capacity.isnull())
            & (cap_not_done.venue.notnull()), 'venue'
    ].apply(lambda x: ex_cap_dict[x])
    # Fill in the capacity of venues with data from ex_cap_done
    # dataframe.

    cap_not_done['division'] = cap_not_done['division'].fillna(1)

    df = cap_not_done.copy()

    return df


#######################################################################
#######################################################################
###################### STEP 3 Setting Datetime Features ###############
#######################################################################
#######################################################################

def set_datetime(df):

    add_datepart(df, 'match_date', drop=False)
    # call the date_part function! to set date variables
    df['match_Is_Weekend'] = (df['match_date'].dt.weekday >= 5)
    df['match_WeekofYear'] = df['match_date'].dt.weekofyear

    df['match_date'] = df['match_date'].astype(np.str)
    df['match_Year'] = df['match_date'].apply(
        lambda x: int(x.split('-')[0]))
    # set the year properly, get the first four numbers and set them
    # as year values
    return df


#######################################################################
#######################################################################
############ STEP 4 Adding Area Name and Code to Stadiums #### ########
#######################################################################
#######################################################################

def get_stadium_area_geoinfo(df, gmaps):
    # Collect the name of the unique stadiums
    unique_stadium = combine['stadium'].dropna().unique()

    # We're going to save the geo-info of the each stadium
    # in this list by crwaling from google maps
    stadium_area_dict = {}

    for stadium in unique_stadium:
        geo_info = gmaps.geocode(address=stadium, language='jp')
        try:
            stadium_area = geo_info[0]['formatted_address']
        except:
            pass
        stadium_area_dict[stadium] = stadium_area

    pd.DataFrame(data=list(stadium_area_dict.items()),
                 columns=['stadium', 'geo_info'])

    return stadium_area_df

    ######----Done by this line------ ######

    def extract_area_name(row):
        pat_normal = re.compile('(\w+ *\w*) \d{3}-\d{4}')
        pat_abnormal = re.compile('〒\d{3}-\d{4} (\w+(-ken|県$| *\w+))')

        address = row['venue_area']

        try:
            area_name1 = pat_abnormal.search(address).groups(0)[0]
            try:

                area_name1_done = re.search('\w+(県)', area_name1)[0]
                return area_name1_done
            except:
                return area_name1
        except:
            area_name2 = pat_normal.search(address).groups(1)[0]
            return area_name2

    # Using area name - area code data

    pref_code = pd.read_csv('prefecture_code.csv')
    pref_code = pref_code[['AREA', 'AREA Code']].drop_duplicates()
    pref_code.columns = ['area', 'area_code']

    venue_area_df['venue_area'] = venue_area_df.apply(
        extract_area_name, axis=1)

    # Cleaning area name got from google api to fit in the format of
    # prefecture_code.csv

    def tidying_area_name(row):
        kanji_name = ['愛知県', '三重県', '富山市 富山県', '福島県', '兵庫県', '宮城県']
        translated_name = ['Aichi-ken', 'Mie-ken', 'Toyama-ken',
                           'Fukushima-ken', 'Hyogo-ken', 'Miyagi-ken']
        kanji_converter = dict(zip(kanji_name, translated_name))

        name = row['venue_area']

        if name.endswith('-ken'):
            return name
        elif name == 'Osaka Prefecture':
            return 'Osaka-fu'
        elif name == 'Ōita Prefecture':
            return 'Oita-ken'
        elif name == 'Kyoto Prefecture':
            return 'Kyoto-fu'
        elif name == 'Hokkaido':
            return 'Hokkaido'
        elif name == 'Hyōgo Prefecture':
            return 'Hyogo-ken'
        elif name == 'Kōchi Prefecture':
            return 'Kochi-ken'
        elif name == 'Gunma Prefecture':
            return 'Gumma-ken'
        elif name == 'Tokyo':
            return 'Tokyo-to'
        elif name in kanji_name:
            return kanji_converter[name]
        else:
            return name.split(' ')[0] + '-ken'

    venue_area_df['venue_area'] = venue_area_df.apply(
        tidying_area_name, axis=1)

    ven_area_code_name = venue_area_df.merge(
        pref_code, how='left', left_on='venue_area', right_on='area')
    del ven_area_code_name['area']
    ven_area_code_name.rename(
        columns={'area_code': 'venue_area_code'}, inplace=True)

    df_done = df.merge(right=ven_area_code_name, how='left', on=['venue'])

    return df_done

#######################################################################
#######################################################################
############ STEP 5 Set Area Info of both Home and Away Teams #########
#######################################################################
#######################################################################


def get_area_info_of_team(df):

    # Load area code dataset
    prefecture_code = pd.read_csv('prefecture_code.csv', names=[
                                  'year', 'code', 'area', 'pop'])
    prefecture_code = prefecture_code[['code', 'area']].drop_duplicates()
    prefecture_code['code'] = pd.to_numeric(
        prefecture_code['code'], errors='coerce')
    prefecture_code = prefecture_code.dropna()

    # Attach area code to each team: There was no other way but
    # attaching by myself.
    unique_teams = [
        '東京Ｖ', 'FC東京', '町田', '横浜FM-F', '横浜FM-M', '湘南', '横浜FM', '川崎F', '横浜FC',
        '金沢', '広島', '鹿島', '水戸', 'G大阪', 'C大阪', '清水', '磐田', '浦和', '大宮', '市原',
        '柏', '千葉', '岐阜', '名古屋', '福岡', '北九州', '京都', '神戸', '札幌', '仙台', '大分',
        '新潟', '甲府', '山形', '鳥栖', '徳島', '松本', '長崎', '群馬', '愛媛', '熊本', '栃木',
        '岡山', '富山', '鳥取', '讃岐', '山口'
    ]

    area_code = [
        13000.0, 13000.0, 13000.0, 14000.0, 14000.0, 14000.0, 14000.0, 14000.0, 14000.0,
        14000.0, 34000.0, 8000.0, 8000.0, 27000.0, 27000.0, 22000.0, 22000.0, 11000.0,
        11000.0, 12000.0, 12000.0, 12000.0, 12000.0, 23000.0, 40000.0, 40000.0, 26000.0,
        28000.0, 1000.0, 4000.0, 44000.0, 15000.0, 19000.0, 6000.0, 41000.0, 36000.0,
        20000.0, 42000.0, 10000.0, 38000.0, 43000.0, 9000.0, 33000.0, 16000.0, 31000.0,
        37000.0, 35000.0
    ]

    soccer_team_area_code = pd.DataFrame(
        data={'team': unique_teams, 'code': area_code})

    # Data set: Team name - area code - area name
    soccer_team_area_code = soccer_team_area_code.merge(
        prefecture_code, how='left', on='code')

    same_team_dict = {
        'C大阪': 'Ｃ大阪', 'G大阪': 'Ｇ大阪', '川崎F': '川崎Ｆ', 'FC東京': 'FC東京',
        '東京Ｖ': '東京Ｖ', '横浜FC': '横浜FC', '横浜FM': '横浜FM', '横浜FM-F': '横浜Ｆ',
        '横浜FM-M': '横浜M', 'Ｖ川崎': '東京Ｖ', 'Ｆ東京': 'FC東京', '草津': '群馬',
        '平塚': '湘南'
    }

    # Unificate teams' name for better prediction.
    soccer_team_area_code['team'] = soccer_team_area_code[
        'team'].replace(to_replace=same_team_dict)

    # Attaching prefecture information about home team and away team
    # in main dataset.
    home_added = df.merge(
        right=soccer_team_area_code, how='left', left_on=['home_team'],
        right_on=['team']).drop(['team'], 1).rename(
            columns={'code': 'home_team_area_code', 'area': 'home_team_area'}
    )
    df_done = home_added.merge(
        right=soccer_team_area_code, how='left', left_on=['away_team'],
        right_on=['team']).drop(['team'], 1).rename(
            columns={'code': 'away_team_area_code', 'area': 'away_team_area'}
    )

    return df_done

#######################################################################
#######################################################################
########################STEP 7 Distance Btwn Teams, ###################
################# Duration Btwn Games For Baseball and Soccer #########
#######################################################################


def get_location_distance_duration(df, google_map_key):

    key = google_map_key
    gmaps = googlemaps.Client(key=key)
    # Soccer
    # DIstance and duration between teams
    # - Distance and duration from the prefecture of away team to
    # stadium where game holds
    unique_match = df[['venue', 'away_team_area']].drop_duplicates()[
        df.venue.notnull()]

    departures = unique_match.away_team_area.values
    arrivals = unique_match.venue.values

    durations = []
    distances = []

    for dep, arr in zip(departures, arrivals):
        if dep == 'Chiba-ken':
            dep = '千葉県'

        raw_data = gmaps.distance_matrix(dep, arr, language='jp')

        dist = raw_data['rows'][0]['elements'][0]['distance']['value'] / 1000
        distances.append(dist)

        duration = float(raw_data['rows'][0]['elements'][
                         0]['duration']['value']) / 60
        if type(duration) == float:
            durations.append(duration)
        else:
            durations.append(NaN)

    football_duration_distance = pd.DataFrame(
        data={
            'venue': arrivals, 'away_team_area': departures,
            'duration': durations, 'distance': distances
        }
    )

    football_duration_distance = football_duration_distance.drop_duplicates()
    football_duration_distance.reset_index(inplace=True)
    del football_duration_distance['index']

    durat_dist_added = df.merge(
        right=football_duration_distance,
        on=['venue', 'away_team_area']
    )

    durat_dist_added.rename(
        columns={
            'duration': 'duration_bet_teams',
            'distance': 'dist_bet_teams'
        }, inplace=True
    )

    d7_temp = durat_dist_added.copy()

    # Adding latitude and longitude information about venues.
    def find_lat(row, var_name, key):

        stadium = row[var_name]

        # read the raw data using geocoding api, provided by Google
        gmaps = googlemaps.Client(key=key)
        raw_data = gmaps.geocode(address=stadium, language='jp')[0]

        # Extracting latitude data and adding it to dataframe
        lat = raw_data['geometry']['location']['lat']
        return lat

    def find_lon(row, var_name, key):

        stadium = row[var_name]

        # read the raw data using geocoding api, provided by Google
        gmaps = googlemaps.Client(key=key)
        raw_data = gmaps.geocode(address=stadium, language='jp')[0]

        # Extracting latitude data and adding it to dataframe
        global lon
        lon = raw_data['geometry']['location']['lng']
        return lon

    d7_temp['lat'] = 0
    d7_temp['lon'] = 0

    venues_loc = d7_temp[['venue', 'lat', 'lon']].drop_duplicates()
    venues_loc['lat'] = venues_loc.apply(
        find_lat, axis=1, var_name='venue', key=key)
    venues_loc['lon'] = venues_loc.apply(
        find_lon, axis=1, var_name='venue', key=key)
    del d7_temp['lat'], d7_temp['lon']

    d7_soccer = d7_temp.merge(right=venues_loc, how='left', on='venue')

    # Location data about baseball venues.
    d7_soccer['lat_b'] = 0
    d7_soccer['lon_b'] = 0

    venues_loc_b = d7_soccer[
        ['venue_b_1', 'lat_b', 'lon_b']].drop_duplicates().dropna()
    venues_loc_b.rename(columns={'venue_b_1': 'venue_b'}, inplace=True)

    del d7_soccer['lat_b'], d7_soccer['lon_b']

    venues_loc_b['lat_b'] = venues_loc_b.apply(
        find_lat, axis=1, var_name='venue_b', key=key)
    venues_loc_b['lon_b'] = venues_loc_b.apply(
        find_lon, axis=1, var_name='venue_b', key=key)

    d7_temp1 = d7_soccer.merge(
        right=venues_loc_b, how='left',
        left_on='venue_b_1', right_on='venue_b'
    )

    del d7_temp1['venue_b']

    d7_temp1.rename(
        columns={'lat_b': 'lat_b_1', 'lon_b': 'lon_b_1'},
        inplace=True
    )

    d7_temp1 = d7_temp1.merge(
        right=venues_loc_b, how='left',
        left_on='venue_b_2', right_on='venue_b'
    )

    del d7_temp1['venue_b']
    d7_temp1.rename(
        columns={'lat_b': 'lat_b_2', 'lon_b': 'lon_b_2'},
        inplace=True
    )

    df_done = d7_temp1.copy().sort_values(by='id')

    return df_done


#######################################################################
#######################################################################
################### STEP 8 Add and Set Holiday Data ###################
#######################################################################
#######################################################################

# Load holiday dataset
# Data source:
# http://zangyoukeisan.cocolog-nifty.com/blog/2011/09/
# post-b23f.html
holiday = pd.read_excel('holiday_extra.xls', sheet_name='振替休日あり')


def add_holiday_feauture(df):
    # Preprcoess the holiday dataset
    holiday = holiday[['年月日', '祝日名']]
    holiday.columns = ['holiday_date', 'description']
    holiday = holiday[holiday['holiday_date'] > '1992-12-21']

    # Make a dataframe date_df which
    # records whether the day is weekend or not ,
    # holiday or not from 1992-12-21 to 2018-12-31
    date_df = pd.date_range(start='1992-12-21', end='2018-12-31', freq='D')
    date_df = pd.DataFrame(date_df, columns=['date'])

    date_df['DayOfWeek'] = date_df['date'].dt.weekday
    date_df['is_Weekend'] = (date_df['DayOfWeek'] >= 5).astype(np.int8)

    date_df = date_df.merge(holiday, left_on='date',
                            right_on='holiday_date', how='left')
    date_df.drop('holiday_date', 1, inplace=True)

    date_df['is_holiday'] = date_df['description'].notnull().astype(np.int8)
    date_df['is_dayoff'] = date_df['is_Weekend'] + date_df['is_holiday']

<< << << < HEAD
# Use get_elapsed function from util.py
== == == =
# Use get_elapsed function from util.py
>>>>>> > 03c9dbdd547b66098ac87c9ef8283fe32fc6fe2e
# and calculate how many days left before next day off and
# how many days passed from the last day off
    fld = 'is_dayoff'
    get_elapsed(fld, 'After')
    date_df = date_df.sort_values('date', ascending=False)
    get_elapsed(fld, 'Before')

    # Calculate how many days left before next holiday and
    # how many days passed from the last holiday
    date_df = date_df.sort_values('date')

    fld = 'is_holiday'
    get_elapsed(fld, 'After')
    date_df = date_df.sort_values('date', ascending=False)
    get_elapsed(fld, 'Before')

    date_df = date_df.sort_values('date')

    df = df.merge(date_df, left_on='match_date', right_on='date', how='left')
    df = preprocess(df)

    return df

combine = add_holiday_feauture(combine, holiday)
#######################################################################
#######################################################################
###### STEP 9 Simple and Exponentially Weighted Win Ratio For Teams ###
#######################################################################
#######################################################################


def set_th_match(df):
    """
    In some years, sections of teams are not ordered in time sequence.
    This means that sometimes even on game is held later than other games,
    that get earlier section number than those matches.
    This might happened because of delay of game for some reason,
    such as earthquakes.
    This causes problem when calculating stats like win ratio,so we
    created new section for home team and away team in time sequence.
    """

    board1 = df[
        ['match_Year', 'match_date', 'home_team']
    ].rename(columns={'home_team': 'team'})

    board2 = df[
        ['match_Year', 'match_date', 'away_team']
    ].rename(columns={'away_team': 'team'})

    board1['home_game'] = 1
    board2['away_game'] = 1

    board = pd.concat([board1, board2])
    board.sort_values(by='match_date', inplace=True)
    board = reset_index(board)

    """
    we had two columns which consists data about the team of the match.
    away_team and home_team. However we set them into the same column
    and delete the away_team and home_team columns
    """

    board['home_game'] = board['home_game'].fillna(0)
    board['away_game'] = board['away_game'].fillna(0)

    new_dataset = pd.DataFrame()

    start_year, end_year = board['match_Year'].min(), board['match_Year'].max()

    for year in tqdm_notebook(range(start_year, end_year + 1)):

        year_df = board[board['match_Year'] == year].copy()

        unique_team = np.unique(year_df['team'])

        for team in tqdm_notebook(unique_team):
            temp = year_df[(year_df['match_Year'] == year)
                           & (year_df['team'] == team)]

            for i, row in temp.iterrows():
                match_date = row['match_date']
                temp.loc[i, 'th_game'] = int(
                    len(temp[temp['match_date'] < match_date])) + 1

            new_dataset = pd.concat([new_dataset, temp])

    # for all years in the dataset, count how many games a team had
    # before each game put them in the 'section' column

    # Make the home_th_game column
    drop_cols = ['away_game', 'home_game', 'team']

    df = df.merge(new_dataset[new_dataset['home_game'] == 1].drop('match_Year', 1),
                  left_on=['match_date', 'home_team'],
                  right_on=['match_date', 'team'], how='left')

    df.drop(drop_cols, 1, inplace=True)
    df.rename(columns={'th_game': 'home_th_game'}, inplace=True)

    # Make the away_th_game column
    df = df.merge(new_dataset[new_dataset['away_game'] == 1].drop('match_Year', 1),
                  left_on=['match_date', 'away_team'],
                  right_on=['match_date', 'team'], how='left')

    df.drop(drop_cols, 1, inplace=True)
    df.rename(columns={'th_game': 'away_th_game'}, inplace=True)

    return df

# Run set_th_match
combine = set_th_match(combine)


def set_exp_win_ratio(df):
    """
    Warning!!! THIS FUNCTION IS EXTREMELY SLOW!!!!
    This function gets the exponentiall weighted win ratio for a
    given team.
    The code is extremely long thus I believe it deserves some
    explanation
    """

    in_match = pd.read_csv('match_reports.csv')
    out_match = pd.read_csv('ex_match_reports.csv')

    match = pd.concat([in_match, out_match])
    match = match.drop_duplicates(subset='id').sort_values(by='id')
    # load the match result data to find which team won and lost

    def df_of_prev_games_team_for_weighted(match_date,
                                           division,
                                           team,
                                           max_num_of_prev_games_to_consider,
                                           match_year,
                                           df_of_match_result,
                                           df_of_game_sections):

        df_of_prev_games = df_of_game_sections[
            (df_of_game_sections['division'] == division)
            & (df_of_game_sections['match_Year'] == match_year)
            & (df_of_game_sections['match_date'] < match_date)
        ]

        df_of_prev_games_with_match_results = df_of_prev_games[
            (df_of_prev_games['home_team'] == team)
            | (df_of_prev_games['away_team'] == team)
        ].sort_values(by=['match_date']).tail(max_num_of_prev_games_to_consider)

        df_of_prev_games_with_match_results = \
            df_of_prev_games_with_match_results.merge(
                df_of_match_result, on='id', how='left')

        df_of_prev_games_with_match_results = \
            df_of_prev_games_with_match_results.sort_values(
                by=['match_date'], ascending=False)

        return df_of_prev_games_with_match_results

    def result_sorter_for_win_weight_calc(df, team):
        """
        will spit out a list [game(t-1), game(t-2), game(t-3)...]
        1 = win, 0 = draw, loss = -1
        """
        return_list = list()
        for i in range(len(df)):

            row = df.iloc[i]
            # reverse the row sequence so that the first item in the
            # list is the most recent
            if row.home_team_score > row.away_team_score:
                val = 1 if row.home_team == team else -1
                return_list.append(val)
            elif row.home_team_score < row.away_team_score:
                val = -1 if row.home_team == team else 1
                return_list.append(val)
            elif row.home_team_score == row.away_team_score:
                return_list.append(0)
            else:
                return_list = False
                return return_list
                # do note that this will run due to the test set which
                # does not have any value yet
        return return_list

    def weighted_win_point_calc(df, team, weight):
        sorted_list = result_sorter_for_win_weight_calc(df=df, team=team)
        weighted_win_score = 0
        num = 0
        if sorted_list != False:
            for val in sorted_list:
                weighted_win_score += float(val / math.exp(num * weight))
                num += 1
        else:
            return None
        return weighted_win_score

    def weighted_exp_win_for_team(row, home_or_away,
                                  df_of_match_result,
                                  df_of_game_sections,
                                  max_num_of_prev_games_to_consider,
                                  weight):

        if home_or_away == 'home':
            team = row['home_team']
        elif home_or_away == 'away':
            team = row['away_team']
        else:
            raise AttributeError("set home_or_away to 'home' or 'away'")
        # df of match result will be the df which consists of all
        # match results.
        # df of game sections will be the main df

        # get the win-ratio for both home and away teams in a given
        # game
        match_year = row['match_Year']
        match_date = row['match_date']
        division = row['division']

        df_for_weighted_win = df_of_prev_games_team_for_weighted(
            match_date=match_date,
            division=division,
            team=team,
            max_num_of_prev_games_to_consider=max_num_of_prev_games_to_consider,
            match_year=match_year,
            df_of_match_result=df_of_match_result,
            df_of_game_sections=df_of_game_sections
        )

        win_weight_for_team = weighted_win_point_calc(
            df=df_for_weighted_win, team=team, weight=weight)

        return win_weight_for_team

    # Finally get the exponentially weighted win ratio for both home
    # and away teams
    # It is a PEP8 Nightmare here!!!
    df['home_team_weighted_win_3Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='home', weight=0.227,
            max_num_of_prev_games_to_consider=3,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    df['away_team_weighted_win_3Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='away', weight=0.227,
            max_num_of_prev_games_to_consider=3,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    print("Exp Weighted Win Ratio for 3G is DONE!!")

    df['home_team_weighted_win_5Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='home', weight=0.227,
            max_num_of_prev_games_to_consider=5,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    df['away_team_weighted_win_5Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='away', weight=0.227,
            max_num_of_prev_games_to_consider=5,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    print("Exp Weighted Win Ratio for 5G is DONE!!")

    df['home_weighted_win_7Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='home', weight=0.227,
            max_num_of_prev_games_to_consider=7,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    df['away_weighted_win_7Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='away', weight=0.227,
            max_num_of_prev_games_to_consider=7,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    print("Exp Weighted Win Ratio for 7G is DONE!!")

    df['home_weighted_win_9Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='home', weight=0.227,
            max_num_of_prev_games_to_consider=9,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )

    df['away_weighted_win_9Games_for_weight_0.227'] = \
        df.apply(lambda row: weighted_exp_win_for_team(
            row=row, home_or_away='away', weight=0.227,
            max_num_of_prev_games_to_consider=9,
            df_of_match_result=match, df_of_game_sections=df
        ), axis=1
    )
    print("Exp Weighted Win Ratio for 9G is DONE!!")

    df = df.merge(match, how='left', on='id')

    return df


#######################################################################
#######################################################################
################### STEP 11 Set Population data of  ###################
############# prefuctures which Stadiums are located on ###############
#######################################################################

# Load populdation data of Japan
population = pd.read_csv('../add_extra_data/Perfect_population.csv',
                         usecols=[
                             'SURVEY YEAR', 'AREA Code', 'AREA',
                             'A1101_Total population (Both sexes)[person]'
                         ]
                         )


def preprocess_population(df):
    # Get only population data from 1992 and set the minto integers
    # and then preprocess the population data!
    df = df[df['SURVEY YEAR'] >= 1992]
    df.columns = ['year', 'area_code', 'area', 'population']
    df['population'] = df['population'].apply(
        lambda x: re.sub(',', '', x))
    df['population'] = df['population'].astype(np.int)
    df = reset_index(df)
    return df

# Since we statistics japan does not yet offer data for
# year 2016 and 2018 we guess the population for the year!


def caculate_change_ratio(df, year):
    assert type(year) == int

    change_ratio = 1 + (
        df[
            (df['area'] == 'All Japan') & (df['year'] == year + 1)
        ]['population'].values[0]
        - df[(df['area'] == 'All Japan')
             & (df['year'] == year)]['population'].values[0]
    ) / df[
        (df['area'] == 'All Japan')
        & (df['year'] == year)
    ]['population'].values[0]

    last_index = len(df)

    return change_ratio, last_index


def insert_unrecored_population_value(df, year, change_ratio, last_index):
    for ind, x in df[df['year'] == year + 1].iterrows():

        vals = x['year'] + \
            1, x['area_code'], x['area'], x['population'] * change_ratio

        df.loc[last_index] = vals
        last_index += 1

    return df


def get_nextyear_population(df, year):
    change_ratio, last_index = caculate_change_ratio(df, year)
    df = insert_unrecored_population_value(df, year, change_ratio, last_index)
    return df


def make_population_data(df):
    """
    Set population data for each stadiums to get how many
    people are living in the prefucture which the stadium
    is located at
    """

    df = preprocess_population(df)
    df = get_nextyear_population(df, 2015)
    df = get_nextyear_population(df, 2016)

    return df


def merge_with_population_data(main_df, right_df):
    df = pd.merge(main_df, right_df, left_on=[
                  'match_Year', 'stadium_area_code'], right_on=['year', 'area_code'], how='left')
    df = df.drop(['year', 'area', 'area_code'], 1)
    return df

population = make_population_data(population)
combine = merge_with_population_data(combine, population)
#######################################################################
#######################################################################
############## STEP 11 Find Team's League 1 and 2 Years ago ###########
#######################################################################
#######################################################################


def set_team_league_previous_years(df):
    unique_team = np.unique(df['home_team'])

    main_j1 = df[df['division'] == 1]
    main_j2 = df[df['division'] == 2]

    # Whether the team joined J1 League & J2 League per year

    team_year_league_dict = {}

    for team in unique_team:
        parcitipated_year = np.unique(
            df.loc[df['home_team'] == team, 'match_Year'])
        team_year_league_dict[team] = parcitipated_year.tolist()

    # Which league each team joined the last year
    # whether it's J1 League or J2 League or didn't exist(noted as 999).

    previous_participated = {}

    for team in unique_team:

        temp_list = []

        for year in range(1993, 2019):
            temp_j1 = main_j1[(main_j1['match_Year'] == year)]
            temp_j2 = main_j2[(main_j2['match_Year'] == year)]

            if len(temp_j1[temp_j1['home_team'] == team]) > 0:
                temp_list.append(1)

            elif len(temp_j2[temp_j2['home_team'] == team]) > 0:
                temp_list.append(2)

            else:
                temp_list.append(999)

        previous_participated[team] = temp_list

    for year in range(1994, 2019):
        df.loc[df['match_Year'] == year, 'home_team_LastYear_league'] = \
            df.loc[df['match_Year'] == year, 'home_team'].apply(
                lambda x: previous_participated[x][year - 1994
                                                   ]
        )
        df.loc[df['match_Year'] == year, 'away_team_LastYear_league'] = \
            df.loc[df['match_Year'] == year, 'away_team'].apply(
                lambda x: previous_participated[x][year - 1994
                                                   ]
        )

    for year in range(1995, 2019):
        df.loc[df['match_Year'] == year, 'home_team_TwoYearsAgo_league'] = \
            df.loc[df['match_Year'] == year, 'home_team'].apply(
            lambda x: previous_participated[x][year - 1995
                                               ]
        )
        df.loc[df['match_Year'] == year, 'away_team_TwoYearsAgo_league'] = \
            df.loc[df['match_Year'] == year, 'away_team'].apply(
            lambda x: previous_participated[x][year - 1995
                                               ]
        )

    for year in range(1994, 2019):
        df.loc[df['match_Year'] == year, 'home_team_LastYear_league'] = \
            df.loc[df['match_Year'] == year, 'home_team'].apply(
            lambda x: previous_participated[x][year - 1994
                                               ]
        )
        df.loc[df['match_Year'] == year, 'away_team_LastYear_league'] = \
            df.loc[df['match_Year'] == year, 'away_team'].apply(
            lambda x: previous_participated[x][year - 1994
                                               ]
        )

    for year in range(1995, 2019):
        df.loc[df['match_Year'] == year, 'home_team_TwoYearsAgo_league'] = \
            df.loc[df['match_Year'] == year, 'home_team'].apply(
            lambda x: previous_participated[x][year - 1995
                                               ]
        )
        df.loc[df['match_Year'] == year, 'away_team_TwoYearsAgo_league'] = \
            df.loc[df['match_Year'] == year, 'away_team'].apply(
            lambda x: previous_participated[x][year - 1995
                                               ]
        )

    df['Sum_LastYear_team_league'] = \
        df['home_team_LastYear_league'] + df['away_team_LastYear_league']
    df['home_away_Lastyear_Comb'] = \
        df['home_team_LastYear_league'].astype(np.str) \
        + '_' + df['away_team_LastYear_league'].astype(np.str)

    df['Sum_twoyearsago_team_league'] = \
        df['home_team_TwoYearsAgo_league'] + \
        df['away_team_TwoYearsAgo_league']
    df['home_away_twoyearsago_Comb'] = \
        df['home_team_TwoYearsAgo_league'].astype(np.str) \
        + '_' + df['away_team_TwoYearsAgo_league'].astype(np.str)

    return df


#######################################################################
#######################################################################
################## STEP 12 Process the Weather Data ###################
#######################################################################
#######################################################################

def set_process_weather_data(df):

    # Divide kick off time to three time zone.
    # This is for matching the weather of time zone when game was hold
    # to each game.
    # For example, if game was started at 5pm, the weather of 12~18pm
    # will be put to weather columns of that game.
    def kick_off_hour_division(var):
        if var < 12:
            return '0712'
        elif var < 18:
            return '1218'
        else:
            return '1824'

    df['kick_off_hour'] = df['kick_off_time'].apply(
        lambda x: x.split(':')[0]).astype(np.int)
    df['kick_off_time_division'] = df[
        'kick_off_hour'].apply(kick_off_hour_division)

    # Divide weather to fit in three time zones.
    # If the weather forcast is only '雨', All dummy variable columns
    # to show whether it was rainy or not at that time will be 1.

    wea_all_y = df.loc[df.weather.notnull()]
    wet_all_n = df.loc[df.weather.isnull()]
    wea_index = wea_all_y.index

    # 'wea1のちwea2': 0712 time zone will get 1 for column about wea1,
    # 1218 & 1824 time zone will get 1 for columns about wea2.
    # 'wea1のちwea2のちwea3': wea1, 2 and 3 will be distributed to
    # three time zones respectively in time sequence.

    wea_val = wea_all_y.weather.tolist()

    wea_val = [x.split('のち') for x in wea_val]

    type_wea = dict()
    for x in wea_val:
        t = type(x)
        if t not in type_wea:
            type_wea[t] = 1
        else:

            type_wea[t] += 1

    wea_val_done = []

    for x in wea_val:
        if len(x) == 1:
            done = [x[0], x[0], x[0]]
            wea_val_done.append(done)
        elif len(x) == 2:
            done = [x[0], x[1], x[1]]
            wea_val_done.append(done)
        elif len(x) == 3:
            wea_val_done.append(x)

    wea_val_0712 = [x[0] for x in wea_val_done]
    wea_val_1218 = [x[1] for x in wea_val_done]
    wea_val_1824 = [x[2] for x in wea_val_done]

    wea_all_y.loc[wea_all_y.index.isin(wea_index), '0712'] = wea_val_0712
    wea_all_y.loc[wea_all_y.index.isin(wea_index), '1218'] = wea_val_1218
    wea_all_y.loc[wea_all_y.index.isin(wea_index), '1824'] = wea_val_1824

    sunny_wea = ['sunny_0712', 'sunny_1218', 'sunny_1824']

    for x in sunny_wea:
        wea_all_y[x] = wea_all_y[x.split('_')[1]]
        wea_all_y[x] = wea_all_y[x].apply(lambda x: x if '晴' in x else 0)

    cloudy_wea = ['cloudy_0712', 'cloudy_1218', 'cloudy_1824']

    for x in cloudy_wea:
        wea_all_y[x] = wea_all_y[x.split('_')[1]]
        wea_all_y[x] = wea_all_y[x].apply(lambda x: x if '曇' in x else 0)

    rainy_wea = ['rainy_0712', 'rainy_1218', 'rainy_1824']

    for x in rainy_wea:
        wea_all_y[x] = wea_all_y[x.split('_')[1]]
        wea_all_y[x] = wea_all_y[x].apply(lambda x: x if '雨' in x else 0)
    # wea_all_y[x] = wea_all_y[x].apply(lambda x: x if  in x else 0)

    inside_wea = ['inside_0712', 'inside_1218', 'inside_1824']

    for x in inside_wea:
        wea_all_y[x] = wea_all_y[x.split('_')[1]]
        wea_all_y[x] = wea_all_y[x].apply(lambda x: x if '屋内' in x else 0)

    snowy_wea = ['snowy_0712', 'snowy_1218', 'snowy_1824']

    for x in snowy_wea:
        wea_all_y[x] = wea_all_y[x.split('_')[1]]
        wea_all_y[x] = wea_all_y[x].apply(lambda x: x if '雪' in x else 0)

    # we made an index about how strong each weather is.
    # Ex1. wea1一時wea2: wea 2 will get 0.125(mean of 0.00~0.25) and
    # wea1 will get remaining 0.875.
    # Ex2. wea1時々wea2: wea 2 will get 0.375(mean of 0.25~0.50) and
    # wea1 will get remaining 0.675
    # Mix of '一時' and '時々' in one forecast was also calulated in
    # same criteria.

    rainy_wea_converter = {
        '雨': 1, '曇時々雨': 0.375, '曇一時雨': 0.125, '雨時々曇': 0.625, '晴一時雨': 0.125,
        '雨一時曇': 0.875, '雨一時雷雨': 1, '曇一時雷雨': 0.125, '雨時々晴': 0.625,
        '曇時々晴一時雨': 0.125, '晴時々雨': 0.375, '晴時々曇一時雨': 0.125, '雷雨': 1,
        '雨一時雷': 1, 0: 0, '曇時々雨一時晴': 0.375, '晴時々曇時々雨': 0.3, '曇一時晴一時雨': 0.125,
        '晴一時雷雨': 0.125, '雪時々雨': 0.375, '雨時々雪': 0.625, '雨一時霧': 0.875
    }

    cloudy_wea_converter = {
        '曇': 1, '曇時々雨': 0.625, '曇時々晴': 0.625, '晴時々曇': 0.325, '曇一時雨': 0.875,
        '雨時々曇': 0.375, '曇一時雷雨': 0.875, '曇時々晴一時雨': 0.5, '晴時々曇一時雨': 0.375,
        '曇晴': 0.5, '曇時々雪一時晴': 0.5, '曇時々雪時々晴': 0.4, '曇時々雪': 0.625,
        '曇一時晴 ': 0.875, 0: 0, '雨一時曇': 0.125, '晴一時曇': 0.125, '曇一時晴': 0.875,
        '曇時々雨一時晴': 0.5, '晴時々曇時々雨': 0.3,  '曇一時晴一時雨': 0.75, '雨一時霧': 0.125,
        '雪一時曇': 0.125, '曇一時雪': 0.875
    }

    sunny_wea_converter = {
        '晴': 1, '曇時々晴': 0.375, '晴時々曇': 0.625, '晴一時雨': 0.875, '晴一時曇': 0.875,
        '曇一時晴': 0.125, '晴時々雪': 0.625, '曇時々雪一時晴': 0.125, '雨時々晴': 0.375,
        '曇時々晴一時雨': 0.375, '曇晴': 0.5, '晴時々雨': 0.625, '曇時々雪時々晴': 0.3,
        '晴時々曇一時雨': 0.5, '曇時々雨一時晴': 0.125, '晴時々曇時々雨': 0.4,  '曇一時晴一時雨': 0.125,
        '晴一時雷雨': 0.875
    }

    inside_wea_converter = {'屋内': 1, 0: 0}

    snowy_wea_converter = {
        '晴時々雪': 0.375, '曇時々雪': 0.375, '雪': 1, '曇時々雪一時晴': 0.375, 0: 0,
        '曇時々雪時々晴': 0.3, '雪時々雨': 0.625, '雨時々雪': 0.375, '雪一時曇': 0.875,
        '曇一時雪': 0.125
    }

    weathers = [sunny_wea, rainy_wea, cloudy_wea, inside_wea, snowy_wea]
    converters = [sunny_wea_converter, rainy_wea_converter,
                  cloudy_wea_converter, inside_wea_converter,
                  snowy_wea_converter]

    for wea, converter in zip(weathers, converters):
        for comp in wea:
            wea_all_y.loc[(wea_all_y[comp] != 0)
                          & (wea_all_y[comp].notnull()), comp] = \
                wea_all_y.loc[
                (wea_all_y[comp] != 0) & (wea_all_y[comp].notnull()), comp
            ].apply(lambda x: converter[x])

    # very few snowy weather: categorize as rainy weather

    for snow, rain in list(zip(snowy_wea, rainy_wea)):
        wea_all_y.loc[(wea_all_y[snow] != 0)
                      & (wea_all_y[snow].notnull()), rain] = \
            wea_all_y.loc[
            (wea_all_y[snow] != 0) & (wea_all_y[snow].notnull()), snow]

    del wea_all_y[snowy_wea[0]], \
        wea_all_y[snowy_wea[1]], \
        wea_all_y[snowy_wea[2]]

    # very foggy weather: categorize as cloudy weather

    wea_all_y.loc[wea_all_y.weather == '霧', 'cloudy_0712'] = 1
    wea_all_y.loc[wea_all_y.weather == '霧', 'cloudy_1218'] = 1
    wea_all_y.loc[wea_all_y.weather == '霧', 'cloudy_1824'] = 1

    weather_index = wea_all_y[
        ['id', 'sunny_0712', 'sunny_1218', 'sunny_1824', 'cloudy_0712',
         'cloudy_1218', 'cloudy_1824', 'rainy_0712', 'rainy_1218',
         'rainy_1824', 'inside_0712', 'inside_1218', 'inside_1824']
    ]

    df1 = df.merge(weather_index, how='left', on='id')

    first_weather = [x for x in list(df1) if '0712' in x]
    second_weather = [x for x in list(df1) if '1218' in x]
    third_weather = [x for x in list(df1) if '1824' in x]

    # We made new columns for 4 categorized weathers.
    # If the game is held at 16pm, we just put weather scores from
    # ~~(weather)_1218 columns to new columns.
    # By doing this, we tried to see the weather of time when the game
    # was done.

    for ind in df.index:

        if df1.loc[ind, 'kick_off_time_division'] == '0712':
            weather_vals = df1.loc[ind, first_weather]

        elif df1.loc[ind, 'kick_off_time_division'] == '1218':
            weather_vals = df1.loc[ind, second_weather]

        else:
            weather_vals = df1.loc[ind, third_weather]

        for col, val in zip(
            ['weather_sunny', 'weather_cloudy',
             'weather_rainy', 'weather_inside'],
                weather_vals.values):

            df1.loc[ind, col] = val

    # If the score (what I made above) for the weather is bigger than
    # 0.8,we thought the weather is dominant at that time zone.
    # We processed like this because if score is less than the
    # threshold(0.8),the effect of weather is obscure so it might
    # induce bad prediction.

    for col in ['weather_sunny', 'weather_cloudy',
                'weather_rainy', 'weather_inside']:
        df1.loc[df1[col] <= 0.8, col] = 0

    # delete weather columns that will not be used anymore.
    weathers = ['sunny', 'cloudy', 'rainy', 'inside']
    morning_wea = [x + '_0712' for x in weathers]
    aftn_wea = [x + '_1218' for x in weathers]
    even_wea = [x + '_1824' for x in weathers]
    wea_drop_col = morning_wea + aftn_wea + even_wea
    for col in wea_drop_col:
        try:
            del df1[col]
        except:
            continue

    df_done = df1.copy()

    return df_done

#######################################################################
#######################################################################
############ STEP 14 Get Weather Oriented Data: Humidexes, etc ########
#######################################################################
#######################################################################


def set_weather_indexes(df):
    # Set humidex and cut them
   # Calculate dew_point and humidity
    def get_dp_hi(temp, humid):
        if (np.isnan(temp) == False) & (np.isnan(humid) == False):
            temp = Temp(temp, 'c')
            dp = dew_point(temperature=temp, humidity=humid)
            hi = heat_index(temperature=temp, humidity=humid)
            return dp.c, hi.c
        else:
            return np.NaN, np.NaN

    # Caculate humidex
    # Formulas is deviced by  J.M. Masterton and F.A. Richardson at
    # AES, 1979. It's a standard for Canada, but variations are
    # used around the world
    def get_humidex(temp, dew):
        if (np.isnan(temp) == False) & (np.isnan(dew) == False):
            first_half = temp
            second_half = \
                0.5555 * (6.11 * math.e ** round(
                    5417.7530 * ((1 / 273.16)
                                 - (1 / (273.15 + dew))), 5
                ) - 10
                )
            return first_half + second_half
        else:
            return np.NaN

    # Categorize humidex values into 4 bins
    # Each subcategory has the following meanings.
        """
	Humidex     :      Degree of Discomfort
	20-29       :      No Discomfort
	30-39       :      Some discomfort
	40-45       :      Great discomfort; avoid exertion
	45+         :      Dangerous; possible heat stroke
	"""
    def humidex_bin(val):
        if np.isnan(val) == True:
            return 'Unknown'
        elif val < 30:
            return 'Low'
        elif val < 40:
            return 'Normal'
        elif val < 45:
            return 'High'
        else:
            return 'Very_High'

    # Make a columns for dew_point, and heat_index
    # from temperature and humidity column
    for ind, row in df[['temperature', 'humidity']].iterrows():
        temp, humid = row.values
        dp, hi = get_dp_hi(temp, humid)
        if dp:
            df.loc[ind, 'dew_point'] = dp
            df.loc[ind, 'heat_index'] = hi
        else:
            df.loc[ind, 'dew_point'] = np.NaN
            df.loc[ind, 'heat_index'] = np.NaN
    for col in ['dew_point', 'heat_index']:
        df[col] = df[col].astype(np.float)

    for ind, row in df[['temperature', 'dew_point']].iterrows():
        temp, dew = row.values
        humidex = get_humidex(temp, dew)
        if humidex:
            df.loc[ind, 'humidex'] = humidex
        else:
            df.loc[ind, 'humidex'] = np.NaN

    df['Humdex_Bin'] = df['humidex'].apply(humidex_bin)

    return df


#######################################################################
#######################################################################
####### STEP 16 Find and Set First, Last Game of Hometeam #############
#######################################################################
#######################################################################


def set_first_last_game_dummy_hometeam(df):
    df['first_game_of_home_team'] = 0

    for i, row in df.iterrows():
        team = row['home_team']
        season = row['season']
        match_date = row['match_date']
        if len(
                df.loc[(df['season'] == season)
                       & (df['match_date'] < match_date)
                       & (df['home_team'] == team)]) == 0:
            df.loc[df.index == i, 'first_game_of_home_team'] = 1

    df['first_game_of_home_team'] = df['first_game_of_home_team'].fillna(0)

    for i, row in df.iterrows():
        team = row['home_team']
        season = row['season']
        match_date = row['match_date']
        if len(
                df.loc[(df['season'] == season)
                       & (df['match_date'] > match_date)
                       & (df['home_team'] == team)]) == 0:
            df.loc[df.index == i, 'last_game_of_home_team'] = 1

    df['last_game_of_home_team'] = df['last_game_of_home_team'].fillna(0)

    return df


#######################################################################
#######################################################################
############ STEP 19 Get and Set Salary Data of Teams #################
#######################################################################
#######################################################################

def add_salary_data(df, crawling=False):
    # If you want this function to crawl then process, set crawling as True
    # Crawling data
    # Source of getting salary data at 2002~2013 and 2014~2018
    # is different.

    # 2002~2013
    def extract_salary(year):

        player = []
        position = []
        age = []
        salary = []
        team = []

        year = str(year)
        url = 'http://jsalary.wiki.fc2.com/wiki/%E2%96%A0{}%E5%B9%B4%E2%96%A0\
              '.format(year)

        r = requests.get(url)
        html_doc = r.text
        soup = BS(html_doc)

        for x in soup.find_all('a'):
            info = x.get('title')
            try:
                if info.startswith('20'):
                    team_name = info.split('年')[1]
                    team_url = 'http://jsalary.wiki.fc2.com/wiki/{}'\
                        .format(info)
                    r_t = requests.get(team_url)
                    html_t = r_t.text
                    soup_t = BS(html_t)

                    for i, x in enumerate(soup_t.find_all('td')):

                        if i % 4 == 0:
                            player.append(x.text)
                        elif i % 4 == 1:
                            position.append(x.text)
                        elif i % 4 == 2:
                            age.append(x.text)
                        elif i % 4 == 3:
                            salary.append(x.text)
                            team.append(team_name)

            except:
                continue

        salary_data = pd.DataFrame(
            data={
                'year': year,
                'team': team,
                'player': player,
                'salary': salary,
                'position': position,
                'age': age
            }
        )

        salary_data = salary_data[salary_data['age'] != '年齢']
        salary_data = salary_data[salary_data['position'] != 'ポジション']

        return salary_data

    # 2013 ~ 2018
    def extract_salary2(year):
        if year == 2018:
            url = 'https://www.soccer-money.net/players/in_players.php'
            year = str(year)

        else:
            year = str(year)
            url = 'https://www.soccer-money.net/players/past_in_players.php?\
                   year={}'.format(year)

        r = requests.get(url)
        r.encoding = 'utf-8'
        html_doc = r.text
        soup = BS(html_doc)

        ranking = []
        player = []
        age = []
        position = []
        team = []
        salary = []

        for i, x in enumerate(soup.find_all('td')):
            info = x.text
            if i == 0 or i == 1:
                continue
            else:
                crt = i - 2
                if crt % 6 == 0:
                    ranking.append(info)
                elif crt % 6 == 1:
                    player.append(info)
                if crt % 6 == 2:
                    age.append(info)
                if crt % 6 == 3:
                    position.append(info)
                if crt % 6 == 4:
                    team.append(info)
                if crt % 6 == 5:
                    salary.append(info)

        salary_df__ = pd.DataFrame(
            data={
                'year': year,
                'team': team,
                'player': player,
                'salary': salary,
                'position': position,
                'age': age,
                'ranking': ranking
            }
        )

        return salary_df__

    ###
    # I crawl == True it will crawl
    ##
    # Else it will just load the given csv data!
    if crawling == True:
        salary_until_2013 = []
        for year in range(2002, 2014):
            salary = extract_salary(year)
            salary_until_2013.append(salary)

        for i, df_ in enumerate(salary_until_2013):
            if i == 0:
                salary1 = df_
            else:
                salary1 = pd.concat([salary1, df_])

        salary_from_2013 = []
        for year in range(2013, 2019):
            salary = extract_salary2(year)
            salary_from_2013.append(salary)

        for i, df_ in enumerate(salary_from_2013):
            if i == 0:
                salary2 = df_
            else:
                salary2 = pd.concat([salary2, df_])
    else:
        print('selected not to crawl salary data loading!')
        salary1 = pd.read_csv('salary1.csv')
        salary2 = pd.read_csv('salary2.csv')

    # Preprocessing crawled data
    # 2002 ~ 2013
    team_name_dict = df['home_team'].unique().tolist()
    sal1 = salary1.copy()
    sal1.reset_index(inplace=True)
    del sal1['index']

    ind_to_fix = []
    for i, row in sal1.iterrows():
        salary = row['salary']
        if '年俸記載なし' in salary:
            try:
                sal1.loc[sal1.index == i, 'salary'] = \
                    re.search('(\d+、*\d+)万', salary).groups(0)[0]
            except:
                print(salary)
                continue

    # change team names in crawled data to fit in main dataset
    team_before = [
        'コンサドーレ札幌', 'ベガルタ仙台', '鹿島アントラーズ', '浦和レッドダイヤモンズ', 'ジェフユナイテッド市原',
        '柏レイソル', 'FC東京', '東京ヴェルディ', '横浜F・マリノス', '清水エスパルス', 'ジュビロ磐田',
        '名古屋グランパスエイト', '京都パープルサンガ', 'ガンバ大阪', 'ヴィッセル神戸', 'サンフレッチェ広島',
        'セレッソ大阪', '大分トリニータ', 'アルビレックス新潟', '大宮アルディージャ', '川崎フロンターレ',
        'ヴァンフォーレ甲府', 'アビスパ福岡', '横浜FC', '名古屋グランパス', '京都サンガF.C.', 'モンテディオ山形',
        'ジェフユナイテッド市原・千葉', '湘南ベルマーレ', 'サガン鳥栖'
    ]

    team_after = [
        '札幌', '仙台', '鹿島', '浦和', '千葉', '柏', 'FC東京', '東京Ｖ', '横浜FM', '清水', '磐田',
        '名古屋', '京都', 'Ｇ大阪', '神戸', '広島', 'Ｃ大阪', '大分', '新潟', '大宮', '川崎Ｆ', '甲府',
        '福岡', '横浜FC', '名古屋', '京都', '山形', '千葉', '湘南', '鳥栖'
    ]

    team_name_changer1 = dict(zip(team_before, team_after))

    sal1['team'] = sal1['team'].apply(lambda x: team_name_changer1[x])
    sal1['year'] = pd.to_numeric(sal1['year'])

    # change salary column to numeric type

    for i, row in sal1.iterrows():
        year = row['year']
        sal = str(row['salary'])
        if year < 2011:
            if len(re.findall('\d', sal)) == 0:
                sal1.loc[sal1.index == i, 'salary'] = np.nan
            elif sal.endswith('億') or sal.endswith('億円'):
                new_sal = int(re.match('\d+', sal)[0]) * 10000
                sal1.loc[sal1.index == i, 'salary'] = new_sal

            elif '億' in sal and (str(sal).endswith('億')) == False:
                oku = int(sal.split('億')[0]) * 10000
                try:
                    man = int(
                        re.search(
                            '\d+',
                            ''.join(sal.split('億')[1].split('、')))[0]
                    )
                except:
                    pass
                sal1.loc[sal1.index == i, 'salary'] = oku + man

            elif sal in ['東京ガス社員契約円', 'アマチュア契約円', '不明円', '不明']:
                sal1.loc[sal1.index == i, 'salary'] = np.nan

            elif sal == '１億円万' or sal == '１億万':
                sal1.loc[sal1.index == i, 'salary'] = 10000

            else:
                try:

                    sal1.loc[sal1.index == i, 'salary'] = \
                        sal1.loc[sal1.index == i, 'salary'].apply(
                            lambda x: int(re.search(
                                '\d+',
                                ''.join(str(x).split('、')))[0]
                            )
                    )
                except:
                    pass

    sal1['salary'] = sal1['salary'].apply(lambda x: str(x).split('万')[0])
    sal1['salary'] = pd.to_numeric(sal1['salary'], errors='coerce')

    # 2013 ~ 2018
    sal2 = salary2.copy()
    team_before = [
        '名古屋グランパス', '横浜F・マリノス', 'ガンバ大阪', 'ヴィッセル神戸', '浦和レッズ', '川崎フロンターレ',
        '柏レイソル', '鹿島アントラーズ', '大宮アルディージャ', '清水エスパルス', 'FC東京', 'サンフレッチェ広島',
        'ベガルタ仙台', 'ヴァンフォーレ甲府', 'アルビレックス新潟', 'サガン鳥栖', 'セレッソ大阪', '徳島ヴォルティス',
        'モンテディオ山形', '湘南ベルマーレ', '松本山雅FC', 'ジュビロ磐田', 'アビスパ福岡', 'コンサドーレ札幌',
        'Ｖ・ファーレン長崎']
    team_after = [
        '名古屋', '横浜FM', 'Ｇ大阪', '神戸', '浦和', '川崎Ｆ', '柏', '鹿島', '大宮', '清水',
        'FC東京', '広島', '仙台', '甲府', '新潟', '鳥栖', 'Ｃ大阪', '徳島', '山形', '湘南',
        '松本', '磐田', '福岡', '札幌', '長崎'
    ]

    team_name_changer2 = dict(zip(team_before, team_after))

    sal2['team'] = sal2['team'].apply(lambda x: team_name_changer2[x])

    sal2['year'] = pd.to_numeric(sal2['year'])

    for i, row in sal2.iterrows():
        sal = row['salary']

        if '億' in sal:
            if '億円' in sal:
                oku = int(sal.split('億')[0]) * 10000
                sal2.loc[sal2.index == i, 'salary'] = oku
            else:

                oku = int(sal.split('億')[0]) * 10000
                man = int(sal.split('億')[1].split('万')[0])
                sal2.loc[sal2.index == i, 'salary'] = oku + man
        else:
            sal2.loc[sal2.index == i, 'salary'] = int(sal.split('万')[0])

    del sal2['ranking']
    sal2 = sal2[sal2.year > 2013]
    sal_df = pd.concat([sal1, sal2], ignore_index=True)
    sal_df['salary'] = pd.to_numeric(sal_df['salary'], errors='coerce')

    # Merging salary data to match reports data
    in_match = pd.read_csv('match_reports.csv')
    out_match = pd.read_csv('ex_match_reports.csv')
    match = pd.concat([in_match, out_match])
    match = match.drop_duplicates(subset='id').sort_values(by='id')

    df_temp = df.merge(match, how='left', on='id')
    home = df_temp[[
        'season', 'home_team', 'home_team_player1', 'home_team_player2',
        'home_team_player3', 'home_team_player4', 'home_team_player5',
        'home_team_player6', 'home_team_player7', 'home_team_player8',
        'home_team_player9', 'home_team_player10', 'home_team_player11'
    ]]

    away = df_temp[[
        'season', 'away_team', 'away_team_player1', 'away_team_player2',
        'away_team_player3', 'away_team_player4', 'away_team_player5',
        'away_team_player6', 'away_team_player7', 'away_team_player8',
        'away_team_player9', 'away_team_player10', 'away_team_player11'
    ]]

    # Unificate player name of two datasets
    home_player = pd.melt(
        frame=home,
        id_vars=['season', 'home_team'],
        var_name='num',
        value_name='player'
    )

    home_player['position'] = np.nan

    home_player.loc[home_player.player.notnull(), 'player'] = \
        home_player.loc[home_player.player.notnull(), 'player'].apply(
            lambda x: re.search('\D+', x)[0].strip())

    home_player.loc[home_player.player.notnull(), 'position'] = \
        home_player.loc[home_player.player.notnull(), 'player'].str[-2:]

    home_player.loc[home_player.player.notnull(), 'player'] = \
        home_player.loc[home_player.player.notnull(), 'player'].str[:-3]

    del home_player['num']

    player1 = home_player.drop_duplicates(
        subset=['season', 'home_team', 'player'])

    away_player = pd.melt(
        frame=away,
        id_vars=['season', 'away_team'],
        var_name='num',
        value_name='player'
    )

    away_player['position'] = np.nan

    away_player.loc[away_player.player.notnull(), 'player'] = \
        away_player.loc[away_player.player.notnull(), 'player'].apply(
            lambda x: re.search('\D+', x)[0].strip())

    away_player.loc[away_player.player.notnull(), 'position'] = \
        away_player.loc[away_player.player.notnull(), 'player'].str[-2:]

    away_player.loc[away_player.player.notnull(), 'player'] = \
        away_player.loc[away_player.player.notnull(), 'player'].str[:-3]

    del away_player['num']

    away_player.head()

    player2 = away_player.drop_duplicates(
        subset=['season', 'away_team', 'player'])

    player1.rename(columns={'home_team': 'team'}, inplace=True)
    player2.rename(columns={'away_team': 'team'}, inplace=True)
    players = pd.concat([player1, player2], ignore_index=True)

    players = players.drop_duplicates(subset=['season', 'team', 'player'])
    players = players[players.season > 2001]

    players['season'] = players['season'].astype(int)

    players['player'] = players['player'].apply(
        lambda x: ''.join(str(x).split(' ')))
    sal_df['player'] = sal_df['player'].apply(
        lambda x: ''.join(str(x).split('　')))

    del players['position'], sal_df['position']

    players = players[players.player.notnull()]

    # unificate kanji with different shape in two datasets

    kanji_converter = {'﨑': '崎', '眞': '真', '澤': '沢'}
    for i, row in sal_df.iterrows():
        name = row['player']
        for kanji in list(kanji_converter.keys()):
            if kanji in name:
                new_name = []
                for x in list(name):
                    try:
                        y = kanji_converter[x]
                        new_name.append(y)
                    except:
                        new_name.append(x)
                new_name = ''.join(new_name)

                sal_df.loc[sal_df.index == i, 'player'] = new_name

    kanji_converter = {'﨑': '崎', '眞': '真', '澤': '沢'}
    for i, row in players.iterrows():
        name = row['player']
        for kanji in list(kanji_converter.keys()):
            if kanji in name:
                new_name = []
                for x in list(name):
                    try:
                        y = kanji_converter[x]
                        new_name.append(y)
                    except:
                        new_name.append(x)
                new_name = ''.join(new_name)

                players.loc[players.index == i, 'player'] = new_name

    sal_added = players.merge(sal_df, how='left',
                              left_on=['season', 'team', 'player'],
                              right_on=['year', 'team', 'player']
                              )

    del sal_added['year'], sal_added['age']

    # Estimated salary of two star players, what is publicly
    # known by media.
    sal_added.loc[sal_added.player == 'フェルナンドトーレス', 'salary'] = 80000
    sal_added.loc[sal_added.player == 'アンドレスイニエスタ', 'salary'] = 320000

    salary_df = sal_added.copy()

    # Merging to main dataset
    in_match = pd.read_csv('match_reports.csv')
    out_match = pd.read_csv('ex_match_reports.csv')
    match = pd.concat([in_match, out_match])
    match = match.drop_duplicates(subset='id').sort_values(by='id')

    df = df.merge(match, how='left', on='id')

    player_cols = [
        'home_team_player1', 'home_team_player2', 'home_team_player3',
        'home_team_player4', 'home_team_player5', 'home_team_player6',
        'home_team_player7', 'home_team_player8', 'home_team_player9',
        'home_team_player10', 'home_team_player11', 'away_team_player1',
        'away_team_player2', 'away_team_player3', 'away_team_player4',
        'away_team_player5', 'away_team_player6', 'away_team_player7',
        'away_team_player8', 'away_team_player9', 'away_team_player10',
        'away_team_player11'
    ]

    for col in player_cols:
        df.loc[df[col].notnull(), col] = \
            df.loc[df[col].notnull(), col].apply(
                lambda x: re.search('\D+', x)[0].strip())
        df.loc[df[col].notnull(), col] = \
            df.loc[df[col].notnull(), col].str[:-3]
        df.loc[df[col].notnull(), col] = df.loc[df[col].notnull(), col].apply(
            lambda x: ''.join(str(x).split(' ')))

    merging_df = df.copy()[['id', 'match_date']]

    for num in range(1, 12):
        num = str(num)

        for place in ['home', 'away']:
            col = '{}_team_player{}'.format(place, num)
            temp = df[[
                'id', 'season', '{}_team'.format(place), col
            ]].merge(salary_df, how='left',
                     left_on=['season', '{}_team'.format(place), col],
                     right_on=['season', 'team', 'player']
                     )

            gc.collect()
            temp = temp[['id', 'salary']]
            merging_df = merging_df.merge(temp, how='left', on='id')
            merging_df.rename(
                columns={'salary': '{}_team_player{}_salary'
                         .format(place, num)},
                inplace=True
            )
            gc.collect()

    df = df.merge(merging_df.drop(['match_date'], 1), how='left', on='id')

    df[(df.season > 2005) & (df.division == 1)].isnull().sum() / \
        len(df[(df.season > 2005) & (df.division == 1)])

    # Creating features with salary data
    # 1. Average salary of team for each season
    sal_temp = salary_df.copy()
    sal_temp['salary_11'] = np.nan
    teams = list(sal_temp.team.unique())
    seasons = range(2002, 2019)

    for team_ in teams:
        for season_ in seasons:
            z = sorted(
                sal_temp.dropna(
                    subset=['salary']).loc[(sal_temp['team'] == team_)
                                           & (sal_temp['season'] == season_),
                                           'salary'].values.tolist()
            )

            if len(z) == 0:
                continue

            crt = z[-11]

            temp = sal_temp[(sal_temp['team'] == team_) &
                            (sal_temp['season'] == season_)]

            for i, row in temp.iterrows():
                sal = row['salary']
                if sal >= crt:
                    sal_temp.loc[sal_temp.index == i, 'salary_11'] = 1
                else:
                    sal_temp.loc[sal_temp.index == i, 'salary_11'] = 0

    players_11 = sal_temp[sal_temp.salary_11 == 1].sort_values(by='season')

    team_sea_avg_sal = players_11.groupby(['season', 'team'])[
        'salary'].agg('mean').reset_index()

    # Merge average salary of specific season to next season

    team_sea_avg_sal['season'] += 1

    avg_sal_added = df.merge(
        team_sea_avg_sal,
        how='left',
        left_on=['season', 'home_team'],
        right_on=['season', 'team']).rename(
            columns={'salary': 'last_year_home_team_avg_salary'}
    )

    del avg_sal_added['team']
    avg_sal_added = avg_sal_added.merge(
        team_sea_avg_sal,
        how='left',
        left_on=['season', 'away_team'],
        right_on=['season', 'team']).rename(
        columns={'salary': 'last_year_away_team_avg_salary'}
    )

    del avg_sal_added['team']

    # 2. Player with highest salary of home & away teams' start lineup
    df1 = avg_sal_added.copy()
    home_sal_cols = [
        'home_team_player10_salary', 'home_team_player11_salary',
        'home_team_player1_salary', 'home_team_player2_salary',
        'home_team_player3_salary', 'home_team_player4_salary',
        'home_team_player5_salary', 'home_team_player6_salary',
        'home_team_player7_salary', 'home_team_player8_salary',
        'home_team_player9_salary'
    ]

    away_sal_cols = [
        'away_team_player10_salary', 'away_team_player11_salary',
        'away_team_player1_salary', 'away_team_player2_salary',
        'away_team_player3_salary', 'away_team_player4_salary',
        'away_team_player5_salary', 'away_team_player6_salary',
        'away_team_player7_salary', 'away_team_player8_salary',
        'away_team_player9_salary'
    ]

    df1['home_team_highest_salary'] = np.nan
    df1['away_team_highest_salary'] = np.nan

    for i, row in df1.iterrows():
        if row['season'] < 2002 or row['division'] == 2:
            continue
        else:
            try:
                home_max = np.max(
                    [x for x in row[home_sal_cols].values.tolist()
                        if ~np.isnan(x)]
                )
                away_max = np.max(
                    [x for x in row[away_sal_cols].values.tolist()
                        if ~np.isnan(x)]
                )
            except:
                pass

            df1.loc[df1.index == i, 'home_team_highest_salary'] = home_max
            df1.loc[df1.index == i, 'away_team_highest_salary'] = away_max

    df_done = df1.copy()

    return df_done


#######################################################################
#######################################################################
############### STEP 19 Set Derby Values for Teams ####################
#######################################################################
#######################################################################

def set_derby_btwn_teams(df):

    # Match derbies with the same area code
    df['derby'] = 0
    for i, row in df.iterrows():
        if row['home_team_area_code'] == row['away_team_area_code']:
            df.loc[df.index == i, 'derby'] = 1

    # Match derbiews with regions are close
    kyushu_derby = ['福岡', '北九州', '鳥栖', '熊本', '大分']
    for i, row in df.iterrows():
        # Kansai derby
        if sorted(row[['home_team', 'away_team']].values.tolist()) \
                == ['京都', '神戸']:
            df.loc[df.index == i, 'derby'] = 1
        # Kyushu derby
        elif row['home_team'] in kyushu_derby and row['away_team'] \
                in kyushu_derby:
            df.loc[df.index == i, 'derby'] = 1

    return df

#######################################################################
#######################################################################
################### STEP 22 Set Lagged Features #######################
#######################################################################
#######################################################################

"""
We skip from step 19 to 22 due to the fact that
step 20, 21 is not needed to recreate the train data.
They are Just Simply Not Used
"""
# Yearly home , stadium's lag attendance


def lag_year_hometeam_stadium_feature(df, lags, col):
    temp = df[['match_Year', 'home_team', 'stadium', col]]
    for i in lags:
        shifted = temp.copy()
        shifted.columns = ['match_Year', 'home_team',
                           'stadium', col + '_Year_Stadium_Lag_' + str(i)]
        shifted['match_Year'] += i

        shifted = shifted.groupby(
            ['match_Year', 'home_team', 'stadium']
        )[col + '_Year_Stadium_Lag_' + str(i)].mean().reset_index()

        df = pd.merge(
            df, shifted,
            on=['match_Year', 'home_team', 'stadium'],
            how='left'
        )
    return df

# Yearly, Monthly home, away's lag attendance


def lag_year_hometeam_awayteam_feature(df, lags, col):
    temp = df[['match_Year', 'home_team', 'away_team', col]]
    for i in lags:
        shifted = temp.copy()
        shifted.columns = ['match_Year', 'home_team',
                           'away_team', col + '_Year_Hometeam_Awayteam_Lag_' + str(i)]
        shifted['match_Year'] += i

        shifted = shifted.groupby(
            ['match_Year', 'home_team', 'away_team']
        )[col + '_Year_Hometeam_Awayteam_Lag_' + str(i)].mean().reset_index()

        df = pd.merge(
            df, shifted,
            on=['match_Year', 'home_team', 'away_team'],
            how='left'
        )
    return df


def lag_year_month_hometeam_awayteam_feature(df, lags, col):
    temp = df[['match_Year', 'match_Month', 'home_team', 'away_team', col]]
    for i in lags:
        shifted = temp.copy()
        shifted.columns = ['match_Year', 'match_Month', 'home_team',
                           'away_team', col + '_Year_Month_Hometeam_Awayteam_Lag_' + str(i)]
        shifted['match_Year'] += i

        shifted = shifted.groupby(
            ['match_Year', 'match_Month', 'home_team', 'away_team']
        )[col + '_Year_Month_Hometeam_Awayteam_Lag_' + str(i)].mean().reset_index()

        df = pd.merge(
            df, shifted,
            on=['match_Year', 'match_Month', 'home_team', 'away_team'],
            how='left'
        )
    return df


def make_lagged_feature(df, lags):
    df = lag_year_hometeam_awayteam_feature(df, lags, 'attendance')
    df = lag_year_month_hometeam_awayteam_feature(df, lags, 'attendance')
    df = lag_year_hometeam_stadium_feature(df, lags, 'attendance')

    return df

combine = make_lagged_feature(combine, [1, 2, 3, 4])


##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################

def put_all_data_together(google_map_key):

    gmaps = googlemaps.Client(key=google_map_key)

    # 1 Load data
    initial_dataframe = load_and_merge_initial_data()

    gc.collect()
    # 2 Extra J1, J2 and data from 1993
    j1_j2_set_dataframe = set_extra_j1_j2_data(initial_dataframe)

    gc.collect()
    # 3 Set Datetime
    datetime_set_dataframe = set_datetime(j1_j2_set_dataframe)

    gc.collect()
    # 4 Adding Area Name and Code to Stadiums
    area_data_set_dataframe = add_venue_area_info(
        datetime_set_dataframe, gmaps)

    gc.collect()
    # 5 Set Area Info of both Home and Away Teams
    team_area_info_set_dataframe = get_area_info_of_team(
        area_data_set_dataframe)

    gc.collect()
    # 6 Get and Set Baseball data
    baseball_set_dataframe = add_baseball_data(
        team_area_info_set_dataframe, gmaps)

    gc.collect()
    # 7 Distance Btwn Teams, Duration Between Games for Soccer and Baseball
    distance_duration_set_dataframe = get_location_distance_duration(
        baseball_set_dataframe, google_map_key)

    gc.collect()
    # 8 Holiday Data Add
    holiday_set_dataframe = add_holiday_data(distance_duration_set_dataframe)

    gc.collect()
    # 9 Win Ratio!
    # First properly set th_match column
    th_match_set_dataframe = set_th_match(holiday_set_dataframe)

    # Get the simple win ratio
    simple_win_ratio_set_dataframe = set_simple_win_ratio(
        th_match_set_dataframe)

    gc.collect()
    # Get the exponentially weighted win ratio
    exp_win_ratio_set_dataframe = set_exp_win_ratio(
        simple_win_ratio_set_dataframe)

    gc.collect()
    # 10 Population Data for Prefuctures
    population_set_dataframe = set_population_data(exp_win_ratio_set_dataframe)

    gc.collect()
    # 11 Find which league team was in previous years
    team_previous_league_set_dataframe = set_team_league_previous_years(
        population_set_dataframe)

    gc.collect()
    # 12 Set the weather data
    weather_set_dataframe = set_process_weather_data(
        team_previous_league_set_dataframe)

    gc.collect()
    # 13 Filter broadcaster data
    broadcaster_set_dataframe = process_broadcasters(weather_set_dataframe)

    gc.collect()
    # 14 Set weather index data
    weather_index_set_dataframe = set_weather_indexes(
        broadcaster_set_dataframe)

    gc.collect()
    # 15 Set emperor cup data
    emperor_cup_set_dataframe = add_emperor_cup_data(
        weather_index_set_dataframe)

    gc.collect()
    # 16 first_last_game_of_dummy
    first_last_game_set_dataframe = set_first_last_game_dummy_hometeam(
        emperor_cup_set_dataframe)

    gc.collect()
    # 17 Add and set AFC data - 이것도 그대로 독립
    afc_set_dataframe = add_afc_data(first_last_game_set_dataframe)

    gc.collect()
    # 18 Salary - 1, 2 분이면 돌아가니까 그냥 둠
    salary_set_dataframe = add_salary_data(afc_set_dataframe)

    gc.collect()
    # 19 Set derby data
    derby_set_dataframe = set_derby_btwn_teams(salary_set_dataframe)

    gc.collect()
    # 22 Set lagged_data
    final = set_lagged_data(derby_set_dataframe)

    final = preprocess(final)

    final.select_dtypes('object').columns

    drop_cols = [
        'away_team_b_1', 'away_team_b_2', 'date_1', 'date_2', 'home_team_b_1',
        'home_team_b_2', 'kick_off_time', 'venue_area_b', 'venue_area_b_1',
        'venue_area_b_2', 'venue_b_1', 'venue_b_2', 'date', 'home_team_area',
        'weather', 'away_team_player1', 'away_team_player10',
        'away_team_player11', 'away_team_player2', 'away_team_player3',
        'away_team_player4', 'away_team_player5', 'away_team_player6',
        'away_team_player7',
        'away_team_player8', 'away_team_player9', 'home_team_player1',
        'home_team_player10', 'home_team_player11', 'home_team_player2',
        'home_team_player3', 'home_team_player4', 'home_team_player5',
        'home_team_player6', 'home_team_player7', 'home_team_player8',
        'home_team_player9', 'away_team_score', 'home_team_score',
        'allstar_1', 'allstar_2',  'attendance_b_1',
        'attendance_b_2', 'away_area_b_1', 'away_area_b_2', 'derby_b_1',
        'derby_b_2', 'home_area_b_1', 'home_area_b_2', 'home_lose_1',
        'home_lose_2', 'home_nodcsn_1', 'home_nodcsn_2',
        'home_win_1', 'home_win_2', 'round_b_1', 'round_b_2', 'tournament_1',
        'tournament_2', 'home_team_area_code', 'venue_area_code_b',
        'venue_area_code_b_1', 'venue_area_code_b_2', 'duration_bet_sports_1',
        'dist_bet_sports_1', 'duration_bet_sports_2', 'dist_bet_sports_2',
        'lat_b_1', 'lon_b_1', 'lat_b_2', 'lon_b_2', 'wod', 'season'
    ]

    final.drop(drop_cols, 1, inplace=True)

    final['round'] = final['round'].apply(
        lambda x: x[1:-1]).astype(np.int)

    final.select_dtypes('object').columns

    cat_cols = ['away_team', 'away_team_area', 'home_team', 'venue',
                'venue_area', 'description', 'home_away_Lastyear_Comb',
                'home_away_twoyearsago_Comb', 'Humdex_Bin', 'match_Is_Weekend',
                'match_Is_month_end', 'match_Is_month_start',
                'match_Is_quarter_end', 'match_Is_quarter_start']

    X_orig = final[[x for x in list(final) if x not in cat_cols]]

    X_cat = final[cat_cols]

    X_cat = pd.get_dummies(X_cat)

    final = pd.concat([X_orig, X_cat], axis=1)

    final = final.drop_duplicates()

    final.loc[final['attendance'] == 0, 'attendance'] = (
        31921 + 35877 + 44424 + 37945.0) / 4

    return final


def get_train_data():
    key_question = input("Please enter the API key for googlemaps. If you want"
                         " to use competitor's key, just press enter."
                         " However, there is a possibilty that competitor's key"
                         " doesn't work due to expenditure problem,"
                         " so we recommend you using your own key."
                         " Your Google Map API Key : ")
    key_question
    if key_question == '':
        key = 'AIzaSyAVCE-eyUNCAvrcDHiWfRgSDorrz41jA5w'
        gmaps = googlemaps.Client(key=key)
        print("You are now using competitor's key.")

    else:
        try:
            key = key_question
            gmaps = googlemaps.Client(key=key)
            print('You are now using your own key.')
        except:
            print('The key is invalid. Please restart the function and '
                  'enter valid key.')
            raise ValueError

    print("Processing all Data. This will take around 18~20 hours max")
    final_df = put_all_data_together(key)

    train_data = final_df[final_df['match_date'] < '2017-01-01']

    train_data.to_csv('train_data.csv', index=False)

    return final_df


def get_test_data(final_df):
    test_ids = [
        19075, 19076, 19077, 19078, 19079, 19080, 19081, 19082, 19083,
        19084, 19085, 19086, 19087, 19088, 19089, 19090, 19091, 19092,
        19093, 19094, 19095, 19096, 19097, 19098, 19099, 19100, 19101,
        19102, 19103, 19104, 19105, 19106, 19107, 19108, 19109, 19110,
        19111, 19112, 19113, 19114, 19115, 19116, 19117, 19118, 19119,
        19120, 19121, 19122, 19123, 19124, 19125, 19126, 19127, 19128,
        19129, 19130, 19131, 19132, 19133, 19134, 19135, 19136, 19137,
        19138, 19139, 19140, 19141, 19142, 19143, 19144, 19145, 19146,
        19147, 19148, 19149, 19150, 19151, 19152, 19153, 19154, 19155,
        19156, 19157, 19158, 19159, 19160, 19161, 19162, 19163, 19164,
        19165, 19166, 19167, 19168, 19169, 19170, 19171, 19172, 19173,
        19174, 19175, 19176, 19177, 19178, 19179, 19180, 19181, 19182,
        19183, 19184, 19185, 19186, 19187, 19188, 19189, 19190, 19191,
        19192, 19193, 19194, 19195, 19196, 19197, 19198, 19199, 19200,
        19201, 19202, 19203, 19204, 19205, 19206, 19207, 19208, 19209,
        19210, 19211, 19212, 19213, 19214, 19215, 19216, 19217, 19218,
        19219, 19220, 19221, 19222, 19223, 19224, 19225, 19226, 19227,
        19228, 19229, 19230, 19231, 19232, 19233, 19234, 19235, 19236,
        19237, 19238, 19239, 19240, 19241, 19242, 19243, 19244, 19245,
        19246, 19247, 19248, 19249, 19250, 19251, 19252, 19253, 19254,
        19255, 19256, 19257, 19258, 19259, 19260, 19261, 19262, 19263,
        19264, 19265, 19266, 19267, 19268, 19269, 19270, 19271, 19272,
        19273, 19274, 19275, 19276, 19277, 19278, 19279, 19280, 19281,
        19282, 19283, 19284, 19285, 19286, 19287, 19288, 19289, 19290,
        19291, 19292, 19293, 19294, 19295, 19296, 19297, 19298, 19299,
        19300, 19301, 19302, 19303, 19304, 19305, 19306, 19307, 19308,
        19309, 19310, 19311, 19312, 19313, 19314, 19315, 19316, 19317,
        19318, 19319, 19320, 19321, 19322, 19323, 19324, 19325, 19326,
        19327, 19328, 19329, 19330, 19331, 19332, 19333, 19334, 19335,
        19336, 19337, 19338, 19339, 19340, 19341, 19342, 19343, 19344,
        19345, 19346, 19347, 19348, 19349, 19350, 19351, 19352, 19353,
        19354, 19355, 19356, 19357, 19358, 19359, 19360, 19361, 19362,
        19363, 19364, 19365, 19366, 19367, 19368, 19369, 19370, 19371,
        19372, 19373, 19374, 19375, 19376, 19377, 19378, 19379, 19380,
        20745, 20746, 20747, 20748, 20749, 20750, 20751, 20752, 20753,
        20754, 20755, 20756, 20757, 20758, 20759, 20760, 20761, 20762,
        20763, 20764, 20765, 20766, 20767, 20768, 20769, 20770, 20771,
        20772, 20773, 20774, 20775, 20776, 20777, 20778, 20779, 20780,
        20781, 20782, 20783, 20784, 20785, 20786, 20787, 20788, 20789,
        20790, 20791, 20792, 20793, 20794, 20795, 20796, 20797, 20798,
        20799, 20800, 20801, 20802, 20803, 20804, 20805, 20806, 20807,
        20808, 20809, 20810, 20811, 20812, 20813, 20814, 20815, 20816,
        20817, 20818, 20819, 20820, 20821, 20822, 20823, 20824, 20825,
        20826, 20827, 20828, 20829, 20830, 20831, 20832, 20833, 20834,
        20835, 20836, 20837, 20838, 20839, 20840, 20841, 20842, 20843,
        20844, 20845, 20846, 20847, 20848, 20849, 20850, 20851, 20852,
        20853, 20854, 20855, 20856, 20857, 20858, 20859, 20860, 20861,
        20862, 20863, 20864, 20865, 20866, 20867, 20868, 20869, 20870,
        20871, 20872, 20873, 20874, 20875, 20876, 20877, 20878, 20879,
        20880, 20881, 20882, 20883, 20884, 20885, 20886, 20887, 20888,
        20889, 20890, 20891, 20892, 20893, 20894, 20895, 20896, 20897,
        30000, 30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008,
        30009, 30010, 30011, 30012, 30013, 30014, 30015, 30016, 30017
    ]

    test_data = alter_final[alter_final['id'].isin(test_ids)]

    test_data.to_csv('test_data.csv', index=False)


if __name__ == "__main__":
    final_df = get_train_data()
    get_test_data(final_df)

    print("preprocess.py done")
