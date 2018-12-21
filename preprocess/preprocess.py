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
from bs4 import BeautifulSoup as BS
# 3rd Party Packages for Web Scraping


import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from numpy import NaN
# Numpy and Pandas

from IPython.display import display
# For Display

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
# Below are General Functions which will be continuosly called by other
# functions In alphabetical order


def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end',
            'Is_quarter_start']
    if time:
        attr = attr + ['Hour']
    for n in attr:
        df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop:
        df.drop(fldname, axis=1, inplace=True)


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'rmse_', rmse(labels, preds), False


def preprocess(df):
    str_dtype, int_dtype, float_dtype = [], [], []

    for col in df.columns:
        if is_string_dtype(df[col]):
            str_dtype.append(col)
        elif is_numeric_dtype(df[col]):
            if str(df[col].dtypes)[:3] == 'int':
                int_dtype.append(col)
            else:
                float_dtype.append(col)

    for col in int_dtype:
        for num in [8, 16, 32, 64]:
            if np.mean(
                    df[col] == df[col].astype('int' + str(num))) == 1:
                df[col] = df[col].astype('int' + str(num))
                break

    for col in float_dtype:
        for num in [16, 32, 64]:
            if np.mean(df[col] == df[col].astype('float' + str(num))
                       ) == 1:
                if np.mean(df[col] == df[col].astype('int' + str(num))
                           ) == 1:
                    df[col] = df[col].astype('int' + str(num))
                else:
                    df[col] = df[col].astype('float' + str(num))
                break

    return df


def preprocess(df):
    str_dtype, int_dtype, float_dtype = [], [], []

    for col in df.columns:
        if is_string_dtype(df[col]):
            str_dtype.append(col)
        elif is_numeric_dtype(df[col]):
            if str(df[col].dtypes)[:3] == 'int':
                int_dtype.append(col)
            else:
                float_dtype.append(col)

    for col in int_dtype:
        for num in [8, 16, 32, 64]:
            if np.mean(df[col] == df[col].astype('int' + str(num))) == 1:
                df[col] = df[col].astype('int' + str(num))
                break

    for col in float_dtype:
        for num in [16, 32, 64]:
            if np.mean(df[col] == df[col].astype('float' + str(num))) == 1:
                if np.mean(df[col] == df[col].astype('int' + str(num))) == 1:
                    df[col] = df[col].astype('int' + str(num))
                else:
                    df[col] = df[col].astype('float' + str(num))
                break

    return df


def reset_index(df):
    df.reset_index(inplace=True)
    df.drop('index', 1, inplace=True)
    return df


"""
Below are functions related to the names
Stadiums, and Teams
"""

# Function to unificate venue name and team name for better prediction.


def tidying_venue_name(df):
    same_stadium_dict = {
        '日産スタジアム': '横浜国際総合競技場',
        '静岡スタジアムエコパ': 'エコパスタジアム',
        '大阪長居スタジアム': 'ヤンマースタジアム長居',
        '長居スタジアム': '大阪長居第２陸上競技場',
        '長居第２陸上競技場': '大阪長居第２陸上競技場',
        '大阪長居第2陸上競技場': '大阪長居第２陸上競技場',
        'デンカビッグスワンスタジアム': '東北電力ビッグスワンスタジアム',
        '大分銀行ドーム': '大分スポーツ公園総合競技場',
        '九州石油ドーム': '大分スポーツ公園総合競技場',
        '市立吹田サッカースタジアム': 'パナソニックスタジアム吹田',
        'エディオンスタジアム広島': '広島ビッグアーチ',
        '熊本県民総合運動公園陸上競技場': 'うまかな・よかなスタジアム',
        'えがお健康スタジアム': 'うまかな・よかなスタジアム',
        'Ｐｉｋａｒａスタジアム': '香川県立丸亀競技場',
        '東平尾公園博多の森陸上競技場': '博多の森陸上競技場',
        'ノエビアスタジアム神戸': 'ホームズスタジアム神戸',
        '草薙総合運動公園陸上競技場': '静岡県草薙総合運動場陸上競技場',
        '静岡県営草薙陸上競技場': '静岡県草薙総合運動場陸上競技場',
        '名古屋市瑞穂陸上競技場': 'パロマ瑞穂スタジアム',
        '新潟スタジアム': '東北電力ビッグスワンスタジアム',
        'ShonanBMWスタジアム平塚': 'ＳｈｏｎａｎＢＭＷスタジアム平塚',
        '瑞穂公園陸上競技場': 'パロマ瑞穂スタジアム',
        '岐阜メモリアルセンター長良川競技場': '岐阜メモリアルセンター長良川球技メドウ',
        '沖縄県総合運動公園陸上競技場': '高知県立春野総合運動公園陸上競技場',
        '鳥栖スタジアム': 'ベストアメニティスタジアム',
        'レベルファイブスタジアム': '東平尾公園博多の森球技場',
        'さいたま市浦和駒場スタジアム': '浦和市駒場スタジアム',
        '京都市西京極総合運動公園陸上競技場': '京都市西京極総合運動公園陸上競技場兼球技場',
        'ＩＡＩスタジアム日本平': '日本平スタジアム',
        'IAIスタジアム日本平': '日本平スタジアム',
        'アウトソーシングスタジアム日本平': '日本平スタジアム',
        '日本平運動公園球技場': '日本平スタジアム',
        '名古屋市瑞穂球技場': 'パロマ瑞穂スタジアム',
        'NDソフトスタジアム山形': 'ＮＤソフトスタジアム山形',
        '山形県総合運動公園陸上競技場': 'ＮＤソフトスタジアム山形',
        'トランスコスモススタジアム長崎': '長崎県立総合運動公園陸上競技場',
        '駒沢陸上競技場': '駒沢オリンピック公園総合運動場陸上競技場',
        '松本平広域公園総合球技場': 'サンプロアルウィン',
        '維新百年記念公園陸上競技場': '維新みらいふスタジアム',
        'ｋａｎｋｏスタジアム': 'シティライトスタジアム',
        '岡山県陸上競技場桃太郎スタジアム': 'シティライトスタジアム',
        'ニンジニアスタジアム': '愛媛県総合運動公園陸上競技場',
        '仙台スタジアム': 'ユアテックスタジアム仙台',
        '七北田公園仙台スタジアム': 'ユアテックスタジアム仙台',
        '栃木県立グリーンスタジアム': '栃木県グリーンスタジアム',
        '徳島県鳴門総合運動公園陸上競技場': '鳴門・大塚スポーツパークポカリスエットスタジアム',
        '長野運動公園総合運動場': '長野市営長野運動公園総合運動場',
        '山梨中銀スタジアム': '山梨県小瀬スポーツ公園陸上競技場',
        'とりぎんバードスタジアム': '鳥取市営サッカー場バードスタジアム',
        '埼玉県営大宮公園サッカー場': 'ＮＡＣＫ５スタジアム大宮',
        'さいたま市大宮公園サッカー場': 'ＮＡＣＫ５スタジアム大宮',
        'NACK5スタジアム大宮': 'ＮＡＣＫ５スタジアム大宮',
        '三協フロンテア柏スタジアム': '日立柏サッカー場',
        '正田醤油スタジアム群馬': '群馬県立敷島公園県営陸上競技場',
        '平塚競技場': 'ＳｈｏｎａｎＢＭＷスタジアム平塚',
        'ジュビロ磐田スタジアム': 'ジュビロ磐田サッカースタジアム',
        'ヤマハスタジアム（磐田）': 'ジュビロ磐田サッカースタジアム',
        '横浜市三ツ沢公園球技場': 'ニッパツ三ツ沢球技場',
        '函館市千代台公園陸上競技場': '千代台公園陸上競技場',
        '水戸市立競技場': 'ケーズデンキスタジアム水戸',
        '西が丘サッカー場': '国立スポーツ科学センター西が丘サッカー場',
        '味の素フィールド西が丘': '国立スポーツ科学センター西が丘サッカー場',
        '国立西が丘サッカー場': '国立スポーツ科学センター西が丘サッカー場',
        '埼玉スタジアム２００２': '埼玉スタジアム2002'
    }

    df['venue'] = df['venue'].replace(to_replace=same_stadium_dict)


def tidying_team_name(df):
    # A football team may have the multiple names due to historical
    # reasons and due to denotation reasons. The Japanese 'G' and
    # the English 'G' are the same G for humans but not for the
    # computer.
    # Teams like Ｖ川崎 = Kawasaky Verdy, 東京Ｖ = Tokyo Verdy are the same
    # team just has different names in a different point of time in history.

    same_team_dict = {
        'C大阪': 'Ｃ大阪', 'G大阪': 'Ｇ大阪', '川崎F': '川崎Ｆ', 'FC東京': 'FC東京',
        '東京Ｖ': '東京Ｖ', '横浜FC': '横浜FC', '横浜FM': '横浜FM', '横浜FM-F': '横浜Ｆ',
        '横浜FM-M': '横浜M', 'Ｖ川崎': '東京Ｖ', 'Ｆ東京': 'FC東京', '草津': '群馬',
        '東京V': '東京Ｖ', '平塚': '湘南', '市原': '千葉'
    }

    df['home_team'] = df['home_team'].replace(to_replace=same_team_dict)
    df['away_team'] = df['away_team'].replace(to_replace=same_team_dict)


# In case it is required to change name of venue at columns what's name
# is not 'venue', which is designated in  tidying_venue_name(df)
# function, we created dictionary to convert venues' name.
same_stadium_dict = {
    '日産スタジアム': '横浜国際総合競技場',
    '静岡スタジアムエコパ': 'エコパスタジアム',
    '大阪長居スタジアム': 'ヤンマースタジアム長居',
    '長居スタジアム': '大阪長居第２陸上競技場',
    '長居第２陸上競技場': '大阪長居第２陸上競技場',
    '大阪長居第2陸上競技場': '大阪長居第２陸上競技場',
    'デンカビッグスワンスタジアム': '東北電力ビッグスワンスタジアム',
    '大分銀行ドーム': '大分スポーツ公園総合競技場',
    '九州石油ドーム': '大分スポーツ公園総合競技場',
    '市立吹田サッカースタジアム': 'パナソニックスタジアム吹田',
    'エディオンスタジアム広島': '広島ビッグアーチ',
    '熊本県民総合運動公園陸上競技場': 'うまかな・よかなスタジアム',
    'えがお健康スタジアム': 'うまかな・よかなスタジアム',
    'Ｐｉｋａｒａスタジアム': '香川県立丸亀競技場',
    '東平尾公園博多の森陸上競技場': '博多の森陸上競技場',
    'ノエビアスタジアム神戸': 'ホームズスタジアム神戸',
    '草薙総合運動公園陸上競技場': '静岡県草薙総合運動場陸上競技場',
    '静岡県営草薙陸上競技場': '静岡県草薙総合運動場陸上競技場',
    '名古屋市瑞穂陸上競技場': 'パロマ瑞穂スタジアム',
    '新潟スタジアム': '東北電力ビッグスワンスタジアム',
    'ShonanBMWスタジアム平塚': 'ＳｈｏｎａｎＢＭＷスタジアム平塚',
    '瑞穂公園陸上競技場': 'パロマ瑞穂スタジアム',
    '岐阜メモリアルセンター長良川競技場': '岐阜メモリアルセンター長良川球技メドウ',
    '沖縄県総合運動公園陸上競技場': '高知県立春野総合運動公園陸上競技場',
    '鳥栖スタジアム': 'ベストアメニティスタジアム',
    'レベルファイブスタジアム': '東平尾公園博多の森球技場',
    'さいたま市浦和駒場スタジアム': '浦和市駒場スタジアム',
    '京都市西京極総合運動公園陸上競技場': '京都市西京極総合運動公園陸上競技場兼球技場',
    'ＩＡＩスタジアム日本平': '日本平スタジアム',
    'IAIスタジアム日本平': '日本平スタジアム',
    'アウトソーシングスタジアム日本平': '日本平スタジアム',
    '日本平運動公園球技場': '日本平スタジアム',
    '名古屋市瑞穂球技場': 'パロマ瑞穂スタジアム',
    'NDソフトスタジアム山形': 'ＮＤソフトスタジアム山形',
    '山形県総合運動公園陸上競技場': 'ＮＤソフトスタジアム山形',
    'トランスコスモススタジアム長崎': '長崎県立総合運動公園陸上競技場',
    '駒沢陸上競技場': '駒沢オリンピック公園総合運動場陸上競技場',
    '松本平広域公園総合球技場': 'サンプロアルウィン',
    '維新百年記念公園陸上競技場': '維新みらいふスタジアム',
    'ｋａｎｋｏスタジアム': 'シティライトスタジアム',
    '岡山県陸上競技場桃太郎スタジアム': 'シティライトスタジアム',
    'ニンジニアスタジアム': '愛媛県総合運動公園陸上競技場',
    '仙台スタジアム': 'ユアテックスタジアム仙台',
    '七北田公園仙台スタジアム': 'ユアテックスタジアム仙台',
    '栃木県立グリーンスタジアム': '栃木県グリーンスタジアム',
    '徳島県鳴門総合運動公園陸上競技場': '鳴門・大塚スポーツパークポカリスエットスタジアム',
    '長野運動公園総合運動場': '長野市営長野運動公園総合運動場',
    '山梨中銀スタジアム': '山梨県小瀬スポーツ公園陸上競技場',
    'とりぎんバードスタジアム': '鳥取市営サッカー場バードスタジアム',
    '埼玉県営大宮公園サッカー場': 'ＮＡＣＫ５スタジアム大宮',
    'さいたま市大宮公園サッカー場': 'ＮＡＣＫ５スタジアム大宮',
    'NACK5スタジアム大宮': 'ＮＡＣＫ５スタジアム大宮',
    '三協フロンテア柏スタジアム': '日立柏サッカー場',
    '正田醤油スタジアム群馬': '群馬県立敷島公園県営陸上競技場',
    '平塚競技場': 'ＳｈｏｎａｎＢＭＷスタジアム平塚',
    'ジュビロ磐田スタジアム': 'ジュビロ磐田サッカースタジアム',
    'ヤマハスタジアム（磐田）': 'ジュビロ磐田サッカースタジアム',
    '横浜市三ツ沢公園球技場': 'ニッパツ三ツ沢球技場',
    '函館市千代台公園陸上競技場': '千代台公園陸上競技場',
    '水戸市立競技場': 'ケーズデンキスタジアム水戸',
    '西が丘サッカー場': '国立スポーツ科学センター西が丘サッカー場',
    '味の素フィールド西が丘': '国立スポーツ科学センター西が丘サッカー場',
    '国立西が丘サッカー場': '国立スポーツ科学センター西が丘サッカー場',
    '埼玉スタジアム２００２': '埼玉スタジアム2002'
}


#######################################################################
#######################################################################
######## STEP 1 Loading Datasets Given By Competition Host ############
#######################################################################
#######################################################################

def load_and_merge_initial_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # Load the train, test csv file

    merged = pd.concat([train, test], ignore_index=True)
    merged = merged[[
        'id', 'attendance', 'away_team', 'broadcasters', 'home_team',
        'humidity', 'kick_off_time', 'match_date', 'round', 'section',
        'temperature', 'venue', 'weather'
    ]]
    # limit the columns

    tidying_venue_name(merged)
    tidying_team_name(merged)
    # Set the team and stadium names straight.

    cap = pd.read_csv('stadium_capacity_mapping.csv')
    cap.rename(columns={'stadium': 'venue'}, inplace=True)
    tidying_venue_name(cap)

    cap_dupl_check = cap.venue.value_counts()
    # check how many duplicate stadium data are there
    more_than_two = cap_dupl_check[cap_dupl_check >= 2].index

    # There are no cases in which attendances change
    # when stadium names are changed.

    cap[cap.venue.isin(more_than_two)]
    cap = cap.drop_duplicates()
    # Drop all duplicates

    df = merged.merge(right=cap, how='left', on='venue')
    # merge the stadium data back in!

    return df


#######################################################################
#######################################################################
############ STEP 2 Set Extra Data J1, J2 League From 1993 ############
#######################################################################
#######################################################################


def set_extra_j1_j2_data(df):
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

def add_venue_area_info(df, gmaps):
    venues = df['venue'].dropna().unique()

    venue_areas = []

    for venue in venues:
        raw_data = gmaps.geocode(address=venue, language='jp')
        try:
            venue_area = raw_data[0]['formatted_address']
            venue_areas.append(venue_area)
        except:
            print('The internet seems to be unstable try again')
            raise IndexError

    venue_area_df = pd.DataFrame(
        data={'venue': venues, 'venue_area': venue_areas})

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
################## STEP 6 Create and Add Baseball Data ################
#######################################################################
#######################################################################

def add_baseball_data(df, gmaps, crawling=False):
        # The entire function which crawls data from the internet, then
        # calls it to merge the data into our trainingset!

    def baseball_extract(year):

        year = str(year)
        month = [str(x) for x in range(4, 12)]

        # round, match date, home team, away team, venue, home team wins
        # in the season before the game,home team loses in the season
        # before the game, home team no decision before the game,
        # attendance, post-season game dummy
        round_b, date, home_team_b, away_team_b, venue_b, home_win, \
            home_lose, home_nodcsn, attendance_b, tournament \
            = [], [], [], [], [], [], [], [], [], []

        for mon in month:

            cal_url = 'http://npb.jp/bis/{}/calendar/index_{}.html'.format(
                year, mon.zfill(2))
            cal_r = requests.get(cal_url)
            cal_r.encoding = 'utf-8'
            cal_html = cal_r.text
            cal_soup = BS(cal_html)

            links = cal_soup.find_all('a')
            game_links = []
            for link in links:
                page = link.get('href')
                # html with length of 35 contains match data.
                if len(page) == 35:
                    game_links.append(page)

            if len(game_links) == 0:
                continue
            else:
                for game_link in game_links:
                    game_url = 'http://npb.jp{}'.format(game_link)
                    game_r = requests.get(game_url)
                    game_r.encoding = 'uft-8'     # change the encoding format
                    game_html = game_r.text
                    game_soup = BS(game_html)

                    try:
                        game_breaker = game_soup.find(
                            'td', {'align': "right", 'class': 'gmout'}).string
                        continue
                    except:
                        if game_soup.find(
                                'div', {'class': 'gmdivnumber'}) is None:
                            continue   # To pass the day of no game
                            # data, like the case of cancellation
                            # due to heavy rain

                        else:
                            round_performance = game_soup.find(
                                'div', {'class': 'gmdivnumber'}).string
                            try:
                                round_val = re.match(
                                    '\d+回戦', round_performance).group()
                            except:
                                pass
                            try:
                                performance = re.search(
                                    '(\d+)勝(\d+)敗(\d+)分',
                                    round_performance).groups()

                            except:
                                continue
                            win_val = int(performance[0])
                            lose_val = int(performance[1])
                            nodcsn_val = int(performance[2])

                            # Code to calculate win, lose, and no
                            # decision just before the game: to
                            # prevent leakage problem

                            score_finder = [x.string for x in game_soup.find_all(
                                'td', {'align': 'center', 'class': 'gmscore'})]
                            away_score = int(score_finder[1])
                            home_score = int(score_finder[-3])
                            if home_score > away_score:
                                win_val = win_val - 1
                            elif home_score < away_score:
                                lose_val = lose_val - 1
                            elif home_score == away_score:
                                nodcsn_val = nodcsn_val - 1
                            else:
                                raise ValueError('{},{}'.format(h, a))

                            # Attendance
                            attend_raw = game_soup.find_all(
                                'td', {'align': 'right'})[0].string
                            attend_val = re.search(
                                '(\d+),(\d+)', attend_raw).groups()
                            attend_val = int(attend_val[0] + attend_val[1])

                            # Venue
                            venue_val = game_soup.find_all('td')[9].string
                            venue_val

                            # Match date / Home and away team
                            date_teams_raw = game_soup.find('title').string
                            date_raw = [x.zfill(2) for x in re.search(
                                '(\d+)年(\d+)月(\d+)日', date_teams_raw).groups()]
                            date_val = '-'.join(date_raw)

                            teams_raw = re.search(
                                '(\w+)vs(\w+)', date_teams_raw).groups()
                            hometeam_val = teams_raw[0]
                            awayteam_val = teams_raw[1]

                            # Dummy variable to indicate whether it is
                            # the post-season game
                            if mon in ['10', '11'] and len(str(round_val)) == 1:
                                tournament_val = 1
                            else:
                                tournament_val = 0

                            round_b.append(round_val)
                            date.append(date_val)
                            home_team_b.append(hometeam_val)
                            away_team_b.append(awayteam_val)
                            venue_b.append(venue_val)
                            home_win.append(win_val)
                            home_lose.append(lose_val)
                            home_nodcsn.append(nodcsn_val)
                            attendance_b.append(attend_val)
                            tournament.append(tournament_val)

        baseball_data = pd.DataFrame(
            data={
                'round_b': round_b, 'date': date, 'home_team_b': home_team_b,
                'away_team_b': away_team_b, 'venue_b': venue_b,
                'home_win': home_win, 'home_lose': home_lose,
                'home_nodcsn': home_nodcsn, 'attendance_b': attendance_b,
                'tournament': tournament
            }
        )

        baseball_data.to_csv('{}.csv'.format(year), index=False)
        print('{} finished'.format(year))

    if crawling == True:
        try:
            # To prevent server error, splited whole period to 2 blocks.
            for x in range(2005, 2013):
                baseball_extract(x)

            for x in range(2013, 2019):
                baseball_extract(x)

            bb_list = glob('20*.csv')
            for i, file in enumerate(bb_list):
                if i == 0:
                    merged_bb_data_1 = pd.read_csv(file)
                else:
                    bb_year = pd.read_csv(file)
                    merged_bb_data_1 = pd.concat([merged_bb_data_1, bb_year])
            case = 1
        # When we crawl on NPB website, we found there were lot of
        # server problems.
        # SO in case the crawling failed, we attached pre-crawled
        # dataset of baseball.
        except:
            bb_list = glob('pre_20*.csv')
            print('baseball crawling erorr occured load pre-crawled dataset')
            for i, file in enumerate(bb_list):
                if i == 0:
                    merged_bb_data_2 = pd.read_csv(file)
                else:
                    bb_year = pd.read_csv(file)
                    merged_bb_data_2 = pd.concat([merged_bb_data_2, bb_year])
            case = 2

    else:
        print('selected not to crawl baseball data')
        print('load pre_crawled dataset')
        bb_list = glob('pre_20*.csv')
        for i, file in enumerate(bb_list):
            if i == 0:
                merged_bb_data_2 = pd.read_csv(file)
            else:
                bb_year = pd.read_csv(file)
                merged_bb_data_2 = pd.concat([merged_bb_data_2, bb_year])
        case = 2

    if case == 1:
        merged_bb_data = merged_bb_data_1.copy()
    elif case == 2:
        merged_bb_data = merged_bb_data_2.copy()

    merged_bb_data['venue_b'] = merged_bb_data[
        'venue_b'].apply(lambda x: ''.join(x.split('\u3000')))

    # If the name of a venue is different by year, we unificated
    # it for better prediction.
    venue_name_changer = {'グッドウィル': '西武ドーム',
                          'インボイス': '西武ドーム',
                          'ＱＶＣマリン': 'マリン',
                          '千葉マリン': 'マリン',
                          'QVCマリン': 'マリン',
                          'ZOZOマリン': 'マリン',
                          'ヤフードーム': 'ヤフオクドーム',
                          'スカイマーク': 'ほっと神戸',
                          'フルスタ宮城': '宮城',
                          'Ｋスタ宮城': '宮城',
                          'Kスタ宮城': '宮城',
                          'コボスタ宮城': '宮城',
                          'Koboスタ宮城': '宮城',
                          'Koboパーク宮城': '宮城',
                          '楽天生命パーク': '宮城',
                          'マツダ': '広島',
                          '西武プリンス': '西武ドーム',
                          'メットライフ': '西武ドーム',
                          '京セラＤ大阪': '大阪ドーム',
                          '京セラD大阪': '大阪ドーム'
                          }

    venue_to_change = venue_name_changer.keys()

    merged_bb_data.loc[merged_bb_data['venue_b'].isin(venue_to_change), 'venue_b'] = merged_bb_data.loc[
        merged_bb_data['venue_b'].isin(venue_to_change), 'venue_b'].apply(lambda x: venue_name_changer[x])

    # In this procedure, we changed venue names to fit with the
    # official name by NPB.
    # Source: http://npb.jp/stadium/
    venues = merged_bb_data.venue_b.unique().tolist()

    official_venues = [
        '西武ドーム', '千葉マリンスタジアム', 'ヤフオクドーム', '札幌ドーム', '大阪ドーム', '東京ドーム', 'ナゴヤドーム',
        '楽天生命パーク宮城', 'ほっと神戸', '明治神宮野球場', '横浜スタジアム', '広島市民球場',
        '福島県営あづま球場', 'いわきグリーンスタジアム', '阪神甲子園球場', '豊橋市民球場', '北九州市民球場',
        '松山中央公園野球場', '長崎県営野球場', '山形県野球場', '佐賀県立森林公園野球場',
        '西京極総合運動公園野球場', '福山市竹ヶ端運動公園野球場', '南長野運動公園野球場',
        '岡山県倉敷スポーツ公園野球場', '札幌市円山球場', '広島県立びんご運動公園野球場',
        '静岡県草薙総合運動場硬式野球場', '秋田県立野球場', '米子市民球場', '富山市民球場',
        '鹿児島県立鴨池野球場', '岩手県営野球場', '福井県営球場', '岐阜県営長良川球場', '石川県立野球場',
        '釧路市民球場', '帯広の森野球場', '浜松市営球場', 'ひたちなか市民球場', '平塚球場',
        '花咲スポーツ公園硬式野球場', '群馬県立敷島公園野球場', '千代台公園野球場', '藤崎台県営野球場',
        '下関球場', '小瀬スポーツ公園野球場', '山口市スポーツの森', '呉市二河野球場', '宇都宮清原球場',
        '宮崎県総合運動公園硬式野球場', '大洲総合運動公園硬式野球場', '埼玉県営大宮公園野球場', '山形県野球場',
        '相模原市立相模原球場', 'みよし運動公園野球場', '松本市野球場', '新潟県立鳥屋野潟公園野球場', '西京極総合運動公園野球場',
        '郡山総合運動場開成山野球場', '埼玉県営大宮公園野球場', '那覇市営奥武山野球場', '宇部市野球場',
        '皇子山総合運動公園野球場', '宮崎県総合運動公園硬式野球場', '弘前市運動公園野球場', '山形市営球場'
    ]

    venues_to_official = dict(zip(venues, official_venues))

    merged_bb_data['venue_b'] = merged_bb_data[
        'venue_b'].apply(lambda x: venues_to_official[x])

    #  Adding dummy about all star game & post-season matches
    merged_bb_data['round_b'] = merged_bb_data[
        'round_b'].apply(lambda x: re.match('\d+', x)[0])

    merged_bb_data['year'] = merged_bb_data[
        'date'].apply(lambda x: x.split('-')[0])
    merged_bb_data['month'] = merged_bb_data[
        'date'].apply(lambda x: x.split('-')[1])
    merged_bb_data['day'] = merged_bb_data[
        'date'].apply(lambda x: x.split('-')[2])

    merged_bb_data['month'] = merged_bb_data['month'].astype(int)
    merged_bb_data['round_b'] = merged_bb_data['round_b'].astype(int)

    july_round = [1, 2, 3]
    aug_round = [2]
    oct_nov_round = [1, 2, 3, 4, 5, 6, 7]
    merged_bb_data['tournament'] = 0
    merged_bb_data['allstar'] = 0

    merged_bb_data.loc[(merged_bb_data.round_b.isin(july_round)) & (
        merged_bb_data.month == 7), 'allstar'] = 1
    merged_bb_data.loc[(merged_bb_data.round_b.isin(aug_round)) & (
        merged_bb_data.month == 8), 'allstar'] = 1
    merged_bb_data.loc[(merged_bb_data.round_b.isin(oct_nov_round)) & (
        merged_bb_data.month.isin([10, 11])), 'tournament'] = 1

    merged_bb_data.loc[merged_bb_data['home_team_b']
                       == 'セ', 'home_team_b'] = 'セントラル'
    merged_bb_data.loc[merged_bb_data['away_team_b']
                       == 'セ', 'away_team_b'] = 'セントラル'
    merged_bb_data.loc[merged_bb_data['home_team_b']
                       == 'パ', 'home_team_b'] = 'パシフィック'
    merged_bb_data.loc[merged_bb_data['away_team_b']
                       == 'パ', 'away_team_b'] = 'パシフィック'

    merged_bb_data.loc[(merged_bb_data['away_team_b'] == 'パシフィック') & (
        merged_bb_data['home_team_b'] == 'リーグ'), 'home_team_b'] = 'セントラル'
    merged_bb_data.loc[(merged_bb_data['away_team_b'] == 'セントラル') & (
        merged_bb_data['home_team_b'] == 'リーグ'), 'home_team_b'] = 'パシフィック'

    # Adding information about team's hometown
    teams_b = {'セ', '東京ヤクルト', '横浜', 'パ', 'ヤクルト', '埼玉西武', 'パシフィック', '広島東洋',
               'セントラル', '読売', '福岡ソフトバンク', 'ＳＫ', 'オリックス', '西武', '中日', 'ＫＩＡ',
               '千葉ロッテ', '阪神', '北海道日本ハム', '東北楽天', 'リーグ', '横浜DeNA'}

    team_area = [None, 13000, 14000, None, 13000, 11000, 50000, 34000, 51000, 13000,
                 40000, 52000, 27000, 11000, 23000, 52000, 12000, 28000, 1000,
                 4000, None, 14000]

    # パシフィック: 50000, セントラル: 51000, SK&KIA(Korean teams): 52000
    #   -> Attached by ourself.
    team_area_b = pd.DataFrame(data={'team_b': teams_b, 'area_b': team_area})

    team_area_dict = dict(zip(teams_b, team_area))

    merged_bb_data['home_area_b'] = merged_bb_data[
        'home_team_b'].apply(lambda x: team_area_dict[x])
    merged_bb_data['away_area_b'] = merged_bb_data[
        'away_team_b'].apply(lambda x: team_area_dict[x])

    # Adding derby match dummy
    kansai_derby = [27000, 28000]

    merged_bb_data['derby_b'] = 0
    merged_bb_data.loc[(merged_bb_data['home_area_b'] ==
                        merged_bb_data['away_area_b']), 'derby_b'] = 1
    merged_bb_data.loc[(merged_bb_data['home_area_b'].isin(kansai_derby)) & (
        merged_bb_data['away_area_b'].isin(kansai_derby)), 'derby_b'] = 1

    # Unificating team name for better prediction
    merged_bb_data.loc[merged_bb_data['home_team_b']
                       == '埼玉西武', 'home_team_b'] = '西武'
    merged_bb_data.loc[merged_bb_data['away_team_b']
                       == '埼玉西武', 'away_team_b'] = '西武'

    bb = merged_bb_data.copy()
    del bb['year'], bb['month'], bb['day']

    # Adding area & area code information to each baseball stadium
    # found by myself
    venue_b = [
        '西武ドーム', '千葉マリンスタジアム', 'ヤフオクドーム', '札幌ドーム', '大阪ドーム', '東京ドーム', 'ナゴヤドーム',
        '楽天生命パーク宮城', 'ほっと神戸', '明治神宮野球場', '横浜スタジアム', '広島市民球場', '福島県営あづま球場',
        'いわきグリーンスタジアム', '阪神甲子園球場', '豊橋市民球場', '北九州市民球場', '松山中央公園野球場',
        '長崎県営野球場', '山形県野球場', '佐賀県立森林公園野球場', '西京極総合運動公園野球場',
        '福山市竹ヶ端運動公園野球場', '南長野運動公園野球場', '岡山県倉敷スポーツ公園野球場', '札幌市円山球場',
        '広島県立びんご運動公園野球場', '静岡県草薙総合運動場硬式野球場', '秋田県立野球場', '米子市民球場',
        '富山市民球場', '鹿児島県立鴨池野球場', '岩手県営野球場', '福井県営球場', '岐阜県営長良川球場', '石川県立野球場',
        '釧路市民球場', '帯広の森野球場', '浜松市営球場', 'ひたちなか市民球場', '平塚球場', '花咲スポーツ公園硬式野球場',
        '群馬県立敷島公園野球場', '千代台公園野球場', '藤崎台県営野球場', '下関球場', '小瀬スポーツ公園野球場',
        '山口市スポーツの森', '呉市二河野球場', '宇都宮清原球場', '宮崎県総合運動公園硬式野球場', '大洲総合運動公園硬式野球場',
        '埼玉県営大宮公園野球場', '相模原市立相模原球場', 'みよし運動公園野球場', '松本市野球場', '新潟県立鳥屋野潟公園野球場',
        '郡山総合運動場開成山野球場'
    ]

    venue_area_code_b = [
        11000, 12000, 40000, 1000, 27000, 13000, 23000, 45000, 28000, 13000,
        14000, 34000, 7000, 7000, 28000, 23000, 40000, 40000, 42000, 6000,
        41000, 26000, 34000, 20000, 33000, 1000, 34000, 22000, 5000, 31000,
        16000, 46000, 3000, 18000, 21000, 17000, 1000, 1000, 22000, 8000,
        14000, 1000, 10000, 1000, 43000, 35000, 19000, 35000, 44000, 34000,
        9000, 45000, 11000, 14000, 34000, 11000, 15000, 7000
    ]

    pref_code = pd.read_csv('prefecture_code.csv')
    pref_code = pref_code[['AREA', 'AREA Code']].drop_duplicates()
    pref_code.columns = ['area', 'area_code']

    bb_venue_area_code = pd.DataFrame(
        data={'venue_b': venue_b, 'venue_area_code_b': venue_area_code_b}
    )

    bb_area_code_done = bb.merge(
        right=bb_venue_area_code, how='left', on='venue_b'
    )

    bb_area_done = bb_area_code_done.merge(
        right=pref_code, how='left',
        left_on='venue_area_code_b', right_on='area_code'
    )

    del bb_area_done['area_code']
    bb_area_done.rename(columns={'area': 'venue_area_b'}, inplace=True)

    # Merging to main dataset
    d6_2 = bb_area_done.copy()
#     d6_2['date'] = pd.to_datetime(d6_2['date'])
    df_b = df.merge(
        right=d6_2, how='left', left_on=['match_date', 'venue_area'],
        right_on=['date', 'venue_area_b']
    )

    df_b['home_area_b'] = df_b['home_area_b'].astype(str)

    # For the case that two games are held same time at one region,
    # We created one more set of columns for baseball features to
    # include data about second game.

    two_finder = df_b.id.value_counts()

    two_index = two_finder[two_finder == 2].index

    df_two = df_b.loc[df_b.id.isin(two_index)]
    df_two.reset_index(inplace=True)
    del df_two['index']

    two_even = df_two.loc[df_two.index % 2 == 0]
    two_even.reset_index(inplace=True)
    del two_even['index']
    two_odd = df_two.loc[df_two.index % 2 == 1]
    two_odd.reset_index(inplace=True)
    del two_odd['index']

    two_add = two_even.merge(
        right=two_odd, how='left',
        on=[
            'attendance', 'away_team', 'broadcasters', 'capacity', 'division',
            'home_team', 'humidity', 'id', 'kick_off_time', 'match_date',
            'round', 'section', 'temperature', 'venue', 'weather',
            'venue_area_code', 'venue_area', 'home_team_area_code',
            'home_team_area', 'away_team_area_code', 'away_team_area'
        ]
    )

    col_x = [x for x in list(two_add) if '_x' in x]
    col_y = [y for y in list(two_add) if '_y' in y]
    col_fix1 = [x[:-2] + '_1' for x in col_x]
    col_fix2 = [x[:-2] + '_2' for x in col_y]

    col_fix = col_fix1 + col_fix2

    col_changer = dict(zip(col_x + col_y, col_fix))

    two_add.rename(columns=col_changer, inplace=True)

    df_col_to_change = [
        'round_b', 'home_team_b', 'away_team_b', 'venue_b', 'home_win',
        'home_lose', 'home_nodcsn', 'attendance_b', 'tournament', 'lat_b',
        'lon_b', 'allstar', 'derby_b', 'home_area_b', 'away_area_b'
    ]

    df_col_changed = [x + '_1' for x in df_col_to_change]

    df_col_changer = dict(zip(df_col_to_change, df_col_changed))

    df_b.rename(columns=df_col_changer, inplace=True)

    df_to_drop = df_b.loc[df_b.id.isin(two_add.id.values.tolist())].index

    df_b.drop(index=df_to_drop, inplace=True)

    df_two = pd.concat([df_b, two_add])

    drop_cols2 = [x for x in list(df_two) if x.startswith(
        'match') and (x.endswith('_1') or x.endswith('_2'))]

    for col in drop_cols2:
        del df_two[col]

    df_two.sort_values(by=['id'], inplace=True)

    d6_3 = df_two.copy()

    # Adding distance between soccer & baseball venue,
    # if both games are at same prefecture at same day.

    obj = d6_3[['id', 'venue', 'venue_b_1', 'venue_b_2']]

    # When two baseball games are held in same region at same day, we
    # put features of second game(chosen randomly) to '~~b_2_' columns
    # combination of stadiums 'venue  - venue_b_2 ' are all included in
    # 'venue - venue_b_1' cases.
    # SO just need to calculate distance & combination of
    # 'venue - venue_b_1' cases and apply it to  'venue  - venue_b_2 '.

    stadiums = obj[['venue', 'venue_b_1']].dropna().drop_duplicates()

    soccer_venue = stadiums.venue.tolist()
    baseball_venue = stadiums.venue_b_1.tolist()

    durations = []
    distances = []

    for dep, arr in zip(soccer_venue, baseball_venue):

        raw_data = gmaps.distance_matrix(dep, arr, language='jp')

        dist = raw_data['rows'][0]['elements'][0]['distance']['value'] / 1000
        distances.append(dist)

        duration = float(raw_data['rows'][0]['elements'][
                         0]['duration']['value']) / 60
        if type(duration) == float:
            durations.append(duration)
        else:
            print(dep, arr)
            durations.append(NaN)

    bf_dist = pd.DataFrame(
        data={'venue': soccer_venue, 'venue_b': baseball_venue,
              'duration': durations, 'distance': distances
              }
    )

    intersport = obj.merge(right=bf_dist, how='left',
                           left_on=['venue', 'venue_b_1'],
                           right_on=['venue', 'venue_b']
                           )

    del intersport['venue_b']

    intersport.rename(
        columns={'duration': 'duration_1', 'distance': 'distance_1'},
        inplace=True
    )

    intersport = intersport.merge(
        right=bf_dist, how='left', left_on=['venue', 'venue_b_2'],
        right_on=['venue', 'venue_b']
    )

    del intersport['venue_b']

    intersport.rename(
        columns={'duration': 'duration_2', 'distance': 'distance_2'},
        inplace=True
    )

    bb_merged_data = d6_3.merge(right=intersport, how='left', on=[
                                'id', 'venue', 'venue_b_1', 'venue_b_2']
                                )

    bb_merged_data.rename(
        columns={
            'duration_1': 'duration_bet_sports_1',
            'distance_1': 'dist_bet_sports_1',
            'duration_2': 'duration_bet_sports_2',
            'distance_2': 'dist_bet_sports_2'
        }, inplace=True
    )

    bb_merged_data = bb_merged_data.drop_duplicates()
    del bb_merged_data['date']

    df_done = bb_merged_data.copy()

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

def add_holiday_data(df):

    # data source:
    # http://zangyoukeisan.cocolog-nifty.com/blog/2011/09/
    # post-b23f.html
    holiday = pd.read_excel('holiday_extra.xls', sheet_name='振替休日あり')
    holiday = holiday[['年月日', '祝日名']]
    holiday.columns = ['holiday_date', 'description']
    holiday = holiday[holiday['holiday_date'] > '1992-12-21']

    date_df = pd.date_range(start='1992-12-21', end='2018-12-31', freq='D')
    date_df = pd.DataFrame(date_df, columns=['date'])

    date_df['wod'] = date_df['date'].dt.weekday
    date_df['is_weekend'] = (date_df['wod'] >= 5).astype(np.int8)

    date_df = date_df.merge(holiday, left_on='date',
                            right_on='holiday_date', how='left')
    date_df.drop('holiday_date', 1, inplace=True)

    date_df['is_holiday'] = date_df['description'].notnull().astype(np.int8)
    date_df['is_dayoff'] = date_df['is_weekend'] + date_df['is_holiday']

    def get_elapsed(fld, pre):
        day1 = np.timedelta64(1, 'D')
        last_date = np.datetime64()
        res = []

        for v, d in zip(date_df[fld].values, date_df['date'].values):

            if v != 0:
                last_date = d

            res.append(((d - last_date).astype('timedelta64[D]') / day1))
        date_df[pre + '_' + fld] = res

    fld = 'is_dayoff'

    get_elapsed(fld, 'After')

    date_df = date_df.sort_values('date', ascending=False)
    get_elapsed(fld, 'Before')

    date_df = date_df.sort_values('date')

    fld = 'is_holiday'

    get_elapsed(fld, 'After')

    date_df = date_df.sort_values('date', ascending=False)
    get_elapsed(fld, 'Before')

    date_df = date_df.sort_values('date')

    df['match_date'] = pd.to_datetime(df['match_date'])

    d8 = df.merge(date_df, left_on='match_date', right_on='date', how='left')
    df_done = preprocess(d8)

    return df_done

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
        ['id', 'match_Year', 'match_date', 'home_team']
    ].drop_duplicates().rename(columns={'home_team': 'team'})

    board1['home_game'] = 1

    board2 = df[
        ['id', 'match_Year', 'match_date', 'away_team']
    ].drop_duplicates().rename(columns={'away_team': 'team'})

    board2['away_game'] = 1

    board = pd.concat([board1, board2])
    """
    we had two columns which consists data about the team of the match.
    away_team and home_team. However we set them into the same column
    and delete the away_team and home_team columns
    """

    board['home_game'] = board['home_game'].fillna(0)
    board['away_game'] = board['away_game'].fillna(0)

    for sea in range(1993, 2019):
        teams = board.loc[(board['match_Year'] == sea),
                          'team'].unique().tolist()

        for team in teams:
            temp = board.loc[(board['match_Year'] == sea)
                             & (board['team'] == team)]
            for i, row in temp.iterrows():
                match_date = row['match_date']
                board.loc[board.index == i, 'section'] = \
                    len(
                        temp.loc[(temp['match_Year'] == sea)
                                 & (temp['match_date'] < match_date)
                                 & (temp['team'] == team)
                                 ]
                )
    # for all years in the dataset, count how many games a team had
    # before each game put them in the 'section' column

    board.loc[board.home_game == 1, 'home_th_match'] = board.loc[
        board.home_game == 1, 'section']
    board.loc[board.away_game == 1, 'away_th_match'] = board.loc[
        board.away_game == 1, 'section']
    # divide the data for home_teams and away_teams

    away_match = board.loc[(board.away_game == 1)][['id', 'away_th_match']]
    home_match = board.loc[(board.home_game == 1)][['id', 'home_th_match']]
    df = df.merge(away_match, how='left', on='id')
    df = df.merge(home_match, how='left', on='id')

    return df


def set_simple_win_ratio(df):
    """
    Warning!!! THIS FUNCTION IS EXTREMELY SLOW!!!!
    This function gets the simple win ratio for home and awayteams
    The code is extremely long thus I believe it deserves some
    explanation
    """

    in_match = pd.read_csv('match_reports.csv')
    out_match = pd.read_csv('ex_match_reports.csv')

    match = pd.concat([in_match, out_match])
    match = match.drop_duplicates(subset='id').sort_values(by='id')
    # load the match result data to find which team won and lost

    def df_of_prev_games_team_for_ratio(match_date,
                                        division,
                                        team,
                                        num_of_prev_games_to_consider,
                                        match_Year,
                                        df_of_match_result,
                                        df_of_game_sections):

        # this function gets the previous games of a team so that
        # the win loss occurences can be calculated. for each row of
        # the df

        df_of_prev_games = df_of_game_sections[
            (df_of_game_sections['division'] == division)
            & (df_of_game_sections['match_Year'] == match_Year)
            & (df_of_game_sections['match_date'] < match_date)
        ]
        # first prepare by division, match_Year and all matches before
        # the date

        df_of_prev_games_with_match_results = df_of_prev_games[
            (df_of_prev_games['home_team'] == team)
            | (df_of_prev_games['away_team'] == team)
        ].sort_values(by=['match_date']
                      ).tail(num_of_prev_games_to_consider)
        # then get all the games for the designated team and cut them
        # by the value we have set.

        df_of_prev_games_with_match_results = \
            df_of_prev_games_with_match_results.merge(
                df_of_match_result, on='id', how='left')
        # then merge it with the result data

        # return False if we don't have enough games.
        # ex) win_ratio for 3G needs at least 3 previous games of a
        # team.
        # if this is the second game of a team, this function will
        # return False
        if df_of_prev_games_with_match_results.shape[0] \
                < num_of_prev_games_to_consider:
            game_number_satisfied = False
            return game_number_satisfied, df_of_prev_games_with_match_results

        else:
            game_number_satisfied = True
            return game_number_satisfied, df_of_prev_games_with_match_results

    def win_ratio_calc(df, team, game_number_satisfied,
                       num_of_prev_games_to_consider):
        # this function calculates the win_ratio of the team
        # from the df given by the
        # "df_of_prev_games_team_for_ratio" function above

        if game_number_satisfied:
            win_freq = int(
                df[(df['away_team'] == team) & (df['away_team_score']
                                                > df['home_team_score'])].shape[0])
            + int(
                df[(df['home_team'] == team) & (df['away_team_score']
                                                < df['home_team_score'])].shape[0])

            draw_freq = int(
                df[df['away_team_score'] == df['home_team_score']].shape[0])

            win_ratio = (win_freq + (draw_freq * 0.5)) / \
                num_of_prev_games_to_consider

        # calculate the ratio in the if statement above
            return win_ratio
        else:
            win_ratio = None
            return win_ratio

    def win_ratio_for_team(row, home_or_away,
                           df_of_match_result,
                           df_of_game_sections,
                           num_of_prev_games_to_consider):

        # This function combines the two functions above and returns
        # the win_loss ratio for a given team!

        if home_or_away == 'home':
            team = row.home_team
        elif home_or_away == 'away':
            team = row.away_team

        # set which team of the row will have their win loss ratio
        # calculated

        # get the win-ratio for both home and away teams in a given
        # game
        match_Year = row.match_Year
        match_date = row.match_date
        division = row.division

        game_num_satisfied, df_for_win_ratio_calc = \
            df_of_prev_games_team_for_ratio(
                match_date=match_date,
                division=division, team=team,
                num_of_prev_games_to_consider=num_of_prev_games_to_consider,
                match_Year=match_Year,
                df_of_match_result=df_of_match_result,
                df_of_game_sections=df_of_game_sections
            )

        win_ratio = win_ratio_calc(
            df=df_for_win_ratio_calc,
            team=team,
            game_number_satisfied=game_num_satisfied,
            num_of_prev_games_to_consider=num_of_prev_games_to_consider)

        return win_ratio

    # Finally get the win ratio for both home and away teams
    df['home_team_avg_win_ratio_3G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='home', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=3
        ), axis=1
    )

    df['away_team_avg_win_ratio_3G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='away', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=3
        ), axis=1
    )

    print("Simple Win 3G is DONE!!")

    df['home_team_avg_win_ratio_5G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='home', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=5
        ), axis=1
    )

    df['away_team_avg_win_ratio_5G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='away', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=5
        ), axis=1
    )

    print("Simple Win 5G is DONE!!")

    df['home_team_avg_win_ratio_7G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='home', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=7
        ), axis=1
    )

    df['away_team_avg_win_ratio_7G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='away', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=7
        ), axis=1
    )

    print("Simple Win 7G is DONE!!")

    df['home_team_avg_win_ratio_9G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='home', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=9
        ), axis=1
    )

    df['away_team_avg_win_ratio_9G'] = df.apply(
        lambda row: win_ratio_for_team(
            row=row, home_or_away='away', df_of_match_result=match,
            df_of_game_sections=df, num_of_prev_games_to_consider=9
        ), axis=1
    )

    print("Simple Win 9G is DONE!!")

    return df


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

def set_population_data(df):
    """
    Set population data for each stadiums to get how many
    people are living in the prefucture which the stadium
    is located at
    """
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['match_year'] = df['match_date'].dt.year
    df['match_month'] = df['match_date'].dt.month
    df['section'] = df['section'].apply(
        lambda x: x[1:-1]).astype(np.int)

    pop = pd.read_csv('Perfect_population.csv',
                      usecols=[
                          'SURVEY YEAR', 'AREA Code', 'AREA',
                          'A1101_Total population (Both sexes)[person]'
                      ]
                      )

    # load the population data!

    pop = pop[pop['SURVEY YEAR'] >= 1992]
    pop.columns = ['year', 'area_code', 'area', 'population']
    pop['population'] = pop['population'].apply(
        lambda x: re.sub(',', '', x))
    pop['population'] = pop['population'].astype(np.int)
    pop = reset_index(pop)
    # get only population data from 1992 and set the minto integers

    # since we statistics japan does not yet offer data for
    # year 2016 and 2018 we guess the population for the year!
    change_ratio = (
        1 + (
            pop[
                (pop['area'] == 'All Japan') & (pop['year'] == 2016)
            ]['population'].values[0]
            - pop[(pop['area'] == 'All Japan')
                  & (pop['year'] == 2015)]['population'].values[0]
        ) / pop[
            (pop['area'] == 'All Japan')
            & (pop['year'] == 2015)
        ]['population'].values[0]
    )

    last_index = len(pop)

    for ind, x in pop[pop['year'] == 2016].iterrows():
        vals = x['year'] + \
            1, x['area_code'], x['area'], x['population'] * change_ratio

        pop.loc[last_index] = vals

        last_index += 1

    change_ratio = (
        1 + (
            pop[
                (pop['area'] == 'All Japan') & (pop['year'] == 2017)
            ]['population'].values[0]
            - pop[(pop['area'] == 'All Japan')
                  & (pop['year'] == 2016)]['population'].values[0]
        ) / pop[
            (pop['area'] == 'All Japan')
            & (pop['year'] == 2016)]['population'].values[0]
    )

    last_index = len(pop)

    for ind, x in pop[pop['year'] == 2017].iterrows():
        vals = x['year'] + \
            1, x['area_code'], x['area'], x['population'] * change_ratio

        pop.loc[last_index] = vals

        last_index += 1

    pop_added = df.merge(
        pop,
        left_on=['match_year', 'venue_area_code'],
        right_on=['year', 'area_code'],
        how='left'
    ).drop(['year', 'area_code'], 1)

    del pop_added['area']

    df = pop_added.copy()

    return df

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
################## STEP 13 Process Broadcaster Data ###################
#######################################################################
#######################################################################


def process_broadcasters(df):
    df['broad_list'] = df.broadcasters.str.split('/')

    nhk = set()
    for i, row in df.iterrows():
        broad_list = row['broad_list']
        if type(broad_list) == list:
            nhk.update(broad_list)

    broadcast = list(nhk)
    nhk_cast = [x for x in broadcast if 'ＮＨＫ' in x] + \
        [x for x in broadcast if 'NHK' in x]
    nhk_bs = [
        'NHK-BSハイビジョン',
        'NHK BS1 ※6 浦和がスルガ銀行チャンピオンシップ2017SAITAMAに出場のため、7月22日(土)に開催',
        'NHK BS1(101ch)',
        'NHK BS1(102ch)',
        'NHK BS1 ※AFCチャンピオンズリーグ2018の影響により、C大阪vs鹿島は7月25日(水)に開催',
        'NHK BS1(録)',
        'NHK BS1 ※4 AFCチャンピオンズリーグ2016の影響により、浦和vsFC東京は6月22日(水)に開催',
        'NHK BS1',
        'ＮＨＫ\u3000ＢＳハイビジョン',
        'ＮＨＫ\u3000ＢＳ１'
    ]

    pure_nhk = nhk_cast.copy()

    # Extract NHK channels except for NHK BS
    for x in nhk_bs:
        pure_nhk.remove(x)

    """
    I saw the data, and found that normal NHK channels like NHK 総合 have
    tendency shows games in unpopular times, like 2pm of weekday.
    Also lot of local NHK channels just show the game of their local
    teams.
    So I thought that NHK channels are showing J-league games
    regardless of the popularity of that game.
    Hence, I excluded NHK channels when I count the number of
    broadcasters in each game.
    However, I made an exception for NHK BS channels since they are pay
    channel, so they might have tendency to show games with popularity.
    I also deleted DAZN and SKY perfect channels,
    because they show almost every game so it is not good to calculate
    popularity of game by counting it.
    """

    for i, row in df.iterrows():
        broad_list = row['broad_list']
        try:
            df.loc[df.index == i, 'broad_list'] = \
                df.loc[df.index == i, 'broad_list'].apply(
                    lambda x: [a for a in x if 'スカ' not in a]
            )

            df.loc[df.index == i, 'broad_list'] = \
                df.loc[df.index == i, 'broad_list'].apply(
                    lambda x: [a for a in x if 'DAZN' not in a]
            )

            df.loc[df.index == i, 'broad_list'] = \
                df.loc[df.index == i, 'broad_list'].apply(
                    lambda x: [a for a in x if 'ＤＡＺＮ' not in a]
            )

            df.loc[df.index == i, 'broad_list'] = \
                df.loc[df.index == i, 'broad_list'].apply(
                lambda x: [a for a in x if a not in pure_nhk]
            )

        except:
            df.loc[df.index == i, 'broad_list'] = np.NaN

    df.loc[df.broad_list.notnull(), 'broad_num'] = df.loc[
        df.broad_list.notnull(), 'broad_list'].apply(lambda x: len(x))

    del df['broad_list'], df['broadcasters']

    df_done = df.copy()

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
################### STEP 15 Get and Set Emperor Cup Data ##############
#######################################################################
#######################################################################


def add_emperor_cup_data(df, crawling=False):
    # Crawling
    # Some pages have different html format, so crawling codes needed
    # to be different.
    # However codes for each format are almost same. But we didn't
    # put them together by using for loop or 'try-except' method,
    # because while we are using google colab we found some unstability
    # while crawling data when we use that kind of method.
    if crawling == True:
        # 85th cup (2005)
        time = '85'
        url = 'https://ja.wikipedia.org/wiki/%E7%AC%AC{}%E5%9B%9E%E5%A4%A9%E7%9A\
               %87%E6%9D%AF%E5%85%A8%E6%97%A5%E6%9C%AC%E3%82%B5%E3%83%83%E3%82%A\
               B%E3%83%BC%E9%81%B8%E6%89%8B%E6%A8%A9%E5%A4%A7%E4%BC%9A'.format(
            time)

        r = requests.get(url)
        r.encoding = 'utf-8'
        html_doc = r.text

        soup = BS(html_doc)

        game_dates = []
        home_teams = []
        away_teams = []
        home_scores = []
        away_scores = []

        for x in soup.find_all('li'):
            try:
                try:
                    match_result = re.search(
                        '\w+ \d - \d \w+',
                        x.text)
                    [0]

                    home_team = re.search(
                        '(\w+ *\w*) \d - \d (\w+ *\w*)',
                        match_result
                    ).groups()[0]

                    away_team = re.search(
                        '(\w+ *\w*) \d - \d (\w+ *\w*)',
                        match_result
                    ).groups()[1]

                    home_score = re.search(
                        '\w+ *\w* ((\d) - (\d)) \w+ *\w*',
                        match_result
                    ).groups()[1]

                    away_score = re.search(
                        '\w+ *\w* ((\d) - (\d)) \w+ *\w*',
                        match_result
                    ).groups()[2]

                    home_teams.append(home_team)
                    away_teams.append(away_team)
                    home_scores.append(home_score)
                    away_scores.append(away_score)
                    game_dates.append(nan)

                except:
                    match_result = re.search(
                        '\w+ \d - \d（延長\d - \d） \w+',
                        x.text
                    )[0]
                    home_team = re.search(
                        '(\w+) \d - \d（延長\d - \d） (\w+)',
                        match_result
                    ).groups()[0]
                    away_team = re.search(
                        '(\w+) \d - \d（延長\d - \d） (\w+)',
                        match_result
                    ).groups()[1]
                    home_score = re.search(
                        '(\w+) ((\d) - (\d))（延長\d - \d） (\w+)',
                        match_result
                    ).groups()[2]
                    away_score = re.search(
                        '(\w+) ((\d) - (\d))（延長\d - \d） (\w+)',
                        match_result
                    ).groups()[3]

                    home_teams.append(home_team)
                    away_teams.append(away_team)
                    home_scores.append(home_score)
                    away_scores.append(away_score)
                    game_dates.append(nan)

            except:
                continue

        for x in soup.find_all('div', {'style': 'text-align:right'}):
            date_raw = x.text
            game_date = date_raw.split('日')[0] + '日'
            game_dates.append(game_date)

        for x in soup.find_all('span',
                               {'class': 'summary',
                                'style': "display: none;"}):
            match_up = x.text
            home_team = match_up.split(' v ')[0]
            away_team = match_up.split(' v ')[1]

            home_teams.append(home_team)
            away_teams.append(away_team)

        game_dates_fixed = ['-'] * (len(home_teams)
                                    - len(game_dates)) + game_dates

        for x in soup.find_all('th', {'style': 'width:22%; text-align:center'}):
            score = x.text
            home_score = score.split(' - ')[0]
            away_score = score.split(' - ')[1].split(' (')[0].rstrip()

            home_scores.append(home_score)
            away_scores.append(away_score)

        emperor_85 = pd.DataFrame(
            data={
                'game_date': game_dates_fixed,
                'home_team': home_teams,
                'away_team': away_teams,
                'home_score': home_scores,
                'away_score': away_scores
            }
        )

        emperor_85 = emperor_85[-15:]
        emperor_85.reset_index(inplace=True)
        del emperor_85['index']

        emperor_85.iloc[:7]['game_date'] = '2005年12月10日'
        emperor_85.iloc[7]['game_date'] = '2005年12月17日'

        # 86 ~ 96th cup (2006~16)
        emperor = emperor_85.copy()

        for t in range(86, 97):
            time = str(t)
            url = 'https://ja.wikipedia.org/wiki/%E7%AC%AC{}%E5%9B%9E%E5%A4%A9%E7\
                   %9A%87%E6%9D%AF%E5%85%A8%E6%97%A5%E6%9C%AC%E3%82%B5%E3%83%83%E\
                   3%82%AB%E3%83%BC%E9%81%B8%E6%89%8B%E6%A8%A9%E5%A4%A7%E4%BC%9A'\
                   .format(time)

            r = requests.get(url)
            r.encoding = 'utf-8'
            html_doc = r.text

            soup = BS(html_doc)

            game_dates = []
            home_teams = []
            away_teams = []
            home_scores = []
            away_scores = []

            for x in \
                    soup.find_all(
                        'td',
                        {'style': 'width:13%; \
                         vertical-align:top; \
                         text-align:center;'}):
                home_score = re.search('(\d{1}) - (\d{1})', x.text).groups()[0]
                away_score = re.search('(\d{1}) - (\d{1})', x.text).groups()[1]
                home_scores.append(home_score)
                away_scores.append(away_score)

            for round_raw in \
                    soup.find_all(
                        'td',
                        {'style': "width:17%; \
                         vertical-align:top; \
                         text-align:right;"}):
                raw_data = round_raw.text.split(' ')
                if int(time) in range(92, 97):
                    game_date = raw_data[1].split('(')[0].rstrip()
                elif int(time) in range(87, 92):
                    game_date = raw_data[1].split('[')[0].rstrip()
                elif int(time) < 87:
                    game_date = raw_data[0]
                game_dates.append(game_date)

            for i, team in enumerate(
                    soup.find_all('span', {'class': 'fn org'})):
                team = team.text
                if i % 2 == 0:
                    home_teams.append(team)
                else:
                    away_teams.append(team)

            game_nums = [x for x in range(1, len(game_dates) + 1)]

            temp = pd.DataFrame(
                data={
                    'game_date': game_dates,
                    'home_team': home_teams,
                    'away_team': away_teams,
                    'home_score': home_scores,
                    'away_score': away_scores
                })[-15:]

            emperor = pd.concat([emperor, temp])

        # 97th cup (2017)
        time = 97
        url = 'https://ja.wikipedia.org/wiki/%E7%AC%AC{}%E5%9B%9E%E5%A4%A9%E7%9A%\
               87%E6%9D%AF%E5%85%A8%E6%97%A5%E6%9C%AC%E3%82%B5%E3%83%83%E3%82%AB%\
               E3%83%BC%E9%81%B8%E6%89%8B%E6%A8%A9%E5%A4%A7%E4%BC%9A'\
               .format(time)

        r = requests.get(url)
        r.encoding = 'utf-8'
        html_doc = r.text

        soup = BS(html_doc)

        game_nums = []
        game_dates = []
        home_teams = []
        away_teams = []
        home_scores = []
        away_scores = []

        for x in \
                soup.find_all(
                'td',
                {'style': 'width:13%; \
                vertical-align:top; \
                text-align:center;'}):
            if len(x.text) < 3:
                home_score = nan
                away_score = nan
                home_scores.append(home_score)
                away_scores.append(away_score)
            else:
                home_score = re.search('(\d{1}) - (\d{1})', x.text).groups()[0]
                away_score = re.search('(\d{1}) - (\d{1})', x.text).groups()[1]
                home_scores.append(home_score)
                away_scores.append(away_score)

        final_score = re.search(
            '\d{1} - \d{1}',
            soup.find('th', {'style': "width:22%; text-align:center"}).text
        )[0]

        final_home_score = final_score.split(' - ')[0]
        final_away_score = final_score.split(' - ')[1]
        home_scores.append(final_home_score)
        away_scores.append(final_away_score)

        for round_raw in \
                soup.find_all(
                    'td',
                    {'style': "width:17%; \
                	vertical-align:top; \
                	text-align:right;"}):
            raw_data = round_raw.text.split(' ')
            game_num = raw_data[0].split('.')[1]
            game_date = raw_data[1].split('(')[0].rstrip()
            game_nums.append(game_num)
            game_dates.append(game_date)

        final_date = re.search(
            '\w+日', soup.find('div', {'style': 'text-align:right'}).text
        )[0]

        game_dates.append(final_date)
        final_num = re.search(
            'No.(\d+)',
            soup.find(
                'div',
                {'style': 'text-align:right'}).text)[0].split('.')[1]

        game_nums.append(final_num)

        for i, team in enumerate(soup.find_all('span', {'class': 'fn org'})):
            team = team.text
            if i % 2 == 0:
                home_teams.append(team)
            else:
                away_teams.append(team)

        emperor_97 = pd.DataFrame(
            data={
                'game_date': game_dates,
                'home_team': home_teams,
                'away_team': away_teams,
                'home_score': home_scores,
                'away_score': away_scores})[-15:]

        emperor = pd.concat([emperor, emperor_97])

        # 98th cup (2018)
        time = 98
        url = 'https://ja.wikipedia.org/wiki/%E7%AC%AC{}%E5%9B%9E%E5%A4%A9%E7%9A%\
               87%E6%9D%AF%E5%85%A8%E6%97%A5%E6%9C%AC%E3%82%B5%E3%83%83%E3%82%AB%\
               E3%83%BC%E9%81%B8%E6%89%8B%E6%A8%A9%E5%A4%A7%E4%BC%9A'.\
            format(time)

        r = requests.get(url)
        r.encoding = 'utf-8'
        html_doc = r.text

        soup = BS(html_doc)

        game_nums = []
        game_dates = []
        home_teams = []
        away_teams = []
        home_scores = []
        away_scores = []

        for x in \
                soup.find_all(
                    'td',
                    {'style': 'width:13%; \
                	vertical-align:top; \
                	text-align:center;'}):
            if len(x.text) < 3:
                home_score = np.nan
                away_score = np.nan
                home_scores.append(home_score)
                away_scores.append(away_score)
            else:
                home_score = re.search('(\d{1}) - (\d{1})', x.text).groups()[0]
                away_score = re.search('(\d{1}) - (\d{1})', x.text).groups()[1]
                home_scores.append(home_score)
                away_scores.append(away_score)

        for round_raw in \
            soup.find_all(
                'td',
                {'style': "width:17%; \
        	    vertical-align:top; \
        	    text-align:right;"}):
            raw_data = round_raw.text.split(' ')
            game_num = raw_data[0].split('.')[1]
            game_date = raw_data[1].split('(')[0].rstrip()
            game_nums.append(game_num)
            game_dates.append(game_date)

        for i, team in enumerate(soup.find_all('span', {'class': 'fn org'})):
            team = team.text
            if i % 2 == 0:
                home_teams.append(team)
            else:
                away_teams.append(team)

        game_dates.insert(56, '2018年6月6日')
        home_scores.insert(56, '1')
        away_scores.insert(56, '1')

        emperor_98 = pd.DataFrame(
            data={
                'game_date': game_dates,
                'home_team': home_teams,
                'away_team': away_teams,
                'home_score': home_scores,
                'away_score': away_scores})[-15:]

        emperor = pd.concat([emperor, emperor_98])

    else:
        # we could or just load things!
        print('selected not to crawl emperor cup data, load!')
        emperor = pd.read_csv('emperor.csv')
    # Merging to main dataset
    temp = emperor.copy().dropna()
    temp.reset_index(inplace=True)
    del temp['index']
    temp['game_date'] = temp['game_date'].apply(lambda x: x.rstrip())
    temp['game_date'] = temp['game_date'].apply(
        lambda x: '-'.join([z.zfill(2) for z in re.findall('\d+', x)]))
    temp.home_score = temp.home_score.astype(int)
    temp.away_score = temp.away_score.astype(int)

    team_name_before = [
        'FC東京', '福島ユナイテッドFC', '湘南ベルマーレ', 'モンテディオ山形', '大分トリニータ', '明治大学',
        'ギラヴァンツ北九州', '水戸ホーリーホック', '横浜F･マリノス', '清水エスパルス', '徳島ヴォルティス',
        'サガン鳥栖', 'ザスパ草津', '京都サンガF.C.', '鹿島アントラーズ', 'アビスパ福岡', 'ジェフユナイテッド千葉',
        'セレッソ大阪', 'アルビレックス新潟', 'AC長野パルセイロ', '大宮アルディージャ', 'ベガルタ仙台', '柏レイソル',
        '筑波大学', '川崎フロンターレ', '愛媛FC', 'ロアッソ熊本', 'ザスパクサツ群馬', 'ガンバ大阪',
        'ヴィッセル神戸', 'サンフレッチェ広島', 'ジェフ千葉', '横浜F・マリノス', 'FC岐阜', '浦和レッズ',
        'コンサドーレ札幌', 'Honda FC', '横河武蔵野FC', 'ジュビロ磐田', '松本山雅FC', 'V・ファーレン長崎',
        '名古屋グランパスエイト', 'ヴァンフォーレ甲府', 'FC町田ゼルビア', '横浜FC', '名古屋グランパス',
        '東京ヴェルディ', '北海道コンサドーレ札幌', 'マリノス'
    ]

    team_name_after = [
        'FC東京', '福島ユナイテッドFC', '湘南', '山形', '大分', '明治大学', '北九州', '水戸', '横浜FM',
        '清水', '徳島', '鳥栖', '群馬', '京都.', '鹿島', '福岡', '千葉', 'Ｃ大阪', '新潟',
        'AC長野パルセイロ', '大宮', '仙台', '柏', '筑波大学', '川崎Ｆ', '愛媛', '熊本', '群馬', 'Ｇ大阪',
        '神戸', '広島', '千葉', '横浜FM', '岐阜', '浦和', '札幌', 'Honda FC', '横河武蔵野FC',
        '磐田', '松本', '長崎', '名古屋', '甲府', '町田', '横浜FC', '名古屋', '東京Ｖ', '札幌',
        '横浜FM'
    ]

    team_name_changer = dict(zip(team_name_before, team_name_after))
    temp['home_team'] = \
        temp['home_team'].apply(lambda x: team_name_changer[x])
    temp['away_team'] = \
        temp['away_team'].apply(lambda x: team_name_changer[x])

    temp['season'] = temp['game_date'].apply(lambda x: int(x.split('-')[0]))
    temp.loc[temp.game_date.str.endswith('-01-01'), 'season'] -= 1

    # Calculate the result of game by using score of both team.
    home_emp = temp[['game_date', 'home_team',
                     'home_score', 'away_score', 'season']]
    home_emp['criteria'] = home_emp['home_score'] - home_emp['away_score']
    del home_emp['home_score'], home_emp['away_score']

    away_emp = temp[['game_date', 'away_team',
                     'home_score', 'away_score', 'season']]
    away_emp['criteria'] = away_emp['away_score'] - away_emp['home_score']
    del away_emp['home_score'], away_emp['away_score']

    home_alive = []
    home_dead = []

    for i, row in home_emp.iterrows():
        if str(row['criteria']) != '0':
            continue
        else:

            if row['home_team'] in \
                    list(
                        temp.loc[(temp.index > i) & (temp.season == row['season']),
                                 'home_team'].values) \
                    + list(
                        temp.loc[(temp.index > i) & (temp.season == row['season']),
                                 'away_team'].values):
                home_alive.append(i)
            else:
                home_dead.append(i)

    home_emp.loc[home_emp.index.isin(home_alive), 'criteria'] = 10
    home_emp.loc[home_emp.index.isin(home_dead), 'criteria'] = -10

    away_alive = []
    away_dead = []

    for i, row in away_emp.iterrows():
        if str(row['criteria']) != '0':
            continue
        else:
            if row['away_team'] in \
                list(
                temp.loc[(temp.index > i) & (temp.season == row['season']),
                         'home_team'].values) \
                + list(temp.loc[(temp.index > i) & (temp.season == row['season']),
                                'away_team'].values):
                away_alive.append(i)
            else:
                away_dead.append(i)

    away_emp.loc[away_emp.index.isin(away_alive), 'criteria'] = 10
    away_emp.loc[away_emp.index.isin(away_dead), 'criteria'] = -10

    emp = pd.concat(
        [
            home_emp.rename(columns={'home_team': 'team'}),
            away_emp.rename(columns={'away_team': 'team'})
        ]
    ).sort_values(by='game_date')

    df['season'] = df['match_Year']

    emp_j1_teams = []
    for t in emp.team.unique():
        if t in df.home_team.unique().tolist():
            emp_j1_teams.append(t)

    emp['start_buff'] = np.nan
    emp['end_buff'] = np.nan

    # Start buff date: If teams proceeded to round 8, match date of round
    # 16 will be date of start buff.
    # End buff date: In case of teams who are not champions, last match of
    # emperor cup of that season will be the day of the buff ends.
    # In case of champion, the final match is first day of year, so we thought
    # that the buff will effective until March, the early part of next season.
    for team in emp_j1_teams:
        for year in range(2005, 2019):
            emp_dates = \
                emp.loc[
                    (emp.season == year) & (emp.team == team)
                ]['game_date'].tolist()

            emp_criteria = \
                emp.loc[
                    (emp.season == year) & (emp.team == team)
                ]['criteria'].tolist()

            if len(emp_dates) != 0:

                # if the team losts at round 16, they can't get buff.
                if emp_criteria[0] < 0:
                    emp.loc[(emp.season == year) & (
                        emp.team == team), 'start_buff'] = np.nan
                elif True:
                    emp.loc[
                        (emp.season == year)
                        & (emp.team == team),
                        'start_buff'
                    ] = emp_dates[0]
                    for i, crt in enumerate(emp_criteria):

                        # end day = the day team lost at emperor cup
                        if crt < 0:
                            emp.loc[
                                (emp.season == year)
                                & (emp.team == team),
                                'end_buff'
                            ] = emp_dates[i]

    # buff for the round 8 teams at 2018: Round 8 was held at 11/21, so
    # winners' buff are prolonged below.
    emp.loc[(emp.start_buff.notnull()) & (
        emp.end_buff.isnull()), 'end_buff'] = '2018-11-20'

    df['home_emp'] = 0
    df['away_emp'] = 0

    emp_start_end = emp[[
        'team',
        'season',
        'start_buff',
        'end_buff'
    ]].drop_duplicates().dropna()

    # End buff; the day before the 2018 final of emperor cup

    emp_start_end.loc[
        (emp_start_end.team == '山形') & (emp_start_end.season == 2018),
        'end_buff'
    ] = '2018-12-15'
    emp_start_end.loc[
        (emp_start_end.team == '鹿島') & (emp_start_end.season == 2018),
        'end_buff'
    ] = '2018-12-15'
    emp_start_end.loc[
        (emp_start_end.team == '浦和') & (emp_start_end.season == 2018),
        'end_buff'
    ] = '2018-12-15'
    emp_start_end.loc[
        (emp_start_end.team == '甲府') & (emp_start_end.season == 2018),
        'end_buff'
    ] = '2018-12-15'

    # As mentioned above, we thought that champion buff will be effective until March of next season,
    # since they get AFC champions league ticket for next season and it might make them more popular.
    # After many analysis, we decided that it is better to give buff only for
    # champion of emperor cup each season.
    for i, row in emp_start_end.iterrows():
        end = row['end_buff']
        if end.endswith('-01-01'):
            team = row['team']
            start = row['start_buff']
            season = row['season']

            df.loc[
                (df['season'] == season) & (df['home_team'] == team)
                & (df['match_date'] > start) & (df['match_date'] < end),
                'home_emp'
            ] = 1

            df.loc[
                (df['season'] == season) & (df['away_team'] == team)
                & (df['match_date'] > start) & (df['match_date'] < end),
                'away_emp'
            ] = 1

            df.loc[
                (df['season'] == season + 1)
                & (df['home_team'] == team)
                & (df['match_date'] > start)
                & (df['match_date'] < '{}-04-01'.format(str(season + 1))),
                'home_emp'
            ] = 1

            df.loc[
                (df['season'] == season + 1)
                & (df['away_team'] == team)
                & (df['match_date'] > start)
                & (df['match_date'] < '{}-04-01'.format(str(season + 1))),
                'away_emp'
            ] = 1

    df.loc[df.season < 2005, 'home_emp'] = np.nan
    df.loc[df.season < 2005, 'away_emp'] = np.nan

    df_done = df.copy()

    return df_done

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
######### STEP 17 Get and Set Data on Teams Playing AFC ###############
#######################################################################
#######################################################################


def add_afc_data(df, crawling=False):
    # Crawling data
    # Like emperor cup, different pages required different codes.

    # 2018
    if crawling == True:
        url = 'https://ja.wikipedia.org/wiki/AFC%E3%83%81%E3%83%A3%E3%83%B3%E3%83%\
              94%E3%82%AA%E3%83%B3%E3%82%BA%E3%83%AA%E3%83%BC%E3%82%B02018_%E6%B1%\
              BA%E5%8B%9D%E3%83%88%E3%83%BC%E3%83%8A%E3%83%A1%E3%83%B3%E3%83%88'

        r = requests.get(url)
        html_doc = r.text
        soup = BS(html_doc)

        game_dates = []
        home_teams = []
        away_teams = []
        home_scores = []
        away_scores = []

        for x in soup.find_all('th', {'style': "width:22%; text-align:center"}):
            score = x.text.split('(')[0]
            home_score = score.split(' - ')[0]
            away_score = score.split(' - ')[1].rstrip()
            home_scores.append(home_score)
            away_scores.append(away_score)

        for x in \
                soup.find_all('span', {'class': 'summary', 'style': 'display: none;'}):
            try:
                teams = x.text
                home_team = teams.split('  v  ')[0]
                away_team = teams.split('  v  ')[1]
            except:
                home_team = teams.split('  v ')[0]
                away_team = teams.split('  v ')[1]

            home_teams.append(home_team)
            away_teams.append(away_team)

        for x in soup.find_all('div', {'style': 'text-align:right'}):
            game_date = x.text.split('日')[0] + '日'
            game_dates.append(game_date)

        afc_18 = pd.DataFrame(
            data={
                'game_date': game_dates,
                'home_team': home_teams,
                'away_team': away_teams,
                'home_score': home_scores,
                'away_score': away_scores
            }
        )

        afc = afc_18[-14:]

        # 2013 ~ 2017
        for year in range(2013, 2018):
            time = str(year)

            url = 'https://ja.wikipedia.org/wiki/AFCチャンピオンズリーグ{}'.format(time)

            r = requests.get(url)
            html_doc = r.text
            soup = BS(html_doc)

            game_dates = []
            home_teams = []
            away_teams = []
            home_scores = []
            away_scores = []

            for x in \
                    soup.find_all('th', {'style': "width:22%; text-align:center"}):
                score = x.text.split('(')[0]
                try:
                    home_score = score.split(' - ')[0]
                    away_score = score.split(' - ')[1].rstrip()
                except:
                    home_score = score.split(' – ')[0]
                    away_score = score.split(' – ')[1].rstrip()

                home_scores.append(home_score)
                away_scores.append(away_score)

            for x in \
                soup.find_all(
                    'span',
                    {'class': 'summary', 'style': 'display: none;'}):
                teams = x.text
                try:
                    home_team = teams.split('  v  ')[0]
                    away_team = teams.split('  v  ')[1]
                except:
                    try:
                        home_team = teams.split('  v ')[0]
                        away_team = teams.split('  v ')[1]
                    except:
                        try:
                            home_team = teams.split('　 v  ')[0]
                            away_team = teams.split('　 v  ')[1]
                        except:
                            home_team = teams.split(' v  ')[0]
                            away_team = teams.split(' v  ')[1]

                home_teams.append(home_team)
                away_teams.append(away_team)

            for x in soup.find_all('div', {'style': 'text-align:right'}):
                game_date = x.text.split('日')[0] + '日'
                game_dates.append(game_date)

            df1 = pd.DataFrame(
                data={
                    'game_date': game_dates,
                    'home_team': home_teams,
                    'away_team': away_teams,
                    'home_score': home_scores,
                    'away_score': away_scores}
            )[-14:]

            afc = pd.concat([afc, df1])

        # 2009~2012
        for time in [str(x) for x in range(2009, 2013)]:

            url = 'https://ja.wikipedia.org/wiki/AFCチャンピオンズリーグ{}'.format(time)

            r = requests.get(url)
            html_doc = r.text
            soup = BS(html_doc)

            game_dates = []
            teams = []
            home_teams = []
            away_teams = []
            home_scores = []
            away_scores = []

            for x in \
                    soup.find_all('th', {'style': "width:22%; text-align:center"}):
                score = x.text.split('(')[0]
                try:
                    home_score = score.split(' - ')[0]
                    away_score = score.split(' - ')[1].rstrip()
                except:
                    try:
                        home_score = score.split(' – ')[0]
                        away_score = score.split(' – ')[1].rstrip()
                    except:
                        home_score = np.nan
                        away_score = np.nan

                home_scores.append(home_score)
                away_scores.append(away_score)

            for x in soup.find_all('span', {'class': 'fn org'}):
                raw = x.find('a').text
                teams.append(raw)

            for i, x in enumerate(teams):
                if i % 2 == 0:
                    home_teams.append(x)
                else:
                    away_teams.append(x)

            for x in soup.find_all('div', {'style': 'text-align:right'}):
                game_date = x.text.split('日')[0] + '日'
                game_dates.append(game_date)
            try:

                df2 = pd.DataFrame(
                    data={
                        'game_date': game_dates,
                        'home_team': home_teams,
                        'away_team': away_teams,
                        'home_score': home_scores,
                        'away_score': away_scores
                    }
                )[-13:]

                afc = pd.concat([afc, df2])
            except:
                print(time)

        # 2007~2008
        for time in [str(x) for x in range(2007, 2009)]:

            url = 'https://ja.wikipedia.org/wiki/AFCチャンピオンズリーグ{}'.format(time)

            r = requests.get(url)
            html_doc = r.text
            soup = BS(html_doc)

            game_dates = []
            teams = []
            home_teams = []
            away_teams = []
            home_scores = []
            away_scores = []

            for x in \
                    soup.find_all('th', {'style': "width:22%; text-align:center"}):
                score = x.text.split('(')[0]
                try:
                    home_score = score.split(' - ')[0]
                    away_score = score.split(' - ')[1].rstrip()
                except:
                    try:
                        home_score = score.split(' – ')[0]
                        away_score = score.split(' – ')[1].rstrip()
                    except:
                        home_score = np.nan
                        away_score = np.nan

                home_scores.append(home_score)
                away_scores.append(away_score)

            for x in soup.find_all('span', {'class': 'fn org'}):
                raw = x.find('a').text
                teams.append(raw)

            for i, x in enumerate(teams):
                if i % 2 == 0:
                    home_teams.append(x)
                else:
                    away_teams.append(x)

            for x in soup.find_all('div', {'style': 'text-align:right'}):
                game_date = x.text.split('日')[0] + '日'
                game_dates.append(game_date)

            try:
                df3 = pd.DataFrame(
                    data={
                        'game_date': game_dates,
                        'home_team': home_teams,
                        'away_team': away_teams,
                        'home_score': home_scores,
                        'away_score': away_scores
                    }
                )[-14:]

                afc = pd.concat([afc, df3])
            except:
                print(time)

    else:
        print('selected not to crawl afc data! loading!')
        afc = pd.read_csv('afc.csv')

    afc_df = afc.copy()

    afc_df.loc[afc_df.away_score == '3（注）', 'away_score'] = '3'

    afc_df['game_date'] = afc_df['game_date'].apply(lambda x: x.rstrip())
    afc_df['game_date'] = afc_df['game_date'].apply(
        lambda x: '-'.join([z.zfill(2) for z in re.findall('\d+', x)])
    )

    afc_df.home_score = afc_df.home_score.astype(int)
    afc_df.away_score = afc_df.away_score.astype(int)

    afc_df['season'] = afc_df['game_date'].apply(
        lambda x: int(x.split('-')[0])
    )

    # afc_df = afc_df.sort_values(by='game_date')
    afc_df.reset_index(inplace=True)
    del afc_df['index']

    afc_df.match_num = np.nan
    for i, row in afc_df.iterrows():
        season = row['season']
        match_num = len(
            afc_df.loc[(afc_df['season'] == season)
                       & (afc_df.index < i)]
        ) + 1

        afc_df.loc[afc_df.index == i, 'match_num'] = match_num

    # Method to calulate how far Japanese teams are went on AFC tournament
    afc_df['until_8'] = 0
    afc_df['until_semi'] = 0
    afc_df['until_final'] = 0
    afc_df['champ'] = 0

    for i, row in afc_df.iterrows():
        team = row['home_team']
        num = row['match_num']
        sea = row['season']
        semi_list = list(
            afc_df.loc[
                (afc_df.season == sea)
                & (afc_df['match_num'] > 8),
                'home_team'
            ].values
        )

        if num < 9:
            if team not in semi_list:
                afc_df.loc[afc_df.index == i, 'until_8'] = 1

    for i, row in afc_df.iterrows():
        team = row['home_team']
        num = row['match_num']
        sea = row['season']

        if num == 9:
            afc_df.loc[afc_df.index == i, 'away_team'] = \
                afc_df.loc[
                (afc_df.season == sea) & (afc_df['match_num'] == 11),
                'home_team'
            ].values[0]

        elif num == 10:
            afc_df.loc[afc_df.index == i, 'away_team'] = \
                afc_df.loc[
                (afc_df.season == sea) & (afc_df['match_num'] == 12),
                'home_team'
            ].values[0]

        elif num == 11:
            afc_df.loc[afc_df.index == i, 'away_team'] = \
                afc_df.loc[
                (afc_df.season == sea) & (afc_df['match_num'] == 9),
                'home_team'
            ].values[0]

        elif num == 12:
            afc_df.loc[afc_df.index == i, 'away_team'] = \
                afc_df.loc[
                (afc_df.season == sea) & (afc_df['match_num'] == 10),
                'home_team'
            ].values[0]

    for i, row in afc_df.iterrows():
        team = row['home_team']
        num = row['match_num']
        sea = row['season']
        if sea not in range(2009, 2013):
            final_list = list(
                afc_df.loc[
                    (afc_df.season == sea)
                    & (afc_df['match_num'] > 12),
                    'home_team'
                ].values
            )
            if num > 8 and num < 13:
                if team not in final_list:
                    afc_df.loc[afc_df.index == i, 'until_semi'] = 1

    afc_df.loc[
        (afc_df.season == 2010)
        & (afc_df['match_num'] == 9), 'until_semi'
    ] = 1

    afc_df.loc[
        (afc_df.season == 2010)
        & (afc_df['match_num'] == 10), 'until_semi'
    ] = 1

    # No Japanese clubs make good result at 2010~2012 season, so don't
    # need to include data of that period.
    afc_df = afc_df[(afc_df['season'] < 2010) | ((afc_df['season'] > 2012))]

    for i, row in afc_df.iterrows():
        team = row['home_team']
        num = row['match_num']
        sea = row['season']

        if sea != 2009:
            if num == 13:
                afc_df.loc[afc_df.index == i, 'away_team'] = \
                    afc_df.loc[
                    (afc_df['season'] == sea) & (afc_df['match_num'] == 14),
                    'home_team'
                ].values[0]

            if num == 14:
                afc_df.loc[afc_df.index == i, 'away_team'] = \
                    afc_df.loc[
                    (afc_df['season'] == sea) & (afc_df['match_num'] == 13),
                    'home_team'
                ].values[0]

    # Check Japanese clubs of AFC champion
    until_final_index = []
    champ_index = [12, 83, 149, 162]

    afc_df.loc[afc_df.index.isin(champ_index), 'champ'] = 1

    afc_df.start_buff = np.nan
    afc_df.end_buff = np.nan

    for i, row in afc_df.iterrows():
        team = row['home_team']
        num = row['match_num']
        sea = row['season']
        until_8 = row['until_8']
        until_semi = row['until_semi']
        until_final = row['until_final']
        champ = row['champ']

        afc_df.loc[afc_df.index == i, 'start_buff'] = \
            afc_df.loc[
                (afc_df['season'] == sea) & (afc_df['match_num'] == 1),
                'game_date'
        ].values[0]

        if sea != 2009:
            if until_8 == 1:
                afc_df.loc[afc_df.index == i, 'end_buff'] = \
                    afc_df.loc[
                    (afc_df['season'] == sea) & (afc_df['match_num'] == 8),
                    'game_date'
                ].values[0]

            elif until_semi == 1:
                afc_df.loc[afc_df.index == i, 'end_buff'] = \
                    afc_df.loc[
                        (afc_df['season'] == sea) & (
                            afc_df['match_num'] == 12),
                        'game_date'
                ].values[0]

            elif until_final == 1:
                afc_df.loc[afc_df.index == i, 'end_buff'] = \
                    afc_df.loc[
                        (afc_df['season'] == sea) & (
                            afc_df['match_num'] == 14),
                        'game_date'
                ].values[0]

            elif champ == 1:
                afc_df.loc[afc_df.index == i, 'end_buff'] = str(sea) + '-12-31'

    afc_df = afc_df.loc[(afc_df['champ'] == 1)]
    afc_data = afc_df.dropna(subset=['end_buff'])[[
        'game_date',
        'home_team',
        'start_buff',
        'end_buff',
        'season']]

    afc_data.rename(columns={'home_team': 'team'}, inplace=True)

    # Merging to dataset
    # change team names to fit in existing dataset
    jap_team = ['柏レイソル', 'ガンバ大阪',
                '川崎フロンターレ', '浦和レッズ',
                '浦和レッドダイヤモンズ', '鹿島アントラーズ']

    afc_data = afc_data[afc_data.team.isin(jap_team)]

    team_name_changed = ['柏', '大阪', '川崎Ｆ', '浦和', '浦和', '鹿島']

    team_name_changer = dict(zip(jap_team, team_name_changed))

    afc_data['team'] = afc_data['team'].apply(lambda x: team_name_changer[x])

    # We thought that AFC champion teams get buff for the remained
    # matches of that season.

    df['home_afc'] = 0
    df['away_afc'] = 0

    for i, row in afc_data.iterrows():
        team = row['team']
        start = row['start_buff']
        end = row['end_buff']
        season = row['season']

        df.loc[
            (df['season'] == season) & (df['home_team'] == team)
            & (df['match_date'] > start), 'home_afc'
        ] = 1

        df.loc[
            (df['season'] == season) & (df['away_team'] == team)
            & (df['match_date'] > start), 'away_afc'
        ] = 1

    # No AFC data before 2005, so make them null value.
    df.loc[df['season'] < 2005, 'home_afc'] = np.nan
    df.loc[df['season'] < 2005, 'away_afc'] = np.nan

    df_done = df.copy()

    return df_done


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


def set_lagged_data(df):
    for year in range(1994, 2019):
        unique_team = df.loc[
            df['match_Year'] == year, 'home_team'].unique()
        unique_stadium = df.loc[
            df['match_Year'] == year, 'venue'].unique()

        for year_, team, stadium in product(
                [str(year)], unique_team, unique_stadium):
            for lag in range(1, 5):
                # For Home
                temp = df[(df['match_Year'] == int(year_) - lag)
                          & (df['home_team'] == team)
                          & (df['venue'] == stadium)]

                if len(temp) != 0:
                    mean_ = temp['attendance'].mean()
                    df.loc[(df['match_Year'] == int(year_))
                           & (df['home_team'] == team)
                           & (df['venue'] == stadium),
                           'Lag{}Year_home_stadium_Mean'.format(lag)
                           ] = mean_
                # For Away
                temp = df[(df['match_Year'] == int(year_) - lag)
                          & (df['away_team'] == team)
                          & (df['venue'] == stadium)]

                if len(temp) > 0:
                    mean_ = temp['attendance'].mean()
                    df.loc[(df['match_Year'] == int(year_))
                           & (df['away_team'] == team)
                           & (df['venue'] == stadium),
                           'Lag{}Year_away_stadium_Mean'.format(lag)
                           ] = mean_

    # Yearly, Monthly home, away's lag attendance
    for year in range(1993, 2019):
        for year_, month, team in product(
                [str(year)], [str(x) for x in range(2, 13)],
                df.loc[df['match_Year'] == year,
                       'home_team'].unique()):
            for lag in range(1, 5):
                # Home
                temp = df[
                    (df['match_Year'] == int(year_) - lag)
                    & (df['match_Month'] == int(month))
                    & (df['home_team'] == team)
                ]

                if len(temp) != 0:
                    mean_ = temp['attendance'].mean()

                    df.loc[
                        (df['match_Year'] == int(year_))
                        & (df['match_Month'] == int(month))
                        & (df['home_team'] == team),
                        'Lag{}Year_Month_Home_Mean'.format(lag)
                    ] = mean_

                # Away
                temp = df[
                    (df['match_Year'] == int(year_) - lag)
                    & (df['match_Month'] == int(month))
                    & (df['away_team'] == team)
                ]

                if len(temp) != 0:
                    mean_ = temp['attendance'].mean()

            df.loc[
                (df['match_Year'] == int(year_))
                & (df['match_Month'] == int(month)) & (df['away_team'] == team),
                'Lag{}Year_Month_Away_Mean'.format(lag)
            ] = mean_

    def lag_feature(df, lags, col):
        temp = df[['match_Year', 'home_team', 'away_team', col]]
        for i in lags:
            shifted = temp.copy()
            shifted.columns = ['match_Year', 'home_team',
                               'away_team', col + '_lag_' + str(i)]
            shifted['match_Year'] += i

            shifted = shifted.groupby(
                ['match_Year', 'home_team', 'away_team']
            )[col + '_lag_' + str(i)].mean().reset_index()

            df = pd.merge(
                df, shifted,
                on=['match_Year', 'home_team', 'away_team'],
                how='left'
            )
        return df

    df = lag_feature(df, [1, 2, 3, 4], 'attendance')
    return df


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
