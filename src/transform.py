import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
import json
import os

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

def transform_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('-1')
        elif df[col].dtype == 'int':
            df[col] = df[col].fillna(-1)
    
    df['is_featured'] = df['artist_name'].apply(is_featured).astype(int)
    
    # if artist is same as composer
    df['artist_composer'] = (df['artist_name'] == df['composer']).astype(int)

    # if artist is same as lyricist
    df['artist_lyricist'] = (df['artist_name'] == df['lyricist']).astype(int)
    
    # if artist, lyricist and composer are all three same
    df['artist_composer_lyricist'] = ((df['artist_name'] == df['composer']) & (df['artist_name'] == df['lyricist']) & (df['composer'] == df['lyricist'])).astype(int)

    # df.to_csv('data/train_before_transformed.csv', index=False)

    split_genre_ids = df['genre_ids'].str.split('|')
    genre_ids = split_genre_ids.explode().reset_index(drop=True)
    counter_genre_ids = Counter(genre_ids)
    genre_counts = split_genre_ids.apply(lambda x: [counter_genre_ids[genre_id] for genre_id in x])
    df["genre_id_1_count"] = genre_counts.apply(lambda x: x[0] if len(x) > 0 else -1).astype(int)
    df["genre_id_2_count"] = genre_counts.apply(lambda x: x[1] if len(x) > 1 else -1).astype(int)
    df["genre_id_n_count"] = genre_counts.apply(lambda x: sum(x[2:]) if len(x) > 2 else -1).astype(int)
    df["genre_id_1"] = split_genre_ids.apply(lambda x: x[0] if len(x) > 0 else "-1")
    df["genre_id_2"] = split_genre_ids.apply(lambda x: x[1] if len(x) > 1 else "-1")
    df["genre_id_n"] = split_genre_ids.copy().apply(lambda x: 1 if len(x) > 2 else 0).astype(int)
    df.drop('genre_ids', axis = 1, inplace = True)

    split_artists_names = df['artist_name'].str.replace('、', '|').str.replace('and', '|').str.replace('feat', '|').str.replace('&', '|').str.split('|')
    split_artists = split_artists_names.copy().explode().reset_index(drop=True)
    counter_artists = Counter(split_artists)
    artist_counts = split_artists_names.apply(lambda x: [counter_artists[artist] for artist in x])
    df["artist_name_1_count"] = artist_counts.apply(lambda x: x[0] if len(x) > 0 else -1).astype(int)
    df["artist_name_2_count"] = artist_counts.apply(lambda x: x[1] if len(x) > 1 else -1).astype(int)
    df["artist_name_n_count"] = artist_counts.apply(lambda x: sum(x[2:]) if len(x) > 2 else -1).astype(int)
    df["artist_name_1"] = split_artists_names.copy().apply(lambda x: x[0] if len(x) > 0 else "-1")
    df["artist_name_2"] = split_artists_names.copy().apply(lambda x: x[1] if len(x) > 1 else "-1")
    df["artist_name_n"] = split_artists_names.copy().apply(lambda x: 1 if len(x) > 2 else 0).astype(int)
    df.drop('artist_name', axis = 1, inplace = True)
    
    mapping = {}
    label_encoding = preprocessing.LabelEncoder()
    # cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in df.columns:
        df[col] = label_encoding.fit_transform(df[col])
        # output the label encoding mapping for test and save as json
        mapping[col] = dict(zip(label_encoding.classes_, label_encoding.transform(label_encoding.classes_)))
    # check if train_mapping.json exists
    filename = 'data/train_mapping.json'
    if 'train_mapping.json' in os.listdir('data'):
        filename = 'data/test_mapping.json'
    with open(filename, 'w') as f:
        json.dump({k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in mapping.items()}, f)