import numpy as np
import pandas as pd
from collections import Counter

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        return 2000 + int(isrc[5:7])
    return -1

def consolidate_data():
    data_path = 'data/'
    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    songs = pd.read_csv(data_path + 'songs.csv')
    songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
    members = pd.read_csv(data_path + 'members.csv')

    train = train.merge(songs, on='song_id', how='left')
    test = test.merge(songs, on='song_id', how='left')
    
    songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
    songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)
    
    train = train.merge(songs_extra, on = 'song_id', how = 'left')
    test = test.merge(songs_extra, on = 'song_id', how = 'left')

    members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
    members.drop('registration_init_time', axis = 1, inplace = True)
    members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
    members.drop('expiration_date', axis = 1, inplace = True)
    members['bd'] = np.clip(members['bd'], 0, 100)
    members['bd'] = members['bd'].replace(0, -1)

    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')

    train.to_csv('data/train_consolidated.csv', index=False)
    test.to_csv('data/test_consolidated.csv', index=False)
    return train, test
