"""
Cleans raw ELEXON data
"""
import pickle

import pandas as pd

price = pd.read_csv('data/B1770.csv', parse_dates=True)

price = price.pivot_table(values='imbalancePriceAmountGBP',
                          index='time_stamp',
                          columns='priceCategory')

price.index = pd.to_datetime(price.index)

vol = pd.read_csv('data/B1780.csv', index_col=0, parse_dates=True)
vol = vol.set_index('time_stamp', drop=True).sort_index()
vol.index = pd.to_datetime(vol.index)

def print_duplicates(df):
    dupes = df[df.index.duplicated()]
    num = dupes.shape[0]

    print('{} duplicates'.format(num))
    if num != 0:
          print('duplicates are:')
          print(dupes.head())

def remove_duplicates(df):
    print('removing duplicates for {}'.format(df.columns))
    print_duplicates(df)
    return df[~df.index.duplicated(keep='first')]

vol = remove_duplicates(vol)
price = remove_duplicates(price)

merged = pd.concat([price, vol], axis=1)

idx = pd.DatetimeIndex(freq='30min', start=merged.index[0], end=merged.index[-1])
out = pd.DataFrame(index=idx)

out.loc[price.index, 'ImbalancePrice_excess_balance_[£/MWh]'] = price.loc[:, 'Excess balance']
out.loc[price.index, 'ImbalancePrice_insufficient_balance_[£/MWh]'] = price.loc[:, 'Insufficient balance']

out.loc[vol.index, 'ImbalanceVol_[MW]'] = vol.loc[:, 'imbalanceQuantityMAW']

def print_nans(df):
    nans = df[df.isnull().any(axis=1)]
    num = nans.shape[0]
    print('{} nans'.format(num))
    if num != 0:
        print('nan values are:')
        print(nans.head())

def fill_nans(df):
    print('filling nans in {}'.format(df.columns))
    print_nans(df)
    df = df.fillna(method='backfill')
    return df

out = fill_nans(out)

def check_output(df):
    print('final checkout of output')
    print_duplicates(df)
    print_nans(df)

check_output(out)

out.to_csv('data/clean.csv')
