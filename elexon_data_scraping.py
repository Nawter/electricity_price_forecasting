"""

"""
import argparse
from collections import defaultdict, namedtuple
from datetime import datetime as dt
from datetime import timedelta
import requests
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pandas as pd


def get_dates(start_date, days):
    """
    Creates a list of dates

    args
        start_date (str)
        days (int) number of days after start date

    returns
        dates (list) list of strings
    """
    start_date = dt.strptime(start_date, '%Y-%m-%d')
    dates = []
    for day in range(days):
        dates.append(start_date + timedelta(days=day))
    return dates


class ReportGrabber(object):
    """
    Grabs data from Elexon

    args
        name (str) name of the Elexon report
        data_cols (list) list of columns to get for the report
        key (str) API key
    """
    def __init__(self, name, data_cols, key):
        self.name = name

        assert isinstance(data_cols, list)
        self.data_cols = data_cols

        self.columns = ['settlementDate', 'settlementPeriod']
        self.columns.extend(data_cols)

        self.key = key

    def scrape_report(self, settlement_date):
        """
        Gets data for one settlement date

        args
            settlement_date (str)

        returns
            output (dict) {column name : data}
        """
        url = self.get_url(settlement_date)
        print('scraping {} {}'.format(self.name, settlement_date))

        #  use the requests library to get the response from this url
        req = requests.get(url)

        if 'An invalid API key has been passed' in req.text:
            raise ValueError('Invalid API key')

        self.root = ET.fromstring(req.content)

        #  iterate over the XML
        #  save each of the columns into a dict
        output = defaultdict(list)

        #  we can narrow down where we need to look in this XML
        for parent in self.root.findall("./responseBody/responseList/item"):

            for child in parent:

                #  condition that only gets the data we want
                #  if we wanted all raw data we wouldn't do this
                if child.tag in self.columns:
                    output[child.tag].append(child.text)

        return output

    def create_dataframe(self, output_dict):
        """
        Creates a dataframe from the output dictionary
        Will create a dataframe for one settlement_date, as the output_dict
        will be data for one settlement_date

        args
            output_dict (dict) {column name : data}

        returns
            output (DataFrame)
        """
        #  create a dataframe
        output = pd.DataFrame().from_dict(output_dict)

        #  create the time stamp by iterating over each row
        #  there must be a better way!
        for row_idx in range(output.shape[0]):

            date = dt.strptime(output.loc[row_idx, 'settlementDate'], '%Y-%m-%d')
            stamp = date + timedelta(minutes=30*int(output.loc[row_idx, 'settlementPeriod']))
            output.loc[row_idx, 'time_stamp'] = stamp

        output.loc[:, 'time_stamp'] = pd.to_datetime(output.loc[:, 'time_stamp'])

        output.index = output.loc[:, 'time_stamp']
        output.drop('time_stamp', inplace=True, axis=1)

        #  iterating through the XML creates duplicates - not sure why
        #  here we drop duplicates and sort the index
        output.drop_duplicates(inplace=True)
        output.sort_index(inplace=True)

        #  finally we set the dype of the columns correctly
        for col in self.data_cols:
            output.loc[:, col] = pd.to_numeric(output.loc[:, col])
        output = output.loc[:, self.data_cols]
        return output

    def get_url(self, settlement_date):
        """
        Forms the URL to query the Elexon API

        args
            settlement_date (str)

        returns
            url (str)
        """
        url = 'https://api.bmreports.com/BMRS/{}/'.format(self.name)
        url += 'v1?APIKey={}&'.format(self.key)
        url += 'ServiceType=xml&'
        url += 'Period=*&SettlementDate={}'.format(settlement_date)
        return url


if __name__ == '__main__':
    #  send in the ELEXON API key from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--key')
    args = parser.parse_args()
    key = args.key

    #  the reports we want data for
    #  format of {name: columns}
    reports = {'B1770': ['imbalancePriceAmountGBP'],
               'B1780': ['imbalanceQuantityMAW']}

    #  the dates we want data for
    settlementdates = get_dates('2015-01-01', 3*365)

    #  report data is a global list our data
    report_data = []
    for name, cols in reports.items():
        report = ReportGrabber(name, cols, key)

        #  dataframes is a list of reports for each date
        dataframes = []
        for date in settlementdates:
            output_dict = report.scrape_report(date)
            dataframes.append(report.create_dataframe(output_dict))

        all_dates = pd.concat(dataframes, axis=0)
        report_data.append(all_dates)

    report_data = pd.concat(report_data, axis=1)
    print('report data starts at {}'.format(report_data.index[0]))
    print('report data ends at {}'.format(report_data.index[-1]))
    print(report_data.head())
    print(report_data.describe())

    report_data.to_csv('elexon_data/elexon_report_data.csv')
