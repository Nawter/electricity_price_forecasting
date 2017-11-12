

from bs4 import BeautifulSoup
import lxml

"""
Purpose of this script is to scrape data from the ELEXON APIKey

Requires using your own API key which can you can get from here ***
"""
SettlementDate = '12'

# dictionary of reports that we want
imbaprice = {'Report':'B1770',
             'SettlementDate':SettlementDate,
             'Period':'*',
             'Col_name':'imbalancePriceAmountGBP'}

imbavol = {'Report':'B1780',
           'SettlementDate':SettlementDate,
           'Period':'*',
           'Col_name':'imbalanceQuantityMAW'}

def BMRS_GetXML(**kwargs):
    """
    args
        dictionary containing arguments to include in url (report specific)

    returns
        xml (object) : the parsed Elexon report
    """
    #  create the base URL
    url = 'https://api.bmreports.com/BMRS/{Report}/v1?APIKey=***YOUR API KEY HERE***&ServiceType=xml'.format(**kwargs)
    #  iterate over the report dictionary
    for key, value in kwargs.items():
        #  ignore report name as it is already in the url
        #  also ignore the column name
        if key == 'Report' or key == 'Col_name':
            pass
        else:
        #  add the report info onto the url
            url += "&%s=%s" % (key, value)

    #  parse the url using lxml
    print('parsing {}'.format(url))
    xml = 1
    lxml.objectify.parse(urllib.request.urlopen(url,timeout=500))
    return url, xml

url, xml = BMRS_GetXML(**imbavol)
import requests
r = requests.get(url)
