# UK Imbalance Price Forecasting

This code is currently being reworked!

**elexton_data_scraping.py** 

This script pulls data from ELEXON using their API.  It requires getting an API key which [you can get here](https://www.elexon.co.uk/guidance-note/bmrs-api-data-push-user-guide/).

Your API key is passed into the script from the command line

``` bash
$ python elexon_data_scraping.py --key YOURKEYHERE
```

The basic flow of the script is
- form a URL to query the ELEXON API for a single SettlementDate
- the requests library is used to parse the response from the API, searching for the columns specified 
- a pandas DataFrame is created from the parsed XML, with the timestamp converted from London time to GMT.  This is to avoid the complication of daylight savings time

The script is setup to iterate over multiple days and to save all data to a csv.  Saving a copy of the raw data is good
practice - cleaning and processing of the ELEXON data is done in make_dataset.py

**clean_data.py**

``` bash
$ python clean_data.py 
```
Bespoke data cleaning for the Elexon data. Care is taken to remove duplicates and fill missing data.

**train_model.py**
```bash
$ python train_model.py
```
This script takes the clean data and processes it for use in training models.  

After processing the data is used to train sklearn models.

The script makes use of some custom build pipelines which are in **pipelines.py**.  The script uses a simple library for
models in **models.py**

