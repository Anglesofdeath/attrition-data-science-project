import os
import logging
import sqlite3
import urllib.request
import pandas as pd
import numpy as np
from . import cleaning_engineering_functions as cef

class DataProcessing:
    def __init__(self) -> None:

        self.logger = logging.getLogger(__name__) #track the package

        url = "https://techassessment.blob.core.windows.net/aiap12-assessment-data/attrition.db"
        name = "attrition.db"

        if os.path.isfile(name):
            os.remove(name) #if already exist delete it

        urllib.request.urlretrieve(url, name)  # download the file

        cnx = sqlite3.connect(name)
        cursor = cnx.cursor()

        self.df = pd.read_sql_query("SELECT * FROM attrition", cnx)

        self.logger.info(f"Size of DataFrame: {self.df.shape[0]}")

    def clean_data(self):
        self.logger.info("Starting data cleaning and Feature Engineering")
        self.df.drop(index = [1833, 2215], inplace=True)
        self.df.at[1992, 'Age'] = 18.0
        self.logger.info("Dropped rows with too many incorrect datapoints as found in eda and changing datapoint with 1 incorrect entry")
        self.logger.info(f"Size of DataFrame: {self.df.shape[0]}")

        self.df['Birth Year'] = self.df['Birth Year'].mask(self.df['Birth Year'] < 0)
        self.df['Age'] = self.df['Age'].mask(self.df['Age'] < 0)
        self.logger.info("Set Birth Year and Age negative values to NaN")
        self.df['Client Age'] = self.df['Birth Year'].apply(cef.convertYearToClientAge)
        self.logger.info("Created Client Age Column with NaN entries")
        self.df['Age'] = self.df.apply(lambda x: cef.fillInMissingAgeValues(x['Age'], x['Client Age'], x['Months']), axis = 1)
        self.df['Birth Year'] = self.df.apply(lambda x: cef.fillInMissingBirthYearValues(x['Age'], x['Birth Year'], x['Months']), axis = 1)
        self.df['Client Age'] = self.df['Birth Year'].apply(cef.convertYearToAge) #to fill in all the previously NaN values in Client Age
        self.logger.info("Created Client Age feature")

        self.df['Work Domain'] = self.df['Work Domain'].apply(cef.changeCategoryOfWorkDomain)
        self.logger.info("Redid categories of Work Domain Feature")

        self.df['Weekly Hours'] = self.df.apply(lambda x: cef.weeklyHours(x['Usage Rate'], x['Usage Time']), axis = 1)
        self.logger.info("Created Weekly Hours feature")

        self.df.reset_index()
        self.logger.info("index reset")

        self.logger.info("Finished Data Cleaning and Feature Engineering")
        return self.df