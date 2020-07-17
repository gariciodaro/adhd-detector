# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to download subject data from HBN. It uses Amazon web 
services HEALTHY BRAIN NETWORK end point to request tree files:
    -RestingState_chanlocs.csv
    -RestingState_data.csv'
    -RestingState_event.csv
If all of them are available, it stores them in the path specified in  
configuration/config.cfg file, in the variable PATH_SIGNALS_CSV.
The ID of the wanted subjects shoud be specified in PATH_SIGNALS_CSV.
"""

#libraries

import wget
import pandas as pd
import requests
import os
import configparser

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

#csv file with the ID you want to download
CSV_SUBJECTS_IDS = root_path + config.get('FILES','CSV_SUBJECTS_IDS')

#AWS HBS URLS.
URL_AWS          = config.get('URL','URL_AWS')
NEXT_URL_AWS     = config.get('URL','NEXT_URL_AWS')

#path to store the csv downloaded csv files.
PATH_SIGNALS_CSV = root_path +  config.get('PATH_STORE',
                                        'PATH_SIGNALS_CSV')

#required files to make diagnosis.
FILES_USE        = config.get('FILES','FILES_USE').split(",")


def main():
    subjects_id=pd.read_csv(CSV_SUBJECTS_IDS)
    list_ids=subjects_id.id.values.tolist()
    for each_subject in list_ids:
            print('Beginning download of subject:',each_subject)
            download_data=True
            try:
                #check that all the 3 required files exits.
                #if one of them is missing abort download.
                for each_file in FILES_USE:
                    url=(URL_AWS+
                        each_subject+
                        NEXT_URL_AWS+
                        each_file)
                    r = requests.head(url).status_code
                    if r!=200:
                        print('URL not found: ', url)
                        download_data=False
                #download the files
                if download_data:
                    for each_file in FILES_USE:
                        url = (
                            URL_AWS + 
                            each_subject + 
                            NEXT_URL_AWS+
                            each_file)
                        wget.download(
                            url,
                            PATH_SIGNALS_CSV + each_subject+"_" + each_file)
            except:
                print('abording dowload of subject: ', each_subject)

if __name__ == "__main__":
    main()