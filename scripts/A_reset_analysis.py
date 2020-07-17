# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to delete files created during execution. Modify
main function to custom the folder files to be deleted.
"""

import os
import configparser

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

#config variables
PATH_SIGNALS_CSV=root_path+config.get('PATH_STORE','PATH_SIGNALS_CSV')
PATH_DATA_ID_2  =root_path+config.get('PATH_STORE','PATH_DATA_ID_2')
PATH_DATA_ID_44 =root_path+config.get('PATH_STORE','PATH_DATA_ID_44')
PATH_DATA       =root_path+config.get('PATH_STORE','PATH_DATA')
PATH_RESULTS    = root_path + config.get('PATH_STORE','PATH_RESULTS')

def empty_signals_csv():
    csvs=os.listdir(PATH_SIGNALS_CSV)
    for each_file in csvs:
        os.remove(PATH_SIGNALS_CSV+each_file)

def empty_id2_data():
    files=[f for f in os.listdir(PATH_DATA_ID_2) if 
                            os.path.isfile(f) or f.endswith('.file')]
    for each_file in files:
        os.remove(PATH_DATA_ID_2+each_file)
    files_in=os.listdir(PATH_DATA_ID_2+'Datasets_delta_theta/')
    for each_file in files_in:
        os.remove(PATH_DATA_ID_2+'Datasets_delta_theta/'+each_file)
    
    os.remove(PATH_DATA_ID_2+'montage_clustered.png')

def empty_id44_data():
    files=[f for f in os.listdir(PATH_DATA_ID_44) if 
                    os.path.isfile(f) or f.endswith('.file') and 
                    not f.endswith('.pkl') ]
    for each_file in files:
        os.remove(PATH_DATA_ID_44+each_file)
    files_in=os.listdir(PATH_DATA_ID_44+'Datasets_alpha_beta/')
    for each_file in files_in:
        os.remove(PATH_DATA_ID_44+'Datasets_alpha_beta/'+each_file)
    images_files= os.listdir(PATH_DATA_ID_44+
                                        'Datasets_image_alpha_beta/')
    for each_image in images_files:
        os.remove(PATH_DATA_ID_44+
                                'Datasets_image_alpha_beta/'+each_image) 

def remove_files_in_data_folder():
    #os.remove(PATH_DATA+'GSN_HydroCel_129_chanlocs.csv')
    os.remove(PATH_DATA+'mapper_subject.file')

def remove_results():
    files=[f for f in os.listdir(PATH_RESULTS) if  f.endswith('.csv') ]
    for each_file in files:
        os.remove(PATH_RESULTS+each_file)

def try_pass(functions):
    try:
        functions()
    except Exception as e: 
        print(e)
        pass

def main():
    try_pass(empty_signals_csv)
    try_pass(empty_id2_data)
    try_pass(empty_id44_data)
    try_pass(remove_files_in_data_folder)
    try_pass(remove_results)

#if __name__ == "__main__":
#    main()