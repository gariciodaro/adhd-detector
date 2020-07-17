# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to use ID44 restnet18 classfier. Remember to run scripts
1-5 before executing this one.

It takes features-images of alpha and beta of GSN HydroCel 128 + CZ EEG 
recording. Read thesis document for more information.

Spatial_pre: None
Feature_Ext : alpha and beta relative power image representation
Ml_Tech : 18-Residual-CNN
"""

import os
import configparser
from OFHandlers import load_object
import pandas as pd
from mapSubjectSegment import from_map_to_df
import numpy as np
from fastai.vision import load_learner,open_image
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

PATH_DATA_ID_44   = root_path + config.get('PATH_STORE','PATH_DATA_ID_44')
PATH_DATA        = root_path + config.get('PATH_STORE','PATH_DATA')
PATH_RESULTS        = root_path + config.get('PATH_STORE','PATH_RESULTS')

path_to_images   =  PATH_DATA_ID_44+'Datasets_image_alpha_beta/'

def main():
    #load trained learner
    learn = load_learner(PATH_DATA_ID_44)
    #load dictionary of subject:index_interval
    subject_segment_map=load_object(PATH_DATA+'mapper_subject.file')
    #load dataframe. with name of files.png
    subject_segment_map_img= load_object(PATH_DATA_ID_44+'image_name_map.file')
    #create dataframe from index interval and name of subject
    map_df=from_map_to_df(subject_segment_map,'subjects')
    #the resulting dataframe has columns subjects('id_subject') 
    # and name('file name of .png')
    df_images_id=map_df.join(subject_segment_map_img)

    predictions_hold=[]
    for each_png in df_images_id.name:
        x=open_image(path_to_images+each_png)
        p=learn.predict(x)
        tensor = p[2]
        single_pred=np.round(tensor.cpu().detach().numpy()[0], 5)
        predictions_hold=predictions_hold+[single_pred]

    df_prediction=\
        df_images_id.join(pd.DataFrame(predictions_hold,
                                        columns=['Predicted_Target_id44']))
    df_prediction_decision=df_prediction.groupby('subjects').mean()
    df_prediction_decision.rename(
                columns={'Predicted_Target_id44':'Decision_id44'},
                inplace=True)
    print(df_prediction.head())
    print(df_prediction_decision.head)

    df_prediction.to_csv(PATH_RESULTS+'Predicted_Target_id44.csv')
    df_prediction_decision.to_csv(PATH_RESULTS+'Decision_id44.csv')


#if __name__ == "__main__":
#    main()