# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to use ID2 XGB classfier. Remember to run scripts
1-5 before executing this one.

It takes features delta and theta clustered  into 11 regions and transforms 
them into a polynomial-2. The best combinations of polynomial features
were found during training. Read thesis document for more information.

Spatial_pre: Cluster 11
Feature_Ext : delta, theta power relatives
Ml_Tech : XGB -> Polynomial
"""


import os
import configparser
from OFHandlers import load_object
import pandas as pd
from mapSubjectSegment import from_map_to_df

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

PATH_DATA_ID_2   = root_path + config.get('PATH_STORE','PATH_DATA_ID_2')
path_id2_clf     = PATH_DATA_ID_2+'ID2_classifier/'
PATH_DATA        = root_path + config.get('PATH_STORE','PATH_DATA')
PATH_RESULTS        = root_path + config.get('PATH_STORE','PATH_RESULTS')
subject_segment_map=load_object(PATH_DATA+'mapper_subject.file')

def prepare_for_predict(X,
                        poly_obj,
                        best_features_poly):
    """pipeline of transformations to make a prediction."""
    X_poly=poly_obj.transform(X)
    cols_poly=poly_obj.get_feature_names(X.columns)

    df_X_poly=pd.DataFrame(X_poly,columns=cols_poly,index=X.index)
    X=df_X_poly[best_features_poly]
    return X

def main():
    # map segments to subject ID
    map_df=from_map_to_df(subject_segment_map,'subjects')
    # load delta theta tensors.
    df_features=load_object(PATH_DATA_ID_2+
                                        'Datasets_delta_theta/features.files')

    # load trained ID_2 parameters.
    id2_clf=load_object(path_id2_clf+'eeg_delta_theta_clf.file')
    poly_obj=load_object(path_id2_clf+'eeg_delta_theta_poly_obj.file')
    best_features_poly=load_object(path_id2_clf+
                                    'eeg_delta_theta_best_features_poly.file')
    
    X=prepare_for_predict(df_features,
                        poly_obj,
                        best_features_poly)
    predictions=id2_clf.predict_proba(X)[:,1]
    df_prediction=\
        map_df.join(pd.DataFrame(predictions,columns=['Predicted_Target_id2']))
    df_prediction_decision=df_prediction.groupby('subjects').mean()
    df_prediction_decision.rename(
                columns={'Predicted_Target_id2':'Decision_id2'},inplace=True)
    print(df_prediction)
    print(df_prediction_decision)
    df_prediction.to_csv(PATH_RESULTS+'Predicted_Target_id2.csv')
    df_prediction_decision.to_csv(PATH_RESULTS+'Decision_id2.csv')

#if __name__ == "__main__":
#    main()
