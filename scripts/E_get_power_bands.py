# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to obtain the required relatives power bands. deta, theta,
and alpha, beta. 
"""

from scipy import signal
import scipy
import os
import sys
import pandas as pd
import numpy as np
from OFHandlers import save_object,load_object
import configparser

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))
PATH_DATA_ID_2   = root_path + config.get('PATH_STORE','PATH_DATA_ID_2')
PATH_DATA_ID_44  = root_path + config.get('PATH_STORE','PATH_DATA_ID_44')
R_S_FREQ         = int(config.get('ML_VARS','R_S_FREQ'))
tensor_segments_id2   = load_object(PATH_DATA_ID_2+'data_set_tensor.file')
tensor_segments_id44  = load_object(PATH_DATA_ID_44+'data_set_tensor.file')

def get_index_band(rate,lower,upper):
    lower_index=int(lower*rate)
    upper_index=int(upper*rate)
    return[lower_index,upper_index]

def get_power_spectrum(X,fs,slow_band):
    total_sample_number=X.shape[0]
    channel=X.shape[1]
    print('channel',channel)
    points_per_signal=X.shape[2]
    sample_holder=[]
    for sample_number in range(0,total_sample_number):
        data_channel_holder=[]
        for each_channel in range(0,channel):
            #print("data_channel_holder",data_channel_holder)
            each_signal=X[sample_number,each_channel,:]
            Pxx_den = signal.periodogram(each_signal, fs,scaling="spectrum")[1]
            rate_equi=(points_per_signal/fs)
            #delta power 0-4Hz
            indexs=get_index_band(rate_equi,0,4)
            #delta_power=Pxx_den[indexs[0]:indexs[1]]
            delta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #theta power 4-7hz
            indexs=get_index_band(rate_equi,4,8)
            #print(1,indexs)
            theta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #Alpha power 8-15hz
            indexs=get_index_band(rate_equi,8,16)
            #print(2,indexs)
            alpha_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])
            #beta power 16-31hz
            indexs=get_index_band(rate_equi,16,32)
            #print(3,indexs)
            beta_power=scipy.integrate.simps(Pxx_den[indexs[0]:indexs[1]])

            total_power=delta_power+theta_power+alpha_power+beta_power
            if slow_band:
                data_channel_holder=np.hstack([data_channel_holder,
                                    delta_power/total_power,
                                    theta_power/total_power])
            else:
                data_channel_holder=np.hstack([data_channel_holder,
                                    alpha_power/total_power,
                                    beta_power/total_power])

            #print(data_channel_holder)
        if(sample_number==0):
            sample_holder=data_channel_holder
        else:
            sample_holder=np.vstack([sample_holder,data_channel_holder])
    return sample_holder

def main():
    band_features_slow = get_power_spectrum(X=tensor_segments_id2.X,
                                            fs=R_S_FREQ,
                                            slow_band=True)
    df_slow=pd.DataFrame(band_features_slow,
            columns=[str(col) for col in range(band_features_slow.shape[1])])
    save_object(PATH_DATA_ID_2+'Datasets_delta_theta/features.files',df_slow)

    band_features_fast = get_power_spectrum(X=tensor_segments_id44.X,
                                            fs=R_S_FREQ,
                                            slow_band=False)
    df_fast=pd.DataFrame(band_features_fast,
            columns=[str(col) for col in range(band_features_fast.shape[1])])
    save_object(PATH_DATA_ID_44+'Datasets_alpha_beta/features.files',df_fast)

#if __name__ == "__main__":
#    main()