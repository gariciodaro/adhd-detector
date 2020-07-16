# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to transform the csv files of the subjects
into a dataset, where each entry correspont to a segment tensor.
"""

import pandas as pd
import configparser
import os
import mne
import numpy as np
from braincodeAux import create_tensor
from OFHandlers import save_object,load_object

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

#csv file with the ID you want to process
CSV_SUBJECTS_IDS = root_path + config.get('FILES','CSV_SUBJECTS_IDS')

#Path to load clustered plot of electrodes.
PATH_SAVE_CLUSTERED_DF = root_path + config.get('PATH_STORE',
                                                    'PATH_SAVE_CLUSTERED_DF')

#path to store the csv downloaded csv files.
PATH_SIGNALS_CSV = root_path + config.get('PATH_STORE','PATH_SIGNALS_CSV')

PATH_DATA_ID_2   = root_path + config.get('PATH_STORE','PATH_DATA_ID_2')
PATH_DATA_ID_44  = root_path + config.get('PATH_STORE','PATH_DATA_ID_44')
PATH_DATA        = root_path + config.get('PATH_STORE','PATH_DATA')

#required files to make diagnosis.
FILES_USE        = config.get('FILES','FILES_USE').split(",")

#process configuration.
MONTAGE          = config.get('ML_VARS','MONTAGE')
SCALE            = float(config.get('ML_VARS','SCALE'))
EVENT_ID         = int(config.get('ML_VARS','EVENT_ID'))
S_FREQ           = int(config.get('ML_VARS','S_FREQ'))
R_S_FREQ         = int(config.get('ML_VARS','R_S_FREQ'))
N_CLUSTERS       = int(config.get('ML_VARS','N_CLUSTERS'))
LOW_CUT_HZ       = int(config.get('ML_VARS','LOW_CUT_HZ'))
HIGH_CUT_HZ      = int(config.get('ML_VARS','HIGH_CUT_HZ'))
INTERVAL_SEC     = int(config.get('ML_VARS','INTERVAL_SEC'))

def csv_to_mne(path_subject_channels,path_subject_signal,path_subject_events,
                s_freq,r_s_freq,montage,scale,event_id):
    #channels
    #get dataframe of channel location and labels
    df_subject_channels=pd.read_csv(path_subject_channels, 
                                    delimiter=",",
                                    decimal=".")
    #channels labels 
    ch_labels=list(df_subject_channels.labels)
    #apply montage
    internal_montage = mne.channels.make_standard_montage(montage)
    #create info for object
    info = mne.create_info(ch_names=ch_labels, 
                            sfreq=s_freq, 
                            ch_types='eeg', 
                            montage=internal_montage)
    #signal
    #load and scale signal
    dat_test=np.loadtxt(path_subject_signal, delimiter=',')*scale

    #Create the MNE Raw data object
    raw = mne.io.RawArray(dat_test, info)

    #create in stimuation channel
    stim_info = mne.create_info(['stim'], s_freq, 'stim')
    #create zero signal to store stimulus
    stim_raw = mne.io.RawArray(np.zeros(shape=[1, len(raw._times)]), stim_info)

    #add stim channle to raw signal
    raw.add_channels([stim_raw], force_update_info=True)

    #events
    #read csv of events
    df_subject_event=pd.read_csv(path_subject_events,
                                delimiter=",",
                                decimal=".")

    #fake structure of events
    evs = np.empty(shape=[0, 3])

    #from HBT, the signals were already marked each 20 seconds.
    for each_element in df_subject_event.values[1:len(df_subject_event)-1]:
        if('break cnt'!=each_element[0]):
            if(int(each_element[0])==event_id):
                evs = np.vstack((evs, np.array([each_element[1], 0,
                                                    int(each_element[0])])))
    #print(evs)
    # Add events to data object
    raw.add_events(evs, stim_channel='stim')

    #Check events
    print(mne.find_events(raw))

    #detect flat channels
    flat_chans = np.mean(raw._data[:len(ch_labels), :], axis=1) == 0

    # Interpolate bad channels
    raw.info['bads'] = \
                    list(np.array(raw.ch_names[:len(ch_labels)])[flat_chans])
    print('Bad channels: ', raw.info['bads'])
    raw.interpolate_bads()

    # Get good eeg channel indices
    #eeg_chans = mne.pick_types(raw.info, meg=False, eeg=True)

    #resample to have to 250 hz, 
    #this will allow us to compare with
    #the HDHD dataset.
    raw.resample(r_s_freq, npad='auto')

    #set reference to Cz
    raw.set_eeg_reference(ref_channels=['Cz'])

    raw.drop_channels(['Cz'])
    #return Raw object from mne class
    return raw



def main():
    subjects_id=pd.read_csv(CSV_SUBJECTS_IDS)
    list_ids=subjects_id.id.values.tolist()
    i=0
    mapper_subject={}
    for each_subject in list_ids:
        path_0=PATH_SIGNALS_CSV+'/'+each_subject+'_'+FILES_USE[0]
        path_1=PATH_SIGNALS_CSV+'/'+each_subject+'_'+FILES_USE[1]
        path_2=PATH_SIGNALS_CSV+'/'+each_subject+'_'+FILES_USE[2]

        #Obtain a mne object from the csv
        #this objects combines the information from the 3 csv
        #downloaded from AWS.
        raw=csv_to_mne(path_subject_channels=path_0,
                        path_subject_signal=path_1,
                        path_subject_events=path_2,
                        s_freq=S_FREQ,
                        r_s_freq=R_S_FREQ,
                        montage=MONTAGE,
                        scale=SCALE,
                        event_id=EVENT_ID)
        #make a copy of the object. one is input 
        #as tolal channels. The other is for clustered signal.
        copy_raw_1=raw.copy()
        copy_raw_2=raw.copy()

        #transforms the mne object into a n-numpy tensor.
        # n is the number of segments. Since the signal was has 5
        # event marked, we have 5 segments.
        #third order Butterworth-filter is applied too.
        tensor_with_segments=create_tensor(input_signal=copy_raw_1,
                                            event_mark=EVENT_ID,
                                            low_cut_hz=LOW_CUT_HZ,
                                            high_cut_hz=HIGH_CUT_HZ,
                                            interval_ms=INTERVAL_SEC*1000,
                                            channels_clusters=None,
                                            n_cluster=None)

        #transforms the mne object into a numpy tensor. 
        #third order Butterworth-filter is applied too.
        #this signal is clustered according to N_CLUSTERS
        tensor_with_segments_clustered=create_tensor(input_signal=copy_raw_2,
                        event_mark=EVENT_ID,
                        low_cut_hz=LOW_CUT_HZ,
                        high_cut_hz=HIGH_CUT_HZ,
                        interval_ms=INTERVAL_SEC*1000,
                        channels_clusters=load_object(PATH_SAVE_CLUSTERED_DF),
                        n_cluster=N_CLUSTERS)

        #concatenate the n-tensor for each subject
        #this will create the final dataset for ID2 and ID44.
        try:
            if(i==0):
                concat_signal=tensor_with_segments
                concat_signal_clustered=tensor_with_segments_clustered
                mapper_subject[each_subject]=[0,len(tensor_with_segments.y)]
            else:
                print(i,each_subject)
                concat_signal.X=np.vstack([concat_signal.X,
                                        tensor_with_segments.X])
                concat_signal_clustered.X=np.vstack([concat_signal_clustered.X,
                                        tensor_with_segments_clustered.X])
                #save subjects positions
                start=len(concat_signal.y)
                concat_signal.y=np.concatenate(
                    (concat_signal.y,tensor_with_segments.y), axis=0)
                end=len(concat_signal.y)
                mapper_subject[each_subject]=[start,end]
                print('concat_signal.X.shape',concat_signal.X.shape)
                print('concat_signal.y.shape',concat_signal.y.shape)
        except Exception as e:
            print('error occured see:')
            print(e)
            pass
        i=i+1
    save_object(PATH_DATA_ID_2+'data_set_tensor.file',
                                                concat_signal_clustered)
    save_object(PATH_DATA_ID_44+'data_set_tensor.file',
                                                concat_signal)
    save_object(PATH_DATA+'mapper_subject.file',mapper_subject)

if __name__ == '__main__':
    main()