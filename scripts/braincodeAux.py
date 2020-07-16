# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to split mne object into segments with
band pass filter signal. Third order Butterworth-filter
"""

from collections import OrderedDict
import sys
import numpy as np
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import bandpass_cnt
import mne


def create_tensor(input_signal,event_mark,low_cut_hz,high_cut_hz,
                        interval_ms,channels_clusters=None,n_cluster=None):

    ival = [0, interval_ms]
    gdf_events = mne.find_events(input_signal)
    sfreq=input_signal.info["sfreq"]
    input_signal=input_signal.drop_channels(["stim"])
    raw_training_signal=input_signal.get_data()
    print("data shape:",raw_training_signal.shape)

    for i_chan in range(raw_training_signal.shape[0]):
        # first set to nan, than replace nans by nanmean.
        this_chan = raw_training_signal[i_chan]
        raw_training_signal[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(raw_training_signal[i_chan])
        chan_mean = np.nanmean(raw_training_signal[i_chan])
        raw_training_signal[i_chan, mask] = chan_mean

    if n_cluster is not None:
        new_signal=np.empty([n_cluster, raw_training_signal.shape[1]])
        #loop over each cluster
        for i_cluster in range(n_cluster):
            #get current channels in the signal
            list_current_channels=input_signal.info["ch_names"]
            
            #get the names of channels in a particular cluster
            channel_name_in_cluster=\
                channels_clusters[channels_clusters["cluster"]==\
                                                            i_cluster].index
            
            #create list to hold the index of a channel that
            #belogs to a particular cluster
            store_cluster=[]
            
            #loop over all channels in use
            for index,each_channel_name in enumerate(list_current_channels):
                #if each_channel_name in a the list for particual
                #cluster, add to store_cluster
                if(each_channel_name in channel_name_in_cluster):
                    store_cluster.append(index)

            new_signal[i_cluster,:]=\
                np.mean(raw_training_signal[store_cluster,:],axis=0)

        info=mne.create_info(ch_names=[str(each_cluster)
                                     for each_cluster in range(n_cluster)],
                        sfreq=sfreq,
                        ch_types='eeg',
                        montage=None,
                        verbose=None)

        # Reconstruct
        input_signal = mne.io.RawArray(new_signal, info,verbose="WARNING")
    else:
        #Reconstruct
        input_signal = mne.io.RawArray(raw_training_signal, 
                                        input_signal.info, 
                                        verbose="WARNING")
    # append the extracted events
    #raw_gdf_training_signal
    input_signal.info["events"] = gdf_events
    train_cnt=input_signal
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(
            a,
            low_cut_hz,
            high_cut_hz,
            train_cnt.info["sfreq"],
            filt_order=3,
            axis=1,
        ),
        train_cnt,
    )

    marker_def = OrderedDict(
        [
            ("ec_healthy", [event_mark])
        ]
    )
    
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)

    return train_set