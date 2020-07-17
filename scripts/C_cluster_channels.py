# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to cluster the channel location of the montage using k means
clustering. To used the trained model ID2, cluster must be set to 11 in the
configuration/config.cfg file,
"""

#libraries
import os
from OFHandlers import save_object
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

#load path of the montage location
CSV_MONTAGE = root_path + config.get('FILES','CSV_MONTAGE')

#Path to store clustered plot of electrodes and Data Frame.
PATH_SAVE_CLUSTERED_DF = root_path + config.get('PATH_STORE',
                                                    'PATH_SAVE_CLUSTERED_DF')
PATH_SAVE_CLUSTERED_PLOT = root_path + config.get('PATH_STORE',
                                                    'PATH_SAVE_CLUSTERED_PLOT')

#Number of clusters.
N_CLUSTERS = int(config.get('ML_VARS','N_CLUSTERS'))

def GSN_HydroCel_129_to_df(path_cvs_channel_loc):
    """ Get dataframe from csv for GSN_HydroCel_129 montage. Extracts x,y,z 
    location and removes facial expressions associated electrodes.

    Paramaters
    ---------
    path_cvs_channel_loc : string
        path to monstage location file.

    Returns
    -------
    channels_pos : pandas.DataFrame
        locations and electrodes labels and labels as index.
    """

    # REF How context influences the interpretation of facial expressions: a 
    # source localization high-density EEG study on the “Kuleshov effect”
    exclude_outermost_channels=["E43", "E48", "E49", 
                                "E56", "E63", "E68",
                                "E73", "E81", "E88",
                                "E94", "E99","E107", 
                                "E113", "E119", "E120", 
                                "E125", "E126", "E127", 
                                "E128"]
    #read coordinates of the montage
    channels_pos=pd.read_csv(path_cvs_channel_loc)[["labels","X","Y","Z"]]
    #remove channels from montage in exclude_outermost_channels
    channels_pos=\
        channels_pos[~channels_pos["labels"].isin(exclude_outermost_channels)]
    channels_pos.set_index("labels",inplace=True)
    return channels_pos

def clustering_channels(channels_pos,n_clusters,path_to_store):
    """ Cluster the channel location of montage using K means clustering.
    
    Parameters
    ---------
    channels_pos : pandas.DataFrame
        locations and electrodes labels as index.
    n_clusters : int
        number of clusters to produce.
    path_to_store : string
        local path to store channels_clusters dataframe.

    Returns
    -------
    channels_clusters : pandas.DataFrame
        index as label electrodes, cartesian coordinates and predicted cluster
        per electrode.
    """
    #Initialize clustering
    clustering=KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    max_iter=300,
                    n_init=1000,
                    random_state=0)
    clustering.fit(channels_pos)
    #get centers
    centers=clustering.cluster_centers_

    #calculate aritmetic sum of componements
    # id_cluster:sum_compo
    r={index:(each_center[0]+each_center[1]+each_center[2]) for 
                            index,each_center in enumerate(centers)}

    #get cluster sorted id_cluster:smallest sum of components
    r_sorted=sorted(r.items(), key=lambda x: x[1])

    #get id of clusters
    new_order_keys=[each[0] for each in r_sorted]

    #make a copy of the centriods
    centers_unsorted=centers.copy()

    #re assigne the clusters centers
    for index,each_sorted_key in enumerate(new_order_keys):
        clustering.cluster_centers_[index]=centers_unsorted[each_sorted_key]

    channels_clusters=pd.DataFrame.from_dict({
                                'channels_pos':channels_pos.index,
                                'cluster':clustering.predict(channels_pos),
                                'X':channels_pos.X,
                                'Y':channels_pos.Y,
                                'Z':channels_pos.Z})
    save_object(path_to_store,channels_clusters)
    return channels_clusters

def plot_save_clustered_montage(channels_clusters,
                                n_clusters,
                                save,
                                path_to_save):
    """ Plots ans stores a 3D representation of clustered electrodes.

    Parameters
    --------
    channels_clusters : pandas.DataFrame
        index as label electrodes, cartesian coordinates and predicted cluster
        per electrode. 
    n_clusters : int
        number of clusters.
    save : boolean
    path_to_save : string
    """

    #colors_list=sns.color_palette("husl", n_clusters)
    #i=0
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    for label,df in channels_clusters.groupby("cluster"):
        #if(label==7 or label==6):
        ax.scatter(df.X, df.Y, df.Z,label=label)
        ax.plot_trisurf(df.X, df.Y, df.Z,alpha=0.5)
        ax.view_init(30)
        #ax.view_init(30)
        #i=i+1
    plt.legend() 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if save==True:
        plt.savefig(path_to_save,
                    pad_inches=0.0,
                    transparent=True,
                    bbox_inches="tight")

def main():
    channels_pos=GSN_HydroCel_129_to_df(path_cvs_channel_loc=CSV_MONTAGE)
    channels_clustered=clustering_channels(channels_pos,
                                        n_clusters=N_CLUSTERS,
                                        path_to_store=PATH_SAVE_CLUSTERED_DF)
    plot_save_clustered_montage(channels_clustered,
                                N_CLUSTERS,
                                save=True,
                                path_to_save=PATH_SAVE_CLUSTERED_PLOT)

#if __name__ == "__main__":
#    main()

