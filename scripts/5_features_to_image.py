# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Utility script to obtain a image representation of the alpha and beta
relative power bands.
"""

from OFHandlers import save_object,load_object
import pandas as pd
import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

PATH_DATA_ID_44  = config.get('PATH_STORE','PATH_DATA_ID_44')
df_fast=load_object(root_path + PATH_DATA_ID_44+
                                        'Datasets_alpha_beta/features.files')

path_to_images=root_path + PATH_DATA_ID_44+'Datasets_image_alpha_beta/'
# 110=10*11. We will an image represating 110 rows, one per channels
# and two columns, alpha and beta.
dim_tuple_images=(110,2)

def generate_images(features,dim_tuple,path_to_save):
    for i in range(0,len(features)):
        sample=np.array(features.iloc[i]).reshape(dim_tuple)
        #save image
        fig=plt.figure(figsize=(7,7))
        ax=fig.add_subplot(1,1,1)
        ax.imshow(sample, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(path_to_save+str(i),
            pad_inches=0.0,transparent=True,bbox_inches='tight')
        plt.close()


def main():
    generate_images(df_fast,
                dim_tuple=dim_tuple_images,
                path_to_save=path_to_images)
    name_test_fast_ai={'name':[str(i)+'.png' for i in range(0,len(df_fast))]}
    name_test_fast_ai=pd.DataFrame.from_dict(name_test_fast_ai)
    save_object(root_path + PATH_DATA_ID_44+'/image_name_map.file',name_test_fast_ai)

if __name__ == '__main__':
    main()
    