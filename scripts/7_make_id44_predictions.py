import os
import configparser
from OFHandlers import load_object
import pandas as pd
from mapSubjectSegment import from_map_to_df
import numpy as np


#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

PATH_DATA_ID_44   = root_path + config.get('PATH_STORE','PATH_DATA_ID_44')
path_images=PATH_DATA_ID_44+'Datasets_image_alpha_beta'
PATH_DATA        = root_path + config.get('PATH_STORE','PATH_DATA')
path_to_images   =  PATH_DATA_ID_44+'Datasets_image_alpha_beta/'


from fastai.vision import load_learner,open_image
#from fastai.callbacks import *
#import torchvision.models as models
#from fastai.callbacks.hooks import *
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#1.0.60


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

    df_prediction=df_images_id.join(pd.DataFrame(predictions_hold))
    return df_prediction

if __name__ == "__main__":
    predictions=main()
    print(predictions)
    print( predictions.groupby('subjects').mean())