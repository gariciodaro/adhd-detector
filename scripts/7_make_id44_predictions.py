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

PATH_DATA_ID_44   = root_path + config.get('PATH_STORE','PATH_DATA_ID_44')
path_images=PATH_DATA_ID_44+'Datasets_image_alpha_beta'
PATH_DATA        = root_path + config.get('PATH_STORE','PATH_DATA')
path_to_images   =  PATH_DATA_ID_44+'Datasets_image_alpha_beta/'

from PIL import Image

from fastai.vision import *
from fastai.callbacks import *
import torchvision.models as models
from fastai.callbacks.hooks import *
import imageio
from scipy.stats import ttest_ind,bartlett
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#1.0.60


def hooked_backward(m,learn,xb,cat,path_image):
    """
    Average and save the last convolutional layer
    activation.
    """
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,cat].backward()
            img = open_image(path_image)
            p=learn.predict(img)
            tensor = p[2]
            single_pred=tensor.cpu().detach().numpy()[0]
    return hook_a,hook_g,single_pred


def explore_activation_subjects(data,
                            m,
                            learn,
                            subject_id,
                            df_images_id,
                            path_to_images,
                            title="Activations"):
    """
    Plot the activations of the convolutional layer
    of the 5 subsets of 20 seconds signal of a particuar
    subject.
    """
    title=title+" "+subject_id
    df=df_images_id[df_images_id["subjects"]==subject_id]
    fig= plt.figure(figsize=(15,11))
    ax0=fig.add_subplot(2,3,1)
    ax1=fig.add_subplot(2,3,2)
    ax2=fig.add_subplot(2,3,3)
    ax3=fig.add_subplot(2,3,4)
    ax4=fig.add_subplot(2,3,5)
    axes_dict={0:ax0,
          1:ax1,
          2:ax2,
          3:ax3,
          4:ax4}
    prediction_holder=[]
    for index,each_png in enumerate(df.name):
        
        x=open_image(path_to_images+each_png)
        xb,_ = data.one_item(x)
        xb = xb.cuda()
        hook_a,hook_g,single_pred= hooked_backward(m,
                                    learn,
                                    xb,
                                    0,
                                    path_to_images+each_png)
        
        acts  = hook_a.stored[0].cpu()
        avg_acts = acts.mean(0)
        x.show(axes_dict.get(index))
        axes_dict.get(index).imshow(avg_acts, alpha=0.69, extent=(0,352,352,0),
                  interpolation='bilinear', cmap='Reds')
        axes_dict.get(index).set_title(" pred:"+str(round(single_pred,3)))
        prediction_holder.append(single_pred)
    fig.suptitle(title, fontsize=14)
    plt.show()
    return prediction_holder

def main():
    #load trained learner
    learn = load_learner(PATH_DATA_ID_44)
    
    #put it in evaluation mode
    m = learn.model.eval()

    #load dictionary of subject:index_interval
    subject_segment_map=load_object(PATH_DATA+'mapper_subject.file')
    #load dataframe. with name of files.png
    subject_segment_map_img= load_object(PATH_DATA_ID_44+'image_name_map.file')
    #create dataframe from index interval and name of subject
    map_df=from_map_to_df(subject_segment_map,'subjects')
    #the resulting dataframe has columns subjects('id_subject') 
    # and name('file name of .png')
    df_images_id=map_df.join(subject_segment_map_img)
    #print(df_images_id)
    data = ImageDataBunch.from_folder(path_to_images)
    #learn.load("selected_model")

    predictions_hold=[]
    for each_subject in df_images_id.subjects.unique():
        predictions=explore_activation_subjects(data=data,
                                    m=m,
                                    learn=learn,
                                    subject_id=each_subject,
                                    df_images_id=df_images_id,
                                    path_to_images=path_to_images,
                                    title="Activations")
        predictions_hold=predictions_hold+predictions

    df_prediction=df_images_id.join(pd.DataFrame(predictions_hold))
    return df_prediction
    
    


if __name__ == "__main__":
    predictions=main()
    print(predictions)
    print( predictions.groupby('subjects').mean())
    

