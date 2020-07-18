# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 2020
@author: gari.ciodaro.guerra

Stremlit app. Rende app. local host 8501
"""


import streamlit as st
import os
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


#read configurarion file
config = configparser.ConfigParser()
root_path= os.getcwd().replace('scripts','')
script_location = os.getcwd().replace('scripts','configuration')
config.read_file(open(script_location+'/config.cfg'))

#csv file with the ID you want to download
CSV_SUBJECTS_IDS = root_path + config.get('FILES','CSV_SUBJECTS_IDS')
PATH_DATA_ID_44   = config.get('PATH_STORE','PATH_DATA_ID_44')
PATH_SIGNALS_CSV = root_path +config.get('PATH_STORE','PATH_SIGNALS_CSV')
PATH_DATA_ID_2 = root_path +config.get('PATH_STORE','PATH_DATA_ID_2')
PATH_DATA_ID_44 = root_path +config.get('PATH_STORE','PATH_DATA_ID_44')
PATH_RESULTS        = root_path + config.get('PATH_STORE','PATH_RESULTS')

path_to_images=root_path + PATH_DATA_ID_44+'Datasets_image_alpha_beta/'

#Create folder structure
try:
    os.mkdir(root_path+'data/ID2_data/Datasets_delta_theta')
except:
    pass
try:
    os.mkdir(root_path+'data/ID44_data/Datasets_alpha_beta')
    os.mkdir(root_path+'data/ID44_data/Datasets_image_alpha_beta')
except:
    pass

try:
    os.mkdir(root_path+'data/results')
except:
    pass

try:
    os.mkdir(root_path+'data/signals')
except:
    pass

def main():
    last_execution_pressed='None'
    msg='Content of folder-files. If empty you need to execute process.'
    st.markdown('''
    # ♏ Super ADHD Detector
---------------------
<center>
<img src="http://garisplace.com/master/img/network.png" alt="cluster11" style="width:500px;"/>
</center>

This application aims to make the results of Master thesis 
(***Resting EEG classification of children with ADHD***) reproducible. 
Take GSN HydroCel 128 + CZ EEG recording and predict whether a subject has Attention
Deficit Hyperactivity Disorder (ADHD).

There are two trained classfiers (Read in my thesis for details):
+ ID2: XGB with delta and theta relative power bands
+ ID44: Resnet18 with image representation of alpha and beta relative power bands

| Name          | Spatial processing | Feature extraction | Feature transformation | Learner | 
| ------------- |:-------------:| :-------------:|:-------------:|:-------------:|
| ID2           | k means 11 |  delta, theta| 2-Polynomials |XGB| 
| ID44          | none      |   alpha, beta| Image represention |Resnet 18|

 ''',unsafe_allow_html=True)

    st.sidebar.markdown('''
    # 🔥 Executor controler
    ---------------------''')
    #st.markdown
    st.sidebar.markdown("👤 Select the ID from HBN data set you want to analyze ")
    if st.sidebar.checkbox("Show IDs of subject",False):
        st.subheader("Subjects to analyze")
        df_subjects=pd.read_csv(CSV_SUBJECTS_IDS)
        st.write(df_subjects)

    st.sidebar.markdown('''
    ## ✨ Download data. 
    ---------------------
    + Verify that you have not downloaded the data with  
    ```check signal tensor ``` checkbox. This process is slow.''')
    
    if st.sidebar.button('Download data'):
        import B_download_data
        B_download_data.main()
        last_execution_pressed='Download data'

    st.sidebar.markdown('''
    ## ✨ Reset app. 
    ---------------------
    + This will empty all the files including the download data.''')
    
    if st.sidebar.button('Reset analysis'):
        import A_reset_analysis
        A_reset_analysis.main()
        last_execution_pressed='Reset analysis'


    st.sidebar.markdown('''
    ## ✨ Execute everything.
    ---------------------''')
    if st.sidebar.button('Run everything.'):
        import C_cluster_channels
        C_cluster_channels.main()
        import D_build_data_sets
        D_build_data_sets.main()
        import E_get_power_bands
        E_get_power_bands.main()
        import F_features_to_image
        F_features_to_image.main()
        import G_make_id2_predictions
        G_make_id2_predictions.main()
        import H_make_id44_predictions
        H_make_id44_predictions.main()

    if st.sidebar.checkbox("Show predictions"):


        try:
            p_id44=pd.read_csv(PATH_RESULTS+'Predicted_Target_id44.csv')
            d_id44=pd.read_csv(PATH_RESULTS+'Decision_id44.csv')

            p_id2=pd.read_csv(PATH_RESULTS+'Predicted_Target_id2.csv')
            d_id2=pd.read_csv(PATH_RESULTS+'Decision_id2.csv')
            
            st.subheader("ID-44 Results")
            st.write(p_id44)
            st.write(d_id44)
            st.subheader("ID-2 Results")
            st.write(p_id2)
            st.write(d_id2)
        except:
            pass
        st.markdown('''
        ### experiment ID-44. Propositions:
        + ***P.2*** ID-44 is suitable for detecting ADHD-Combined Type and 
                    Hyperactive. A subject predicted with 
                    probability $0.63\pm 0.08$ is likely to be 
                    ADHD-Combined Type. A subject predicted with probability 
                    $0.74\pm 0.198$ is likely to be ADHD-Hyperactive Type. 
                    Subjects with a predicted probability of $0.32\pm 0.12$ 
                    are likely to be Healthy. $95\%$ confidence level.
        + ***P.3*** ADHD subjects exhibit symmetrical over activation on 
                    general frontal region, with localized over activation on 
                    temporal region. Additionally, alpha times beta 
                    symmetrical over activation  of frontal and parietal area, 
                    with localized temporal region.
        
        ### experiment ID-2. Propositions:
        + ***P.4***  ID-2 is suitable for detecting ADHD-Combined Type and 
                     ADHD-Inattentive. A subject predicted with probability 
                     $0.55\pm 0.02$ is likely to be ADHD-Combined Type. A 
                     subject predicted with probability $0.59\pm 0.09$ is 
                     likely to be ADHD-Inattentive Type. A subject predicted 
                     with probability $0.43\pm 0.05$ is likely to be Healthy. 
                     $95\%$ confidence level.
        + ***P.5***  ADHD subjects exhibit similarity under activation of 
                    frontal lobe in delta power band. asymmetrical over 
                    activation 
                    of theta in frontal, central and temporal regions.
        ''')
        
        
        


    st.sidebar.markdown('''
    ## ↪️ Check current status of folders
    ---------------------
    To be able to make a predictions, none of the folder list should
    empty.''')

    if st.sidebar.checkbox("Check content csv files Signals",False):
        st.subheader(msg)
        signal_files=os.listdir(PATH_SIGNALS_CSV)
        st.write(signal_files)
    
    if st.sidebar.checkbox("Check clustered montage",False):
        st.subheader(msg)
        content_list=os.listdir(PATH_DATA_ID_2)
        content_list= [each for each in content_list if 
                        len(each.split('.'))>1]
        st.write(content_list)

    if st.sidebar.checkbox("check signal tensor"):
        st.subheader(msg)
        content_list_id2=os.listdir(PATH_DATA_ID_2)
        content_list_id2= [each for each in content_list_id2 if 
                        each in ['data_set_tensor.file']]

        content_list_id44=os.listdir(PATH_DATA_ID_44)
        content_list_id44= [each for each in content_list_id44 if 
                        each in ['data_set_tensor.file']]
        st.write(content_list_id2)
        st.write(content_list_id44)
    
    if st.sidebar.checkbox("Check prepared power band datasets",False):
        st.subheader(msg)
        content_list_id2=os.listdir(PATH_DATA_ID_2+'Datasets_delta_theta')
        content_list_id44=os.listdir(PATH_DATA_ID_44+'Datasets_alpha_beta')
        st.write(content_list_id2)
        st.write(content_list_id44)

    if st.sidebar.checkbox("Check sample of image for id44",False):
        st.subheader(msg)
        content_list_id44=os.listdir(path_to_images)[0:3]
        st.write(content_list_id44)
    
    st.sidebar.markdown('''
    ## ✨ Select single process to execute
    ---------------------''')
    if st.sidebar.button('build cluster of channels'):
        import C_cluster_channels
        C_cluster_channels.main()
        last_execution_pressed='build cluster of channels'

    if st.sidebar.button('Build data set'):
        import D_build_data_sets
        D_build_data_sets.main()
        last_execution_pressed='Build data set'

    if st.sidebar.button('Get power bands'):
        import E_get_power_bands
        E_get_power_bands.main()
        last_execution_pressed='Get power bands'

    if st.sidebar.button('Alpha and Beta into images'):
        import F_features_to_image
        F_features_to_image.main()
        list_images=os.listdir(path_to_images)
        st.subheader('Sample of Alpha and Beta powers into images')
        fig=plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,3,1)
        ax2=fig.add_subplot(1,3,2)
        ax3=fig.add_subplot(1,3,3)
        img = Image.open(path_to_images+list_images[0])
        img2 = Image.open(path_to_images+list_images[1])
        img3 = Image.open(path_to_images+list_images[3])
        ax.imshow(img )
        ax2.imshow(img2 )
        ax3.imshow(img3 )
        st.pyplot()
        last_execution_pressed='Alpha and Beta into images'
    
    if st.sidebar.button('Make predictions ID2'):
        import G_make_id2_predictions
        G_make_id2_predictions.main()
        last_execution_pressed='Make predictions ID2'

    if st.sidebar.button('Make predictions ID44'):
        import H_make_id44_predictions
        H_make_id44_predictions.main()
        last_execution_pressed='Make predictions ID44'

    st.markdown('## 🕛 Last executed process: '+last_execution_pressed)


if __name__ == "__main__":
    main()