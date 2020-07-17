import streamlit as st
import os
import configparser
import pandas as pd
import A_reset_analysis
import B_download_data,C_cluster_channels
import D_build_data_sets,E_get_power_bands
import F_features_to_image
import G_make_id2_predictions,H_make_id44_predictions
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
path_to_images=root_path + PATH_DATA_ID_44+'Datasets_image_alpha_beta/'

def main():
    st.title("Super ADHD Detector")
    st.sidebar.title("Executor controler")
    #st.markdown
    st.sidebar.markdown("Select the ID from HBN data set you want to analyze  👤")
    if st.sidebar.checkbox("Show IDs of subject",False):
        st.subheader("Subjects to analyze")
        df_subjects=pd.read_csv(CSV_SUBJECTS_IDS)
        st.write(df_subjects)
    
    st.sidebar.subheader("select process to execute")
    process=st.sidebar.selectbox("method",('select','Complete analysis',
                                    'Reset analysis',
                                    'Download data',
                                    'build cluster of channels',
                                    'Build data set',
                                    'Get power bands',
                                    'Transform Alpha and Beta into images'))

    if process=='Download data':
        B_download_data.main()

    if process=='Transform Alpha and Beta into images':
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
            

    # st.pyplot()
    #if st.sidebar.checkbox("Download data from HBN in AWS",False):
    #    execute_1_script=True
    #latest_iteration = st.empty()
    #bar = st.progress(0)

if __name__ == "__main__":
    main()