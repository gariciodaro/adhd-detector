# Super ADHD Detector
---------------------

Clustered Montage for ID2             |  Activation map ID44
:-------------------------:|:-------------------------:
![](http://garisplace.com/master/img_ID_2/montage_rotation.gif)  |  ![](http://garisplace.com/master/img_ID_44/miss_combined.png)


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


### Instalation 
Only tested on Ubuntu 14.
+ Create folder to hold the project.
+ Clone or download this repository into that folder.
+ Install [anaconda](https://www.anaconda.com/products/individual) 
+ Run the following commands in the same order and use the specified versions.

```
-conda create -n ADHD python=3.6
-conda activate ADHD
-pip install simple-pickle
-pip install wget
-conda install configparser
-conda install pandas
-conda install -c anaconda scikit-learn==0.22.1 
-conda install -c conda-forge matplotlib==3.1.3 
-conda install -c anaconda tornado 
-conda install -c conda-forge mne==0.19.2
-ubunto: conda install -c conda-forge xgboost==0.90
-Windows: conda install -c anaconda py-xgboost==0.90
-pip install Braindecode==0.4.85
-conda install -c fastai fastai==1.0.60
-conda install -c anaconda seaborn==0.10.0
-pip install streamlit
```

### Usage

Modify the ```data/subjects_id/subjects_id.csv``` to select the HBN subject. 
you want to analyse
You can run the scripts sequently from terminal:
+ cd to /scripts
+ run ```python   A_reset_analysis.py```...```python  H_make_id44_prediction.py```
+ The results will be stored in ```data/results/``` as csv files.

Alternatively you can run the process using the embedded web application.
+ cd to /scripts
+ run ```streamlit run app.py```.
This should open your web browser as:

![](http://garisplace.com/master/app.png) 

+ When executing a process, you will see in the top left corner of the App the 
message ```streamlit run app.py```. 

+ Start by downloading the data, and then hit ```Run everything```.


*** Structure of the repository. ***
```
├── app_description.md
├── configuration
│   └── config.cfg
├── data
│   ├── GSN_HydroCel_129_chanlocs.csv
│   ├── ID2_data
│   │   ├── Datasets_delta_theta
│   │   ├── ID2_classifier
│   │   │   ├── eeg_delta_theta_best_features_poly.file
│   │   │   ├── eeg_delta_theta_clf.file
│   │   │   ├── eeg_delta_theta.html
│   │   │   ├── eeg_delta_theta_poly_obj.file
│   │   │   └── img
│   │   │       ├── base_line.png
│   │   │       ├── clustering_report.png
│   │   │       └── roc_plot.png
│   │   └── montage_clustered.png
│   ├── ID44_data
│   │   ├── Datasets_alpha_beta
│   │   ├── Datasets_image_alpha_beta
│   │   └── export.pkl
│   ├── results
│   ├── signals
│   └── subjects_id
│       └── subjects_id.csv
├── instalation.txt
├── LICENSE
├── README.md
└── scripts
    ├── app_dev.py
    ├── app.py
    ├── A_reset_analysis.py
    ├── B_download_data.py
    ├── braincodeAux.py
    ├── C_cluster_channels.py
    ├── D_build_data_sets.py
    ├── E_get_power_bands.py
    ├── F_features_to_image.py
    ├── G_make_id2_predictions.py
    ├── H_make_id44_predictions.py
    ├── mapSubjectSegment.py
```

## Files descriptions
+ **data/subjects_id.csv:** main source of ID of subjects to analyse. This is the only
file that need to modify manually.
+ **A_reset_analysis.py:** Utility script to delete files created during execution. Modify
main function to custom the folder files to be deleted.
+ **B_download_data.py**: 
Utility script to download subject data from HBN. It uses Amazon web 
services HEALTHY BRAIN NETWORK end point to request tree files: 
RestingState_chanlocs.csv,RestingState_data.csv,RestingState_event.csv.
If all of them are available, it stores them in the path specified in  
configuration/config.cfg file, in the variable PATH_SIGNALS_CSV.
The ID of the wanted subjects shoud be specified in PATH_SIGNALS_CSV.
+ **C_cluster_channels.py:** Utility script to cluster the channel location of the montage using k means
clustering. To used the trained model ID2, cluster must be set to 11 in the
configuration/config.cfg file,
+ **D_build_data_sets.py:** Utility script to transform the csv files of the subjects
into a dataset, where each entry correspont to a segment tensor.
+ **E_get_power_bands.py:** Utility script to obtain the required relatives power bands. deta, theta,
and alpha, beta. 
+ **F_features_to_image.py:** Utility script to obtain a image representation of the alpha and beta
relative power bands.
+ **G_make_id2_predictions.py:** Utility script to use ID2 XGB classfier. Remember to run scripts
1-5 before executing this one. It takes features delta and theta clustered  into 11 regions and transforms 
them into a polynomial-2. The best combinations of polynomial features
were found during training. Read thesis document for more information.
+ **H_make_id44_predictions.py:** Utility script to use ID44 restnet18 classfier. Remember to run scripts
1-5 before executing this one. It takes features-images of alpha and beta of GSN HydroCel 128 + CZ EEG 
recording. Read thesis document for more information.
+ **data/ID2_data/ID2_classifier/eeg_delta_theta_clf.file** Main serialized ID2 classifier. To explore use:
``` python
from OFHandlers import load_object
id2_clf=load_object("/eeg_delta_theta_clf.file") 
```
+ **export.pkl** Main serialized ID44 classifier. To explore use: 
``` python
from fastai.vision import load_learner
id44_clf = load_learner("/export.pkl")
```

## Notice
Even though you can not train the classifiers here, note the following:
+ We extracted resting EEG signals from the Healthy Brain Network dataset, made
available by The Child Mind Institute through the 1000 Functional Connec-
tomes Project / INDI. The EEG signals are publicly available, however pheno-
typical data must be accessed through the Healthy Brain Network-dedicated instance
of the Longitudinal Online Research and Imaging System (LORIS).

+ [Neuroimaging Data Access](http://fcon\_1000.projects.nitrc.org/indi/cmi\_healthy\_brain\_network/sharing\_neuro.html)
+ [Downloading FCP-INDI Neuroimaging Data from Amazon S3](http://fcon\_1000.projects.nitrc.org/indi/s3/index.html)
+ [LORIS instance of Healthy Brain Network](https://data.healthybrainnetwork.org/main.php)

## Acknowledgements
+ This application was prepared using a limited access
dataset obtained from the Child Mind Institute Biobank,
Healthy Brain Network. This manuscript reflects the
views of the authors and does not necessarily reflect
the opinions or views of the Child Mind Institute.
Research reported in this publication was supported
by Jacobs University and exposes the results of the
master thesis for the Data Engineering program. Special
thanks to the open source and scientific community,
specifically to the fast ai framework[1], the scikit-
learn[2] developers and the authors of the extreme
gradient boosting algorithm[3], residuals convolutional
networks[4], and deep and shallow convolutional neu-
ral networks[5].

## References

+ [1]J. Howard et al., “fastai,” https://github.com/fastai/fastai, 2018.
+ [2]F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion,
O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg,
J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot,
+ [3]T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting
system,” Proceedings of the 22nd acm sigkdd international
conference on knowledge discovery and data mining, 2016, pp.
785–794.
+ [4]K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning
for image recognition,” Proceedings of the IEEE conference on
computer vision and pattern recognition, 2016, pp. 770–778.
+ [5]R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer,
M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter,
W. Burgard, and T. Ball, “Deep learning with convolutional
neural networks for eeg decoding and visualization,” Human
Brain Mapping, aug 2017. (available at: http:// dx.doi.org/ 10.
1002/ hbm.23730).