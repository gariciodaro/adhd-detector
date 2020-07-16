
import pandas as pd
import numpy as np


def from_map_to_df(mappers,col_name):
    """
    Auxiliar function to create
    a dataframe of subjects ID
    with the same index as the dataset
    """
    list_holder=[(str(key)+",")*(value[1]-value[0]) for
                                    key,value in mappers.items()]
    indexes=[]
    for each in list_holder:
        current_list=each.split(',')
        while('' in current_list) : 
            current_list.remove('') 
        indexes=indexes+current_list
    df=pd.DataFrame({col_name:indexes})
    return df
