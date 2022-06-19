# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:21:36 2022

@author: RH
"""


#%%

import os
import pathlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import implicit

from scipy.sparse import csr_matrix
from implicit import als
from lightgbm import LGBMClassifier

from metrics import precision_at_k, recall_at_k
from utils import prefilter_items
from recommenders import MainRecommender


#%%

#pip install pipreqs
pip freeze > requirements.txt


#%%

path = os.getcwd()
items_features = pd.read_csv(f'../Project_data/product.csv', sep = ',', encoding = 'utf8')
user_features = pd.read_csv(f'../Project_data/hh_demographic.csv', sep = ',', encoding = 'utf8')
df_train = pd.read_csv(f'../Project_data/retail_train.csv', sep = ',', encoding = 'utf8')




#%%

