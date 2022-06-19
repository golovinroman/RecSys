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
items_features = pd.read_csv('./data/product.zip', compression='zip',sep = ',', encoding = 'utf8')
user_features = pd.read_csv('./data/hh_demographic.zip', compression='zip',sep = ',', encoding = 'utf8')
df_train = pd.read_csv('./data/retail_train.zip', compression='zip', sep = ',', encoding = 'utf8')
df_test = pd.read_csv('./data/retail_test1.zip', compression='zip', sep = ',', encoding = 'utf8')




#%%


output = pd.DataFrame()

output.to_csv(f'./recommendations.csv', index=False, sep= ',', encoding = 'utf8')
