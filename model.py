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

#pip install pipreqs
#pip freeze > requirements.txt

path = os.getcwd()
item_features = pd.read_csv('./data/product.zip', compression='zip',sep = ',', encoding = 'utf8')
user_features = pd.read_csv('./data/hh_demographic.zip', compression='zip',sep = ',', encoding = 'utf8')
df_train = pd.read_csv('./data/retail_train.zip', compression='zip', sep = ',', encoding = 'utf8')
df_test = pd.read_csv('./data/retail_test1.zip', compression='zip', sep = ',', encoding = 'utf8')



#%%

def calc_recall(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()


def calc_precision(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()


#%%

ITEM_COL = 'item_id'
USER_COL = 'user_id'
ACTUAL_COL = 'actual'
VAL_MATCHER_WEEKS = 6

# column processing
item_features.columns = [col.lower() for col in item_features.columns]
user_features.columns = [col.lower() for col in user_features.columns]

item_features.rename(columns={'product_id': ITEM_COL}, inplace=True)
user_features.rename(columns={'household_key': USER_COL }, inplace=True)



df_train = prefilter_items(df_train, item_features=item_features, take_n_popular=5000)
#data[ITEM_COL] = data[ITEM_COL].astype(int)

# берем данные для тренировки matching модели
data_train_matcher = df_train[df_train['week_no'] < (df_train['week_no'].max() - VAL_MATCHER_WEEKS)]

# берем данные для валидации matching модели
data_val_matcher = df_train[df_train['week_no'] >= (df_train['week_no'].max() - VAL_MATCHER_WEEKS)]

# берем данные для тренировки ranking модели
data_train_ranker = data_val_matcher.copy()  # Для наглядности. Далее мы добавим изменения, и они будут отличаться

# берем данные для теста ranking, matching модели
data_val_ranker = df_test.copy()


# make cold-start to warm-start

#data_train_matcher = prefilter_items(data_train_matcher, item_features=item_features, take_n_popular=5000)

# ищем общих пользователей
common_users = data_train_matcher.user_id.values

data_val_matcher = data_val_matcher[data_val_matcher.user_id.isin(common_users)]
data_train_ranker = data_train_ranker[data_train_ranker.user_id.isin(common_users)]
data_val_ranker = data_val_ranker[data_val_ranker.user_id.isin(common_users)]

# Теперь warm-start по пользователям



#%%

%%time

recommender = MainRecommender(data_train_matcher)

N_PREDICT = 50 

TOPK_RECALL = 50
TOPK_PRECISION = 5

result_eval_matcher = data_val_matcher.groupby(USER_COL)[ITEM_COL].unique().reset_index()
result_eval_matcher.columns=[USER_COL, ACTUAL_COL]


#%%

%%time
result_eval_matcher['own_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))


#%%
%%time
result_eval_matcher['sim_item_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_similar_items_recommendation(x, N=N_PREDICT))


#%%
%%time
result_eval_matcher['sim_user_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_similar_users_recommendation(x, N=N_PREDICT))


#%%
%%time
result_eval_matcher['als_rec'] = result_eval_matcher[USER_COL].apply(lambda x: recommender.get_als_recommendations(x, N=N_PREDICT))


#%%


result_eval_matcher_recall = sorted(calc_recall(result_eval_matcher, TOPK_RECALL), key=lambda x: x[1],reverse=True)
result_eval_matcher_precision  = sorted(calc_precision(result_eval_matcher, TOPK_PRECISION), key=lambda x: x[1],reverse=True)


#%%








#%%






#%%



output = pd.DataFrame()

output.to_csv(f'./recommendations.csv', index=False, sep= ',', encoding = 'utf8')
