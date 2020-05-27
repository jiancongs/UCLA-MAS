import os
import codecs
import json
import spacy
import pandas as pd
import numpy as np
import itertools as it
import sqlite3
import pickle
from collections import defaultdict
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from geopy.distance import distance
from geopy.geocoders import Nominatim
from mlxtend.feature_selection import ColumnSelector  
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
# from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.sparse import coo_matrix
from datetime import datetime
from numpy.linalg import norm
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import SVD,KNNBasic,KNNWithMeans,KNNBaseline,NMF,SlopeOne,SVDpp
from surprise.model_selection import cross_validate
from surprise.model_selection.split import train_test_split
from surprise.model_selection.search import GridSearchCV

conn = sqlite3.connect('../Dataset/yelp.db')

def data_prep():
    sql = """
        select user_id, business_id, stars from review
        where review_id in (select review_id from parsed_review)
        and user_id in (select user_id from parsed_review group by user_id having count(user_id)>20)
    """
    user_review = pd.read_sql(sql,conn)
    return user_review

all_user_id = pd.read_sql("select user_id from parsed_review group by user_id having count(user_id)>20",conn)
all_buss_id = pd.read_sql("select distinct business_id from parsed_review where user_id in ('{}')".format("','".join(all_user_id['user_id'].values.tolist())),conn)

#load model
svd_load = pickle.load(open('../Models/Collaborative_filtering2.model', 'rb'))


# user_review = pd.read_sql("select user_id, business_id, stars from review where review_id in (select review_id from parsed_review)",conn)
def select_model(user_review):
    user_review = data_prep()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_review[['user_id', 'business_id', 'stars']], reader)
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [KNNBasic(),KNNBaseline(),  KNNWithMeans(), SVD(),SVDpp(), SlopeOne(), NMF()]:
        # Perform cross validation
        print(algorithm)
        print('start ......')
        results = cross_validate(algorithm, data, measures=['RMSE','MAE'], cv=3, verbose=False)
        
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
        print(benchmark)

def find_best_parameter():
    # it shows that SVD and SVDpp have best and similar results. However, SVD cost much fewer time in training.
    # Select SVD as final model, use Grid Search CV to find best parameter
    svd_param_grid = {'n_epochs': [35, 37,40], 
                    'lr_all': [0.005,0.007, 0.008],
                    'reg_all': [0.15,0.10,0.05]}

    svd_gs = S(SVD, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
    svd_gs.fit(data)
    print('SVD   - RMSE:', round(svd_gs.best_score['rmse'], 4), '; MAE:', round(svd_gs.best_score['mae'], 4))
    # RMSE 0.9584 MAE 0.7506
    print(svd_gs.best_params['mae']) # {'n_epochs': 25, 'lr_all': 0.01, 'reg_all': 0.4} # best parameter
    print(svd_gs.best_params['rmse']) # {'n_epochs': 25, 'lr_all': 0.009, 'reg_all': 0.4}


# fit and save model
def fit_model(data):
    train, test = train_test_split(data,test_size = 0.25)
    svd = SVD(n_epochs=25,lr_all=0.01,reg_all=0.4)
    svd.fit(train)
    pred = svd.test(test)
    print('RMSE for test set: {}'.format(accuracy.rmse(pred)))
    print('MAE for test set: {}'.format(accuracy.mae(pred)))
    # save model
    path = '../Models/Collaborative_filtering2.model'
    pickle.dump(svd, open(path, 'wb'))
    print("Model is saved to: {}".format(path))


# predict score given user_id and business_id
def predict_score(user_id,business_id):
    return svd_load.predict(uid=user_id,iid=business_id)


# given user_id, top 10 recommendations
def recommend_user(user_id, n=10):
    # df = user_review[user_review['user_id']==user_id]
    testset = [[user_id,iid,0] for iid in all_buss_id['business_id']]
    result = svd_load.test(testset)
    lst = []
    for uid, iid, true_r, est, _ in result:
        lst.append((iid, est))
    pred_df = pd.DataFrame(lst,columns=['business_id','est_score']).sort_values('est_score',ascending=False) 
    top_n_pred = pred_df.head(3*n)
    sql = "select * from business where business_id in ('{}') and is_open = 1".format("','".join(top_n_pred['business_id'].values.tolist()))
    buss_detail = pd.read_sql(sql,conn)
    ranked_business_detail = top_n_pred.merge(buss_detail,on='business_id',how='inner')
    # ranked_business_detail = ranked_business_detail[ranked_business_detail['is_open']==1]
    ranked_business_detail.drop(columns=['categories','attributes','hours','review_count','is_open'])
    return ranked_business_detail[:n]


def sort_distance(user_id, n, rslt, address, dist_limit):
    geolocator = Nominatim(user_agent="stat418_project")
    location = geolocator.geocode(address) 
    location.longitude
    coord1 = [location.latitude,location.longitude]
    coord2 = rslt[['latitude','longitude']].values.tolist()
    distance_lst = []
    for i in range(0,max(50,10*n)):
        distance_lst.append(distance(coord1,coord2[i]).mi)
    rslt['distance'] = distance_lst
    sorted_rslt = rslt.sort_values('distance')
    return sorted_rslt[sorted_rslt['distance']<=dist_limit][:n]


def Collaborative_filtering(user_id,n=10,address='', dist_limit=15):
    if not address:
        result = recommend_user(user_id, n=n)[['business_id','name','address','city','stars','est_score']]
    else:
        recomm_df = recommend_user(user_id, n=max(50,10*n))
        result = sort_distance(user_id, n, recomm_df, address, dist_limit)[['business_id','name','address','city','stars','est_score','distance']]
        if len(result) < n:
            print('NOTE: distance is too far from restaurants in Las Vegas. Please check your address again or increase distance limit.')
    return result
    
## test
# Collaborative_filtering('hyhvm_7sZV_zBqFicN7v5A',n=10,address='3655 S Las Vegas Blvd, Las Vegas, NV 89109', dist_limit=1)
# Collaborative_filtering('tL2pS5UOmN6aAOi3Z-qFGg',n=5)


