# content_based_recommendation
import os
import codecs
import json
import spacy
import pandas as pd
import numpy as np
import itertools as it
import sqlite3
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from geopy.distance import distance
from geopy.geocoders import Nominatim
from sklearn.preprocessing import OneHotEncoder
from mlxtend.feature_selection import ColumnSelector  
from sklearn.pipeline import FeatureUnion
from scipy.sparse import coo_matrix
import simplejson as json
from datetime import datetime
from sklearn.model_selection import train_test_split
from numpy.linalg import norm

# Read in Datasets from yelp.db Using sqlite3
conn = sqlite3.connect('../Dataset/yelp.db')

sql=""" 
   select business_id, name, address, city, latitude, longitude, stars
   from business  
   where categories like '%Restaurants%'  
   and city = 'Las Vegas';
   """
restaurant_lv = pd.read_sql(sql,conn)

sql="""  
    select review_id, business_id, user_id, stars 
    from review 
    where business_id in ('{}');  
    """.format("','".join(restaurant_lv['business_id'].values.tolist()))
review_res = pd.read_sql(sql, conn)

sql="""    
    select user_id, name, average_stars from user;
    """
user_df = pd.read_sql(sql, conn)

#add attributes
buss = pd.read_csv('../Dataset/buss_attributes.csv')  
usr = pd.read_csv('../Dataset/user_attributes.csv') 

business = pd.merge(restaurant_lv,buss,on = 'business_id',how='inner')
user = pd.merge(user_df,usr,on = 'user_id',how='inner')
review = pd.merge(review_res,usr,on = 'user_id',how='inner')


# Generate Profile of the User

# avg = float(user_df.average_stars[user_df.user_id == user_id])

def gen_user_profile(user_id):
    # reviews given by the specified user
    avg = float(user_df.average_stars[user_df.user_id == user_id])
    reviews_given = review[review.user_id == user_id]
    reviews_given['stars'] = (reviews_given['stars'] - avg)/20
    reviews_given['avg_star'] = (reviews_given['avg_star'] - avg)/20
    reviews_given = reviews_given.sort_values('business_id')

    # list of ids of the restaurants reviewed by the user
    # reviewed_busi_list = reviews_given['business_id'].tolist()
    reviewed_busi = pd.DataFrame(reviews_given['business_id']).merge(business,on='business_id',how='left')
    # reviewed_busi = business[business['business_id'].isin(reviewed_busi_list)]
    reviewed_busi = reviewed_busi.sort_values('business_id')
    reviewed_busi['avg_star'] = (reviewed_busi['avg_star'] - avg)/20
    features = reviewed_busi.iloc[:,4:].to_numpy()
    profile = np.matrix(reviews_given.stars) * features
    return profile, avg

def calculate_cosine_sim(profile,avg):
    # Calculate Cosine Similarity of the Unreviewed Restaurants with the User's Profile
    res_test = business
    res_test = res_test.sort_values('business_id')
    res_test_list = res_test['business_id'].tolist()
    res_test['avg_star'] = (res_test['avg_star'] - avg)/20
    fea_test = res_test.iloc[:,4:].to_numpy()
    similarity = np.asarray(profile * fea_test.T) * 1./(norm(profile) * norm(fea_test, axis = 1))
    # similarity = cosine_similarity(profile,fea_test)
    return similarity, res_test_list

# Output the Recommended Restaurants
def recommend_user_cb(user_id,n):
    profile, avg = gen_user_profile(user_id)
    similarity, res_test_list = calculate_cosine_sim(profile, avg)
    index_arr = (-similarity).argsort()[:n][0][0:n]
    print ('Hi ' + user_df.name[user_df.user_id == user_id].values[0] + '\nCheck out these restaurants: ')
    recommend = pd.DataFrame(columns=['business_id','name','address','city','latitude','longitude','stars'])
    for i in index_arr:
        r = restaurant_lv[restaurant_lv['business_id'] == res_test_list[i]]
        recommend = recommend.append(r,ignore_index=True)
    # print(recommend)
    return recommend
    # return recommend.to_json(orient='records') # return in json format


def sort_distance_cb(user_id, n, rslt, address, dist_limit):
    geolocator = Nominatim(user_agent="stat418_project")
    location = geolocator.geocode(address) 
    coord1 = [location.latitude,location.longitude]
    coord2 = rslt[['latitude','longitude']].values.tolist()
    distance_lst = []
    for i in range(0,max(50,10*n)):
        distance_lst.append(distance(coord1,coord2[i]).mi)
    rslt['distance'] = distance_lst
    sorted_rslt = rslt.sort_values('distance')
    return sorted_rslt[sorted_rslt['distance']<=dist_limit][:n]


def content_based_recommendation(user_id,n=10,address='', dist_limit=15):
    if not address:
        result = recommend_user_cb(user_id, n=n)[['business_id','name','address', 'city','stars']]
    else:
        recomm_df = recommend_user_cb(user_id, n=max(50,10*n))
        result = sort_distance_cb(user_id, n, recomm_df, address, dist_limit)[['business_id','name','address','city','stars','distance']]
        if len(result) < n:
            print('NOTE: distance is too far from restaurants in Las Vegas. Please check your address again or increase distance limit.')
    return result
