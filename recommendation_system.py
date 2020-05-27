import pandas as pd
import numpy as np
import sqlite3
from collaborative_filtering import Collaborative_filtering
from content_based_rec import content_based_recommendation
from location_based_rec import location_based_recommendation

conn = sqlite3.connect('../Dataset/yelp.db')
users = pd.read_sql('select user_id from parsed_review group by user_id having count(user_id)>=10',conn)

def recommend_system(user_id,address='',n=10,dist_limit=1):
    users = pd.read_sql('select user_id from parsed_review group by user_id having count(user_id)>=10',conn)
    if user_id in users['user_id'].values.tolist():
        collab_recomm = Collaborative_filtering(user_id,n=n,address=address, dist_limit=dist_limit)
        content_recomm = content_based_recommendation(user_id,n=n,address=address, dist_limit=dist_limit)
        return collab_recomm,content_recomm
    else:
        if address:
            location_recomm = location_based_recommendation(address,n=n)
            return location_recomm
        else:
            sql = """select business_id,name,address,city,stars from business 
                where city="Las Vegas" 
                and is_open=1 
                and categories like "%Restaurants%" 
                and stars >=4 
                limit {}""".format(max(50,10*n))
            top_recomm = pd.read_sql(sql,conn).sample(n = n)
            return top_recomm

# sample
# user_id = 'tL2pS5UOmN6aAOi3Z-qFGg'
# user_id = 'zzUlFuJ5HFNEm15o9YC9Qg'
# user_id = 'zzK05ZbEva9FGAjEFWlGFg'
# address1 = '3655 S Las Vegas Blvd, Las Vegas, NV 89109'