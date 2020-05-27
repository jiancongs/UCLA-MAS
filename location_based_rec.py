# Location-Based Recommendation
import sqlite3
import pandas as pd
import numpy as np
from geopy.distance import distance
from geopy.geocoders import Nominatim
import pickle
import matplotlib.pyplot as plt
import plotly_express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Read in Datasets from yelp.db Using sqlite3
conn = sqlite3.connect('../Dataset/yelp.db')
kmeans_load = pickle.load(open('../Models/Collaborative_filtering.model', 'rb'))

def data_prep():
    sql=""" 
    select business_id, name, address, review_count, stars, latitude, longitude
    from business  
    where categories like '%Restaurants%'  
    and city = 'Las Vegas' 
    and is_open = 1; 
    """
    restaurant_lv = pd.read_sql(sql,conn)
    return restaurant_lv
    # Location-Based Recommendation
    # top_res_lv = restaurant_lv.sort_values(by=['review_count', 'stars'], ascending=False)

# K-Means Clustering
# determing the number of clusters by the Elbow plot
def find_param_Elbow(coords):
    distortions = []
    K = range(1,25)
    for k in K:
        kmeansModel = KMeans(n_clusters=k)
        kmeansModel = kmeansModel.fit(coords)
        distortions.append(kmeansModel.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(K, distortions, marker='o')
    plt.xlabel('k')
    plt.ylabel('Distortions')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('kmean_elbow.png')
# plt.show()

# silhouette score method
def find_param_silhouette(coords):
    sil = []
    kmax = 50
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(coords)
        labels = kmeans.labels_
        sil.append(silhouette_score(coords, labels, metric = 'euclidean'))

# fit a K-Means model with 5 clusters based on the Elbow plot
def fid_kmeans_model():
    coords = restaurant_lv[['longitude','latitude']]
    kmeans = KMeans(n_clusters=5, init='k-means++')
    kmeans.fit(coords)
    y = kmeans.labels_
    print("k = 5", " silhouette_score ", silhouette_score(coords, y, metric='euclidean'))
    path = '../Models/location_based.model'
    pickle.dump(kmeans, open(path, 'wb'))
    print("Model is saved to: {}".format(path))

def location_based_recommendation(n=10):
    restaurant_lv['cluster'] = kmeans.predict(restaurant_lv[['longitude','latitude']])

# plot the restaurant clusters
def plot_cluster():
    fig1 = px.scatter_mapbox(restaurant_lv, lat="latitude", lon="longitude", color="cluster", size='stars',
                hover_data= ['name', 'latitude', 'longitude'], zoom=10, width=1200, height=800)
# fig1.write_image('lvres_cluster.png')
# need to install plotly-orca to export the interactive image to a static image

# define a function to recommend the best restaurants
def location_based_recommendation(address,n=10):
    # Predict the cluster for longitude and latitude provided
    path = '../Models/location_based.model'
    kmeans = pickle.load(open(path, 'rb'))
    restaurant_lv = data_prep()
    df = restaurant_lv.sort_values(by=['review_count', 'stars'], ascending=False)   
    geolocator = Nominatim(user_agent="stat418_project")
    location = geolocator.geocode(address) 
    coord1 = [location.latitude,location.longitude]
    df['cluster'] = kmeans.predict(df[['longitude','latitude']])
    cluster = kmeans.predict(np.array(coord1).reshape(1, -1))[0]
    print(cluster)
    # Get the top N restaurant in this cluster
    # return df[df['cluster'] == cluster].iloc[0:5][['name', 'latitude', 'longitude']]
    return df[(df['cluster'] == cluster) & (df['stars']>=4)].iloc[0:n]

# Test for Recommendation
def test_location_based():
    test_coordinates = {
        'latitude' : [36.1017316],
        'longitude' : [-115.1891691],
    }
    user1 = pd.DataFrame(test_coordinates)
    recommend_restaurants(top_res_lv, user1.longitude, user1.latitude)

    # plot the locations of the restaurants and the user
    fig = px.scatter_mapbox(recommend_restaurants(top_res_lv, user1.longitude, user1.latitude),
                            lat="latitude", lon="longitude",
                            zoom=10, width=1200, height=800,
                            hover_data= ['name', 'latitude', 'longitude'])
    fig2 = fig.add_scattermapbox(
        lat=user1["latitude"],lon= user1["longitude"]).update_traces(dict(mode='markers', marker = dict(size = 15)))
    # fig2.write_image('recommend_plot.png')

def plots():
    # print the top 2 restaurants in each cluster
    restaurant_lv = data_prep()
    res_c = restaurant_lv.sort_values(['stars'],ascending = False).groupby('cluster').head(2)
    res_c2 = res_c.sort_values(by='cluster')
    res_c2[['cluster','name','stars']]

    # print recommendations to the user
    print(user1)
    print("\nHi, check out these restaurants in the neighborhood:")
    print(recommend_restaurants(top_res_lv, user1.longitude, user1.latitude))