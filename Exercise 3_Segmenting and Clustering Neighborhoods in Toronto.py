# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:45:01 2020

@author: marin
"""

#Importing table from HTML 

import requests
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
html = requests.get(url).content
df = pd.read_html(html)
df_canada = df[0]

#Ignoring cells with a borough that is Not assigned

df_canada = df_canada.drop(df_canada[df_canada.Borough=="Not assigned"].index)

#Aggregate neighborhoods per Postal Code

df_canada=df_canada.groupby("Postal Code").agg(lambda x:','.join(set(x)))

#If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough

df_canada.loc[df_canada.Neighborhood == "Not assigned", "Neighborhood"] = df_canada.Borough

# The dataframe will consist of three columns: PostalCode, Borough, and Neighborhood

df_canada.reset_index(level=0, inplace=True)

df_canada.columns=['Postcode','Borough','Neighborhood']
df_canada.shape

#Importing CSV file

url = 'http://cocl.us/Geospatial_data'
df=pd.read_csv(url)
df.head()

df.columns=['Postcode','Latitude','Longitude']

#Merging df_canada dataframe with Lat/Long dataframe (df) based on Postcode

results = df_canada.merge(df,on='Postcode')

results

##Importing Libraries

import numpy as np # library to handle data in a vectorized manner

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[43.651070, -79.347015], zoom_start=11)


# add markers to map
for lat, lng, label in zip(results['Latitude'], results['Longitude'], results['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto

# Foursquare identification
CLIENT_ID = 'A4E2RJ3RMEKUDGR4XYKXXXQGORQ2AM5KIB1NJWZDTAFRN14T' # your Foursquare ID
CLIENT_SECRET = 'VMOWVQJ3LVU2JDWL0N2WS11W5KGVQJIWRQ4UPTG14UOJB5G1' # your Foursquare Secret
VERSION = '20190530' # Foursquare API version

#Get the top 100 venues in Scarborough
scarborough_data = results[results['Borough'] == 'Scarborough'].reset_index(drop=True)
scarborough_data.head(5)

address_scar = 'Scarborough,Toronto'
latitude_scar = 43.773077
longitude_scar = -79.257774

LIMIT = 100
radius = 500
url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude_scar, longitude_scar, VERSION, radius, LIMIT)                                                              

Scarborough = requests.get(url).json()

def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

import json # library to handle JSON files
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

venues = Scarborough['response']['groups'][0]['items']  
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head(10)

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

scarborough_venues = getNearbyVenues(names=scarborough_data['Neighborhood'],
                                   latitudes=scarborough_data['Latitude'],
                                   longitudes=scarborough_data['Longitude']
                                  )
scarborough_venues.head(10)

# one hot encoding
scarb_onehot = pd.get_dummies(scarborough_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
scarb_onehot['Neighborhood'] = scarborough_venues['Neighborhood']
 
# move neighborhood column to the first column
fixed_columns = [scarb_onehot.columns[-1]] + list(scarb_onehot.columns[:-1])
scarb_onehot = scarb_onehot[fixed_columns]

scarb_onehot.head()

scarb_grouped = scarb_onehot.groupby('Neighborhood').mean().reset_index()
scarb_grouped.head(7)

# import k-means from clustering stage
from sklearn.cluster import KMeans

scarb_data = scarborough_data.drop(14)
# set number of clusters
kclusters = 5

scarb_grouped_clustering = scarb_grouped.drop('Neighborhood', 1)


# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(scarb_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
len(kmeans.labels_)#=16
scarborough_data.shape

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = scarb_grouped['Neighborhood']

for ind in np.arange(scarb_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(scarb_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted

scarb_merged = scarb_data

# add clustering labels
scarb_merged['Cluster Labels'] = kmeans.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
scarb_merged = scarb_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

scarb_merged

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location = [latitude_scar, longitude_scar], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(scarb_merged['Latitude'], scarb_merged['Longitude'], scarb_merged['Neighborhood'], scarb_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters.save('my_map3.html')

from IPython.display import HTML

HTML('<iframe src=my_map3.html width=700 height=450></iframe>')

##Examine each of the five clusters

scarb_merged.loc[scarb_merged['Cluster Labels'] == 0, scarb_merged.columns[[1] + list(range(5, scarb_merged.shape[1]))]]
scarb_merged.loc[scarb_merged['Cluster Labels'] == 1, scarb_merged.columns[[1] + list(range(5, scarb_merged.shape[1]))]]
scarb_merged.loc[scarb_merged['Cluster Labels'] == 2, scarb_merged.columns[[1] + list(range(5, scarb_merged.shape[1]))]]
scarb_merged.loc[scarb_merged['Cluster Labels'] == 3, scarb_merged.columns[[1] + list(range(5, scarb_merged.shape[1]))]]
scarb_merged.loc[scarb_merged['Cluster Labels'] == 4, scarb_merged.columns[[1] + list(range(5, scarb_merged.shape[1]))]]

