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

df_canada.columns=['Postcode','Borough','Neighbourhood']
df_canada.shape
