# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:44:39 2023

@author: udehs
"""

# importing libraries

# libraries for computation and data manipulation
import numpy as np
import pandas as pd

# libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# library for preprocessing
from sklearn.preprocessing import LabelEncoder

# library for scaling
from sklearn.preprocessing import MinMaxScaler

# library for evaluating cluster quality
from sklearn.metrics import silhouette_score

# Read in the GDP per capita data
GDP_capita =  pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv', skiprows = 4)

# drop irrelevant column
GDP_capita = GDP_capita.drop('Unnamed: 66', axis =1)

# Reshape dataframe 
GDP_data = GDP_capita.melt(id_vars = ['Country Name',
                                           'Country Code',
                                           'Indicator Name',
                                           'Indicator Code'],
                                var_name = 'year',
                                value_name = 'value') 
                                
print(GDP_capita)

# Read in the GDP per capita data
pop = pd.read_csv('API_EN.POP.DNST_DS2_en_csv_v2_4770491.csv', skiprows=4)

# drop the unwanted drop column
pop = pop.drop('Unnamed: 66', axis =1)

# Reshape dataframe 
pop_data = pop.melt(id_vars = ['Country Name', 'Indicator Name', 'Country Code', 'Indicator Code'], var_name = 'year', value_name = 'value')

print(pop)

# Concatenate the two indicators
Data = pd.concat([GDP_data,pop_data])

# select important columns
Data = Data[['Country Name', 'Indicator Name', 'year', 'value']].copy()

# make the dataframe a pivot
Data_pivot = Data.pivot(index=['Country Name', 'year'],
                                columns='Indicator Name', 
                                values='value').reset_index()

# convert year to integer
Data_pivot['year'] = Data_pivot['year'].astype(int)

# list of irrelevant rows
irrelevant = ['Africa Eastern and Southern','Arab World','Caribbean small states','Central African Republic', 'Central Europe and the Baltics',
'Early-demographic dividend', 'East Asia & Pacific',
       'East Asia & Pacific (IDA & IBRD countries)',
       'East Asia & Pacific (excluding high income)','Europe & Central Asia',
       'Europe & Central Asia (IDA & IBRD countries)',
       'Europe & Central Asia (excluding high income)', 'European Union',
 'Fragile and conflict affected situations','French Polynesia','Heavily indebted poor countries (HIPC)',
 'High income', 'IBRD only',
       'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total','Late-demographic dividend',
 'Latin America & Caribbean',
       'Latin America & Caribbean (excluding high income)',
       'Latin America & the Caribbean (IDA & IBRD countries)',
       'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
 'Middle East & North Africa',
 'Middle East & North Africa (IDA & IBRD countries)',
       'Middle East & North Africa (excluding high income)',
       'Middle income', 'Not classified',
       'OECD members', 'Other small states',
       'Pacific island small states','Post-demographic dividend',
       'Pre-demographic dividend','Small states','South Asia (IDA & IBRD)','Sub-Saharan Africa', 
 'Sub-Saharan Africa (IDA & IBRD countries)',
       'Sub-Saharan Africa (excluding high income)','Upper middle income', 'West Bank and Gaza',
                 'World','Africa Western and Central'
]


# sort countries that are not in the GDP_irrelevant row
Data_1 = Data_pivot[~Data_pivot['Country Name'].isin(irrelevant)]

def grouped_mean(Data, group_by, mean_columns):
    
    """
    This function groups a DataFrame and compute the mean value of multiple columns.
    
    """
    
    grouped = Data.groupby(group_by)
    # Compute the mean value of the specified columns
    mean_df = grouped[mean_columns].mean()
    return mean_df.reset_index()

# Generate the data for the years between 2006 and 2021
Latter_data = Data_1[Data_1['year'].between(2010, 2021)]

# Compute the mean values for the recent data
Latter = grouped_mean(Latter_data, 'Country Name', ['GDP per capita (current US$)','Population density (people per sq. km of land area)'])

# Remove empty values
Latter = Latter.dropna()

# Normalize the data

scaler = MinMaxScaler()
scale = scaler.fit_transform(Latter.drop('Country Name', axis =1))

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []

#Test 1 to 10 differnt clusters
for i in range(1, 11): 
    '''
   Loop over the indented statements 10 times to see how distance reduces till it becomes steady.
   '''

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    
    #Train the model for all the 10 clusters and append into wcss list
    kmeans.fit(scale)  
    
    '''kmeans.fit(X) generates the following parameters:
    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=5, n_init=10, n_jobs=1, precompute_distances='auto', random_state=42, tol=0.0001, verbose=0)
    inertia: Sum of squared distances of samples to their closest cluster center
    '''
    #Add each Sum of squared distances to wcss[] list
    wcss.append(kmeans.inertia_) 
    
# plot the ten different wcss against the number of clusters    
plt.plot(range(1, 11), wcss) 
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#The Sum of squared distances started to stabilize from the 4th cluster. Therefore, the best number of clusters (k) in this task is 4

#Fitting  K-Means to the dataset 
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter=300, random_state = 42)

# Fit the model to the data
kmeans.fit(scale)

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(scale, kmeans.labels_)}')

# Get centroid values for the three clusters
centroids = kmeans.cluster_centers_
centroids = pd.DataFrame(centroids, columns=['GDP per capita (current US$)','Population density (people per sq. km of land area)'])
centroids.index = np.arange(1, len(centroids)+1)

# Plot the scatter plot of the clusters
plt.scatter(scale[:, 0], scale[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.xlabel('Input A')
plt.ylabel('Input B')
plt.show()


# Assign each country to a cluster
y_pred = kmeans.fit_predict(Latter.drop('Country Name', axis =1))
Latter['cluster'] = y_pred +1

# Plot the scatterplot of clusters against GDP per density and population density
plt.figure(figsize=(14,8))
sns.set_palette('dark')  
sns.scatterplot(x= Latter['GDP per capita (current US$)'],
                y= Latter['Population density (people per sq. km of land area)'], 
                hue= Latter['cluster'], 
                palette='muted')
plt.title('Country Clusters Based on GDP per Capita and population density', fontsize = 20)
plt.show()


# forecasting with Logistic function 
def logistic(t, n0, g, t0):
    
    """
    Calculates the logistic function 
    """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

import scipy.optimize as opt
import matplotlib.pyplot as plt
































































































































































































