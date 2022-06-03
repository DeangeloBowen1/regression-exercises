import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydataset
import seaborn as sns


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

from sklearn.model_selection import train_test_split
from scipy import stats as stats
from itertools import combinations


import warnings
warnings.filterwarnings("ignore")

import acquire as aq
import prepare as prep



def plot_variable_pairs(df):
    
    column = ['bedrooms', 'bathrooms', 'tax_amount', 'sqft', 
                'year_built']
    columns= combinations(column, 2)
    
    for col in columns:
        sns.lmplot(data = df.sample(1000), x=col[1] , y=col[0],
                   col = 'fips',
                   hue='fips',line_kws={'color': 'red'} )
    return



def plot_categorical_and_continuous_vars(df):
    
    columns = ['bedrooms', 'bathrooms', 'tax_amount', 'sqft',
               'year_built']
    
    columns2 = ['bedrooms', 'bathrooms', 'tax_amount', 'sqft',
                'tax_value']

    for col in columns:
        plt.figure(figsize=(10, 13))
        sns.displot(data = df.sample(10_000), x=col, hue='fips', element="step")

    for col in columns2:
        plt.figure(figsize=(7, 8))
        sns.swarmplot(data=df.sample(1000), x='fips', y=col)
    
    for col in columns2:
        plt.figure(figsize=(10,5))
        plt.xlabel(col)
        sns.violinplot(x = df.fips, y = df[col], split =True,
                       scale='count')
        plt.show()
    
    return
