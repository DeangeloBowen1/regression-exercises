import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import acquire as aq


# function to calculate quartile range and remove outliers
def omit_outliers(df, calc, columns):
    for col in columns:
        
        # select quartiles
        q1, q3 = df[col].quantile([.25,.75]) 
        
        # calculate interquartile range
        iqr = q3 - q1
        
        upper_bound = q3 + calc * iqr
        lower_bound = q1 - calc * iqr
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df




def wrangle_zillow():
    zillow_df = aq.get_zillow_data()
    df = zillow_df.dropna()
    df = df.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqft', 'yearbuilt':'year_built', 
                          'taxamount':'tax_amount', 'taxvaluedollarcnt':'tax_value'})
    
    column_list = df.columns[0:-1]
    df = omit_outliers(df, 1.5, column_list)
    
    df.fips = df.fips.map({6037.0: 'Los Angeles', 6059.0:'Orange County', 6111.0:'Ventura County'})
    
    return df
