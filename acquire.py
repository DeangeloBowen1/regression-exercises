from env import host, user, password, get_db_url
import pandas as pd 
import os

def get_zillow_data(use_cache=True):
    filename = 'zillow.csv'
    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''
        SELECT
        bathroomcnt, 
        bedroomcnt, 
        calculatedfinishedsquarefeet, 
        yearbuilt,
        taxamount,
        taxvaluedollarcnt,
        fips
        FROM properties_2017
        JOIN propertylandusetype USING(propertylandusetypeid)
        Where propertylandusetypeid = 261;''' , get_db_url('zillow'))
        df.to_csv(filename, index=False)
        return df
