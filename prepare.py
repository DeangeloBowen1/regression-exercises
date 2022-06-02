from env import get_db_url
from sklearn.model_selection import train_test_split

import pandas as pd
import acquire as aq
import wrangle as wr

"""
Contains CodeUp dataset functions for prepping data.
Author: Deangelo Bowen


Splt:


Prep:

prep_iris_data():

preps and cleans iris data

prep_titanic_data():

preps and cleans titanic data

prep_telco_data():

preps and cleans telco data



"""

"""
Sample Code from fred

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test.
    Test is 20% of the original dataset, validate is .30*.80= 24% of the
    original dataset, and train is .70*.80= 56% of the original dataset.
    The function returns, in this order, train, validate and test dataframes.
    '''
    train_validate, test = train_test_split(df, test_size=0.2,
                                            random_state=seed,
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3,
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

"""

#split/prep iris data function------------------------------------------------------
def split_iris_data(iris):
    train_validate, test = train_test_split(iris, test_size=.2,
                                           random_state=123,
                                           stratify=iris.species)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123,
                                     stratify=train_validate.species)
    return train, validate, test

def prep_iris(iris):
    iris = iris.drop(['species_id','measurement_id'], axis = 1)
    iris = iris.rename(columns={'species_name': 'species'})
    dummy_var = pd.get_dummies(iris[['species']], dummy_na = False, drop_first=True)
    iris = pd.concat([iris, dummy_var], axis = 1)

    # the following portion is used only for training purposes with
    # the training dataset. Removing redundant cols:

    iris = iris.drop(['species_virginica', 'species_versicolor'], axis= 1)

    

    train, validate, test = split_iris_data(iris)
    
    return train, validate, test


#---------------------------------------------------------------------------------






#split/prep titanic data function------------------------------------------------------
def split_titanic_data(titanic):
    train_validate, test = train_test_split(titanic, test_size=.2,
                                           random_state=123,
                                           stratify=titanic.survived)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123,
                                     stratify=train_validate.survived)
    return train, validate, test

# prep titanic data
def prep_titanic(titanic):
    titanic = titanic.drop(['passenger_id',
                            'class','deck', 'embarked'], axis= 1)
    titanic.drop_duplicates(inplace=True)
    titanic['age'] = titanic.age.fillna(titanic.age.mean())
    titanic['embark_town'] = titanic.embark_town.fillna('Southampton')
    dummy_titanic = pd.get_dummies(titanic[['sex', 'embark_town']],
                              dummy_na = False,
                              drop_first = [True, True])
    titanic = titanic.drop(['sex','embark_town'], axis= 1)
    titanic = pd.concat([titanic, dummy_titanic], axis = 1)

    train, validate, test = split_titanic_data(titanic)
    
    return train, validate, test
#------------------------------------------------------------------------------------





#split/prep telco data function--------------------------------------------------------
def split_telco_data(telco):
    train_validate, test = train_test_split(telco, test_size=.2,
                                           random_state=123,
                                           stratify=telco.churn)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123,
                                     stratify=train_validate.churn)

    # Test Dataset
    
    # mapping gender
    train.gender = train.gender.map({'Male':1 , 'Female':0})
    train.gender.head()

    #mapping yes = 1, no = 1
    train.multiple_lines = train.multiple_lines.map({'Yes': 1, 'No': 0,
                                                     'No phone service':2})

    # drop columns for which I have a numerical dummy column for
    drop_columns1 = ['contract_type', 'payment_type', 'internet_service_type']
    train = train.drop(columns = drop_columns1)

    # Validate Dataset

    # mapping gender
    validate.gender = validate.gender.map({'Male':1 , 'Female':0})
    validate.gender.head()

    #mapping yes = 1, no = 1
    validate.multiple_lines = validate.multiple_lines.map({'Yes': 1, 'No': 0,
                                                     'No phone service':2})

    # drop columns for which I have a numerical dummy column for
    drop_columns2 = ['contract_type', 'payment_type', 'internet_service_type']
    validate = validate.drop(columns = drop_columns2)

    # Test Dataset
    # mapping gender
    test.gender = test.gender.map({'Male':1 , 'Female':0})
    test.gender.head()

    #mapping yes = 1, no = 1
    test.multiple_lines = test.multiple_lines.map({'Yes': 1, 'No': 0,
                                                     'No phone service':2})

    # drop columns for which I have a numerical dummy column for
    drop_columns3 = ['contract_type', 'payment_type', 'internet_service_type']
    test = test.drop(columns = drop_columns3)

    # renaming columns
    train = train.rename(columns={'gender': 'is_male',
                              'contract_type_Month-to-month': 'month_to_month_contract',
                             'contract_type_One year': 'one_year_contract',
                              'ontract_type_Two year': 'two_year_contract', 
                              'payment_type_Bank transfer (automatic)': 'bank_transfer_pay',
                              'payment_type_Credit card (automatic)': 'credit_card_pay', 
                              'payment_type_Electronic check': 'electronic_check_pay', 
                              'payment_type_Mailed check': 'mailed_check_pay', 
                              'internet_service_type_DSL': 'DSL_internet', 
                              'internet_service_type_Fiber optic': 'fiber_optic_internet', 
                              'internet_service_type_None': 'no_internet'})
    
    return train, validate, test





#Prep telco data function
def prep_telco(telco):
    
    # drop duplicates
    telco.drop_duplicates(inplace=True)
    
    # drop specific columns
    telco = telco.drop(['internet_service_type_id', 'payment_type_id', 'contract_type_id',
                  'online_security', 'online_backup', 'device_protection',
                  'tech_support', 'streaming_tv', 'streaming_movies'], axis= 1)
    
    # strip spaces from total charges, turn into a float
    telco['total_charges'] = telco['total_charges'].str.strip()
    telco = telco[telco.total_charges != '']
    telco['total_charges']= telco.total_charges.astype('float')
    
    # map Yes = 1, No = 0
    telco['partner'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco['dependents'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco['phone_service'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco['paperless_billing'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    # create dummy for cat cols to numeric cols
    dummy_telco = pd.get_dummies(telco[['multiple_lines', 'contract_type',
                                     'payment_type', 'internet_service_type']])
    
    # concat
    telco = pd.concat([telco, dummy_telco], axis = 1)

    # drop unneeded columns from the dummies concat
    telco = telco.drop(['multiple_lines_No', 'multiple_lines_No phone service',
                        'multiple_lines_Yes'], axis =1)
    
    # train validate test
    train, validate, test = split_telco_data(telco)
    
    return train, validate, test
#---------------------------------------------------------------------------------

def split_zillow_data(zillow):
    train_validate, test = train_test_split(zillow, test_size=.2,
                                           random_state=123)
    train,validate = train_test_split(train_validate, test_size=.3,
                                     random_state=123)
    return train, validate, test


def prep_zillow(zillow):
    
    zillow = zillow.dropna()
    zillow = zillow.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqft', 'yearbuilt':'year_built', 
                          'taxamount':'tax_amount', 'taxvaluedollarcnt':'tax_value'})
    
    column_list = zillow.columns[0:-1]
    zillow = wr.omit_outliers(zillow, 1.5, column_list)
    
    zillow.fips = zillow.fips.map({6037.0: 'Los Angeles', 6059.0:'Orange County', 6111.0:'Ventura County'})

    train, validate, test = split_zillow_data(zillow)

    return train, validate, test
