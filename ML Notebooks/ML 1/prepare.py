import pandas as pd
import numpy as np
import random
from random import randrange
from random import seed
import time
import datetime
from random import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns

data = pd.read_csv('virus_data.csv')
train_data = pd.read_csv('x_train_final.csv')


def prepare_data(data: pd.DataFrame, training_data: pd.DataFrame):
    seed(139)
    new_data = data.copy()

    # Create a OHE for blood type
    new_data_encoded = pd.get_dummies(new_data, columns=['blood_type'])

    # Transform the string to a list of symptoms.
    def split(s):
        l = None
        if not pd.isna(s):
            l = s.split(';')
        else:
            l = ['NaN']
        return l

    # Get every type of symptoms.
    def get_items(df):
        s = set()
        for item in df:
            for value in item:
                s.add(value)
        return list(s)

    symptoms_as_list_df = new_data_encoded['symptoms'].apply(split)
    items = get_items(symptoms_as_list_df)

    # Transform a column containing list to a boolean df
    def boolean_df(item_lists, unique_items):
        bool_dict = {}
        for i, item in enumerate(unique_items):
            bool_dict[item] = item_lists.apply(lambda x: int(item in x))
        return pd.DataFrame(bool_dict)

    symptoms_df = boolean_df(symptoms_as_list_df, items)
    symptoms_df = symptoms_df.drop(['NaN'], axis=1)

    # Join it the training set and drop the old symptoms col
    new_data_encoded2 = new_data_encoded.join(symptoms_df).drop(['symptoms'], axis=1)

    # Convert a string of coordinates to a two columns dataframe

    # Convert a date to a timestamp (int)
    def convert_date(s: str):
        if not pd.isna(s):
            s = s.replace('-','/' )
            return time.mktime(datetime.datetime.strptime(s, "%d/%m/%y").timetuple())

    date_df = pd.DataFrame(new_data_encoded2['pcr_date'].apply(convert_date).to_list(), columns=['date'])
    new_data_encoded3 = new_data_encoded2.join(date_df).drop(['pcr_date'], axis = 1)

    # Drop the address columns
    new_data_final = new_data_encoded3.drop(['address'], axis=1)

    # __________IMPUTATION__________


    # imputation of the PCR_Values
    for i in ['PCR_01', 'PCR_04', 'PCR_05', 'PCR_08', 'PCR_10']:
        new_data_final[i] = new_data_final[i].fillna(training_data[i].median())

    # Imputation of the continuous columns
    cont_cols = ['household_income', 'sugar_levels', 'sport_activity']

    for col in cont_cols:
        new_data_final[col] = new_data_final[col].fillna(training_data[col].median())

    # ____________SCALING________________
    new_data_scaled = new_data_final.copy()

    standard_col = ['household_income',
                    'sport_activity', 'PCR_01',
                    'PCR_04', 'PCR_05', 'PCR_08', 'PCR_10','sugar_levels']

    for col in standard_col:
        std_scaler = StandardScaler()
        std_scaler.fit(training_data[[col]])
        new_data_scaled[col] = std_scaler.transform(new_data_final[[col]])

    # ________ADDING NEW FEATURES________
    r = pd.DataFrame(np.sqrt(new_data_scaled.PCR_05**2+new_data_scaled.sugar_levels**2), columns=['radial_distance_pcr5_sugar'])
    new_data_scaled = new_data_scaled.join(r)

    covid_symptoms = pd.DataFrame(new_data_scaled.sore_throat|new_data_scaled.shortness_of_breath|new_data_scaled.cough, columns=['covid_symptoms'])
    df_final = new_data_scaled.join(covid_symptoms)


    return df_final.drop([ 'age', 'sex', 'weight', 'num_of_siblings','sugar_levels',
       'happiness_score', 'conversations_per_day',
        'PCR_02', 'PCR_03',
        'PCR_06', 'PCR_07',  'PCR_09',
       'sore_throat', 'low_appetite', 'cough','shortness_of_breath'
       ,'current_location' ,'date'], axis = 1).replace({'High': 1, 'Low': 0,True: 1, False:0})







