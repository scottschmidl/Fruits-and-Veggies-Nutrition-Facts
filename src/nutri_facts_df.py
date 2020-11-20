### this file is for displaying nutrition facts on the flask app ###
import pandas as pd

def get_nutri_facts(data):
    '''import csv with nutrition facts into data frame'''
    df = pd.read_csv(data)
    return df

if __name__ == "__main__":
    get_dat_data = get_nutri_facts('data/nutri_facts_name.csv')
    print(get_dat_data['Fruits_Vegetables_Name'].tolist())