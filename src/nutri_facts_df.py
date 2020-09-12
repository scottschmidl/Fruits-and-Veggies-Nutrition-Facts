import pandas as pd

'''import csv with nutrition facts into data frame'''

def get_nutri_facts(data):
    df = pd.read_csv(data)
    return df

if __name__ == "__main__":
    get_dat_data = get_nutri_facts('data/nutri_facts_name.csv')
    print(get_dat_data.head())