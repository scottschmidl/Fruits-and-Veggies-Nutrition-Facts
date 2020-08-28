import pandas as pd

'''import csv with nutrition facts into data frame'''

df = pd.read_csv('data/nutri_facts_name.csv')
print(df.head())