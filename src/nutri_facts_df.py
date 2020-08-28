#import csv with nutrition facts
import pandas as pd

df = pd.read_csv('data/nutri_facts_name.csv')
print(df.head())