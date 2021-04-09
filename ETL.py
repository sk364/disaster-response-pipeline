#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sqlalchemy import create_engine

# read datasets
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

# merge datasets
df = messages.merge(categories)

# get list of categories
categories = df['categories'].str.split(pat=';', expand=True)

# get first row
row = categories.iloc[0, :]

# get category column names
category_colnames = row.apply(lambda cell: cell[:-2])
categories.columns = category_colnames

# store value for each category and convert to numeric
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

# drop original categories column and join the category column names
df = df.drop(columns=['categories'])
df = df.join(categories)

# check if there are any duplicates
row_counts = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'count'})
dups = row_counts[row_counts['count'] > 1]

# drop duplicates, if any
if dups.shape[0] > 1:
    df = df.drop_duplicates()

# store the dataset to a sqlite DB
engine = create_engine('sqlite:///DisasterMessages.db')
df.to_sql('MessageCategory', engine, index=False)
