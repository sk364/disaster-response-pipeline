#!/usr/bin/env python
# coding: utf-8

"""
This module performs the Extract, Transform, Load (ETL) operations on the CSVs, provided in
the same directory as this module named `messages.csv` and `categories.csv`, outputting a SQLite DB
named `DisasterResponse.db` containing one table named `MessageCategories`. The below methods
contain relevant comments to indicate the ETL operations perfomed on the given data.
"""

import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_data_path, categories_data_path):
    """
    Loads messages & categories CSVs, returning the merged DataFrame of the two
    :param: messages_data_path: `str` storing the file path to messages csv data
    :param: categories_data_path: `str` storing the file path to categories csv data
    :return: `pandas.DataFrame` object
    """

    # Extract
    # read datasets
    try:
        messages = pd.read_csv(messages_data_path)
        categories = pd.read_csv(categories_data_path)
    except:
        print("CSV data could not be loaded. Please check if file exists and contains valid data.")
        sys.exit()

    # merge datasets
    df = messages.merge(categories)

    return df


def transform_data(df):
    """
    Merges the datasets such that all categories are flattened out as columns having values 0 or 1.
    Also, drops the duplicates.
    :param: df: `pandas.DataFrame` object
    :return: `pandas.DataFrame` object
    """

    # Transform
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
    row_counts = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
    dups = row_counts[row_counts['count'] > 1]

    # drop duplicates, if any
    if dups.shape[0] > 1:
        df = df.drop_duplicates()

    # drop rows which have categories with value other than 0 or 1
    df = df[(df[category_colnames].eq(0)) | (df[category_colnames].eq(1))]

    return df


def store_data_to_sqlite(df, database_name, table_name):
    """
    Store `df` to a sqlite DB.
    :param: df: `pandas.DataFrame` object
    :param: database_name: `str` object to name the database
    :param: table_name: `str` object naming the table
    :return: None
    """

    # Load
    engine = create_engine(f'sqlite:///{database_name}')
    df.to_sql(table_name, engine, index=False)


if __name__ == '__main__':
    args = sys.argv

    if len(args) < 5:
        print("Usage: python process_data.py [messages_csv_path] [categories_csv_path] [database_name] [table_name]")
        sys.exit()

    messages_csv_path, categories_csv_path, database_name, table_name = args[1:5]

    df = load_data(messages_csv_path, categories_csv_path)
    df = transform_data(df)
    store_data_to_sqlite(df, database_name, table_name)
