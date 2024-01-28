import pandas as pd


def one_hot_encode(df, column_name):
    one_hot_encoded = pd.get_dummies(df[column_name], dtype=int)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df.drop(column_name, axis=1, inplace=True)
    return df
