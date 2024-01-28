import pandas as pd


def one_hot_encode(df, column_name):
    """
    One-hot encodes a categorical column in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the categorical column.
    column_name (str): The name of the categorical column to be one-hot encoded.

    Returns:
    tuple: A tuple containing the modified DataFrame with the categorical column dropped and the one-hot encoded column.
    """
    one_hot_encoded = pd.get_dummies(df[column_name], dtype=int)
    df.drop(column_name, axis=1, inplace=True)
    return df, one_hot_encoded
