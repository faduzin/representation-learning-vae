import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def clean_data(df,
               remove_na=True,
               remove_duplicates=True,
               remove_outliers=True):
    if remove_na:
        try:
            df = df.dropna()
        except:
            print('Error removing NA values')

    if remove_duplicates:
        try:
            df = df.drop_duplicates()
        except:
            print('Error removing duplicates')

    if remove_outliers:
        try:
            # Remove outliers
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        except:
            print('Error removing outliers')
    
    return df


def encode_labels(df, columns):
    le = LabelEncoder()
    for col in columns:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            print(f'Error encoding column {col} labels')
    
    return df


def normalize_data(df,
                   columns,
                   method='standard'):
    match method:
        case 'standard':
            scaler = StandardScaler()
        case 'minmax':
            scaler = MinMaxScaler()
        case _:
            print('Invalid normalization method')
            return df
        
    try:
        df[columns] = scaler.fit_transform(df[columns])
    except:
        print('Error normalizing data')

    return df


def preprocess_data(df,
                    remove_na=True,
                    remove_duplicates=True,
                    remove_outliers=True,
                    encode_labels=True,
                    normalize_data=True,
                    label_columns=[],
                    normalize_columns=[],
                    normalize_method='standard'):
    df = clean_data(df, remove_na, remove_duplicates, remove_outliers)
    
    if encode_labels:
        df = encode_labels(df, label_columns)
    
    if normalize_data:
        df = normalize_data(df, normalize_columns, normalize_method)
    
    return df


def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=42)