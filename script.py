import numpy as np
import pandas as pd
from pathlib import Path
import holidays
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

df_train = pd.read_parquet(Path('data') / 'train.parquet')
df_test = pd.read_parquet(Path('data') / 'final_test.parquet')

X_train = df_train.drop(columns=['log_bike_count', 'bike_count'])
y_train = df_train['log_bike_count']
X_test = df_test
X_train.info()
X_test.info()
def _encode_dates(X):
    X = X.copy()
    X['date'] = pd.to_datetime(X['date'])
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['weekday'] = X['date'].dt.weekday
    X['hour'] = X['date'].dt.hour

    fr_holidays = holidays.France(years=[2020, 2021])
    def is_holiday(date):
        weekday = date.weekday()
        if weekday > 4 or date in fr_holidays:
            return 1
        else:
            return 0
    X['is_holiday'] = X['date'].apply(is_holiday)
    return X


def _merge_external_data(X):
    file_path = Path('data') / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    X = X.copy()
    X['date'] = X['date'].astype('datetime64[ns]')
    X['orig_index'] = np.arange(X.shape[0])
    cols_to_merge = ['date', 'pmer', 'tend', 'cod_tend', 'tend24',
                                        'dd', 'ff', 't', 'td', 'u', 'vv',  
                                        'n', 'pres', 'raf10', 'ww', 'nbas',
                                        'ht_neige', 'rr1', 'rr6',]
    X = pd.merge_asof(
        X.sort_values('date'), df_ext[cols_to_merge].sort_values('date'), on='date'
    )  
    X = X.sort_values('orig_index')
    del X['orig_index']
    for col in cols_to_merge:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)
    return X


def one_hot_encode_and_concat(X):
    categorical_columns = ['counter_id']
    one_hot = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoded_data = one_hot.fit_transform(X[categorical_columns])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data.toarray(), 
            columns=one_hot.get_feature_names_out(categorical_columns))
    X_dropped = X.drop(columns=categorical_columns)
    X_encoded = pd.concat([X_dropped.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)
    del X_encoded['date']

    return X_encoded


def _drop_columns(X):
    res = X.copy()
    res = res.drop(columns=['counter_name',
        'coordinates',
        'site_name',
        'site_id',
        'counter_technical_id',
        'counter_installation_date',
        'latitude', 
        'longitude'
        ])
    return res
def preprocessing(X):
    X = _encode_dates(X)
    X = _merge_external_data(X)
    X = one_hot_encode_and_concat(X)
    X = _drop_columns(X)
    return X

X_train = preprocessing(X_train)
X_test = preprocessing(X_test)
regressor = XGBRegressor(

)

pipeline = Pipeline([
    #('scaler', ),
    ('regressor', regressor)
])

pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(prediction.shape[0]),
        log_bike_count=prediction,
    )
)

results.to_csv("submission.csv", index=False)
print("CSV file created: submission.csv")