import pandas as pd
from sklearn.preprocessing import LabelEncoder


def make_mystep(df):
    df['step'] = 1577836800 + df['step'] * 3600 * 24
    df['step'] = pd.to_datetime(df['step'], unit='s')
    return df


compute_features = {
    "nbre_trans_7jrs": 7,
    "nbre_trans_15jrs": 15,
    "nbre_trans_30jrs": 30
}
compute_features_cm = {
    "count_cust_merch_1_day": 1,
    "count_cust_merch_7_day": 7,
    "count_cust_merch_15_day": 15
}


def create_transaction_customer_historic(data):
    for key, value in compute_features.items():
        temp = pd.Series(data.index, index=data.step, name=key).sort_index()
        count_day = temp.rolling(str(value) + 'd').count() - 1
        count_day.index = temp.values
        data[key] = count_day.reindex(data.index)
    return data


# Historique de transaction du client sur un même marchant
def create_transaction_customer_merchant_historic(data):
    for key, value in compute_features_cm.items():
        temp = pd.Series(data.index, index=data.step, name=key).sort_index()
        count_day = temp.rolling(str(value) + 'd').count() - 1
        count_day.index = temp.values
        data[key] = count_day.reindex(data.index)
    return data


def encoder(data):
    cat_cols = ['age', 'gender', 'category']
    enc = LabelEncoder()
    for col in cat_cols:
        data[col] = enc.fit_transform(data[col])
    return data


def pipeline_transform(data_init):
    df = make_mystep(data_init)
    df = df.groupby(['customer', 'merchant']).apply(create_transaction_customer_merchant_historic)
    df = df.groupby('customer').apply(create_transaction_customer_historic)
    df = df.drop(['customer', 'merchant', 'step'], axis=1)
    df = encoder(df)
    return df


def predict(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config, index=[0])
    else:
        df = config
    data = pipeline_transform(df)
    y_pred = model.predict(data)
    return y_pred
