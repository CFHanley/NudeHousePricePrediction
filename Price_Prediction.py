from datetime import datetime
from time import gmtime, strftime

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

ML_COLS = ['Price-Paid', 'Date', 'Full-Postcode', 'Postcode', 'Property-Type', 'Freehold/Leasehold',
           'New-Build',
           'UK region',
           'Town/City', 'Town/Area', 'Region', 'District', 'County',
           'Transaction Type', 'Grid Reference',  # 'Active postcodes',
           'Population',
           'Households',  # 'Nearby districts',
           'Latitude', 'Longitude', 'Easting',
           'Northing',  # 'Postcodes',
           'imputed-meta-data-info',
           'imputed-pcode-and-imputed-info']

REDUCED_COLS_FOR_PREDICTION_ASSISTANCE = ['Postcode', 'New-Build', 'UK region', 'Town/City',
                                          'Town/Area', 'Region', 'District', 'County',
                                          'Transaction Type', 'Grid Reference', 'Population', 'Households',
                                          'Latitude', 'Longitude', 'Easting', 'Northing']


def machine_learning_preprocessing(dataframe, customer_dataframe_requiring_encoding):
    # Selecting Columns
    dataframe = dataframe[ML_COLS]

    # Date Feature-Engineering
    dataframe['datetime'] = dataframe['Date'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))

    dataframe['Year'] = dataframe['datetime'].apply(
        lambda x: x.year)

    dataframe.drop(['datetime', 'Date'], inplace=True, axis=1)

    # Creating lists of columns names that are numeric vs non floats. Will allow for automated encoding
    cols = dataframe.columns
    print(cols)
    numeric_columns = dataframe._get_numeric_data().columns
    categorical_cols = list(set(cols) - set(numeric_columns))

    dataframe[categorical_cols] = dataframe[categorical_cols].astype('category')

    dataframe.dropna(inplace=True)
    enc = OrdinalEncoder()
    dataframe[categorical_cols] = enc.fit_transform(dataframe[categorical_cols])

    customer_dataframe_requiring_encoding['Year'] = customer_dataframe_requiring_encoding['Year'].astype('float64')
    customer_dataframe_requiring_encoding[categorical_cols] = enc.transform(
        customer_dataframe_requiring_encoding[categorical_cols])

    return dataframe, customer_dataframe_requiring_encoding


def split_to_predictors_and_target(dataframe):
    y = dataframe['Price-Paid']
    X = dataframe.drop('Price-Paid', inplace=False, axis=1)

    return X, y


def split_into_training_and_testing(predictors, targets):
    X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size=0.95, random_state=42)

    return X_train, X_test, y_train, y_test


def prediction_assistance_join(customer_dataframe, clean_transformed_data):
    trimmed_unprocessed_data = clean_transformed_data[REDUCED_COLS_FOR_PREDICTION_ASSISTANCE]
    customer_dataframe = customer_dataframe.merge(
        trimmed_unprocessed_data, left_on='Postcode', right_on='Postcode', how='left')

    return customer_dataframe[:1]


def make_dataframe_for_customer_prediction(date_today, full_postcode, property_type, leasehold_type):
    customer_request = {'Year': [date_today],
                        'Full-Postcode': [full_postcode],
                        'Postcode': [str(full_postcode).split()[0]],
                        'Property-Type': [property_type],
                        'Freehold/Leasehold': [leasehold_type],
                        'imputed-meta-data-info': False,
                        'imputed-pcode-and-imputed-info': False
                        }

    customer_request_df = pd.DataFrame(data=customer_request)

    return customer_request_df


if __name__ == '__main__':
    Prediction_Postcode = 'OL9 7FN'
    Prediction_Property_Type = 'Detached'
    Prediction_Date = strftime("%Y", gmtime())
    Prediction_leasehold = 'Leasehold'

    prediction_dataframe = make_dataframe_for_customer_prediction(Prediction_Date,
                                                                  Prediction_Postcode, Prediction_Property_Type,
                                                                  Prediction_leasehold)

    prediction_dataframe_with_meta_data = prediction_assistance_join(prediction_dataframe,
                                                                     pd.read_csv('Final_unprocessed_dataset.csv'))
    print(prediction_dataframe_with_meta_data)

    price_predictor = xgb.XGBRegressor(
        tree_method="gpu_hist",
        enable_categorical=True,
        use_label_encoder=False,
        n_estimators=100,
        max_depth=10,
        max_leaves=100,
        gamma=0.05,
        subsample=0.5,
        learning_rate=0.05)

    processed_data, customer_prediction_processed_data = machine_learning_preprocessing(
        pd.read_csv("Final_unprocessed_dataset.csv"), prediction_dataframe_with_meta_data)

    Prediction, Target = split_to_predictors_and_target(processed_data)
    X_train, X_test, y_train, y_test = split_into_training_and_testing(Prediction, Target)

    price_predictor.fit(X_train, y_train)
    price_predictor.save_model("xgboost_price_prediction_model_final.json")

    print(price_predictor.predict(customer_prediction_processed_data))
