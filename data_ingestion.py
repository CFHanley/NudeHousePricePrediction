import pandas as pd

ASSIGNED_CSV_HEADERS = ['Transaction-Unique-Identifier', 'Price-Paid', 'Date', 'Full-Postcode', 'Property-Type',
                        'New-Build',
                        'Freehold/Leasehold', 'House-Number', 'Flat-Number', 'Street', 'Locality', 'Town/City',
                        'District', 'County', 'Transaction Type', 'Record Status']

WANTED_COLUMNS = ['Price-Paid', 'Date', 'Full-Postcode', 'Property-Type', 'New-Build',
                  'Freehold/Leasehold', 'House-Number', 'Flat-Number', 'Street', 'Locality', 'Town/City',
                  'District', 'County', 'Transaction Type']

MASTER_COLUMNS = ['Price-Paid', 'Date', 'Full-Postcode', 'Property-Type', 'New-Build',
                  'Freehold/Leasehold', 'House-Number', 'Flat-Number', 'Street', 'Locality', 'Town/City',
                  'District', 'County', 'Transaction Type', 'Postcode']

CSV_FILES = ("2018-Housing-Price-Paid.csv", "2019-Housing-Price-Paid.csv", "2020-Housing-Price-Paid.csv")

NEW_BUILD_MAPPING = {"N": "New-Build",
                     "Y": "Non-New-Build"}

PROPERTY_TYPE_MAPPING = {"D": "Detached",
                         "S": "Semi-Detached",
                         "T": "Terraced",
                         "F": "Flats/Maisonettes",
                         "O": "Other Buildings"}

FREEHOLD_LEASEHOLD_MAPPING = {"F": "Freehold",
                              "L": "Leasehold"}

TRANSACTION_TYPE = {"A": "Citizen Transaction",
                    "B": "Business / Other Transaction"}


def ingest_house_price_data(filelist: list):
    """ Ingesting house prices """
    # Creating an empty master dataframe to append cleaned csv files to.
    master_dataframe = pd.DataFrame(columns=WANTED_COLUMNS)

    # Looping through all the files listed in the CSV_FILES variable. Each iteration of the loop will read in the
    # file, apply headers to the dataframe, selected only the wanted columns, append the transformed dataframe to
    # master dataframe. Once each iteration is complete, then the data is saved to the local directory.
    for file in filelist:
        data = pd.read_csv(file)
        data.columns = ASSIGNED_CSV_HEADERS
        data_wanted_cols = data[WANTED_COLUMNS]
        data_wanted_cols.loc[:, 'New-Build'] = data_wanted_cols.loc[:, 'New-Build'].map(NEW_BUILD_MAPPING)
        data_wanted_cols.loc[:, 'Property-Type'] = data_wanted_cols.loc[:, 'Property-Type'].map(PROPERTY_TYPE_MAPPING)
        data_wanted_cols.loc[:, 'Freehold/Leasehold'] = data_wanted_cols.loc[:, 'Freehold/Leasehold'].map(
            FREEHOLD_LEASEHOLD_MAPPING)
        data_wanted_cols.loc[:, 'Transaction Type'] = data_wanted_cols.loc[:, 'Transaction Type'].map(TRANSACTION_TYPE)
        data_wanted_cols.loc[:, 'Postcode'] = data_wanted_cols.loc[:, 'Full-Postcode'].apply(
            lambda x: str(x).split()[0])
        master_dataframe = pd.concat([master_dataframe, data_wanted_cols])

    master_dataframe.to_csv("Master_Housing_Data.csv", index=False)
    return master_dataframe


def ingest_required_postcodes():
    """ Ingesting postcode data - removing Scottish postcodes since they are not needed """
    post_codes_metadata = pd.read_csv('postcode-districts.csv')

    # Removed scotland and Na's since they were not required. Unsure about removing post codes with zero population.
    # Left them in to be safe
    post_codes_metadata = post_codes_metadata[~post_codes_metadata["UK region"].isin(['Scotland', 'nan'])]
    return post_codes_metadata


def paid_price_data_postcode_merge(paid_price, postcodes):
    """ Merging paid prices with postcode data """
    merged_data = paid_price.merge(postcodes, on='Postcode', how='left')
    return merged_data


def impute_missing_postcode_metadata(paid_price_data, meta_data):
    """ Impute missing postcode metadata - matching with the closest postcode to impute matadata """
    data_without_meta = paid_price_data[paid_price_data['UK region'].isna()]
    postcodes_without_meta = data_without_meta[data_without_meta['Full-Postcode'].notna()]['Postcode'].unique()

    postcodes_with_meta = paid_price_data[paid_price_data['UK region'].notna()]['Postcode'].unique()
    matches = {}

    for codes in postcodes_without_meta:
        for codes2 in postcodes_with_meta:
            if str(codes)[:2] == str(codes2)[:2]:
                matches[codes] = codes2
                break

    postcodes_for_imputation = data_without_meta[data_without_meta['Full-Postcode'].notna()]
    postcodes_for_imputation = postcodes_for_imputation[MASTER_COLUMNS]
    postcodes_for_imputation['temp-postcode'] = postcodes_for_imputation['Postcode'].map(matches)
    postcodes_for_imputation = postcodes_for_imputation.merge(meta_data, left_on='temp-postcode',
                                                              right_on='Postcode', how='left')

    postcodes_for_imputation['imputed-meta-data-info'] = True
    postcodes_for_imputation.drop(['temp-postcode', 'Postcode_y'], axis=1, inplace=True)
    postcodes_for_imputation.rename(columns={'Postcode_x': 'Postcode'}, inplace=True)
    postcodes_for_imputation['imputed-pcode-and-imputed-info'] = False

    return postcodes_for_imputation


def impute_missing_postcodes(dataframe, meta_data):
    """ Imputing postcodes for transactions without recorded postcode bsed on Town/City, District and County"""
    # Finding Town/City + District + Counties combinations where the property doesn't have a postcode.
    no_postcode = dataframe[dataframe['Full-Postcode'].isna()]
    no_postcode['unique-concat'] = unique_address(no_postcode)

    # Finding Town/City + District + Counties combinations where the property does have a postcode.
    with_post_codes = dataframe[dataframe['UK region'].notna()][['Town/City', 'District', 'County', 'Postcode']]
    with_post_codes['unique-concat'] = unique_address(with_post_codes)
    with_post_codes.drop_duplicates(inplace=True)

    id_to_postcode_dict = {with_post_codes['unique-concat'][idx]: with_post_codes['Postcode'][idx] for idx in with_post_codes.index}

    no_postcode['Postcode'] = no_postcode['unique-concat'].map(id_to_postcode_dict)
    no_postcode = no_postcode[MASTER_COLUMNS]
    no_postcode = no_postcode.merge(meta_data, left_on='Postcode', right_on='Postcode', how='left')

    no_postcode['imputed-meta-data-info'] = False
    no_postcode['imputed-pcode-and-imputed-info'] = True

    return no_postcode


def unique_address(postcode_df):
    """ Returns colum with concatenated Town/City, District and County"""
    return postcode_df['Town/City'] + postcode_df['District'] + postcode_df['County']


def final_merge(main_data, imputed_meta, imputed_postcode):
    """ Union imputed dataframes to main_data dataframe """
    main_data_without_na_region = main_data[main_data['UK region'].notna()]
    main_data_without_na_region['imputed-meta-data-info'] = False
    main_data_without_na_region['imputed-pcode-and-imputed-info'] = False

    return pd.concat([main_data_without_na_region, imputed_meta, imputed_postcode])
