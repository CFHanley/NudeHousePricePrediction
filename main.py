from data_ingestion import (paid_price_data_postcode_merge,
                            ingest_required_postcodes,
                            ingest_house_price_data,
                            CSV_FILES,
                            impute_missing_postcode_metadata,
                            impute_missing_postcodes,
                            final_merge)


def clean_and_transform():
    paid_price = ingest_house_price_data(CSV_FILES)
    postcode_metadata = ingest_required_postcodes()

    merged_data = paid_price_data_postcode_merge(paid_price, postcode_metadata)

    imputed_metadata = impute_missing_postcode_metadata(merged_data, postcode_metadata)
    imputed_postcodes = impute_missing_postcodes(merged_data, postcode_metadata)

    final_unprocessed_dataset = final_merge(merged_data, imputed_metadata, imputed_postcodes)
    final_unprocessed_dataset.to_csv("Final_unprocessed_dataset.csv", index=False)


if __name__ == '__main__':
    clean_and_transform()
