

def remove_nan_collumns(df, limit=0.2):
    """
    Input:
        limit: float, ratio
    Output:
        Returns a new dataframe without columns which had more than limit missing data
    """

    no_of_rows = df.shape[0]

    for col_name in df.columns.values:
        # count how many missing values in collumn
        no_of_missing_values = df[col_name].isnull().sum()
        if no_of_missing_values > no_of_rows * limit:
            df = df.drop([col_name], axis=1)
            miss_proc = (no_of_missing_values / no_of_rows) * 100
            print("{:30s} attribute removed, {:.2f} % of data was missing".format(col_name, miss_proc))

    return df