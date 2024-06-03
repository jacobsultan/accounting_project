# Import necessary libraries and modules
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


class DataEncoder:
    def __init__(self, whole_df):
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.label_encoder = LabelEncoder()
        self.onehot_columns = None

        # Fit the encoders on the whole dataset
        self.onehot_encoder.fit(whole_df[["account", "day_of_week", "bank"]])
        self.label_encoder.fit(whole_df["direction"])
        # Retrieving the feature names after one-hot encoding

        self.onehot_columns = self.onehot_encoder.get_feature_names_out(
            ["account", "day_of_week", "bank"]
        )

    def transform(self, df):
        # Apply the trained encoders to any dataframe
        onehot_encoded_data = pd.DataFrame(
            self.onehot_encoder.transform(df[["account", "day_of_week", "bank"]])
        )
        # Transforming the categorical columns with the fitted OneHotEncoder

        onehot_encoded_data.columns = self.onehot_columns

        df["direction_encoded"] = self.label_encoder.transform(df["direction"])

        encoded_data = pd.concat(
            [onehot_encoded_data, df[["amount", "direction_encoded", "label"]]], axis=1
        )
        return encoded_data

    def split_data(self, df, test_size=0.2, random_state=1, stratify_column="label"):
        # Splitting the data into training and testing sets
        X = df.drop(stratify_column, axis=1)
        y = df[stratify_column]
        if test_size == 0:
            return X, y
        else:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=df[stratify_column],
            )
