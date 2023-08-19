import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders.binary import BinaryEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


def create_new_feat(df: pd.DataFrame, inplace=False):
    df_new = df if inplace else df.copy()
    df_new['screen_area'] = df_new.device_screen_resolution.apply(lambda screen: np.prod([int(size) for size in screen.split('x')]))
    df_new = df_new.drop(['device_screen_resolution'], axis=1)
    return df_new


class DataTransformer(TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = BinaryEncoder()
        self.numeric_columns = ['screen_area']
        self.category_columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                                 'device_category', 'device_brand', 'device_browser',
                                 'geo_country', 'geo_city']

    def fit(self, x):
        self.scaler.fit(x[self.numeric_columns])
        self.encoder.fit(x[self.category_columns])
        return self

    def transform(self, data):
        data_out = data.copy()
        data_out[self.numeric_columns] = self.scaler.transform(data_out[self.numeric_columns])
        data_out[self.encoder.get_feature_names_out()] = self.encoder.transform(data_out[self.category_columns])
        data_out.drop(self.category_columns, axis=1, inplace=True)
        return data_out


class LinearRegressionClassifier(LinearRegression):
    def __init__(self, threshold=0.08, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def predict(self, x):
        pred = super().predict(x)
        return np.array([1 if val > self.threshold else 0 for val in pred])


def main():
    df = pd.read_csv(r"data/sessions.csv", index_col=0)
    x = df.drop(columns=['is_target_action'])
    y = df.is_target_action


if __name__ == '__main__':
    main()
