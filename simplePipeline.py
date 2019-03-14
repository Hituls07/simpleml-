from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Features Selection
    """
    def __init__(self, features = None):
        self.features = features

    def fit(self, x, Y = None):
        return self

    def transform(self, x, Y = None):
        if self.features is None:
            return x[:]
        else:
            return x[self.features]


class DropColumn(BaseEstimator,  TransformerMixin):
    """
    Drops columns if its more than threshold value
    """
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.drop_ = []

    def fit(self, x, Y = None):
        size = len(x)
        for col in x.columns:
            if len(x[x[col].isnull() == True]) / size > self.threshold:
                self.drop_.append(col)
        return self

    def transform(self, x, Y = None):
        return x.drop(axis = 1, columns = self.drop_)


def num_scaler_pipeline(feat):
    """
    :return: Numerical pipeline with standardize continuous variables.
    """
    return Pipeline([
                    ('selector', FeatureSelector(features=feat)),
                    ('column', DropColumn()),
                    ('Impute', SimpleImputer(strategy= 'mean')),
                    ('scaler', StandardScaler())
                    ])


def num_sparse_pipeline(feat):
    """
    :return: Numerical pipeline with normalize continuous variables.
    """
    return Pipeline([
                    ('selector', FeatureSelector(features=feat)),
                    ('column', DropColumn()),
                    ('Impute', SimpleImputer(strategy= 'mean')),
                    ('scaler', MinMaxScaler())
                    ])


def cat_encoder_pipeline(feat):
    """
    :return: Pipeline for categorical features.
    """
    return Pipeline([
                    ('selector', FeatureSelector(features=feat)),
                    ('column', DropColumn()),
                    ('Impute', SimpleImputer(strategy= 'most_frequent')),
                    ('Ordinal', OrdinalEncoder()),
                    ('OneHotEncoder', OneHotEncoder(sparse=False, categories='auto'))
                    ])

