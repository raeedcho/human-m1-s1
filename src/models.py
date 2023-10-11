import src
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class ReducedRankRegression(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, rank):
        self.rank = rank

    def fit(self, X, y):
        """
        Fit the model to the data.
        """
        # center X and Y
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean(axis=0)
        X_centered = X - self.X_mean_
        Y_centered = y - self.y_mean_

        lr_model = LinearRegression(fit_intercept=False).fit(X_centered, Y_centered)
        b_ols = lr_model.coef_.T

        _, _, decoder_matrix = np.linalg.svd(X_centered @ b_ols, full_matrices=False)

        self.decoder_ = decoder_matrix[:self.rank, :]

        self.encoder_, s, vh = np.linalg.svd(b_ols @ self.decoder_.T, full_matrices=False)
        self.transformer_ = s @ vh

        return self

    def predict(self, X):
        """
        Predict new data.
        """
        return self.y_mean_ + (X-self.X_mean_) @ self.encoder_ @ self.transformer_ @ self.decoder_

    def transform(self, X):
        """
        Transform new data.
        """
        return (X-self.X_mean_) @ self.encoder_
