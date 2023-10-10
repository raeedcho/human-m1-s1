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
        X_centered = X - X.mean(axis=0)
        Y_centered = y - y.mean(axis=0)

        lr_model = LinearRegression(fit_intercept=False).fit(X_centered, Y_centered)
        b_ols = lr_model.coef_.T

        _, _, Vh = np.linalg.svd(X @ b_ols, full_matrices=False)

        projector_matrix = Vh[:self.rank, :].T

        self.encoder_ = b_ols @ projector_matrix
        self.decoder_ = projector_matrix.T

        self.coef_ = self.encoder_ @ self.decoder_

        return self

    def predict(self, X):
        """
        Predict new data.
        """
        return X @ self.coef_

    def transform(self, X):
        """
        Transform new data.
        """
        return X @ self.encoder_
