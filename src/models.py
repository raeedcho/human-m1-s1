from . import munge
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class ReducedRankRegression(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, rank=None, mean_in_transform=True, multioutput_score='raw_values'):
        self.rank = rank
        self.mean_in_transform = mean_in_transform
        self.multioutput_score = multioutput_score

    def fit(self, X, y):
        """
        Fit the model to the data.
        """
        self.linreg_model = LinearRegression(fit_intercept=False)
        self.pred_pca = PCA(n_components=self.rank)

        # center X and Y
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean(axis=0)
        X_centered = X - self.X_mean_
        Y_centered = y - self.y_mean_

        self.linreg_model.fit(X_centered, Y_centered)
        self.pred_pca.fit(self.linreg_model.predict(X_centered))
        
        self.decoder_ = self.pred_pca.components_
        self.encoder_, s, vh = np.linalg.svd(self.linreg_model.coef_.T @ self.decoder_.T, full_matrices=False)
        self.transformer_ = np.diag(s) @ vh

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
        if self.mean_in_transform:
            return (X-self.X_mean_) @ self.encoder_
        else:
            return X @ self.encoder_

    def inverse_transform(self, X):
        """
        Transform data from latent space back to original space
        (such that inverse_transform -> transform gives back the same signals)
        """
        return X @ self.encoder_.T + self.X_mean_

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight, multioutput=self.multioutput_score)

class DataFrameTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self,X,y=None):
        self.transformer.fit(X,y)
        return self

    def transform(self, X):
        output = self.transformer.transform(X)
        return pd.DataFrame(
            output,
            index=X.index,
            columns=range(output.shape[1]),
        )

class NoMeanTransformPCA(PCA):
    def transform(self,X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        X_transformed = np.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class DataFramePCA(BaseEstimator,TransformerMixin):
    def __init__(self, **kwargs):
        self.pca_model=PCA(**kwargs)
    
    def fit(self,X,y=None):
        self.pca_model.fit(X,y)
        return self

    def transform(self, X):
        return pd.DataFrame(
            self.pca_model.transform(X),
            index=X.index,
            columns=range(self.pca_model.n_components),
        )

    # def __sklearn_clone__(self):
    #     pass
    
class VarimaxTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,X,y=None):
        """
        Return rotation matrix that implements varimax (or quartimax) rotation.

        Adapted from _ortho_rotation from scikit-learn _factor_analysis.py module.
        """
        nrow, ncol = X.shape
        rotation_matrix = np.eye(ncol)
        var = 0

        for _ in range(self.max_iter):
            comp_rot = np.dot(X, rotation_matrix)
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
            u, s, v = np.linalg.svd(np.dot(X.T, comp_rot**3 - tmp))
            rotation_matrix = np.dot(u, v)
            var_new = np.sum(s)
            if var != 0 and var_new < var * (1 + self.tol):
                break
            var = var_new

        # return np.dot(X, rotation_matrix).T
        self.rotation_matrix_ = rotation_matrix
        return self

    def transform(self,X):
        return X @ self.rotation_matrix_