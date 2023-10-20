import pytest

from sklearn.utils.estimator_checks import check_estimator

from src.models import ReducedRankRegression,DataFramePCA


@pytest.mark.parametrize(
    "estimator",
    [DataFramePCA(), ReducedRankRegression()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)