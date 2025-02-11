import numpy as np
from DataModel.dataset import DataSet, generate_data
from DataModel.data_model import DataModel, KFeaturesDefinition, k_features_factory


def test_dataset():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, -1])
    X_test = np.array([[5, 6], [7, 8]])
    y_test = np.array([1, -1])
    θ = np.array([0.5, 0.5])

    dataset = DataSet(X, y, X_test, y_test, θ)

    assert np.array_equal(dataset.X, X), f"Expected {X}, but got {dataset.X}"
    assert np.array_equal(dataset.y, y), f"Expected {y}, but got {dataset.y}"
    assert np.array_equal(
        dataset.X_test, X_test
    ), f"Expected {X_test}, but got {dataset.X_test}"
    assert np.array_equal(
        dataset.y_test, y_test
    ), f"Expected {y_test}, but got {dataset.y_test}"
    assert np.array_equal(dataset.θ, θ), f"Expected {θ}, but got {dataset.θ}"


def test_generate_data():
    d = 10
    x_diagonal = KFeaturesDefinition(diagonal=[(10, 2), (5, 3), (1, 5)])
    θ_diagonal = KFeaturesDefinition(diagonal=[(1, 10)])
    ω_diagonal = KFeaturesDefinition(diagonal=[(2, 5), (3, 5)])
    δ_diagonal = KFeaturesDefinition(diagonal=[(4, 4), (6, 6)])
    ν_diagonal = KFeaturesDefinition(diagonal=[(7, 7), (8, 3)])

    k_features_kwargs = {
        "x_diagonal": x_diagonal,
        "θ_diagonal": θ_diagonal,
        "ω_diagonal": ω_diagonal,
        "δ_diagonal": δ_diagonal,
        "ν_diagonal": ν_diagonal,
    }

    data_model = DataModel(
        d,
        normalize_matrices=False,
        data_model_factory=k_features_factory,
        factory_kwargs=k_features_kwargs,
    )

    data_model.generate_model_matrices()

    n = 100
    tau = 0.1
    dataset = generate_data(data_model, n, tau)

    assert dataset.X.shape == (
        n,
        d,
    ), f"Expected shape {(n, d)}, but got {dataset.X.shape}"
    assert dataset.y.shape == (n,), f"Expected shape {(n,)}, but got {dataset.y.shape}"
    assert dataset.X_test.shape == (
        10000,
        d,
    ), f"Expected shape {(10000, d)}, but got {dataset.X_test.shape}"
    assert dataset.y_test.shape == (
        10000,
    ), f"Expected shape {(10000,)}, but got {dataset.y_test.shape}"
    assert dataset.θ.shape == (d,), f"Expected shape {(d,)}, but got {dataset.θ.shape}"

    # TODO add tests for the covariance matrices to match Σ_x.
    # TODO add tests for the label y
