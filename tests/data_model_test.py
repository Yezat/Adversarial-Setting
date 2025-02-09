from DataModel.data_model import KFeaturesModelDefinition, k_features_factory, DataModel
import numpy as np


def test_kfeatures_model_definition():
    # Define the feature values and sizes
    diagonal = [(10, 2), (5, 3), (1, 5)]
    kfeatures_model_def = KFeaturesModelDefinition(diagonal=diagonal)

    # Generate the ndarray
    d = 10
    result = kfeatures_model_def.get_nd_array(d)

    # Expected result
    expected = np.array([10, 10, 5, 5, 5, 1, 1, 1, 1, 1])

    # Assert the result matches the expected output
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    expected = np.diag(expected)
    result = kfeatures_model_def.get_nd_matrix(d)

    # Assert the result matches the expected output
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_k_features_factory():
    d = 10
    x_diagonal = KFeaturesModelDefinition(diagonal=[(10, 2), (5, 3), (1, 5)])
    θ_diagonal = KFeaturesModelDefinition(diagonal=[(1, 10)])
    ω_diagonal = KFeaturesModelDefinition(diagonal=[(2, 5), (3, 5)])
    δ_diagonal = KFeaturesModelDefinition(diagonal=[(4, 4), (6, 6)])
    ν_diagonal = KFeaturesModelDefinition(diagonal=[(7, 7), (8, 3)])

    Σ_x, θ, Σ_ω, Σ_δ, Σ_ν = k_features_factory(
        d, x_diagonal, θ_diagonal, ω_diagonal, δ_diagonal, ν_diagonal
    )

    expected_Σ_x = np.diag([10, 10, 5, 5, 5, 1, 1, 1, 1, 1])
    expected_θ = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expected_Σ_ω = np.diag([2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    expected_Σ_δ = np.diag([4, 4, 4, 4, 6, 6, 6, 6, 6, 6])
    expected_Σ_ν = np.diag([7, 7, 7, 7, 7, 7, 7, 8, 8, 8])

    assert np.array_equal(Σ_x, expected_Σ_x), f"Expected {expected_Σ_x}, but got {Σ_x}"
    assert np.array_equal(θ, expected_θ), f"Expected {expected_θ}, but got {θ}"
    assert np.array_equal(Σ_ω, expected_Σ_ω), f"Expected {expected_Σ_ω}, but got {Σ_ω}"
    assert np.array_equal(Σ_δ, expected_Σ_δ), f"Expected {expected_Σ_δ}, but got {Σ_δ}"
    assert np.array_equal(Σ_ν, expected_Σ_ν), f"Expected {expected_Σ_ν}, but got {Σ_ν}"


def test_k_features_data_model():
    d = 10
    x_diagonal = KFeaturesModelDefinition(diagonal=[(10, 2), (5, 3), (1, 5)])
    θ_diagonal = KFeaturesModelDefinition(diagonal=[(1, 10)])
    ω_diagonal = KFeaturesModelDefinition(diagonal=[(2, 5), (3, 5)])
    δ_diagonal = KFeaturesModelDefinition(diagonal=[(4, 4), (6, 6)])
    ν_diagonal = KFeaturesModelDefinition(diagonal=[(7, 7), (8, 3)])

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

    data_model = eval(repr(data_model))

    expected_Σ_x = np.diag([10, 10, 5, 5, 5, 1, 1, 1, 1, 1])
    expected_θ = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expected_Σ_ω = np.diag([2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    expected_Σ_δ = np.diag([4, 4, 4, 4, 6, 6, 6, 6, 6, 6])
    expected_Σ_ν = np.diag([7, 7, 7, 7, 7, 7, 7, 8, 8, 8])

    assert np.array_equal(
        data_model.Σ_x, expected_Σ_x
    ), f"Expected {expected_Σ_x}, but got {data_model.Σ_x}"
    assert np.array_equal(
        data_model.θ, expected_θ
    ), f"Expected {expected_θ}, but got {data_model.θ}"
    assert np.array_equal(
        data_model.Σ_ω, expected_Σ_ω
    ), f"Expected {expected_Σ_ω}, but got {data_model.Σ_ω}"
    assert np.array_equal(
        data_model.Σ_δ, expected_Σ_δ
    ), f"Expected {expected_Σ_δ}, but got {data_model.Σ_δ}"
    assert np.array_equal(
        data_model.Σ_ν, expected_Σ_ν
    ), f"Expected {expected_Σ_ν}, but got {data_model.Σ_ν}"

    # TODO, add tests for the rest of the matrices
