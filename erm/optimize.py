import numpy as np
from erm.problems.problems import ProblemType, LOSS_GRADIENT_MAP
from scipy.optimize import minimize
from model.data import DataModel
from model.dataset import DataSet
from sklearn.utils.validation import check_array, check_consistent_length
from util.task import Task


def manage_optimization(task: Task, data_model: DataModel, data: DataSet, logger):
    values = task.get_values()

    w_gd, values = optimize(
        np.random.normal(0, 1, (task.d,)),
        data.X,
        data.y,
        task.lam,
        task.epsilon,
        task.problem_type,
        covariance_prior=data_model.Σ_ω,
        Σ_δ=data_model.Σ_δ,
        logger=logger,
        values=values,
    )

    return w_gd, values


def preprocessing(coef, X, y, lam, epsilon, problem_type: ProblemType):
    # sklearn - this method expects labels as -1 and 1 and converts them to 0 and 1
    # heavily inspired by the sklearn code, with hopefully all the relevant bits copied over to make it work using lbfgs
    solver = "lbfgs"
    X = check_array(
        X,
        accept_sparse="csr",
        dtype=np.float64,
        accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
    )
    y = check_array(y, ensure_2d=False, dtype=None)
    check_consistent_length(X, y)

    _, n_features = X.shape

    w0 = np.zeros(n_features, dtype=X.dtype)

    if coef.size not in (n_features, w0.size):
        raise ValueError(
            "Initialization coef is of shape %d, expected shape %d or %d"
            % (coef.size, n_features, w0.size)
        )
    w0[: coef.size] = coef

    if problem_type == ProblemType.Logistic:
        mask = y == 1
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = 0.0
        target = y_bin
    elif (
        problem_type == ProblemType.PerturbedLogistic or ProblemType.CoefficientLogistic
    ):
        target = y
    else:
        raise Exception(
            f"Preprocessing not implemented for problem type {problem_type}"
        )

    return w0, X, target, lam, epsilon


def optimize(
    coef,
    X,
    y,
    lam,
    epsilon,
    problem_type: ProblemType,
    covariance_prior=None,
    Σ_δ=None,
    values=None,
):
    w0, X, target, lam, epsilon = preprocessing(coef, X, y, lam, epsilon, problem_type)

    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])
    if Σ_δ is None:
        Σ_δ = np.eye(X.shape[1])

    method = "L-BFGS-B"

    loss_gd = LOSS_GRADIENT_MAP[problem_type]

    opt_res = minimize(
        loss_gd,
        w0,
        method=method,
        jac=True,
        args=(X, target, lam, epsilon, covariance_prior, Σ_δ, values),
        options={"maxiter": 1000, "disp": False},
    )

    w0, _ = opt_res.x, opt_res.fun
    return w0, values
