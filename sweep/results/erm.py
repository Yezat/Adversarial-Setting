from erm.problems.problems import ProblemType
from util.task import Task
from model.dataset import DataSet
from model.data import DataModel
from state_evolution.overlaps import Overlaps
from erm.prediction import predict_erm
from erm.errors import (
    error,
    adversarial_error,
    adversarial_error_teacher,
    fair_adversarial_error_erm,
    compute_boundary_loss,
    logistic_training_loss,
)
from state_evolution.observables import (
    generalization_error,
    MAP_PROBLEM_TYPE_ADV_GEN_ERR_OVERLAP,
)
import numpy as np
from sweep.results.result import Result


class ERMResult(Result):
    def __init__(
        self,
        task: Task,
        data_model: DataModel,
        data: DataSet,
        weights: np.ndarray,
        values: dict,
    ) -> None:
        super().__init__(task)

        # let's compute and store the overlaps
        self.ρ: float = data_model.θ.dot(data_model.Σ_x @ data_model.θ) / task.d
        self.m = weights.dot(data_model.Σ_x @ data_model.θ) / task.d
        self.F: float = weights.dot(data_model.Σ_ν @ data_model.θ) / task.d
        self.Q = weights.dot(data_model.Σ_x @ weights) / task.d
        self.A: float = weights.dot(data_model.Σ_ν @ weights) / task.d
        self.P: float = weights.dot(data_model.Σ_δ @ weights) / task.d

        overlaps = Overlaps.from_se_problem_type(task.se_problem_type)
        overlaps.A = self.A
        overlaps.m = self.m
        overlaps.q = self.Q
        overlaps.P = self.P
        overlaps.F = self.F

        # Angle
        self.angle: float = self.m / np.sqrt(self.Q * self.ρ)

        # Generalization Error
        yhat_gd = predict_erm(data.X_test, weights)
        self.generalization_error_erm: float = error(data.y_test, yhat_gd)
        self.generalization_error_overlap: float = generalization_error(
            self.ρ, self.m, self.Q, task.tau
        )

        # Adversarial Generalization Error
        self.adversarial_generalization_errors: np.ndarray = np.array(
            [
                (
                    eps,
                    adversarial_error(
                        data.y_test, data.X_test, weights, eps, data_model.Σ_ν
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )
        self.boundary_errors: np.ndarray = np.array(
            [
                (
                    eps,
                    adv_error - self.generalization_error_erm,
                )
                for eps, adv_error in self.adversarial_generalization_errors
            ]
        )
        self.adversarial_generalization_errors_teacher: np.ndarray = np.array(
            [
                (
                    eps,
                    adversarial_error_teacher(
                        data.y_test, data.X_test, weights, data.θ, eps, data_model
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )
        self.adversarial_generalization_errors_overlap: np.ndarray = np.array(
            [
                (
                    eps,
                    MAP_PROBLEM_TYPE_ADV_GEN_ERR_OVERLAP[task.se_problem_type](
                        overlaps, task, data_model, eps
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )
        self.fair_adversarial_errors: np.ndarray = np.array(
            [
                (
                    eps,
                    fair_adversarial_error_erm(
                        data.X_test,
                        weights,
                        data.θ,
                        eps,
                        task.gamma_fair_error,
                        data_model,
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )

        # Training Error
        yhat_gd_train = predict_erm(data.X, weights)
        self.training_error: float = error(data.y, yhat_gd_train)

        # boundary loss
        self.boundary_loss_train: float = compute_boundary_loss(
            data.y, data.X, task.epsilon, data_model.Σ_δ, weights, task.lam
        )
        self.boundary_loss_test_es: np.ndarray = np.array(
            [
                (
                    eps,
                    compute_boundary_loss(
                        data.y_test,
                        data.X_test,
                        eps,
                        data_model.Σ_ν,
                        weights,
                        task.lam,
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )

        # Loss
        self.training_loss: float = logistic_training_loss(
            weights, data.X, data.y, task.epsilon, Σ_δ=data_model.Σ_δ
        )
        self.test_losses: np.ndarray = np.array(
            [
                (
                    eps,
                    logistic_training_loss(
                        weights,
                        data.X_test,
                        data.y_test,
                        eps,
                        Σ_δ=data_model.Σ_ν,
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )

        if task.erm_problem_type == ProblemType.PerturbedLogistic:
            self.lambda_twiddle_1 = values["lambda_twiddle_1"]
            self.lambda_twiddle_2 = values["lambda_twiddle_2"]
