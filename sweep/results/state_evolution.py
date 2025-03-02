from util.task import Task
from model.data import DataModel
from state_evolution.overlaps import Overlaps
from state_evolution.observables import (
    generalization_error,
    adversarial_generalization_error_overlaps,
    asymptotic_adversarial_generalization_error,
    compute_data_model_angle,
    compute_data_model_attackability,
    training_loss,
    test_loss,
    training_error,
)
from sweep.results.result import Result
from state_evolution.constants import INT_LIMS
import numpy as np


class SEResult(Result):
    # define a constructor with all attributes
    def __init__(self, task: Task, overlaps: Overlaps, data_model: DataModel) -> None:
        super().__init__(task)

        # Generalization Error
        self.generalization_error: float = generalization_error(
            data_model.ρ, overlaps.m, overlaps.q, task.tau
        )

        # Adversarial Generalization Error
        self.adversarial_generalization_errors: np.ndarray = np.array(
            [
                (
                    eps,
                    adversarial_generalization_error_overlaps(
                        overlaps, task, data_model, eps
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )

        # Training Error
        self.training_error: float = training_error(
            task, overlaps, data_model, INT_LIMS
        )

        # Loss
        self.training_loss: float = training_loss(task, overlaps, data_model, INT_LIMS)
        self.test_losses: np.ndarray = np.array(
            [
                (
                    eps,
                    test_loss(task, overlaps, data_model, eps, INT_LIMS),
                )
                for eps in task.test_against_epsilons
            ]
        )

        # Overlaps
        self.__dict__.update(overlaps._overlaps)
        self.__dict__.update(overlaps._hat_overlaps)

        # Angle
        self.angle: float = self.m / np.sqrt((self.q) * data_model.ρ)

        # data_model angle and attackability
        self.data_model_angle: float = compute_data_model_angle(data_model, overlaps, 0)
        self.data_model_attackability: float = compute_data_model_attackability(
            data_model, overlaps
        )
        self.data_model_adversarial_test_errors: np.ndarray = np.array(
            [
                (
                    eps,
                    asymptotic_adversarial_generalization_error(
                        data_model, overlaps, eps, task.tau
                    ),
                )
                for eps in task.test_against_epsilons
            ]
        )

        # Let's compute the two eigenvalues of Σ_x, Σ_θ and Σ_x * Σ_θ
        self.sigmax_eigenvalues = np.array(
            [
                np.linalg.eigvals(data_model.Σ_x)[0],
                np.linalg.eigvals(data_model.Σ_x)[-1],
            ]
        )
        self.sigmaθ_eigenvalues = np.array(
            [
                np.linalg.eigvals(data_model.Σ_θ)[0],
                np.linalg.eigvals(data_model.Σ_θ)[-1],
            ]
        )
        self.xθ_eigenvalues = np.array(
            [
                np.linalg.eigvals(data_model.Σ_x @ data_model.Σ_θ)[0],
                np.linalg.eigvals(data_model.Σ_x @ data_model.Σ_θ)[-1],
            ]
        )

        self.mu_usefulness = (
            np.sqrt(2 / np.pi) * data_model.ρ / np.sqrt(data_model.ρ + task.tau**2)
        )
        self.gamma_robustness_es: np.ndarray = np.array(
            [
                (
                    eps,
                    self.mu_usefulness
                    - (eps / np.sqrt(task.d))
                    * np.trace(data_model.Σ_θ @ data_model.Σ_ν)
                    / np.trace(data_model.Σ_θ),
                )
                for eps in task.test_against_epsilons
            ]
        )
        self.mu_margin = (
            np.sqrt(2 / np.pi) * overlaps.m / np.sqrt(data_model.ρ + task.tau**2)
        )
