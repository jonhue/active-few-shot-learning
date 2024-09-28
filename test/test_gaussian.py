from pytest import approx
import torch
from activeft.gaussian import GaussianCovarianceMatrix

matrix = torch.tensor([[1, 0.5], [0.5, 3]])
noise_std = 1

gaussian = GaussianCovarianceMatrix(matrix)


def test_getitem():
    assert gaussian[0, 0] == 1
    assert gaussian[1, 1] == 3


def test_dim():
    assert gaussian.dim == 2


def test_condition_on():
    conditioned_gaussian = gaussian.condition_on(1, noise_std=noise_std)
    assert conditioned_gaussian.dim == 2
    assert conditioned_gaussian[0, 0] == 0.9375
    assert (
        conditioned_gaussian[0, 0]
        == gaussian.condition_on([1], noise_std=noise_std)[0, 0]
    )

    conditioned_gaussian = gaussian.condition_on([1, 1], noise_std=noise_std)
    assert conditioned_gaussian[0, 0] == approx(0.9286, abs=1e-4)
