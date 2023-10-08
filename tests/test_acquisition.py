import pytest
import torch
from unittest.mock import MagicMock
from discobax.acquisition import DiscoBAXAdditive


# Mock the required components
class MockDataSource:
    def get_data(self):
        return torch.tensor([[1.], [2.], [3.]])


class MockSubsetSelect:
    def __init__(self, X, noise_type, n_samples, k):
        self.exe_path = MagicMock(x=[], y=[])

    def get_exe_paths(self, model):
        return self.exe_path


class MockModel:
    class InnerModel:
        def __call__(self, x):
            return [torch.tensor([0.5])]

        def eval(self):
            pass

        def posterior(self, x):
            class MockPosterior:
                mean = torch.tensor([0.5])
                variance = torch.tensor([0.2])

            return MockPosterior()

    def __init__(self):
        self.model = self.InnerModel()

    def get_posterior(self, x):
        return self.model.posterior(x)

    def predict(self, x):
        return [torch.tensor([0.5])]


@pytest.fixture
def disco_bax_additive():
    budget = 10
    mc_samples = 5
    noise_type = 'additive'
    k = 1
    return DiscoBAXAdditive(budget, mc_samples, noise_type, k)


def test_expected_information_gain(disco_bax_additive):
    dataset_x = MockDataSource()
    subset_selector = MockSubsetSelect(X=dataset_x, noise_type='additive', n_samples=5, k=1)
    last_model = MockModel()

    eig = disco_bax_additive.expected_information_gain(dataset_x, subset_selector, last_model)

    assert isinstance(eig, torch.Tensor)
    assert eig.dim() == 0  # EIG should be a scalar value


def test_call(disco_bax_additive):
    dataset_x = MockDataSource()
    available_indices = ["a", "b", "c"]
    last_selected_indices = []
    last_model = MockModel()

    selected_indices = disco_bax_additive(dataset_x, 2, available_indices, last_selected_indices, last_model)

    assert isinstance(selected_indices, list)
    assert len(selected_indices) == len(available_indices)
