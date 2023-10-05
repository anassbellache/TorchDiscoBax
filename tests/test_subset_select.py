import torch
import gpytorch
import pytest

# Assuming your class and dependencies are in module "module_name"
from discobax import SubsetSelect, NeuralGPModel, BaseGPModel, SimpleGPClassifier, SimpleGPRegressor
from mock import Mock, MagicMock


@pytest.fixture
def simple_training_data():
    train_x = torch.randn(10, 5)
    train_y = torch.randn(10, 1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    return train_x, train_y, likelihood


@pytest.fixture
def neural_gp_model(simple_training_data):
    # Assuming NeuralGPModel needs some basic parameters for initialization
    train_x, train_y, likelihood = simple_training_data
    model = NeuralGPModel(train_x, train_y, likelihood)
    model.eval()
    return model


@pytest.fixture
def base_gp_model(simple_training_data):
    # Assuming NeuralGPModel needs some basic parameters for initialization
    train_x, train_y, likelihood = simple_training_data
    model = BaseGPModel(train_x, train_y)
    model.model.eval()
    return model


@pytest.fixture
def subset_select_instance_add():
    # Mocking the necessary components for initialization
    X = Mock()
    X.get_data.return_value = [0, 1, 2, 3]
    noise_type = "additive"
    return SubsetSelect(X, noise_type)


@pytest.fixture
def subset_select_instance_mul():
    # Mocking the necessary components for initialization
    X = Mock()
    X.get_data.return_value = [0, 1, 2, 3]
    noise_type = "multiplicative"
    return SubsetSelect(X, noise_type)


def test_init_add(subset_select_instance_add):
    assert subset_select_instance_add.noise_type == "additive"
    assert subset_select_instance_add.k == 1
    assert subset_select_instance_add.mc_samples == 1000
    assert isinstance(subset_select_instance_add.noise_gp, SimpleGPRegressor)


def test_init_mul(subset_select_instance_mul):
    assert subset_select_instance_mul.noise_type == "multiplicative"
    assert subset_select_instance_mul.k == 1
    assert subset_select_instance_mul.mc_samples == 1000
    assert isinstance(subset_select_instance_mul.noise_gp, SimpleGPClassifier)


def test_initialize(subset_select_instance_mul):
    subset_select_instance_mul.initialize()

    assert len(subset_select_instance_mul.exe_path.x) == 0
    assert len(subset_select_instance_mul.exe_path.y) == 0


def test_monte_carlo_expectation(subset_select_instance_add, neural_gp_model):
    # Define S as a torch.Tensor with some sample points.
    S = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]])

    expected_max = subset_select_instance_add.monte_carlo_expectation(S, neural_gp_model)

    # Check if the result is a torch.Tensor
    assert isinstance(expected_max, torch.Tensor)

    # Check if the size of the output tensor matches the number of points in S.
    assert expected_max.size(0) == S.size(0)


def test_select_next(subset_select_instance_add, neural_gp_model):
    # Mock the monte_carlo_expectation method to return known values
    subset_select_instance_add.monte_carlo_expectation = MagicMock(return_value=torch.tensor([1.0, 2.0, 3.0]))

    result = subset_select_instance_add.select_next(neural_gp_model)
    # Validate that the point with the highest score (last in the mocked tensor) was selected
    assert torch.equal(result, subset_select_instance_add.X[-1])  # Assuming X has been set in the fixture


def test_take_step(subset_select_instance, neural_gp_model):
    # Mock the select_next method
    subset_select_instance.select_next = MagicMock(return_value=torch.tensor([2.0]))

    result = subset_select_instance.take_step(neural_gp_model)

    # Validate
    assert result == torch.tensor([2.0])
    assert torch.tensor([2.0]) in subset_select_instance.selected_subset


def test_take_step_selection_complete(subset_select_instance, neural_gp_model):
    # Set the state so that selection is complete
    subset_select_instance.k = 3
    subset_select_instance.selected_subset = [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0])]

    result = subset_select_instance.take_step(neural_gp_model)

    # Validate
    assert result is None
