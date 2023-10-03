import torch
import gpytorch
import pytest
from botorch.models import SingleTaskGP
from discobax import LargeFeatureExtractor, NeuralGPModel


# LargeFeatureExtractor tests

def test_large_feature_extractor_forward_pass():
    data_dim = 5
    feature_extractor = LargeFeatureExtractor(data_dim)
    input_tensor = torch.randn((10, data_dim))
    output_tensor = feature_extractor(input_tensor)
    assert output_tensor.shape == (10, 2)


# NeuralGPModel tests

@pytest.fixture
def simple_training_data():
    train_x = torch.randn(10, 5)
    train_y = torch.randn(10, 1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    return train_x, train_y, likelihood


def test_neural_gp_model_initialization(simple_training_data):
    train_x, train_y, likelihood = simple_training_data
    model = NeuralGPModel(train_x, train_y, likelihood)
    assert isinstance(model, NeuralGPModel)


def test_neural_gp_model_posterior(simple_training_data):
    train_x, train_y, likelihood = simple_training_data
    model = NeuralGPModel(train_x, train_y, likelihood)
    test_x = torch.randn(3, 5)
    posterior = model.posterior(test_x)
    assert posterior.mean.shape == (3, 1)
    assert posterior.variance.shape == (3, 1)


def test_neural_gp_model_condition_on_observations(simple_training_data):
    train_x, train_y, likelihood = simple_training_data
    model = NeuralGPModel(train_x, train_y, likelihood)
    new_x = torch.randn(1, 5)
    new_y = torch.randn(1, 1)
    new_model = model.condition_on_observations(new_x, new_y)
    assert isinstance(new_model, NeuralGPModel)
    assert new_model.train_targets.shape[0] == train_y.shape[0] + 1
    assert new_model.train_inputs[0].shape[0] == train_x.shape[0] + 1

#
