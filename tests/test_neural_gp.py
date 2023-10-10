import gpytorch
import pytest
import torch
from discobax import LargeFeatureExtractor, NeuralGPModel


def test_large_feature_extractor_forward_pass():
    data_dim = 5
    feature_extractor = LargeFeatureExtractor(data_dim)
    input_tensor = torch.randn((10, data_dim))
    output_tensor = feature_extractor(input_tensor)
    assert output_tensor.shape == (10, 2)  # This assumes your LargeFeatureExtractor produces outputs of shape (N, 2)


@pytest.fixture
def simple_training_data():
    train_x = torch.randn(10, 5)
    train_y = torch.randn(10, 1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    return train_x, train_y, likelihood


def test_neural_gp_model_initialization(simple_training_data):
    train_x, _, likelihood = simple_training_data
    model = NeuralGPModel(data_dim=train_x.size(-1), likelihood=likelihood)
    assert isinstance(model, NeuralGPModel)


def test_neural_gp_model_posterior(simple_training_data):
    train_x, train_y, likelihood = simple_training_data
    model = NeuralGPModel(data_dim=train_x.size(-1), likelihood=likelihood)
    model.eval()
    test_x = torch.randn(3, 5)
    posterior = model.posterior(test_x)

    samples = posterior.sample(torch.Size([1000]))  # Draw 1000 samples
    mean = samples.mean(dim=0).squeeze(-1)
    variance = samples.var(dim=0).squeeze(-1)

    assert mean.shape == (3,)
    assert variance.shape == (3,)

