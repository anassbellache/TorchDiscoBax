# TODO:
#  model must be a GP
#  It must have sampling
#  It has to be trainable
#  There are two models, one for experiment, one for noise
#  use gpytorch
#  try to add a bayseian neural net
#  Ensure compatilibility with slingpy and AbstractBaseModel, AbstractDataSource
#  Update posterior with new data
#  Scales up to large datasets
#  Handles batches

from typing import AnyStr, Optional, List, Type, Any

import botorch
import gpytorch
import numpy as np
import torch
import torch.optim
from botorch.models.model import TFantasizeMixin
from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors import Posterior
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.mlls import ExactMarginalLogLikelihood
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor, optim
from torch.nn import Module


class BaseGPModel(AbstractBaseModel):
    def __init__(self, dim_input):
        super(BaseGPModel).__init__()
        self.num_samples = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = NeuralGPModel(dim_input, self.likelihood).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.return_samples = True
        self.data_dim = dim_input

    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256, row_names: List[AnyStr] = None) -> List[
        np.ndarray]:
        # Convert dataset_x to a PyTorch tensor
        x_tensor = torch.tensor(dataset_x.get_data(), dtype=torch.float32)

        self.model.eval()
        self.likelihood.eval()

        # Split the data into batches
        num_samples = x_tensor.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size

        all_pred_means = []
        all_pred_stds = []
        all_samples = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(num_batches):
                start_i = i * batch_size
                end_i = min((i + 1) * batch_size, num_samples)
                batch_x = x_tensor[start_i:end_i]

                observed_pred = self.likelihood(self.model(batch_x))
                all_pred_means.append(observed_pred.mean.numpy())
                all_pred_stds.append(observed_pred.stddev.numpy())

                # Sample from the predictive distribution if required
                if self.return_samples:
                    batch_samples = observed_pred.sample(sample_shape=torch.Size([self.num_samples]))
                    all_samples.append(batch_samples.numpy())

        # Concatenate results from all batches
        pred_mean = np.concatenate(all_pred_means, axis=0)
        pred_std = np.concatenate(all_pred_stds, axis=0)

        # Compute the 95% confidence bounds
        upper_bound = pred_mean + 1.96 * pred_std
        lower_bound = pred_mean - 1.96 * pred_std

        # Compute the margins
        y_margins = upper_bound - lower_bound

        if self.num_samples is None:
            raise ValueError("The model hasn't been fit yet. Please fit the model before predicting.")

        if self.return_samples:
            samples = np.concatenate(all_samples, axis=0)
            return [pred_mean, pred_std, y_margins, samples]
        else:
            return [pred_mean, pred_std, y_margins]

    def get_posterior(self, x_tensor):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x_tensor))
        return observed_pred

    def fit(self, train_x: AbstractDataSource, train_y: Optional[AbstractDataSource] = None,
            validation_set_x: Optional[AbstractDataSource] = None,
            validation_set_y: Optional[AbstractDataSource] = None) -> "AbstractBaseModel":

        if train_y is None:
            raise ValueError("train_y cannot be None")

        if train_x is None:
            raise ValueError("train_x cannot be None")

        # Convert AbstractDataSource to torch.Tensor
        train_x = torch.tensor(train_x.get_data(), dtype=torch.float32)
        train_y = torch.tensor(train_y.get_data(), dtype=torch.float32)
        self.num_samples = train_y.size(0)

        noise = 1e-4
        self.likelihood.noise = noise
        self.model.train()
        self.likelihood.train()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            train_x, train_y = train_x.cuda(), train_y.cuda()

        # 2. Define an optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.1)  # 0.1 is the learning rate

        # 3. Define the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # 4. Define the training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()
        return self

    @classmethod
    def load(cls: Type["BaseGPModel"], file_path: AnyStr) -> "BaseGPModel":
        """
        Load the model from the specified file path.

        Parameters:
            - file_path (str): The path from where the model should be loaded.

        Returns:
            - model (BaseGPModel): The loaded model.
        """
        # Load the saved state dictionary
        state_dict = torch.load(file_path)

        # Extract data_dim from the saved state
        data_dim = state_dict["data_dim"]

        # Create a new model instance with the extracted data_dim
        model = cls(data_dim)

        # Restore the state of the model and the likelihood
        model.model.load_state_dict(state_dict["model"])
        model.likelihood.load_state_dict(state_dict["likelihood"])

        return model

    @staticmethod
    def get_save_file_extension() -> AnyStr:
        return '.pt'

    def save(self, file_path: AnyStr):
        """
        Save the model to the specified file path.

        Parameters:
            - file_path (str): The path to where the model should be saved.
        """
        state_dict = {
            "data_dim": self.data_dim,
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
        }
        torch.save(state_dict, file_path)


class SimpleGPRegressor(gpytorch.models.ExactGP, botorch.models.model.FantasizeMixin):
    def __init__(self, train_x, train_y, likelihood):
        super(SimpleGPRegressor, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # Process the input through the neural network.
        # Obtain the prior distribution.
        mvn = self(X)

        if observation_noise:
            if isinstance(self.likelihood, _GaussianLikelihoodBase):
                # Adjust the variance using the likelihood's noise.
                noise = self.likelihood.noise
                mvn = MultitaskMultivariateNormal(
                    mvn.mean, mvn.lazy_covariance_matrix.add_diag(noise)
                )

        # Return the botorch wrapper around GPyTorch's posterior.
        return GPyTorchPosterior(mvn)

    def condition_on_observations(self: TFantasizeMixin, X: Tensor, Y: Tensor, **kwargs: Any) -> TFantasizeMixin:
        """
                Condition the NeuralGP on new observations (X, Y) and return a new NeuralGPModel.
                """
        # Ensure that the new data is processed using the feature extractor

        # Make sure self.train_inputs[0] is the projected version
        train_inputs_projected = self.train_inputs[0]

        if train_inputs_projected.dim == 1:
            updated_train_x = torch.cat([train_inputs_projected, X.squeeze(0)], dim=0)
        else:
            updated_train_x = torch.cat([train_inputs_projected, X], dim=0)

        updated_train_y = torch.cat([self.train_targets, Y], dim=0)

        # Create a new model with the updated data.
        new_model = self.__class__(updated_train_x, updated_train_y, self.likelihood)
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module

        return new_model

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        pass

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x):
        return self.forward(x)


class SimpleGPClassifier(gpytorch.models.ApproximateGP, botorch.models.model.FantasizeMixin):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)

        super(SimpleGPClassifier, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def condition_on_observations(self: TFantasizeMixin, X: Tensor, Y: Tensor, **kwargs: Any) -> TFantasizeMixin:
        """
                Condition the NeuralGP on new observations (X, Y) and return a new NeuralGPModel.
                """
        # Ensure that the new data is processed using the feature extractor
        X_projected = self.feature_extractor(X)

        # Make sure self.train_inputs[0] is the projected version
        train_inputs_projected = self.feature_extractor(self.train_inputs[0])

        if train_inputs_projected.dim() == 1:
            updated_train_x = torch.cat([train_inputs_projected, X_projected.squeeze(0)], dim=0)
        else:
            updated_train_x = torch.cat([train_inputs_projected, X_projected], dim=0)

        updated_train_y = torch.cat([self.train_targets, Y], dim=0)

        # Create a new model with the updated data.
        new_dim = updated_train_x.size(-1)
        new_model = self.__class__(new_dim, self.likelihood)
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module

        return new_model

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # Process the input through the neural network.
        # Obtain the prior distribution.
        mvn = self(X)

        if observation_noise:
            if isinstance(self.likelihood, _GaussianLikelihoodBase):
                # Adjust the variance using the likelihood's noise.
                noise = self.likelihood.noise
                mvn = MultitaskMultivariateNormal(
                    mvn.mean, mvn.lazy_covariance_matrix.add_diag(noise)
                )

        # Return the botorch wrapper around GPyTorch's posterior.
        return GPyTorchPosterior(mvn)

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        pass

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def __call__(self, x, **kwargs):
        return self.forward(x)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))


class NeuralGPModel(gpytorch.models.ExactGP, botorch.models.model.FantasizeMixin):
    def __init__(self, data_dim, likelihood):
        super().__init__(None, None, likelihood)

        self.data_dim = data_dim
        self.feature_extractor = LargeFeatureExtractor(data_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def update_train_data(self, new_x: Tensor, new_y: Tensor) -> None:
        """
        Update the model's training data with new observations.

        Args:
            new_x: New training inputs.
            new_y: New training targets.
        """
        new_x_projected = self.feature_extractor(new_x)

        if self.train_inputs[0] is None:
            updated_train_x = new_x_projected
            updated_train_y = new_y
        else:
            # Make sure self.train_inputs[0] is the projected version
            train_inputs_projected = self.feature_extractor(self.train_inputs[0])

            # Concatenate old and new data
            if train_inputs_projected.dim() == 1:
                updated_train_x = torch.cat([train_inputs_projected, new_x_projected.squeeze(0)], dim=0)
            else:
                updated_train_x = torch.cat([train_inputs_projected, new_x_projected], dim=0)

            updated_train_y = torch.cat([self.train_targets, new_y], dim=0)

        self.set_train_data(updated_train_x, updated_train_y, strict=False)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> 'NeuralGPModel':
        """
        Condition the NeuralGP on new observations (X, Y) and return a new NeuralGPModel.
        """
        # Ensure that the new data is processed using the feature extractor
        X_projected = self.feature_extractor(X)

        # Make sure self.train_inputs[0] is the projected version
        train_inputs_projected = self.feature_extractor(self.train_inputs[0])

        if train_inputs_projected.dim() == 1:
            updated_train_x = torch.cat([train_inputs_projected, X_projected.squeeze(0)], dim=0)
        else:
            updated_train_x = torch.cat([train_inputs_projected, X_projected], dim=0)

        updated_train_y = torch.cat([self.train_targets, Y], dim=0)
        data_dim = updated_train_x.shape(-1)

        new_model = self.__class__(data_dim, self.likelihood)
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module
        new_model.feature_extractor = self.feature_extractor
        new_model.set_train_data(updated_train_x, updated_train_y)

        return new_model

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # Process the input through the neural network.
        # Obtain the prior distribution.
        mvn = self(X)

        # If observation noise should be added and the likelihood is GaussianLikelihood
        if observation_noise and isinstance(self.likelihood, GaussianLikelihood):
            noise = self.likelihood.noise
            mvn = MultivariateNormal(mvn.mean, mvn.lazy_covariance_matrix.add_diag(noise))

        # Return the botorch wrapper around GPyTorch's posterior.
        return GPyTorchPosterior(mvn)

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        pass

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x):
        return self.forward(x)
