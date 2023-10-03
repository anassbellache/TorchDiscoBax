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

import gpytorch
import botorch
import numpy as np
import torch
import torch.optim
from botorch.models.model import TFantasizeMixin
from botorch.posteriors import Posterior
from gpytorch.distributions import MultivariateNormal
from slingpy import AbstractBaseModel, AbstractDataSource
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from botorch.posteriors import GPyTorchPosterior


class BaseGPModel(AbstractBaseModel):
    def __init__(self):
        super().__init__()
        self.num_samples = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = NeuralGPModel(None, None, self.likelihood)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.return_samples = True

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

        # Process each batch
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(num_batches):
                start_i = i * batch_size
                end_i = min((i + 1) * batch_size, num_samples)
                batch_x = x_tensor[start_i:end_i]

                observed_pred = self.likelihood(self.model(batch_x))
                all_pred_means.append(observed_pred.mean.numpy())
                all_pred_stds.append(observed_pred.stddev.numpy())

        # Concatenate results from all batches
        pred_mean = np.concatenate(all_pred_means, axis=0)
        pred_std = np.concatenate(all_pred_stds, axis=0)

        # Compute the 95% confidence bounds
        upper_bound = pred_mean + 1.96 * pred_std
        lower_bound = pred_mean - 1.96 * pred_std

        # Compute the margins
        y_margins = upper_bound - lower_bound

        if self.return_samples:
            samples = observed_pred.sample(sample_shape=torch.Size([num_samples]))
            return [pred_mean, pred_std, y_margins, samples.numpy()]
        else:
            return [pred_mean, pred_std, y_margins]

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
        num_train_iters = 500
        self.likelihood.noise = noise

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.model.train()
        self.likelihood.train()

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=100)

        losses = []
        for i in tqdm(range(num_train_iters)):
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()

                output = self.model(x_batch)
                loss = -mll(output, y_batch)

                loss.backward()

                losses.append(loss.item())
                self.optimizer.step()

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

        # Create a new model instance
        model = cls()

        # Restore the state of the model and the likelihood
        model.model.load_state_dict(state_dict["model"])  # <-- This line was modified
        model.likelihood.load_state_dict(state_dict["likelihood"])

        return model

    @staticmethod
    def get_save_file_extension() -> AnyStr:
        return '.pt'

    def save(self, file_path: AnyStr):
        """
        Save the model to the specified file path.

        Parameters:
        - file_path (str): The path where the model should be saved.
        """
        # Create a dictionary to store the model and likelihood state dicts
        state_dict = {
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "num_data": self.num_samples
        }

        # Only save the state dictionary of the optimizers if they are not None
        if self.optimizer is not None:
            state_dict["optimizer"] = self.optimizer.state_dict()

        # Save the state dictionary to the specified file path
        torch.save(state_dict, file_path)


class GPModel(gpytorch.models.ExactGP,
              botorch.models.model.FantasizeMixin):
    """
    Plain simple GP model with RBF kernel
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> 'GPModel':
        """
        Condition the GP on new observations (X, Y) and return a new GPModel.
        """

        # Combine the existing training data with the new data.
        if self.train_inputs[0].dim() == 1:
            updated_train_x = torch.cat([self.train_inputs[0], X.squeeze(0)], dim=0)
        else:
            updated_train_x = torch.cat([self.train_inputs[0], X], dim=0)

        updated_train_y = torch.cat([self.train_targets, Y], dim=0)

        # Create a new model with the updated data.
        new_model = self.__class__(updated_train_x, updated_train_y, self.likelihood)
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module

        # If your model has more components (e.g., hyperparameters, etc.), you might need to copy those over as well.
        # Here's a basic copy:
        # new_model.covar_module.base_kernel.lengthscale = self.covar_module.base_kernel.lengthscale
        # ... repeat for other hyperparameters and components as necessary ...

        return new_model

    def posterior(self, X: Tensor, *args, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # The ExactGP's __call__ method provides the posterior predictive distribution
        mvn = self(X)
        if observation_noise:
            # If considering observation noise, add it to the predictive covariance
            noise_diag = self.likelihood.noise_covar.noise
            mvn = gpytorch.distributions.MultivariateNormal(
                mvn.mean, mvn.lazy_covariance_matrix + noise_diag.diag_embed()
            )
        return GPyTorchPosterior(mvn)

    def transform_inputs(self, X: Tensor, input_transform: Optional[Module] = None) -> Tensor:
        # Transform the inputs if necessary, placeholder for now
        return X

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        data_dim = train_x.size(-1)
        self.feature_extractor = LargeFeatureExtractor(data_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> 'NeuralGPModel':
        """
        Condition the NeuralGP on new observations (X, Y) and return a new NeuralGPModel.
        """
        # Ensure that the new data is processed using the feature extractor
        X_projected = self.feature_extractor(X)
        if self.train_inputs[0].dim() == 1:
            updated_train_x = torch.cat([self.train_inputs[0], X_projected.squeeze(0)], dim=0)
        else:
            updated_train_x = torch.cat([self.train_inputs[0], X_projected], dim=0)

        updated_train_y = torch.cat([self.train_targets, Y], dim=0)

        # Create a new model with the updated data.
        new_model = self.__class__(updated_train_x, updated_train_y, self.likelihood)
        new_model.likelihood = self.likelihood
        new_model.mean_module = self.mean_module
        new_model.covar_module = self.covar_module
        new_model.feature_extractor = self.feature_extractors

        return new_model

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs: Any) -> Posterior:
        # Process the input through the neural network.
        X_projected = self.feature_extractor(X)
        X_projected = self.scale_to_bounds(X_projected)

        # Obtain the prior distribution.
        mvn = self(X_projected)

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
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
