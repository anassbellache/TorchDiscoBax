from typing import List, AnyStr

import torch
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel
from .gp_model import BaseGPModel
from .algorithm import SubsetSelect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscoBAXAdditive(BaseBatchAcquisitionFunction):

    def __init__(self,
                 budget: int,
                 mc_samples: int,
                 noise_type: str = 'additive',
                 k: int = 1):
        super().__init__()
        self.budget = budget
        self.mc_samples = mc_samples
        self.noise_type = noise_type
        self.k = k
        self.data = []

    def expected_information_gain(self, x: torch.Tensor, subset_selector: SubsetSelect, last_model: BaseGPModel) -> torch.Tensor:
        # Calculate the Expected Information Gain (EIG) for x

        # Step 1: Compute the current entropy of the model's predictions at the locations of previously evaluated points
        current_posterior = last_model.get_posterior(subset_selector.exe_path.x)
        current_entropy = -torch.sum(current_posterior.mean * torch.log(current_posterior.mean))

        # Step 2: Compute the expected entropy after hypothetically adding x to the training set
        x_posterior = last_model.get_posterior(x)
        expected_entropy_at_x = -torch.sum(x_posterior.mean * torch.log(x_posterior.mean))

        # Step 3: The EIG is the difference between the current entropy and the expected entropy
        eig = current_entropy - expected_entropy_at_x
        return eig

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel) -> List:
        selected_indices = []

        for _ in range(len(available_indices)):
            # Sample functions from the model's posterior and get the subsets S_j
            subset_selector = SubsetSelect(X=dataset_x, noise_type=self.noise_type, n_samples=self.mc_samples, k=self.k)
            exe_path = subset_selector.get_exe_paths(last_model)

            eig_values = [self.expected_information_gain(torch.tensor([x]), subset_selector, last_model) for x in available_indices]

            # Select the point with the maximum EIG
            max_index = torch.argmax(torch.tensor(eig_values))
            xi = available_indices[max_index]

            # Append to the selected indices
            selected_indices.append(xi)

            # Update the dataset
            y_pred = last_model.predict(torch.tensor([xi]).unsqueeze(0))
            y = y_pred[0] if isinstance(y_pred, list) else y_pred
            self.data.append((xi, y))

            # Remove the selected index from available_indices
            available_indices.pop(max_index)

        return selected_indices
