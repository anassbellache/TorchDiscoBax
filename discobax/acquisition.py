import torch
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel

from algorithm import SubsetSelect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import List, AnyStr


class DiscoBAXAdditive(BaseBatchAcquisitionFunction):

    def __init__(self,
                 budget: int,
                 mc_samples: int,
                 noise_type: str = 'additive',
                 k: int = 1):
        """
        Initialize the DiscoBAXAdditive algorithm.

        :param budget: Number of iterations.
        :param mc_samples: Number of Monte Carlo samples.
        :param noise_type: Type of noise - either 'additive' or 'multiplicative'.
        :param k: Number of subset points to select.
        """
        super().__init__()
        self.budget = budget
        self.mc_samples = mc_samples
        self.noise_type = noise_type
        self.k = k
        self.data = []

    def expected_information_gain(self, x: torch.Tensor, subset_selector: SubsetSelect, model: AbstractBaseModel) -> torch.Tensor:
        # Calculate the Expected Information Gain (EIG) for x using the subset_selector
        # This requires more information about the form of EIG. For now, I'll assume it's
        # based on the `monte_carlo_expectation` method of the SubsetSelect class.
        S = torch.tensor([x]).float()
        return subset_selector.monte_carlo_expectation(S, model)

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel) -> List:
        selected_indices = []

        for _ in range(self.budget):
            subset_selector = SubsetSelect(X=dataset_x, noise_type=self.noise_type, n_samples=self.mc_samples, k=self.k)
            # Sample functions from the model's posterior and get the subsets S_j
            subset_selector.get_exe_paths(last_model)

            # Calculate EIG for each point in available_indices
            eig_values = [self.expected_information_gain(x, subset_selector, last_model) for x in available_indices]

            # Select the point with the maximum EIG
            max_index = torch.argmax(torch.tensor(eig_values))
            xi = available_indices[max_index]

            # Append to the selected indices
            selected_indices.append(xi)

            # "Query" and update the dataset (assuming querying here means adding to the dataset)
            # This part requires more context on how you'd like to "query" the data.
            # For now, I'll just append it to our dataset `self.data`.
            self.data.append((xi, last_model.model(torch.tensor([xi]))))

            # Remove the selected index from available_indices
            available_indices.pop(max_index)

        return selected_indices
