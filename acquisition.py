from typing import List, AnyStr

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel

from algorithm import SubsetSelect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscoBAXAdditive(BaseBatchAcquisitionFunction):
    def __init__(self, budget, num_samples):
        self.budget = budget
        self.num_samples = num_samples
        self.D = []
        self.algo = None

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel,
                 ):
        """
        Nominate experiments for the next learning round using the DiscoBAX algorithm.
        """
        avail_dataset_x = dataset_x.subset(available_indices)
        self.algo = SubsetSelect(avail_dataset_x, "additive")
        self.algo.initialize()
        exe_path = self.algo.get_exe_paths(last_model)
        new_x = torch.tensor(exe_path.x, device=device)
        new_y = torch.tensor(exe_path.y, device=device)
        fantasy_model = last_model.model.get_fantasy_model(new_x, new_y)
        # calculate the variance of the posterior at this iteration for each input x
        y_pred = last_model.predict(new_x)  # Using the predict method
        var_p = y_pred[1]

        # Calculate H(S), the entropy of the current model's predictions on subset S
        h_s = 0.5 * torch.log(2 * torch.pi * var_p) + 0.5

        eig_values = {}
        for x in available_indices:
            # For each available index x, create a fantasy model and calculate H(S|x)
            # ... (Your method to create/update fantasy model)
            # Assuming `fantasy_model` is the updated/fantasy model
            fantasy_y_pred = fantasy_model.predict(dataset_x.subset(x))
            fantasy_var_p = fantasy_y_pred[1]

            h_s_x = 0.5 * torch.log(2 * torch.pi * fantasy_var_p) + 0.5

            # Calculate EIG(x, S) for this x
            eig = h_s - h_s_x
            eig_values[x] = eig

        # Select the index with the maximum EIG value
        next_index = max(eig_values, key=eig_values.get)

        return next_index


class DiscoBAXMultiplicative(BaseBatchAcquisitionFunction):
    def __init__(self, budget, num_samples):
        self.budget = budget
        self.num_samples = num_samples
        self.noise_type = "multiplicative"
        self.D = []
        self.params = {}

    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model: AbstractBaseModel,
                 ):
        """
        Nominate experiments for the next learning round using the DiscoBAX algorithm.
        """
        avail_dataset_x = dataset_x.subset(available_indices)
        self.algo = SubsetSelect(avail_dataset_x, "multiplicative")
        self.algo.initialize()

