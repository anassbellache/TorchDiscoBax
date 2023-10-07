from typing import List, AnyStr

import torch
from botorch.acquisition import qMaxValueEntropy
from botorch.optim import optimize_acqf
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from slingpy import AbstractDataSource, AbstractBaseModel

from algorithm import SubsetSelect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscoBAXAdditive(BaseBatchAcquisitionFunction):
    def __init__(self, budget, num_samples, mc_samples):
        self.budget = budget
        self.num_samples = num_samples
        self.mc_samples = mc_samples
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
        Nominate experiments for the next learning rounds using the DiscoBAX algorithm.
        """
        # List to hold selected indices
        selected_indices = []

        for i in range(self.budget):
            # Sample l functions from the posterior
            functions = [last_model.posterior(dataset_x.subset(available_indices)).rsample() for _ in
                         range(self.mc_samples)]

            S = []
            for j, f in enumerate(functions):
                self.algo = SubsetSelect(f, "additive")
                self.algo.initialize()
                exe_path = self.algo.get_exe_paths(last_model)
                new_x = torch.tensor(exe_path.x, device=device)
                new_y = torch.tensor(exe_path.y, device=device)
                S.append((new_x, new_y))

            # Create a combined acquisition function over all S samples
            acq_func = qMaxValueEntropy(
                model=last_model.model,
                train_x=torch.cat([s[0] for s in S], dim=0)
            )

            # Use optimize_acqf to get the best next point
            bounds = torch.stack([torch.min(torch.cat([s[0] for s in S], dim=0), dim=0).values,
                                  torch.max(torch.cat([s[0] for s in S], dim=0), dim=0).values])
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=100,  # Number of initial raw samples
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )

            # Find the index of the candidate in the available_indices
            candidate_idx = available_indices[
                torch.argmin(torch.norm(dataset_x.subset(available_indices) - candidate, dim=1))]

            # Append to selected_indices
            selected_indices.append(candidate_idx)

            # Remove the selected index from available_indices to prevent reselection in the next iteration
            available_indices.remove(candidate_idx)

        return selected_indices


