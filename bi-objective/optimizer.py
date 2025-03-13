from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
import torch


class BiBO:
    def __init__(self, model, data_handler, bounds, ref_point, q):
        """
        Initialize the TensorComparison class with the necessary parameters for optimization and comparison.
        
        :param model: The predictive model to use for generating observations.
        :param bounds: The bounds within which to optimize the acquisition function.
        :param data_handler: GPDataHandler class
        :param ref_point: Reference point for computing hypervolume.
        :param q: The number of candidates to generate.
        """
        self.model = model
        self.bounds = bounds
        self.train_x = data_handler.X_scaled_density
        self.ref_point = ref_point
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
        self.q = q

    def optimize_qehvi_and_get_observation(self, num_restarts=10, raw_samples=512):
        """
        Optimizes the qEHVI acquisition function and returns a new candidate and observation.
        """
        with torch.no_grad():
            pred = self.model.posterior(self.train_x).mean
        partitioning = FastNondominatedPartitioning(ref_point=self.ref_point, Y=pred)
        acq_func = qExpectedHypervolumeImprovement(model=self.model, ref_point=self.ref_point,
                                                   partitioning=partitioning, sampler=self.sampler)
        
        candidates, _ = optimize_acqf(acq_function=acq_func, bounds=self.bounds, q=self.q,
                                      num_restarts=num_restarts, raw_samples=raw_samples,
                                      options={"batch_limit": 5, "maxiter": 200}, sequential=True)
        
        return candidates.detach()

    @staticmethod
    def arrays_equal_with_permutation(A, B, tol=0.01):
        """
        Checks if two arrays are equal with permutation, within a tolerance.
        
        :param A: First array.
        :param B: Second array.
        :param tol: Tolerance for the comparison.
        :return: Boolean indicating if arrays are equal within the tolerance.
        """
        A_sorted = np.array(sorted(A.tolist()))
        B_sorted = np.array(sorted(B.tolist()))
        return np.all(np.isclose(A_sorted, B_sorted, atol=tol))

    def generate_tensors_and_count_matches(self, n_attempts, num_restarts=10, raw_samples=512):
        """
        Generates tensors for a given number of attempts and counts the matches for each tensor.
        
        :param n_attempts: Number of tensors to generate and compare.
        :return: Tensors and their corresponding match counts.
        """
        tensors = [self.optimize_qehvi_and_get_observation(num_restarts, raw_samples) for _ in range(n_attempts)]
        matches_count = [0] * n_attempts

        for i in range(n_attempts):
            for j in range(i + 1, n_attempts):
                if self.arrays_equal_with_permutation(tensors[i], tensors[j]):
                    matches_count[i] += 1
                    matches_count[j] += 1

        return tensors, matches_count