import numpy as np
import gpytorch
import torch
from wdbo_algo.model import learn_model_space_time
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import wdbo_criterion

class WDBOOptimizer:
	"""WDBO Optimizer class, interfaces with the user
	"""

	def __init__(self, spatial_domain, spatial_kernel, temporal_kernel, spatial_kernel_args=[], temporal_kernel_args=[], n_initial_observations=15, alpha=0.25):
		"""Build the WDBO algorithm

		Args:
				spatial_domain (np.array): `d x 2`-array describing a `d`-dimensional hyperrectangle
				spatial_kernel (gpytorch.kernels.Kernel class): the spatial kernel class
				temporal_kernel (gpytorch.kernels.Kernel class): the temporal kernel class
				spatial_kernel_args (list, optional): the arguments for building the spatial kernel. Defaults to [].
				temporal_kernel_args (list, optional): the arguments for building the temporal kernel. Defaults to [].
				n_initial_observations (int, optional): the number of observations to collect before starting the optimization.
				Defaults to 15.
				alpha (float, optional): the WDBO hyperparameter, control the removal budget. Defaults to 0.25.
		"""
		self._spatial_domain = spatial_domain
		self._d = self._spatial_domain.shape[0]
		
		self._n_initial_observations = n_initial_observations
		self._alpha = alpha

		self._spatial_kernel = spatial_kernel
		self._spatial_kernel_wdbo = self.get_wdbo_kernel_class(self._spatial_kernel)
		self._spatial_kernel_args = spatial_kernel_args

		self._temporal_kernel = temporal_kernel
		self._temporal_kernel_wdbo = self.get_wdbo_kernel_class(self._temporal_kernel)
		self._temporal_kernel_args = temporal_kernel_args

		self._xx_tt = None
		self._yy = None
		self._budget = 1.0

		self._gpr = None
		self._lambda, self._lS, self._lT, self._noise = None, None, None, None
		self._current_time = None

	def get_wdbo_kernel_class(self, gpytorch_kernel_class):
		"""Correspondance between gpytorch kernels classes and wdbo-criterion kernels classes.

		Args:
				gpytorch_kernel_class (gpytorch.kernels.Kernel class): the kernel class in gpytorch

		Returns:
				wdbo_criterion.Kernel class: the kernel class in wdbo_criterion
		"""
		if gpytorch_kernel_class == gpytorch.kernels.RBFKernel:
			return wdbo_criterion.RBFKernel
		if gpytorch_kernel_class == gpytorch.kernels.MaternKernel:
			return wdbo_criterion.MaternKernel
		
		return None
	
	def dataset_size(self):
		"""Compute the dataset size of the DBO algorithm

		Returns:
				int: the dataset size
		"""
		return self._xx_tt.shape[0]

	def denormalize_x(self, x):
		"""Linear map from [0, 1]^d to the spatial domain of the objective function

		Args:
				x (np.array): the input

		Returns:
				np.array: the input mapped in the function domain
		"""
		return x * (self._spatial_domain[:, 1] - self._spatial_domain[:, 0]) + self._spatial_domain[:, 0]
	
	def normalize_x(self, x):
		"""Linear map from the spatial domain of the objective function to [0, 1]^d

		Args:
				x (np.array): the input

		Returns:
				np.array: the input mapped in [0, 1]^d
		"""
		return (x - self._spatial_domain[:, 0]) / (self._spatial_domain[:, 1] - self._spatial_domain[:, 0])
	
	def normalize_y(self, y):
		"""Standardize the input (i.e. subtract the empirical mean, divide by the empirical standard deviation)

		Args:
				y (np.array): the input

		Returns:
				np.array: the input standardized
		"""
		return (y - np.mean(y)) / np.std(y)
	
	def update_surrogate_model(self, verbose=False):
		"""Conditon a Gaussian Process on the collected data

		Args:
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		self._yy_normalized = self.normalize_y(self._yy)
		self._gpr = learn_model_space_time(self._xx_tt, self._spatial_kernel, self._spatial_kernel_args, self._temporal_kernel, self._temporal_kernel_args, self._yy_normalized)
		self._lambda, self._lS, self._lT, self._noise = np.exp(self._gpr.get_kernel_log_hyperparameters())
		if verbose:
			print(f"Hyperparameters (lambda, lS, lT, noise variance) = {(self._lambda, self._lS, self._lT, self._noise)}")
	
	def tell(self, x, t, y, verbose=False):
		"""Add an observation to the dataset and update the surrogate model.

		Args:
				x (np.array): the input in the space domain
				t (float): the input in the time domain
				y (float): the noisy output
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		normalized_x = self.normalize_x(x)

		# Add the input output pair to the dataset
		if self._xx_tt is None:
			self._xx_tt = np.array([np.concatenate((normalized_x, np.array([t])))])
			self._yy = np.array([y])
		else:
			self._xx_tt = np.concatenate((self._xx_tt, np.array([np.concatenate((normalized_x, np.array([t])))])))
			self._yy = np.concatenate((self._yy, np.array([y])))
		
		if self._n_initial_observations > 0:
			self._n_initial_observations -= 1

		# Update the surrogate model and the hyperparameters
		if self._n_initial_observations <= 0:
			self.update_surrogate_model(verbose=verbose)

	def next_query(self, current_time):
		"""Find the next relevant input to query.

		Args:
				current_time (float): the present time

		Returns:
				np.array: a relevant input to query in the space domain
		"""
		# Initial observations, without optimization of the acquisition function
		if self._n_initial_observations > 0:
			random_normalized_x = np.random.uniform(low=0.0, high=1.0, size=(self._d,))
			return self.denormalize_x(random_normalized_x)
		
		# The initial observations are gathered, we now have to optimize the acquisition function
		UCB = UpperConfidenceBound(self._gpr, beta=0.2 * self._d * np.log(2 * self._xx_tt.shape[0]))
		low_bounds = torch.zeros(self._d+1)
		low_bounds[-1] = current_time
		up_bounds = torch.ones(self._d+1)
		up_bounds[-1] = current_time
		bounds = torch.stack([low_bounds, up_bounds])
		candidate, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=512,)
		next_sample = candidate[0].numpy()

		return self.denormalize_x(next_sample[:-1])
	
	def clean(self, t, verbose=False):
		"""Remove irrelevant observations from the dataset

		Args:
				t (float): the present time
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		if self._n_initial_observations > 0:
			return
		
		if self._current_time is None:
			self._current_time = t
		
		self._budget *= (1.0 + self._alpha) ** ((t - self._current_time) / self._lT)

		# Cleaning loop for the dataset
		min_crit = 0
		while self._xx_tt.shape[0] > 2 and self._budget > min_crit:
			# Measures observations relevancy
			criteria = wdbo_criterion.wasserstein_criterion(
					np.ascontiguousarray(self._xx_tt[:, :-1]),
					np.ascontiguousarray(self._yy_normalized),
					np.ascontiguousarray(self._xx_tt[:, -1]),
					self._xx_tt.shape[0],
					self._d,
					self._lambda, self._noise,
					self._spatial_kernel_wdbo(*([self._lS] + self._spatial_kernel_args)), self._temporal_kernel_wdbo(*([self._lT] + self._temporal_kernel_args)),
					t,
					0, 1)
			
			# Find the least relevant observation
			sorted_args = criteria.argsort()
			indices, criteria = sorted_args, criteria[sorted_args]
			idx_min, min_crit = (indices[0], criteria[0] + 1.0)

			if verbose:
				print(f"Removal Budget: {self._budget} // Least Relevant Observation: {idx_min} // Relevancy: {min_crit} (i.e. {round(100 * min_crit / self._budget, 2)}% of budget)")

			# Remove it if the budget allows it
			if min_crit < self._budget:
				# Budget consumption
				if min_crit > 1.0:
					self._budget = self._budget / min_crit

				if verbose:
					print(f"Observation {idx_min} is removed")

				# Dataset update
				self._xx_tt = np.delete(self._xx_tt, (idx_min), axis=0)
				self._yy = np.delete(self._yy, (idx_min), axis=0)
				self.update_surrogate_model(verbose=verbose)

		self._current_time = t
