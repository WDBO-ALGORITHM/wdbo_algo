import numpy as np
import gpytorch
import torch

class SpaceTimeGPModel(gpytorch.models.ExactGP):
	num_outputs = 1

	def __init__(self, space_kernel, space_args, time_kernel, time_args, train_x, train_y, likelihood):
		"""Build the surrogate model

		Args:
				space_kernel (gpytorch.kernels.Kernel class): the spatial kernel
				space_args (list): the spatial kernel arguments
				time_kernel (gpytorch.kernels.Kernel class): the temporal kernel class
				time_args (list): the temporal kernel arguments
				train_x (np.array): the training dataset
				train_y (np.array): the labels
				likelihood (gpytorch.likelihood): the likelihood function
		"""
		super(SpaceTimeGPModel, self).__init__(train_x, train_y, likelihood)
		if train_x.ndim != 1:
			self.d = train_x.shape[1]
			self.train_x = train_x
		else:
			self.d = train_x.shape[0]
			self.train_x = train_x.unsqueeze(0)
	
		self.train_y = train_y
		self.likelihood = likelihood
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(space_kernel(*space_args, active_dims=torch.tensor(range(self.d-1))) * time_kernel(*time_args, active_dims=torch.tensor([self.d - 1])))

	def forward(self, x):
		"""Use the model on the input

		Args:
				x (torch.Tensor): the input

		Returns:
				gpytorch.distributions.MultivariateNormal: the posterior distribution for input
		"""
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
	
	def fit(self):
		"""Find the hyperparameters from data
		"""
		# Find optimal model hyperparameters
		self.train()
		self.likelihood.train()

		# Use the adam optimizer
		optimizer = torch.optim.Adam(self.parameters(), lr=0.1)	# Includes GaussianLikelihood parameters

		# "Loss" for GPs - the marginal log likelihood
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

		n_iterations = 150
		for i in range(n_iterations):
			# Zero gradients from previous iteration
			optimizer.zero_grad()
			# Output from model
			output = self(self.train_x)
			# Calc loss and backprop gradients
			loss = -mll(output, self.train_y)
			loss.backward()

			optimizer.step()

		self.eval()
		self.likelihood.eval()
	
	def posterior(self, X, posterior_transform=None):
		return self.likelihood(self(X.double()))
	
	def get_kernel_log_hyperparameters(self):
		"""Return the log of kernel hyperparameters

		Returns:
				np.array: the log of the kernel hyperparameters
		"""
		return np.log(np.array([self.covar_module.outputscale.item(), self.covar_module.base_kernel.kernels[0].lengthscale.item(), self.covar_module.base_kernel.kernels[1].lengthscale.item(), self.likelihood.noise.item()]))
	
	def set_parameters(self, lmbd, lS, lT):
		"""Setter for the kernel parameters

		Args:
				lmbd (float): the scale of the covariance function
				lS (float): the spatial lengthscale of the covariance function
				lT (float): the temporal lengthscale for the covariance function
		"""
		self.covar_module.outputscale = lmbd
		self.covar_module.base_kernel.kernels[0].lengthscale = lS
		self.covar_module.base_kernel.kernels[1].lengthscale = lT


def learn_model_space_time(xx_tt, space_kernel, space_kernel_args, time_kernel, time_kernel_args, yy_normalized):
	"""Helper function to build a surrogate model

	Args:
			xx_tt (np.float): training inputs
			space_kernel (gpytorch.kernels.Kernel class): the spatial kernel class
			space_kernel_args (list): the arguments for the spatial kernel class
			time_kernel (gpytorch.kernels.Kernel class): the temporal kernel class
			time_kernel_args (list): the arguments for the temporal kernel class
			yy_normalized (np.array): the training labels

	Returns:
			SpaceTimeModelGP: the surrogate model, trained on the training data
	"""
	likelihood = gpytorch.likelihoods.GaussianLikelihood()
	gpr = SpaceTimeGPModel(space_kernel, space_kernel_args, time_kernel, time_kernel_args, torch.tensor(xx_tt), torch.tensor(yy_normalized), likelihood)
	gpr.fit()

	return gpr