import torch
import gpytorch


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, kernel,
                 mean_module):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            mean_module, num_tasks=self.num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=self.num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,
                                                                  covar_x)


def create_gp_factory(self, likelihood, model, num_tasks, kernel, mean_module):
    def build_gp(train_x, train_y):
        mod = model(train_x=train_x, train_y=train_y, likelihood=likelihood,
                    kernel=kernel, mean_module=mean_module)
        return mod, likelihood
    return build_gp


def build_network(arch_args):
    dim = arch_args["dim"]
    kernel_type = arch_args.get("kernel", "rbf")
    mean_type = arch_args.get("mean", "constant")
    likelihood_type = arch_args.get("likelihood", "multitask-gaussian")
    # Construct kernel
    if kernel_type == "rbf":
        kernel = gpytorch.kernels.RBFKernel()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    # Construct mean
    if mean_type == "constant":
        mean = gpytorch.means.ConstantMean()
    else:
        raise ValueError(f"Unknown mean type: {mean_type}")
    # Construct likelihood
    if likelihood_type == "multitask-gaussian":
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=dim)  # noqa: E501
    else:
        raise ValueError(f"Unknown likelihood type: {likelihood_type}")

    return create_gp_factory(likelihood=likelihood, model=MultitaskGPModel,
                             num_tasks=dim, kernel=kernel, mean_module=mean)
