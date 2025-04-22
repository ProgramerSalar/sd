import torch, numpy as np 




class DiagonalGaussianDistribution(object):

    """ 
    A diagonal Gaussian distribution parametrized by mean and log variance.

    Args:
        parameters: Tensor containing concatenated mean and logvar (shape: [batch, 2*dim, ....])
        deterministic: If True, reduces to a deterministic distribution
    """


    def __init__(self, 
                 parameters,
                 deterministic=False):
        
        self.deterministic = deterministic
        
        # Split input tensor into mean and log variance 
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        
        # Clamp log variance for numerical stability
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        
        # Calculate standard deviation and variance 
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        # Handle deterministic case (zero variance)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

        
    def sample(self):
        """ 
        Generate samples using reparametrization trick:
        z = μ + σ*ε where ε ~ N(0,I)
        """

        if self.deterministic:
            return self.mean
        
        # Random sample from standard normal distribution 
        noise = torch.rand_like(self.mean)
        return self.mean + self.std * noise 
    

    

    def kl(self, other=None):
        """ 
        Compute KL divergence KL(self || other)

        Args:
            other: Another DiagonalGaussianDistribution or None (standard normal)

        Returns:
            KL divergence for each sample in batch
        """
        if self.deterministic:
            return torch.Tensor([0.], device=self.mean.device)
        
        if other is None:
            # KL divergance with standard normal N(0, I)
            return 0.5 * torch.sum(
                input=torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3]
            )
        
        else:
            # KL divergance between two Gaussians 
            return 0.5 * torch.sum(
                input = torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var  - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3]
            )
            

    def nll(self, sample, dims=[1, 2, 3]):
        """ 
        Negative log likelihood of samples under the distribution 

        Args:
            sample: Input samples to evaluate 
            dims: Dims to sum over 

        Returns:
            Negative log likelihood for each sample 
        """
        if self.deterministic:
            return torch.Tensor([0.], device=self.mean.device)
        
        logtwopi = np.log(2.0 * np.pi)

        return 0.5 * torch.sum(
            input = logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims
        )
    

    def mode(self):
        """Return the mode of the distribution (same as mean for Gaussian)"""
        return self.mean
    


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )