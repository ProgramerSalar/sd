import torch.nn as nn, torch 
from vae.lpips import LPIPS
from vae.discriminator import NLayerDiscriminator, weight_init
from torch.nn import functional as F
from vae.autoencoder import AutoencoderKL


def hinge_d_loss(logits_real, logits_fake):

    """ 
    Computes the hinge loss for GAN discriminator.

    The hinge loss encurages:
        - Real samples (logits_real) to output >= 1 
        - Fake samples (logits_fake) to output <= -1 

    Args:
        logits_real (torch.Tensor): Discriminator outputs for real images (shape: [N, *]).
        logits_fake (torch.Tensor): Discriminator outputs for fake images (shape: [N, *]).

    Returns:
        torch.Tensor: Scaler discriminator loss.

    Example:
        >>> real_logits = torch.tensor([2.0, 1.5, 0.5])  # Real samples 
        >>> fake_logits = torch.tensor([-1.5, -0.5, 1.0])   # Fake samples 
        >>> loss = hinge_d_loss(real_logits, fake_logits)
        >>> print(loss.item())  # eg. 0.4167
    """

    loss_real = torch.mean(F.relu(input=1. - logits_real))
    loss_fake = torch.mean(F.relu(input=1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):

    """
    Computes the vanilla (non-saturating) GAN discriminator loss using softplus.
    
    This is a numerically stable version of the original GAN discriminator loss:
    - Real samples: -log(sigmoid(logits_real)) → softplus(-logits_real)
    - Fake samples: -log(1 - sigmoid(logits_fake)) → softplus(logits_fake)

    Args:
        logits_real (torch.Tensor): Discriminator outputs for real images (shape: [N, *]).
        logits_fake (torch.Tensor): Discriminator outputs for fake images (shape: [N, *]).
        
    Returns:
        torch.Tensor: Scalar discriminator loss.
        
    Example:
        >>> real_logits = torch.tensor([2.0, -1.0, 0.5])  # Real samples
        >>> fake_logits = torch.tensor([-1.5, 0.5, 1.0])  # Fake samples
        >>> loss = vanilla_d_loss(real_logits, fake_logits)
        >>> print(loss.item())  # e.g., 0.8765
    """

    d_loss = 0.5 * (
        # Loss for real samples: -log(sigmoid(logits_real)) → softplus(-logits_real)
        torch.mean(torch.nn.functional.softplus(-logits_real)) + 
        # Loss for fake samples: -log(1 - sigmoid(logits_fake)) → softplus(logits_fake)
        torch.mean(torch.nn.functional.softplus(logits_fake))
    )

    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):

    """ 
    Dynamically adjusts a weights value based on the training progress.

    This function is commonly used in GAN training or curriculum lerning to:
    - Disable a loss term early in training (by setting weight=0)
    - Gradually introduce or modify loss weights 

    Args:
        weight (float): Original weight value to return after threshold is reached.
        global_step (int): Current training step/iteration.
        threshold (int): Step before which the weight is replaced with `value`.
        value (float): Temporary value to use before threshold is reached.

    Returns:
        float: The original weight if global_step >= threshold, else `value`

    Example:
        >>> # Gradually introduce perceptual loss after 1000 steps
        >>> current_step = 500
        >>> loss_weight = adopt_weight(weight=1.0,
        ...                             global_step=current_step,
        ...                             threshold=1000,
        ...                             value=0.0)
        >>> print(loss_weight)  # Output: 0.0 (since 500 < 1000)
    """
    if global_step < threshold:
        weight = value
    return weight





class LPIPSWithDiscriminator(nn.Module):

    """ 
    A composite loss function combining:
    - LPIPS perceptual loss 
    - KL divergence (for VAE training)
    - Pixel-wise reconstruction loss (L1)
    - GAN adversarial loss with optional discriminator

    Args:
        disc_start (int): Step at which to start training discriminator.
        logvar_init (float): Initial value for the learned log variance.
        KL_weight (float): Weight for KL divergance term.
        pixelloss_weight (float): Weight for pixel-wise L1 loss.
        disc_num_layers (int): Number of layer in discriminator.
        disc_num_channels (int): Input channels for discriminator.
        disc_factor (float): Weighting factor for discriminator loss.
        disc_weight (float): Overall weight for discriminator component.
        use_actnorm (bool): Whether to use ActNorm in dicriminator.
        disc_conditional (bool): Whether discriminator is conditional on additional input.
        disc_loss (str): Type of discriminator loss ("hinge" or "vanilla")
    """

    
    def __init__(self, 
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge"):
        
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]

        # Initialize loss weights
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        # Initialize LPIPS perceptual loss (fixed, non-trainable)
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_loss.requires_grad_(False)

        # Learnable log variance parameter
        self.logvar = nn.Parameter(data=torch.ones(size=()) * logvar_init)

        # Initialize discriminator
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm).apply(weight_init)
        
        # Discriminator training parameters
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional


    def calculation_adaptive_weight(self, nll_loss, g_loss, last_layer=None):

        """ 
        Compute adaptive weight for balancing generator and reconstruction losses.

        Args:
            nll_loss: Negative log likelihood (reconstruction) loss 
            g_loss: Generator loss 
            last_layer: Last layer of the decoder network

        Returns:
            Computed adaptive weight tensor
        """

        # Compute gradients 
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grad = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        else:
            nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
            g_grad = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

        # Compute weight based on gradient magnitudes
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grad) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight 
        return d_weight
    




    def forward(self, 
                inputs,
                recontructions,
                posteriors,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train",
                weights=None
                ):
        
        """ 
        Forward pass with loss computation.

        Args:
            inputs: Ground truth images
            reconstructions: Reconstructed images for model
            posteriors: Latent distribution from encoder
            optimizer_idx: 0 for generator, 1 for discriminator 
            global_step: Current training step 
            last_layer: Last layer of decoder (for adaptive weighting)
            cond: Conditional input (optional)
            split: "train" or "val" for logging 
            weights: per-Pixel loss weights (optional)

        Returns:
            (loss, log_dict) tuple 
        """
        
        # Compute reconstruction loss (L1 + optional LPIPS)
        rec_loss = torch.abs(inputs.contiguous() - recontructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), recontructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # Weighted NLL loss with learned variance
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss

        # Average losses 
        weighted_nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]


        # Generator update 
        if optimizer_idx == 0:
            # Compute GAN loss 
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(recontructions.contiguous())

            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat(tensors=(recontructions.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            # Compute adaptive weight
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculation_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)


            # Apply disc factor scheduling 
            disc_factor = adopt_weight(weight=self.disc_factor,
                                       global_step=global_step,
                                       threshold=self.discriminator_iter_start)
            
            # Total loss 
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss


            # Logging dictionary 
            log = {
                f"{split}/total_loss" : loss.clone().detach().mean(), 
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(), 
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean()
            }

            return loss, log 
        
        # Discriminator update 
        if optimizer_idx == 1:
            # Get discriminator predictions 
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(recontructions.contiguous().detach())

            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((recontructions.contiguous().detach(), cond), dim=1))


            # Compute discriminator loss with scheduling 
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            # Logging dictionary
            log = {
                "{split}/disc_loss": d_loss.clone().detach().mean(),
                "{split}/logits_real": logits_real.detach().mean(),
                "{split}/logits_fake": logits_fake.detach().mean()
            }

            return d_loss, log 







