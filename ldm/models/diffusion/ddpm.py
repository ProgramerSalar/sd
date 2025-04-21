import torch 
import torch.nn as nn 
from vae.utils.utils import instantiate_from_config
from .utils import count_params, default, noise_like, extract_into_tensor,  make_beta_schedule
from VQ.ema import LitEma
from tqdm import tqdm 
import numpy as np 
from functools import partial
from einops import rearrange, repeat
from contextlib import contextmanager
from torchvision.utils import make_grid
import numpy as np
import pytorch_lightning as pl
from VQ.autoencoder import VQModelInterface
from .utils import log_txt_as_img, isimage, ismap
from .ddim import DDIMSampler
from vae.autoencoder import AutoencoderKL
from vae.distribution import DiagonalGaussianDistribution
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from vae.autoencoder import IdentityFirstStage
from torch.optim.lr_scheduler import LambdaLR

def disable_train(self, mode=True):

    """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
    
    return self 


__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}



class DDPM(pl.LightningModule):

    """ 
    Denoising Diffusion Probabilistic Models (DDPM) implementation.
    This class implements the core diffusion process with various configuration options for 
    noise scheduling, loss types and conditioning approaches.

    Args:
        unet_config (dict): Configuration for the underlying Unet model 
        timesteps (int): Number of diffusion timesteps (default: 1000)
        beta_scheduler (str): Type of noise scheduler ('linear', 'cosine') (default: 'linear')
        loss_type (str): Loss function type ('l1', 'l2') (default: 'l2')
        ckpt_path (str): Path to checkpoint for loading pretrained weights (default: None)
        ignore_keys (list): Keys to ignore when loading checkpoint (default: [])
        load_only_unet (bool): Whether to load only  UNet weights (default: False)
        monitor (str): Metric to monitor for checkpointing (default: 'val/loss')
        use_ema (bool): Whether to use exponential moving average (default: True)
        first_stage_key (str): Key for first stage model (default: 'image')
        image_size (int): Input image size (default: 256)
        channels (int): Input image channels (default: 3)
        log_every_t (int): Log interval duration training (default: 100)
        clip_denoised (bool): Whether to clip denoised sample (default: True)
        linear_start (float): Starting beta values for linear scheduler (default: 1e-4)
        linear_end (float): End beta values for linear scheduler (default: 2e-2)
        cosine_s (float): Cosine scheduler parameters (default: 8e-3)
        given_betas (Tensor): Precomputed betas (default: True)
        original_elbo_weight (float): Weight for ELBO loss term (default: 0.)
        v_posterior (float): Weight for posterior variance (default: 0.)
        l_simple_weight (float): Weight for simple loss (default: 1.)
        conditioning_key (str): Type of conditioning (None, 'concat', 'crossattn', etc.) (default: None)
        parametrization (str): Prediction target ('eps' or 'x0') (default: 'eps')
        scheduler_config (dict): Learning rate scheduler config (default: None)
        use_position_encoding (bool): Whether to use positional encoding (default: False)
        learn_logvar (bool): Whether to Learn log variance (default: False)
        logvar_init (float): Initial value of log variance (default: 0.)

    Examples:
        >>> # Basic unconditional generation
        >>> unet_config = {'target': 'my_module.UNet', 'params': {'in_channels': 3}}
        >>> model = DDPM(unet_config, timesteps=1000)
        >>> x = torch.randn(4, 3, 256, 256)  # Random input
        >>> loss, loss_dict = model(x)  # Forward pass

        >>> # With EMA and monitoring
        >>> model = DDPM(unet_config, use_ema=True, monitor='val/loss')
        >>> # During training:
        >>> loss.backward()
        >>> if model.use_ema:
        >>>     model.model_ema(model.model)

    """

    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_scheduler="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,   # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta 
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",    # all assuming fixed variance scheduler 
                 scheduler_config=None,
                 use_position_encoding=False,
                 learn_logvar=False,
                 logvar_init=0.
                 ):
        
        super().__init__()

        # validate parametrization type 
        assert parameterization in ["eps", "x0"], "currently only supporting 'eps' and 'x0' "
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        # Initialize base attributes 
        self.cond_stage_model = None     # First stage model (eg. VAE) 
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size # try conv?
        self.channels = channels
        self.use_positional_encoding = use_position_encoding
        


        # Initialize the diffusion model (UNet wrapped with conditioning)
        self.model = DiffusionWrapper(diff_model_config=unet_config,
                                      conditioning_key=conditioning_key)
        count_params(self.model, verbose=True) # Print parameter count 


        # Exponential Moving Average Setup
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        
        # Learning rate scheduler configuration 
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        # Loss weights and parameters 
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        # Monitoring and checkpoint
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            pass 

        # Noise scheduler initialization 
        self.register_schedule(
                             beta_scheduler=beta_scheduler,
                             timesteps=timesteps,
                             linear_start=linear_start,
                             linear_end=linear_end,
                             cosine_s=cosine_s)
        
        # Loss configuration 
        self.loss_type = loss_type

        # Log variance learning 
        self.learn_logvar = learn_logvar
        self.num_timesteps = int(timesteps)
        self.logvar = torch.full(fill_value=logvar_init,
                                 size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar,
                                       requires_grad=True)
            

    def register_schedule(self, 
                          given_bitas=None,
                          beta_scheduler="linear",
                          timesteps=1000,
                          linear_start=1e-4,
                          linear_end=2e-2,
                          cosine_s=8e-3):
        
        """
        Register the noise schedule for the diffusion process 
        Computes and stores all necessary coefficients for the forward and reverse diffusion process.

        Args:
            given_betas (np.array, optional): Precomputed beta values. If None, will compute based on schedule.
            beta_scheduler (str): Type of noise scheduler ('linear' or 'cosine')
            timesteps (int):  Total number of diffusion timesteps 
            linear_start (float): Starting beta value for linear scheduler.
            linear_end (float): Ending beta value for linear scheduler.
            cosine_s (float): Cosine schedule parameter.

        Notes:
            - Computes all stores all coefficients needed for:
                - Foward process q(x_t | x_{t-1})
                - Reverse process posterior q(x_{t-1} | x_t, x_0)
                - Loss weighting term 
            - Supports both predefined betas and automatic schedule  generation
        """
        
        # USE   given betas if provided, otherwise create schedule
        if exit(given_bitas):
            betas = given_bitas

        else:
            betas = make_beta_schedule(schedule=beta_scheduler,
                                       timestep=timesteps,
                                       linear_start=linear_start,
                                       linear_end=linear_end,
                                       cosine_s=8e-3)
            

        # Compute alpha values from betas 
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # Store basic parameters 
        timesteps, = betas.shape 
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        # Validate shape 
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        # Helper function for tensor conversion 
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Register to basic buffer 
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alpha_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculates for diffusion q(x_t | x_(t-1)) and others | Register forward process coefficents 
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Calculation for posterior q(x_{t-1} | x_t, x_0) | Compute and register posterior variance parameters 
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # Below log calculation clipped because the posterior variance is 0 at the begining of the diffusion chain  | Clip and register log variance to avoid numerical instability
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        # Register posterior mean coefficients 
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        ))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        ))

        if self.parameterization == "eps":
            lvlb_weight = self.betas ** 2 / (
                2 * posterior_variance * to_torch(alphas) * (1 - alphas_cumprod)
            )

        elif self.parameterization == "x0":
            lvlb_weight = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))

        else:
            raise NotImplementedError('mu not supported')
        

        # TODO: how to choose this term | Adjust first timestep weight 
        lvlb_weight[0] = lvlb_weight[1]
        self.register_buffer('lvlb_weight', lvlb_weight, persistent=False)

        # Validate no NaN values 
        assert not torch.isnan(lvlb_weight).all()


            
    
    def q_sample(self, x_start, t, noise=None):


        """ 
        Diffuse the data for a given number of diffusion steps (forward process).

        Args:
            x_start (Tensor): Original clean samples 
            t (Tensor): Timesteps to diffuse to 
            noise (Tensor): Optional pre-generated noise 

        Returns:
            Tensor: Diffused samples at timestep t
        """


        noise = default(val=noise, d=lambda: torch.rand_like(x_start))
        return (extract_into_tensor(a=self.sqrt_alphas_cumprod, t=t, x_shape=x_start.shape) * x_start + 
                extract_into_tensor(a=self.sqrt_one_minus_cumprod, t=t, x_shape=x_start.shape) * noise)
    
    def get_loss(self, pred, target, mean=True):

        """ 
        Compute loss between predictions and targets.

        Args:
            pred (Tensor): Model predictions 
            target (Tensor): Ground truth targets 
            mean (bool): Whether to average the loss

        Returns:
            Tensor: Computed loss 
        """

        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()

        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')

        else:
            raise NotImplementedError("unknown loss type '{loss_type}' ")
        
        return loss 

    

    def p_losses(self, x_start, t, noise=None):

        """
        Compute loss for given x_start and timestep t.

        Args:
            x_start (Tensor): Original clean samples 
            t (Tensor): Timesteps 
            noise (Tensor): Optional pre-generated noise

        Returns:
            tuple: (total_loss, loss_dict) containing the combination loss and individual loss term 
        """

        noise = default(val=noise, d=lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise 

        elif self.parameterization == "x0":
            target = x_start

        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported.")
        
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({
            f'{log_prefix}/loss_simple': loss.mean()
        })
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weight[t] * loss).mean() # ------------------->
        loss_dict.update({
            f'{log_prefix}/loss_vlb': loss_vlb
        })

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({
            f'{log_prefix}/loss': loss 
        })

        return loss, loss_dict




    def forward(self, x, *args, **kwargs):

        """ 
        Forward pass for training.

        Args:
            x (Tensor): Input samples 
            *args: Additional positional arguments
            **kwargs: Aditional keyward arguments

        Returns:
            tuple : (loss, loss_dict) from p_losses 
        """

        # B, C, H, W, device, image_size = *x.shape, x.device, self.image_size 
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)
    

    def get_input(self, batch, k):

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(momery_format=torch.contiguous_format).float()
        return x 
    

    def shared_step(self, batch):

        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict
    

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss 
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)

        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}

        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)











    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise 
        )
    



    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    




    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        elif self.parameterization == "x0":
            x_recon = model_out

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device 
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t==0 
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise 
    



    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device 

        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling t", total=self.num_timesteps):
            img = self.p_sample(img, t=torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        
        return img 
    


            


    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):

        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)
    


    def _get_row_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoised_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoised_grid = rearrange(denoised_grid, 'b n c h w -> (b n) c h w')
        denoised_grid = make_grid(denoised_grid, nrow=n_imgs_per_row)
        return denoised_grid
    
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_key=None, **kwargs):
        log = dict()

        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x 

        # get diffusion row 
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start,
                                        t=t,
                                        noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_row_from_list(diffusion_row)

        if sample:
            # get denoised row 
            with self.ema_scope("Plotting"):
                samples, denoised_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["diffusion_row"] = self._get_row_from_list(denoised_row)


        



        if return_key:
            if np.intersect1d(list(log.keys()), return_key).shape[0] == 0:
                return log 
            
            else:
                return {key: log[key] for key in return_key}
            
        return log 
    

    def configure_optimizers(self):
        lr = self.learning_rate 
        params = list(self.model.parameters())

        if self.learn_logvar:
            params = params + [self.logvar]


        opt = torch.optim.AdamW(params, lr=lr)
        return opt


    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, 'split_input_params'):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]
                stride = self.split_input_params["stride"]
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]

                bs, nc, h, w = x.shape 
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(stride[1], w)) 
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fol


    







class DiffusionWrapper(nn.Module):

    """ 
    A Wrapper class for diffusion models that handles differenet types of conditioning.
    This class provides a unified interface for various conditioning approaches (concat, cross-attention, etc.)

    used in differenet models like Stable Diffusion.

    Args:
        diff_model_config (dict): Configuration dictionary for the underlying diffusion model.
                                    This is typically passed to `instantiate_from_config` to create the model.
        conditioning_key (str): Type of conditioning to use. Must be one of:
                                    - None: No conditioning 
                                    - 'concat': Concatenate conditioning to input 
                                    - 'crossattn': Use cross-attention conditioning 
                                    - 'hybrid': Combine concat and cross-attention
                                    - 'adm': use adaptive conditioning (e.g.,  for class labels)

        Example:
            >>> # Example 1: Cross-attention conditioning
            >>> config = {
            ...     'target': 'my_diffusion_package.DiffusionModel',
            ...     'params': {'in_channels': 4, 'context_dim': 768}
            ... }
            >>> wrapper = DiffusionWrapper(config, 'crossattn')
            >>> x = torch.randn(2, 4, 32, 32)  # Latent representation
            >>> t = torch.randint(0, 1000, (2,))  # Timesteps
            >>> context = [torch.randn(2, 10, 768)]  # Text embeddings
            >>> out = wrapper(x, t, c_crossattn=context)

            >>> # Example 2: Concatenation conditioning
            >>> config = {
            ...     'target': 'my_diffusion_package.DiffusionModel',
            ...     'params': {'in_channels': 7}  # 4 (x) + 3 (concat)
            ... }
            >>> wrapper = DiffusionWrapper(config, 'concat')
            >>> cond = [torch.randn(2, 3, 32, 32)]  # Low-res image to concatenate
            >>> out = wrapper(x, t, c_concat=cond)
    """

    def __init__(self, 
                 diff_model_config,
                 conditioning_key):
        
        """ 
        Initialize the DiffusionWrapper with configuration and conditioning type.

        Args:
            diff_model_config: Configuration for the base diffusion model 
            conditioning_key: Type of conditioning to apply
        """
        super().__init__()

        # Instantiate the underlying diffusion model from config
        self.diffusion_model = instantiate_from_config(diff_model_config)

        # set and validate conditioning type 
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, 
                x, 
                t,
                c_concat: list = None,
                c_crossattn: list=None):
        
        """ 
        Forward pass with optional conditioning.

        Args:
            x (Tensor): Input tensor (typically latent representation)
            t (Tensor): Diffusion timesteps 
            c_concat (list): List of tensors to concatenate (for 'concat'/'hybrid')
            c_crossattn (list): List of context tensors (for 'crossattn'/'hybrid'/'adm')

        Returns:
            Tensor: Output of the diffusion model
        """

        # No conditioning case 
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)

        # Concatenation conditioning (e.g., for low-res image conditioning)
        elif self.conditioning_key == 'concat':
            # concat input with all conditioning tensors along channel dim
            xc = torch.cat([x] + c_concat, dim=1) 
            out = self.diffusion_model(xc, t)

        # Cross-attention conditioning (e.g. for text embeddings)
        elif self.conditioning_key == "crossattn":
            # Concat multiple context sources if needed 
            cc = torch.cat(c_crossattn, 1) if len(c_crossattn) > 1 else c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)

        # Hybrid conditioning (both concat and cross-attention)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1) if len(c_crossattn) > 1 else c_crossattn[0]
            out = self.diffusion_model(xc,  t, context=cc)

        # Additive conditioning (e.g. for class labels)
        elif self.conditioning_key == "adm":
            # use first (and typically only) cross-attention input
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)

        else:
            raise NotImplementedError(f"Conditioning type {self.conditioning_key} not implemented.")
        
        return out 
    










    
    


    
    

if __name__ == "__main__":
    
    import torch 
    import torch.nn as nn 
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    
    from ImageDataset import ImageDataset
    from vae.utils.utils import load_config
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning import Trainer
    


    # Prepare Dataset 
    config = "E:\\YouTube\\stable-diffusion\ldm\\config.yaml"
    config = load_config(config_path=config)
    # print("COnfig: ", config)
    
    train_dataset = ImageDataset(root_dir="E:\\YouTube\\stable-diffusion\\dataset\\cat_dog_images",
                                split="train",
                                image_size=256)
    

    val_dataset = ImageDataset(root_dir="E:\\YouTube\\stable-diffusion\\dataset\\cat_dog_images",
                                split="val",
                                image_size=256)
    
    train_datloader = DataLoader(dataset=train_dataset,
                               batch_size=1,
                               shuffle=True)
    
    val_datloader = DataLoader(dataset=val_dataset,
                               batch_size=1,
                               shuffle=True)


    # Initialize model 
    model = DDPM(
                unet_config=config['model']['params']['unet_config']
                            )
    model = model.to("cuda")
    print(model)
    
    callbacks = [
        ModelCheckpoint(
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            dirpath="./checkpoints",
            filename='vae-{epoch:02d}-{val_loss:.2f}',
            save_last=True
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]


    # trainer configuration 
    trainer = Trainer(
        max_epochs=1,
        callbacks=callbacks,
        devices=1 if torch.cuda.is_available() else None,  # More flexible device handling
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision=32,  # Can use '16-mixed' for FP16
        accumulate_grad_batches=1
    )

    # Training 
    trainer.fit(model, train_datloader, val_datloader)
    print("Training completed!")
    
    
    # python -m ldm.models.diffusion.test