import torch 
import torch.nn as nn 
from vae.utils.utils import instantiate_from_config
from .utils import exists, count_params, default, noise_like, extract_into_tensor,  make_beta_schedule
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
import os 


# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        

        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            print("diffusion_model : ", self.diffusion_model)
            print(f"class DiffusionWrapper what is the shape of the data: {x.shape}")
            out = self.diffusion_model(x, t)



        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out

# Alternative manual gradient checkpointing implementation
from torch.utils.checkpoint import checkpoint
from ldm.models.openaimodel import UNetModel

class MemoryOptimizedUNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.use_checkpoint = True
        
        # Convert model to half precision but keep normalization in float32
        self.half()
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
                m.float()
    
    def forward(self, x, t, **kwargs):
        # Ensure input dtype matches model dtype
        input_dtype = x.dtype
        model_dtype = next(self.parameters()).dtype
        
        if input_dtype != model_dtype:
            x = x.to(model_dtype)
        
        if self.use_checkpoint and self.training:
            def create_custom_forward():
                def custom_forward(*inputs):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        return self._forward(*inputs)
                return custom_forward
            
            out = checkpoint(create_custom_forward(), x, t, **kwargs, 
                           use_reentrant=False,
                           preserve_rng_state=False)
        else:
            out = self._forward(x, t, **kwargs)
        
        return out.to(input_dtype)
    
    
    



class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
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
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()

        base_model = instantiate_from_config(unet_config)
        self.model = DiffusionWrapper(MemoryOptimizedUNet(base_model), conditioning_key)
        
        # Configure automatic mixed precision
        self.automatic_optimization = False  # Manual optimization for better control



        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.learning_rate = 2.0e-06
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # Add these memory optimizations
        self.model.half()  # Convert model to half precision
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.float()  # Keep normalization in float32


        # if hasattr(self.model.diffusion_model, 'enable_gradient_checkpointing'):
        #     self.model.diffusion_model.enable_gradient_checkpointing()

        # else:
        #     print("Warning: Gradient checkpointing not available for this model.")

        # print(f"check the diffusion model have found the checkpoint gradiant: {self.model.diffusion_model}")


        # # Enable gradient checkpointing to save memory
        # self.model.diffusion_model.enable_gradient_checkpointing()






    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

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

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

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
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
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
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        print(f"function p loss of check the data shape: {x_start.shape}")
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        print(f"function p loss of check the data shape: {x_noisy.shape}")
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        print(f"class p_loss in DDPM class {x.shape}")
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        
        for i in batch['image']:
            print(f"function of get_input data shape: {i.shape}")


        x = batch[k]

        if x.ndim == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        

        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        print(f"function shared_step what is the shape of data: {x.shape}")

        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        # Manual optimization with mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, loss_dict = self.shared_step(batch)
        
        # Manual backward and optimization
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        self.log_dict(loss_dict, prog_bar=True)
        return loss

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # print("function of validation_step: ", next(iter(batch['image'].shape)))
        for i in batch['image']:
            print("image of image", i.shape)
        
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
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
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        # Ensure optimizer works with mixed precision
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    


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
    

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    


    # Prepare Dataset 
    config = "E:\\YouTube\\Git\ldm\\config.yaml"
    # config = "/content/sd/ldm/config.yaml"
    config = load_config(config_path=config)
    # root_dir = "/content/sd/dataset/cat_dog_images"
    root_dir = "E:\\YouTube\\Git\\dataset\\cat_dog_images"
    # print("COnfig: ", config)
    

    train_dataset = ImageDataset(
                                root_dir=root_dir,
                                split="train",
                                image_size=256,
                                )
    

    val_dataset = ImageDataset(root_dir=root_dir,
                                split="val",
                                image_size=256)
    
    train_datloader = DataLoader(dataset=train_dataset,
                               batch_size=1,
                               shuffle=True,
                                pin_memory=True,
                                num_workers=2,
                                persistent_workers=True,
                                prefetch_factor=2)
    
    val_datloader = DataLoader(dataset=val_dataset,
                               batch_size=1,
                               shuffle=True,
                                pin_memory=True,
                                num_workers=2,
                                persistent_workers=True,
                                prefetch_factor=2)


    # Initialize model 
    model = DDPM(
                unet_config=config['model']['params']['unet_config'],
                use_ema=False 
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
        precision="16-mixed",  # Can use '16-mixed' for FP16
        accumulate_grad_batches=1,
        # pin_memory=True,
        # num_workers=4,
        # persistent_workers=True
        # gradient_clip_val=0.5,  # Reduced from 1.0
        # accumulate_grad_batches=4,  # Increased accumulation
        # enable_progress_bar=True,
        enable_model_summary=False,
        limit_train_batches=0.1,  # Train on 10% of data initially
        limit_val_batches=0.1,    # Validate on 10% of data initially
        deterministic=True,       # For reproducibility
        # amp_backend='native',
        strategy='ddp_find_unused_parameters_false'  # More efficient distributed training
    )

    # # Add memory management at the start of training
    # torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    # torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention if available

    # # Training 
    # trainer.fit(model, train_datloader, val_datloader)
    # print("Training completed!")

    # Add memory monitoring
    def print_memory():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    print("Memory before training:")
    print_memory()
    
    try:
        trainer.fit(model, train_datloader, val_datloader)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("Out of memory error caught! Trying recovery...")
            torch.cuda.empty_cache()
            gc.collect()
            # Reduce memory usage and try again
            model.half()
            trainer.fit(model, train_datloader, val_datloader)
        else:
            raise
    
    print("Training completed!")
    print("Final memory stats:")
    print_memory()
    
    
    # python -m ldm.models.diffusion.test