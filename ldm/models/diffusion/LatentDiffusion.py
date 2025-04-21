
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


class LatentDiffusion(pl.LightningModule):

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args,
                 **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        # print("kwargs: -------------->", kwargs['model']['params']['timesteps'])
        assert self.num_timesteps_cond <= kwargs['timesteps']

        # For backward compatibility after implementation of DiffusionWrapper 
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None 

        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])

        super().__init__(*args, **kwargs)
        # super().__init__()
        
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key 

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) -1
        except:
            self.num_downs = 0 

        if not scale_by_std:
            self.scale_factor = scale_factor

        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_first_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False 
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False 
        if ckpt_path is not None:
            pass 






    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disable_train 
        for param in self.first_stage_model.parameters():
            param.requires_grad = False 


    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), 
                                   fill_value=self.num_timesteps -1,
                                   dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps -1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids 



    def register_schedule(self, given_bitas=None, beta_scheduler="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_bitas, beta_scheduler, timesteps, linear_start, linear_end, cosine_s)
        
        self.shorten_cond_schedule = self.num_timesteps_cond > 1 
        if self.shorten_cond_schedule:
            self.make_cond_schedule()



    
    def get_learned_conditioning(self, c):

        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_forward, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()

            else:
                c = self.cond_stage_model(c)


        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c 


    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

            if self.shorten_cond_schedule:  # TODO: drop this option 
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c,
                                  t=tc,
                                  noise=torch.rand_like(c.float()))
                
        return self.p_losses(x, c, t, *args, **kwargs)
    
    def meshgrid(self, h, w):
        
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr 
    
    def delta_border(self, h, w):
        """ 
            Compute normalized distance to the nearest image border for each pixel.

            Create a gradient map where:
            - Border pixels have value 0 (min distance)
            - Center pixels have value 0.5 (max distance)
            - Value increase smoothly from border to center 

            Args:
                h (int): Height of the image/feature map 
                w (int): Width of the image/feature map 

            Returns:
                torch.Tensor: A 2D tensor of shape (h, w) containing normalized distances 
                            to the nearest border, ranging from 0 (border) to 0.5 (center)

            Note:
                The 0.5 maximum value means the distance is normalized such that the center 
                of the image is considered exactly halfway between any two opposite borders.
        """

        # Create a tensor containing the coordinate of the lower-right corner 
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)

        # Generate normalized cordinates using meshgrid (assumed to produce a (h, w, 2) tensor.)
        arr = self.meshgrid(h, w) / lower_right_corner

        # compute minimum distance to lef or top border and right and bottom border  
        dist_left_up = torch.min(arr, dim=-1, keepdim=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdim=True)[0]

        # Combines both distances measures and takes the minimum
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)

        return edge_dist


    


    def get_weighting(self, h, w, Ly, Lx, device):

        """ 
            Compute a weighting matrix for handling overlapping regions in image patches.

            This method creates a weighting matrix that helps in properly normalizing overlapping 
            regions when reconstructing an image from its patches. It can apply different weights 
            to border regions and includes an optional tie-breaker machanism for patch boundaries.

            Args:
                h (int): Height of the patches 
                w (int): Width of the patches 
                Ly (int): Number of patches in vertical direction 
                Lx (int): Number of patches in horizontal direrction 

            Returns:
                torch.Tensor: Weighting tensor of shape (1, h*w, Ly*Lx) that contains:
                    - Higher weighs for central pixels 
                    - Lower weights for border pixels 
                    - Optional tie-breaker weights for path bounderies
            
            The method use configuration perameters from self.split_input_params:
                - clip_min_weight: Minimum clip value for main weights 
                - clip_max_weight: Maximum clip value for main weights 
                - tie_breaker: Whether to apply tie-breaker weights 
                - clip_min_tie_weight: Minimum clip value for tie-breaker weights 
                - clip_max_tie_weight: Maximum clip value for tie-breaker weights 
        """

        weighting = self.delta_border(h, w)

        # clip the weighting values between configured min/max bounds 
        # Prevents extreme values that could couse numerical instability 
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        
        # reshape and expand the weighting 
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        # check if tie-breaker weighting is enabled in config
        if self.split_input_params["tie_braker"]:

            # Compute tie-breaker weights using same delta_border method but for path grid 
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting, 
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])
            
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting

        return weighting

    
    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # TODO: load once not every time shorten code 

        """
        Compute fold and upfold operatations with normalization weighting for image processing.

        This function prepares fold and unfold operations for processing image patches (crops).
        along with normalization weights to handle overlapping regions when reconstructing the image.

        Args:
            x (torch.Tensor): Imput image tensor of shape (batch_size, channels, height, width)
            kernel_size (tuple): Size of the sliding blocks (patch_height, patch_width)
            stride (tuple): Stride of the sliding blocks (vertical_stride, horozontal_stride)
            uf (int, optional): Upscaling factor, Default to 1.
            df (int, optional): Downscaling factor. Default to 1.

        Returns:
            tuple: 
                - fold: torch.nn.Fold instance for reconstruction
                - unfold: torch.nn.Unfold instance for extracting patches 
                - normalization: Tensor for normalizing overlapping regions 
                - weighting: Weighting tensor used for normalization

        Note:
            Currently only suppors uf>1 with df=1 or both uf=1 and df=1 
            Other combinations raise NotImplementedError.
        """

        # Extract batch_size, number_of_channels, height and width from the input tensor 
        bs, nc, h, w = x.shape 

        # Calculate number of vertical/horizontal patches (crops) and fit in the image 
        Ly = (h - kernel_size[0]) // stride[0] + 1 
        Lx = (w - kernel_size[1]) // stride[1] + 1 


        # Case for no upscaling or downscaling 
        if uf == 1 and df == 1:

            # Create parameters for fold/unfold operations with given kernel and stride 
            fold_params = dict(kernel_size=kernel_size,
                               dilation=1,
                               padding=0,
                               stride=stride)
            
            # Create Unfold instance to extract sliding patches from input.
            unfold = torch.nn.Unfold(**fold_params)
            # Create fold instance to reconstruct image from patches 
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            # Get weighting matrix for normalization (method not shown in code)
            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            # Compute normalization factor by folding weights and reshaping 
            normalization = fold(weighting).view(1, 1, h, 1)  
            # Reshape weighting tensor for later use 
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        # Case for upscaling only
        elif uf > 1 and df == 1:

            # Parameters for unfold operation (original scale)
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            # create fold instance for upscaled reconstruction
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)


            # Get weighting matrix for upscaled size 
            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            # Compute normalization factor for upscaled case 
            normalization = fold(weighting).view(1, 1, h // df, w // df) 
            # Reshape upscale weighting tensor 
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))


        else:
            # Error for unsupported parameters combinations
            raise NotImplementedError
        
        # Return all computed operations and tensors
        return fold, unfold, normalization, weighting
    

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):

        """ 
            Hook that executes at the start of each training batch to potentially rescale inputs by their standard deviation.

            This method performs automatic std-based rescaling ONLY on the very first training batch 
            of the first epoch (epoch 0) when these conditions are met:

            - Scale_by_std is True 
            - It's the first batch (batch_idx == 0)
            - Training is starting fresh (not restored from checkpoint)
            - Global step is 0 

            When triggered: 
            1. Gets input batch and passes through first stage encoder 
            2. Calculates standard deviation of the encoding 
            3. Sets scale_factor to 1/std of the encodings 
            4. Register scale_factor as a buffer to persist across checkpoint saves 

            Decorates:
                @rank_zero_only: Ensure this only runs on rank 0 in distributed training 
                @torch.no_grad(): Disables gradient tracking for memory efficiency 

            Args:
                batch: Current training batch 
                batch_idx: Index of current batch 
                dataloader_idx: Index of current dataloader
        """

        print(f"Batch {batch_idx} and Batch: {batch}")
        
        # Compound condition checking if we should perform std rescaling.
        # flag indicating if std rescaling is enabled and only on first epoch 
        # only at very start of training and only on first batch
        # only if not resuming from checkpoint.
        if self.scale_by_std and self.current_epoch == 0 \
            and self.global_step == 0 and batch_idx == 0 \
            and not self.restarted_from_ckpt:

            # Safety check ensuring we are not already using custom rescaling 
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            print('### USING STD-RESCALING ###')

            # Get input data from batch using first stage key
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)

            # Encodes input through first stage (typically a VAE encoder)
            encoder_posterior = self.encode_first_stage(x)

            # Gets latent representation and detaches from computation graph
            z = self.get_first_stage_encoding(encoder_posterior).detach()

            # Remove existing scale_factor if present 
            del self.scale_factor

            # Buffer regestration ensures it's saved with checkpoint 
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")


    @torch.no_grad()
    def encode_first_stage(self, x):

        """ 
            Encodes input images through the first stage model (typically a VAE) with optional patch-based processing.

            this method handles two scenarios:
                1. When split_input_params are configured (patch_distributed_vq=True) 
                    - Processing image in overlapping patches 
                    - Encodes each path separately 
                    - Recombines patches with proper weighting 

                2. Normal case: Directly encodes full image 
                    
                Args:
                    x (torch.Tensor): Input image tensor of shape (B, C, H, W)

                Returns:
                    torch.Tensor: Encoded representation of same shape as first stage model output 

                Decorates:
                    @torch.no_grad(): Disables gradient computation from memory efficiency.
        """
        
        # check if patch-based processing is configured 
        if hasattr(self, "split_input_params"):
            # check if patch-based encoding is enabled 
            if self.split_input_params["patch_distributed_vq"]:

                # Get's kernel/patch size (eg. 128, 128)
                ks = self.split_input_params["ks"]  
                # Get stride between patches (eg. 64, 64)
                stride = self.split_input_params["stride"]
                # Get downscaling factor 
                df = self.split_input_params["vqf"]

                # Stores original image dimensions 
                self.split_input_params["original_image_size"] = x.shape[-2:] 
                # Extracts batch_size, channels, height, width
                bs, c, h, w = x.shape 

                # Adjust kernel_size if larger than input image 
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                # Adjust stride if larger than input image
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w)) 
                    print("reducing stride")

                # Get's patch processing operations 
                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                # Extract overlapping patches 
                z = unfold(x) # (bc, nc * prod(**ks), L)
                
                # Reshaping to image shape 
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1])) # (bn, nc, ks[0], ks[1], L)

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]
                

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape 
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # Stitch crops together 
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded
            

            else:
                return self.first_stage_model.encode(x)
            

        else:
            return self.first_stage_model.encode(x)
        

    def get_first_stage_encoding(self, encoder_posterior):

        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()

        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior

        else:
            raise NotImplementedError(f"encoder_posterior of type `{type(encoder_posterior)}` not yet implemted ")
        
        return self.scale_factor * z 
    

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):

        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()

            z = self.first_stage_key.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()


        z = 1. / self.scale_factor * z 

        if hasattr(self, 'split_input_params'):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"] 
                stride = self.split_input_params["stride"]
                uf = self.split_input_params["vqf"]

                bs, nc, h, w = z.shape 
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w)) 
                    print("reducing kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing Stride")


                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  

                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i], 
                                                                 force_not_quantize=predict_cids or force_not_quantize) 
                                    for i in range(z.shape[-1])]
                    

                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]
                    

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                o = o.view((o.shape[0], -1, o.shape[-1]))
                decoded = fold(o)
                decoded = decoded / normalization 
                return decoded
            
            else: 
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                
                else:
                    return self.first_stage_model.decode(z)
                

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            
            else:
                return self.first_stage_model.decode(z)

    

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        super().get_input(batch, k)


        print("Batch: --------->", batch)

        if bs is not None:
            x = x[:bs]

        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()


        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key

            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]

                elif cond_key == 'class_label':
                    xc = batch 

                else: 
                    xc = super().get_input(batch, cond_key).to(self.device)


            else:
                xc = x 

            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(x)

                else:
                    c = self.get_learned_conditioning(xc.to(self.device))


            else:
                c = xc 

            if bs is not None:
                c = c[:bs]


            if self.use_positional_encoding:
                pos_x = pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 
                     'pos_x': pos_x,
                     'pos_y': pos_y
                     }
                


        else:
            c = None 
            xc = None 
            if self.use_positional_encoding:
                pos_x, pos_y = self.compute_latent_shift(batch)
                c  = {'pos_x': pos_x,
                      'pos_y': pos_y}
                
            
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])

        if return_original_cond:
            out.append(xc)

        return out 
    

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, 'colorize'):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)

        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1 
        return x 
    
    @torch.no_grad()
    def sample_log(self, 
                   cond, 
                   batch_size, 
                   ddim,
                   ddim_steps,
                   **kwargs):
        
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, 
                                                         batch_size,
                                                         shape,
                                                         cond,
                                                         verbose=False,
                                                         **kwargs)
            

        else:
            samples, intermediates = self.sample(cond=cond,
                                                 batch_size=batch_size,
                                                 return_intermediates=True,
                                                 **kwargs)
            

        return samples, intermediates
    
    def _get_denoise_row_from_list(self,
                                   samples,
                                   desc='',
                                   force_no_decoder_quantization=False):
        
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                       force_not_quantize=force_no_decoder_quantization))
            
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W 
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
    

    @torch.no_grad()
    def progressive_denoising(self,
                              cond,
                              shape,
                              verbose=True,
                              callback=None,
                              quantize_denoise=False,
                              img_callback=None,
                              mask=None,
                              x0=None,
                              temperture=1.,
                              noise_dropout=0.,
                              score_corrector=None,
                              corrector_kwargs=None,
                              batch_size=None,
                              X_T=None,
                              start_T=None,
                              log_every_t=None):
        
        if not log_every_t:
            log_every_t = self.log_every_t

        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)

        else:
            b = batch_size = shape[0]

        if X_T is None:
            img = torch.randn(shape, device=self.device)

        else:
            img = X_T

        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else 
                        list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
                

            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]


        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc="Progressive Generation",
                        total=timesteps) if verbose else reversed(
                            range(0, timesteps)
                        )
        
        if type(temperture) == float:
            temperture = [temperture] * timesteps


        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond,
                                     t=tc,
                                     noise=torch.rand_like)
                
            
            img, x0_partial = self.p_sample(img, 
                                            cond, 
                                            ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoise=quantize_denoise,
                                            return_x0=True,
                                            temperture=temperture[i],
                                            noise_dropout=noise_dropout,
                                            score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs
                                            )
            
            if mask is not None:
                assert x0 is not None 
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img 

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)

            if callback: callback(i)
            if img_callback: img_callback(img, i)

        return img, intermediates


    @torch.no_grad()
    def log_images(self,
                   batch,
                   N=8,
                   n_row=4,
                   sample=True,
                   ddim_steps=200,
                   ddim_eta=1.,
                   return_keys=None,
                   quantize_denoised=True,
                   inpaint=True,
                   plot_denoise_rows=False,
                   plot_progressive_rows=True,
                   plot_diffusion_rows=True,
                   **kwargs):
        
        use_ddim = ddim_steps is not None 

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, 
                                           self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x 
        log["reconstruction"] = xrec

        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc 

            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img(wh=(x.shape[2], x.shape[3]), xc=batch["caption"])
                log["conditioning"] = xc 

            elif self.cond_stage_key == "class_label":
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log["conditioning"] = xc 

            elif isimage(xc):
                log["conditioning"] = xc 

            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)


        if plot_diffusion_rows:
            # get diffusion row 
            diffusion_row = list()
            z_start = z[:n_row]

            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))


            diffusion_row = torch.stack(diffusion_row) # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid


        if sample:

            # get denoise row 
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,
                                                         batch_size=N,
                                                         ddim=use_ddim,
                                                         ddim_steps=ddim_steps,
                                                         eta=ddim_eta)
                
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                self.first_stage_model, IdentityFirstStage
            ):
                
                # also display when quantizing x0 while sampling 
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    

                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples


            if inpaint:
                # make a simple center square 
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)

                # zero will be filled in 
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0. 
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,
                                                 batch_size=N,
                                                 ddim=use_ddim,
                                                 eta=ddim_eta,
                                                 ddim_steps=ddim_steps,
                                                 x0=z[:N],
                                                 mask=mask)
                    
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask 

                # outpaint 
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                    
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples


        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
                
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row 

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log 
            
            else:
                return {key: log[key] for key in return_keys}
            
        return log 
    


    def configure_optimizers(self):
        lr = self.learning_rate 
        params = list(self.model.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())

        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config 
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]

            return [opt], scheduler
        
        return opt 
    


    