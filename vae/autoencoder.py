import torch 
import pytorch_lightning as pl 
import torch.nn.functional as F
import torch.torch_version 
from vae.unet import Encoder, Decoder
import importlib
from vae.distribution import DiagonalGaussianDistribution
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vae.utils.utils import instantiate_from_config
from torch import nn 
from vae.utils.utils import load_config

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer

class AutoencoderKL(pl.LightningModule):

    """ 
    A Variational Autoencoder (VAE) with KL divergance regularization and optional adversarial training.

    Args:
        ddconfig (dict): Configuration dictionary for encoder/decoder 
        emb_dim (int): Dimension of the latent embedding space 
        loss_config (dict): Configuration for the loss function 
        ckpt_path (str, optional): Path to checkpoint for loading pretrained weights 
        ignore_keys (list, optional): Keys to ignore when loading from checkpoint
        image_key (str): Key for accessing images in the batch dictionary 
        colorize_nlabels (int, optional): Number of labels for segmentation colorization
        monitor (str, optional): Metric to monitor for checkpointing
    """



    def __init__(self,
                 ddconfig,
                 emb_dim,
                 loss_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None):
        

        super().__init__()
        self.image_key = image_key
        self.learning_rate = 4.5e-6  # Default learning rate 
        self.automatic_optimization = False 

        # encoder network
        self.encoder = Encoder(**ddconfig)
        # decoder network
        self.decoder = Decoder(**ddconfig)
        # Loss function (typically LPIPSWithDiscriminator)
        self.loss = instantiate_from_config(loss_config)

        # Verify double_z is enable in config 
        assert ddconfig["double_z"], "Config must have double_z=True for variance output"

        # Projects encoder output to latent space 
        self.quant_conv = torch.nn.Conv2d(in_channels=2*ddconfig["z_channels"],  # Far mean and variance
                                          out_channels=2*emb_dim,               # Project to embedding dim
                                          kernel_size=1)
        
        # Projects latent back to decoder input 
        self.post_quant_conv = torch.nn.Conv2d(in_channels=emb_dim,
                                               out_channels=ddconfig["z_channels"],
                                               kernel_size=1)
        

        


    

    def encode(self, x):
        """Encode input into latent distribution parameters"""
        h = self.encoder(x)  # get encoder features
        moments = self.quant_conv(h)  # Projects to latent space parameters
        posterior = DiagonalGaussianDistribution(moments)   # Create distribution
        return posterior
    

    def decode(self, z):
        """Decode latent samples into reconstructions"""
        z = self.post_quant_conv(z)   # Project latent to decoder input dims
        dec = self.decoder(z)       # decode to image space 
        return dec 


    def forward(self, input, sample_posterior=True):
        """Full forward pass with optional sampling"""
        posterior = self.encode(input)   # Get latent distribution

        # sample from posterior or take mode (deterministic)
        z = posterior.sample() if sample_posterior else posterior.mode()

        dec = self.decode(z)
        return dec, posterior
    

    def get_input(self, batch, k):

        x = batch[k]
        
        if x.ndim == 4 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        x = x.to(memory_format=torch.contiguous_format).float()

        return x 
    

    def training_step(self, batch, batch_idx):


        # get optimizers 
        opt_ae, opt_disc = self.optimizers()

        # x = batch['image']
        x = self.get_input(batch, self.image_key)
        
        # Foward pass - returns continous latents 
        h = self.encode(x)

        # Decode with quantization (normal training mode)
        reconstructions = self.decode(h)


        # Get quantization (normal training mode)
        if isinstance(h, tuple):
            h = h[0]
        _, codebook_loss, _ = self.quantize(h)

        # calculate loss 
        aeloss, log_dict_ae = self.loss(
            codebook_loss = codebook_loss,
            inputs=x,
            reconstructions=reconstructions,
            optimizer_idx=1,
            global_step=10,
            split="train",
            last_layer = self.get_last_layer()
        )

        # Generator update 
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # Discriminator update (if using adversarial loss)
        discloss, log_dict_disc = self.loss(
            codebook_loss=codebook_loss,
            inputs=x,
            reconstructions=reconstructions,
            optimizer_idx=1,
            global_step=1,
            split="train",
            last_layer = self.get_last_layer()
        )

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        # Logging 
        self.log_dict(log_dict_ae, prog_bar=False)
        self.log_dict(log_dict_disc, prog_bar=False)

        return aeloss + discloss
    

    def validation_step(self, batch, batch_idx):

        x = self.get_input(batch, self.image_key)
        print("data:", x)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer = self.get_last_layer(),
                                        split="val",
                                        )
        
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val",
                                            )
        
        self.log(f"val/rec_loss", log_dict_ae[f"val/rec_loss"],
             prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log(f"val/aeloss", aeloss,
                prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # Compute and log total loss 
        total_loss = aeloss + discloss
        self.log(f"val/total_loss", total_loss, 
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return log_dict_ae

    def configure_optimizers(self):
        # Create optimizers
        lr = self.hparams.get('lr', 4.5e-6)
        opt_ae = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.quant_conv.parameters()) + 
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        
        opt_disc = optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        
        # # Create scheduler
        # scheduler_ae = {
        #     'scheduler': ReduceLROnPlateau(opt_ae, mode='min', factor=0.5, patience=5),
        #     'monitor': 'val/rec_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }
        
        return [opt_ae, opt_disc], []
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):

        assert isinstance(batch, (dict, torch.Tensor)), "Batch must be dict or tensor"
        log = dict()

        x = batch[self.image_key] if isinstance(batch, dict) else batch 
        x = x.to(self.device)

        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection 
                assert xrec.shape[1] > 3 
                x = self.to_rgb(xrec)

            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec 

        log["inputs"] = x 
        return log 
    
    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))

        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min()) / (x.max()-x.min()) -1.
        return x 
    

    


class IdentityFirstStage(torch.nn.Module):

    def __init__(self, *args, vq_interface=False, **kwargs):
        self.q_interface = vq_interface  # TODO: should be true by default but check to not break older stuff 
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x 
    
    def decode(self, x, *args, **kwargs):
        return x 
    
    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x 
    

    def forward(self, x, *args, **kwargs):
        return x
    

    