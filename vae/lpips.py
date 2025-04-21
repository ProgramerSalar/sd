import torch.nn as nn
import torch 
from torchvision import models
from collections import namedtuple
import os, hashlib, requests
from tqdm import tqdm
from vae.utils.utils import get_ckpt_path



class ScalingLayer(nn.Module):

    """ 
    A normalization layer to scale and shift the input tensor 
    to match the statistics of pretrained VGG networks.
    """

    def __init__(self):
        super().__init__()
        # Shift tensor used for normalization
        self.register_buffer('shift',
                             torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        # Scale tensor used for normalization
        self.register_buffer('scale', 
                             torch.Tensor([.458, .448, .450])[None, :, None, None])
        
    def forward(self, inp):
        """Normalize the input tensor by subtracting the shift and dividing by scale."""
        return (inp - self.shift) / self.scale 

class Vgg16(torch.nn.Module):
    
    """A model that extracts features from different layers of a pretrained VGG model."""

    def __init__(self, 
                 requires_grad=False,
                 pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features

        # Defining different slices of the VGG16 model to extract features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
    

        for x in range(4):
            self.slice1.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(4, 9):
            self.slice2.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(9, 16):
            self.slice3.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(16, 23):
            self.slice4.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        for x in range(23, 30):
            self.slice5.add_module(name=str(x),
                                   module=vgg_pretrained_features[x])
            
        # Freeze parameters if requires_grad is False 
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False




    def forward(self, x):

        """Passes the input through different slices and collects featues map """

        h = self.slice1(x)
        print(f"Slice 1 output shape: {h.shape}")  # Should be [B, 64, H, W]
        h_relu1_2 = h 

        h = self.slice2(h)
        print(f"Slice 2 output shape: {h.shape}")  # Should be [B, 128, H/2, W/2]
        h_relu2_2 = h 

        h = self.slice3(h)
        print(f"Slice 3 output shape: {h.shape}")  # Should be [B, 256, H/4, W/4]
        h_relu3_3 = h 

        h = self.slice4(h)
        print(f"Slice 4 output shape: {h.shape}")  # Should be [B, 512, H/8, W/8]
        h_relu4_3 = h 

        h = self.slice5(h)
        print(f"Slice 5 output shape: {h.shape}")  # Should be [B, 512, H/16, W/16]
        h_relu5_3 = h 

        vgg_outputs = namedtuple(typename="VggOutputs",
                                 field_names=['relu1_2', 'relu2_2', 'relu3_3', 'relu_4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out
    

class NetLinLayer(nn.Module):
    """A single linear layer performing 1x1 convolution for features transformation"""

    def __init__(self, 
                 chn_in,
                 chn_out=1,
                 use_dropout=False
                 ):
        super().__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
        
def normalize_tensor(x, eps=1e-10):
        
        """ 
        Normalize a tensor along channel dimension.
        """

        norm_factor = torch.sqrt(torch.sum(x **2, dim=1, keepdim=True))
        return x / (norm_factor+eps)
    
def spatial_average(x, keepdim=True):
    """ 
    Computes spatial mean over height and width dimensions.
    """
    return x.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):

    """ 
    Learned Perceptual Image Path Similarity (LPIPS) model,
    used for comparing perceptual differences between images.
    """

    def __init__(self, 
                 use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512] 

        # Initializing layers for features comparison
        self.net = Vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(chn_in=self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(chn_in=self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(chn_in=self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(chn_in=self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(chn_in=self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()

        # Freeze model parameters 
        for param in self.parameters():
            param.requires_grad = False


    def load_from_pretrained(self, name="vgg_lpips"):

        """
        Load pretrained weight for LPIPS computation.
        """
        ckpt = get_ckpt_path(name, root="lpips_weight")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print(f"loaded pretrained LPIPS loss from {ckpt}")

    
    def forward(self, input, target):
        """ 
        Computes LPIPS similarity between input and target images.
        """

        
    
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}

       

        # Compute features differences
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2 


        # Compute perceptual similarity score 
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val 
    


