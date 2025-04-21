import functools
import torch.nn as nn 
import torch 



class ActNorm(nn.Module):

    """ 
    Activation Normalization (ActNorm) layer for normalizing activations in flow-based models.
    This implementation ensures `numerical stability` and `correct gradients` for flow-based generative
    models.

    Args:
        num_features (int): Number of channels in the input.
        logdet (bool, optional): Whether to compute log determinant for flow models. default: False 
        affine (bool, optional): Whether to use learnable scale and shift. Default: True.
        allow_reverse_init (bool, optional): Allow initialization in reverse pass. Default: False.

    Note:
        - Used in Glow and other normalizing flow arch.
        - Initializes scale/shift using data statistics in the first forward pass.
    """


    def __init__(self, 
                 num_features,
                 logdet=False,
                 affine=True,
                 allow_reverse_init=False):
        
        # ActNorm requires affine=True (learnable parameters)
        assert affine
        super().__init__()
        self.logdet = logdet

        # Learnable parameters (initialized as zeros/ones)
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))   # Shift parameter
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))    # Scale parameter
        self.allow_reverse_init = allow_reverse_init


        # Tracks whether initialization has occured 
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))


    def initialize(self, input):

        """Initialize scale and shift using input statistics."""

        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)

            # Compute mean and std per chanenls 
            mean = (
                flatten.mean(1)
                .unsqueeze(1).unsqueeze(2).unsqueeze(3)   # Reshape to [1, C, 1, 1]
                .permute(1, 0, 2, 3)
            )

            std = (
                flatten.std(1)
                .unsqueeze(1).unsqueeze(2).unsqueeze(3)     # Reshape to [1, C, 1, 1]
                .permute(1, 0, 2, 3)
            )

            # Initialize parameters
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))   # Scale = 1/std (prevent div-by-zero)


    def forward(self, input, reverse=False):

        """Forward pass (normalization) or reverse pass (denormalization)."""

        if reverse:
            return self.reverse(output=input)
        
        # Handle 2D input (eg. [B, C] -> [B, C, 1, 1])
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True 
        else:
            squeeze = False 
        _, _, height, width = input.shape 


        # Initialize on first forward pass in training
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        # Apply normalization: h = scale * (x + loc)
        h = self.scale * (input + self.loc)

        # Revert squeezing if needed 
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)


        # Compute log determinant for flows 
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)   # Sum over channels
            logdet = logdet * torch.ones(input.shape[0]).to(input)  # Per-sample logdet 
            return h, logdet
        
        return h 
    

    def reverse(self, output):

        """Reverse pass: denormalize the output. """


        # Initialize if needed (with safety check)
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm is reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        # Handle 2D inputs
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True 
        else:
            squeeze = False 

        # Denormalize: x = (h / scale) - loc 
        h = output / self.scale - self.loc 

        if squeeze:
            h = h.squeeze(-1).squeeze(-1) 
        return h 
    

def weight_init(m):

    """ 
    Initializes neural network weights for `conv` and `batch normalization layers`.

    Applies custom normal distribution initialization to:
    - Conv layers: weight ~ N(0.1, 0.02)
    - BatchNorm layers: weight ~ N(1.0, 0.02), biases = 0

    Args:
        m (nn.Module): A PyTorch module (layer) to initialize

    Example: 
        >>> net = nn.Sequential(nn.Conv2d(3, 64, 3), nn.BatchNorm2d(64))
        >>> net.apply(weight_init) # Applies initialization to all layers
    """

    # Get the class name of the module
    classname = m.__class__.__name__ 

    # Initialize conv layers
    if classname.find('Conv') != -1:  # Checks if `Conv` is in the class name 
        nn.init.normal_(m.weight.data, 0.1, 0.02)   # weights ~ N(0.1, 0.02)

        # Note: Typically no bias init needed for Conv when followed by BatchNorm
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)  # Initialize bias to 0 if exists

    # Initialize batch normalization layers
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)   # weights ~ N(1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)   # biases = 0
       



class NLayerDiscriminator(nn.Module):
    """ 
    A PatchGAN discriminator with configurable depth, as used in Pix2Pix/CycleGAN.
        
    Args:
        input_nc (int): Number of input channels (eg. 3 for RGB images).
        ndf (int): Number of filters in the first conv layer.
        n_layers (int): Number of intermediate conv layer (default=3).
        use_actnorm (bool): Whether to use ActNorm instead of BatchNorm (default=False).

    Attributes:
        main (nn.Sequential): The sequential model containing all layers.
    """

    def __init__(self, 
                 input_nc=3, 
                 ndf=64, 
                 n_layers=3, 
                 use_actnorm=False):
        super().__init__()

        # Choose normalization layer
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d  # Default: Batch Normalization
        else:
            norm_layer = ActNorm  # Alternative: Activation Normalization (for flow-based model)

        # Determine if conv layers should use bias terms 
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func != nn.BatchNorm2d)  # No bias for BatchNorm

        else:
            use_bias = (norm_layer != nn.BatchNorm2d)


        # Kernel configuration 
        kw = 4   # kernel size 
        padw = 1  # padding to maintain spatial dim after stride-2 conv

        # Initialize downsampling layer 
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        

        # Gradually increase filters count (up to 8x ndf)
        nf_mult = 1 
        nf_mult_prev = 1 
        for n in range(1, n_layers):  # gradually increase the number of filters 
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)    # Cap at 8x ndf (eg. 64*8=512)

            
            sequence += [
                nn.Conv2d(in_channels=ndf * nf_mult_prev,
                          out_channels=ndf * nf_mult,
                          kernel_size=kw,
                          stride=2,
                          padding=padw,
                          bias=use_bias),
                norm_layer(num_features=ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final downsampling (stride=1)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult_prev,
                      out_channels=ndf * nf_mult,
                      kernel_size=kw,
                      stride=1,
                      padding=padw,
                      bias=use_bias),
            norm_layer(num_features=ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer (1-channel prediction map)
        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult,
                      out_channels=1,  # Binary real/fake output 
                      kernel_size=kw,
                      stride=1,
                      padding=padw)  # output 1 channel prediction map
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """
        Forward pass returning patch-wise discriminator scores.

        Args:
            input (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, 1, H', W'] where H' and W' 
                            depend on the number of layers. 
        """
        return self.main(input)


