import torch.nn as nn 
import torch 
import numpy as np 
from einops import rearrange
import math 



def Normalize(in_channels,
              num_groups=32):
    
    return torch.nn.GroupNorm(num_groups=num_groups,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)



def nonlinearity(x):
    return x * torch.sigmoid(x)




class ResnetBlock(nn.Module):

    """ 
    ResNet block with optional timestep embedding 

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels)
        conv_shortcut: Whether to use convolutional shortcut (vs 1x1 conv)
        dropout: Dropout probability 
        temb_channels: Timestep embedding dimension (0 to disable)
    """

    def __init__(self,
                 *,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.0,
                 temb_channels=512):
        
        super().__init__()
        self.in_channels = in_channels

        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        # First normalization and convolution
        self.norm1 = Normalize(in_channels=in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        # Timestep embedding projection
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_features=self.out_channels)
            
        # Second normalization and convolution
        self.norm2 = Normalize(self.out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=3,
                                                     padding=1,
                                                     stride=1
                                                     )
                
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels=self.in_channels,
                                                    out_channels=self.out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
                


    def forward(self, x, temb):

        """ 
        Forward pass 

        Args:
            x: Input tensor of shape (B, C, H, W)
            temb: Optional timestep embedding of shape (B, temb_channels)

        Returns:
            output tensor of shape (B, out_channels, H, W)
        """
        # Ensure input is not None 
        if x is None:
            raise ValueError("Input tensor cannot be None")

        h = x 

        # First conv block 
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # Add timestep embedding if available 
        if temb is not None and hasattr(self, 'temb_proj'):
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]


        # second conv block 
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)


        # Shortcut connection 
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)

            else:
                x = self.nin_shortcut(x)


        return x + h 
    

class Downsample(nn.Module):

    def __init__(self,
                 in_channels, 
                 with_conv=True):
        
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:

            # no asymmetric padding in torch conv, must do it ourselves 
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
            

        
    def forward(self, x):
        
        if x is None:
            raise ValueError("Downsample input cannot be None")
        
        if self.with_conv:
            # No need for extra padding since we are using padding=1 
            return self.conv(x)
        
        else:
            # single average polling 
            return torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        
       
       
class AttnBlock(nn.Module):

    """ 
    Self-Attention block for convolutional networks.

    Args:
        in_channel (int): Number of input channels (same as output channels)

    Input:
        x (Tensor): Input tensor of shape (B, C, H, W)

    Output:
        Tensor of shape (B, C, H, W) with attention features added to input
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Normalization layer 
        self.norm = Normalize(in_channels)

        # Query, key, value projection (1x1 convolutions)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        # output projection 
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        


    def forward(self, x):

        # store original input for residual connection
        h_ = x 

        # Normalize input 
        h_ = self.norm(h_)

        # Compute query, key, value projection
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for attention computation
        b, c, h, w = q.shape 

        # Reshape Q: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        
        # Reshape K: (B, C, H, W) -> (B, C, H*W)
        k = k.reshape(b, c, h * w)
        # Compute attention weights (dot product)
        # (B, H*W, C) @ (B, C, H*W) -> (B, H*W, H*W)
        attn_weight = torch.bmm(q, k)

        # scale by sqrt(channels) and apply the softmax
        attn_weight = attn_weight * (c ** -0.5)
        attn_weight = torch.nn.functional.softmax(attn_weight, dim=2)


        # Reshape V: (B, C, H, W) -> (B, C, H*W)
        v = v.reshape(b, c, h*w)
        
        # Attend to values 
        # (B, C, H*W) @ (B, H*W, H*W) -> (B, c, H*W)
        h_ = torch.bmm(v, attn_weight.permute(0, 2, 1))

        # Reshape back to original dimensions 
        h_ = h_.reshape(b, c, h, w)

        # Final projection 
        h_ = self.proj_out(h_)

        return x+h_ 
    

class LinearAttention(nn.Module):

    """ 
    Linear Attention module with multi-head support.

    Args:
        dim (int): Input channels dimension 
        heads (int): Number of attention heads (default: 4)
        dim_head (int): Dimension of each attention head (default: 32)

    Input:
        x (Tensor): Input tensor of shape (B, C, H, W)

    Output:
        Tensor of shape (B, C, H, W) with attention features
    """

    def __init__(self,
                 dim,
                 heads=4,
                 dim_head=32):
        
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        # Single conv to proj Q, K, V simultaneously
        self.to_qkv = nn.Conv2d(dim,  hidden_dim * 3, 1, bias=False)
        # Output projection 
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)


    def forward(self, x):

        b, c, h, w = x.shape 
        # project input to query, key, value with a single convolution
        qkv = self.to_qkv(x)
        # split into query, key, value and rearrange dimensions
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)

        # Apply softmax to keys along the sequence dimension
        k = k.softmax(dim=-1)
        # Compute context (linear attention)
        # bhde = (bhdc @ bhce) where c is seq_len
        context = torch.einsum('bhdn, bhen -> bhde', k, v)
        
        # compute output (bhce = bhde @ bhdc)
        out = torch.einsum('bhde, bhdn -> bhen', context, q)
        # Rearrange back to original spatial dim
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        # Final projection 
        return self.to_out(out)
    






    
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels,
                         heads=1,
                         dim_head=in_channels)
        





def make_attn(in_channels, attn_type="vanilla"):

    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type , {attn_type} with {in_channels} in_channels")

    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    
    elif attn_type == "none":
        return nn.Identity(in_channels)
    
    else:
        return LinAttnBlock(in_channels)

    






class Encoder(nn.Module):

    """ 
    Hierarchical encoder architecture with residual blocks and attention.

    Args:
        ch (int): Base channel count 
        out_ch (int): Output channels (unused in current implementation)
        ch_mult (tuple): Channel multipliers for each resolution level
        num_res_blocks (int): Number of residual blocks per resolution
        attn_resolution (list): Resolutions to apply attention at 
        dropout (float): Dropout probability 
        resamp_with_conv (bool): Use convolution in downsampling 
        in_channels (int): Input image channels 
        resolution (int): Input image resolution 
        z_channels (int): Latent space channels 
        double_z (bool): Double output channels for mean/logvar 
        use_linear_attn (bool): Use linear attention variant 
        attn_type (str): "vanilla" or "linear" attention
    """




    def __init__(self, 
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolution,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 double_z=True,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignore_kwargs
                 ):
        

        super().__init__()

        # Handle attention type 
        if use_linear_attn: attn_type = "linear"

        # store parameters 
        self.ch = ch 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Initial convolutions
        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        

        # Downsampling blocks 
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            
            # Create blocks for this resolution level
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout
                                         ))
                block_in = block_out


                # Add attention if at specified resolution
                if curr_res in attn_resolution:
                    attn.append(make_attn(block_in, attn_type=attn_type))


            # Create downsampling module 
            down = nn.Module()
            down.block = block 
            down.attn = attn 


            # Add downsampler if not last level 
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(in_channels=block_in,
                                             with_conv=resamp_with_conv)
                curr_res = curr_res // 2 

            self.down.append(down)



        # middle blocks 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                      out_channels=block_in,
                                      temb_channels=self.temb_ch,
                                      dropout=dropout)
        
        self.mid.attn_1 = make_attn(in_channels=block_in,
                                    attn_type=attn_type)
        
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        


        # output 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(in_channels=block_in,
                                        out_channels= 2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        



    def forward(self, x):

        """ 
            1. Resolution Progression:
            Input: 256x256
                |         
            After Level 0: 256x256 → 128x128
                |
            After Level 1: 128x128 → 64x64
                |
            After Level 2: 64x64 → 32x32
                |
            After Level 3: 32x32 (final resolution)

            2. Channel Progression (for ch=128, ch_mult=(1,2,4,8)):
            conv_in: 3 → 128
                |
            Level 0: 128 → 128
                |
            Level 1: 128 → 256
                |
            Level 2: 256 → 512
                |
            Level 3: 512 → 1024

            3. Output:
            Final resolution: 16x16 (for 256 input)
            Channels: 2*z_channels (e.g., 8 for z_channels=4)
        """


        # Debug input 
        print(f"Initial input shape: {x.shape if x is not None else 'None'}")

        # timestep embedding 
        temb = None 

        print("Shape of X -->", x.shape)

        # downsampling embedding
        hs = [self.conv_in(x)]
        # print("hs Testing: -->", hs[-1].shape, temb)  # shape of hs: torch.Size([4, 128, 256, 256])
        print(f"After conv_in: {hs[-1].shape if hs[-1] is not None else 'None'}")

        for i_level in range(self.num_resolutions):
            print(f"\nProcessing level: {i_level}")
            # Process all blocks at this resolution
            for i_block in range(self.num_res_blocks):
                print(f" Block {i_level}, input shape: {hs[-1].shape if hs[-1] is not None else 'None'}")
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                
                # get current block 
                current_block = self.down[i_level].block[i_block]

                # processing block 
                h = current_block(hs[-1], temb)
                print(f" After resblock: {h.shape if h is not None else 'None'}")

                # Apply attention if exists
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                    print(f" After attention: {h.shape if h is not None else 'None'}")
                hs.append(h)

            # Downsample if not last level 
            if i_level != self.num_resolutions -1:
                downsample = self.down[i_level].downsample 
                h_down = downsample(hs[-1])
                print(f"After downsample: {h_down.shape if h_down is not None else 'None'}")
                hs.append(h_down)


        # middle blocks 
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Output 
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h 
    

class Upsample(nn.Module):

    def __init__(self, 
                 in_channels,
                 with_conv):
        
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0)
        if self.with_conv:
            x = self.conv(x)

        return x 
    




class Decoder(nn.Module):

    def __init__(self, 
                 *,                            
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolution,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 give_pre_end=False,
                 tanh_out=False,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 **ignorekwargs):
        

        """ 
            Decoder archecture.

            * (config): config file 
            ch (int): Base channel count 
            out_ch (int): Output channels (e.g 3 for RGB)
            num_res_blocks (int): ResBlocks for resolutions 
            attn_resolutions (list[int]): Resolution to apply attention 
            dropout (float): Dropout probability 
            in_channels (int): Latent channels (maches encoders z_channels)
            resolution (int): Target output resolution 
            z_channels (int): Latent space channels 
            give_pre_end (bool): Return pre-final layer 
            tanh_out (bool): Use tanh on output 
        """
        

        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch 
        self.temb_ch = 0 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute channel multipliers (1, 1, 2, 4, 8)
        in_ch_mult = (1,)+tuple(ch_mult) 

        # Start with smallest resolution (bootleneck)
        block_in = ch * ch_mult[self.num_resolutions-1]
        print("block Input: ", block_in)
        curr_res = resolution // 2 **(self.num_resolutions-1)
        print(f"Current Resolutions: {curr_res}")
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)}")


        # Initial convolutions
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        

        # Middle Blocks 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        self.mid.attn_1 = make_attn(in_channels=block_in,
                                    attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        

        # upsampling 
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            print(f"Block output: {block_out}")

            # Extra +1 block compared to encoder 
            for i_bloek in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out 
                print("Block Input: ", {block_in})
                if curr_res in attn_resolution:
                    attn.append(make_attn(block_in, attn_type=attn_type))


            # Create Upsampling module 
            up = nn.Module()
            up.block = block 
            up.attn = attn


            # Add upsampler if not final layer 
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2  # Double resolution
                print("After upsampling current Resolution: ", curr_res)
            # Insert at begining to maintain order 
            self.up.insert(0, up)
            print("upblock: ", self.up)


        # End 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        



    def forward(self, z):

        """ 
            1. Input:
            z_shape: [B, 4, 16, 16] (for 4 latent channels)

            2. Through Network:
            conv_in: [B, 1024, 16, 16] (ch * ch_mult[-1])
                    |
            After middle: [B, 1024, 16, 16]
                    |
            Level 3: [B, 1024, 16, 16] → [B, 512, 32, 32]
                    |
            Level 2: [B, 512, 32, 32] → [B, 256, 64, 64]
                    |
            Level 1: [B, 256, 64, 64] → [B, 128, 128, 128]
                    |
            Level 0: [B, 128, 128, 128] → [B, 128, 256, 256]

            3. Output:
            Final: [B, 3, 256, 256] (RGB image)
        """

        # Store input shape 
        print("Z :", z)
        self.last_z_shape = z.shape 
        

        # timestep embedding 
        temb = None 

        # Initial projection 
        h = self.conv_in(z)

        # middle Block 
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling Block 
        for i_level in reversed(range(self.num_resolutions)):

            # Process all blocks at the level
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)

                # Apply attention if exists
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            # Upsample if not final level 
            if i_level != 0:
                h = self.up[i_level].upsample(h)


        # Early return optional
        if self.give_pre_end:
            return h 
        
        # Final processing 
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # Optional tanh activation 
        if self.tanh_out:
            h = torch.tanh(h)

        return h 
    

def get_timestep_embedding(timesteps, embedding_dim):

    """ 
    Creates sinusoidal embeddings for diffusion model timestep following the original DDPM paper implementation.

    Args:
        timesteps: 1D tensor of timesteps (batch_size,)
        embedding_dim: Dimension of the output embeddings

    Returns:
        Tensor of shape (batch_size, embedding_dim) containing the embeddings
    """


    # verify input shape is 1D (vector of temesteps)
    assert len(timesteps.shape) == 1 

    # Calculate half the embedding dimension (for sin/cos pairs)
    half_dim = embedding_dim // 2 
    
    # Create the base log scale factor
    emb = math.log(10000) / (half_dim - 1)

    # Generate the exponential sequence [0, -emb, -2*emb, ....]
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    # Move to same device as timesteps
    emb = emb.to(device=timesteps.device)

    # Outer product: timesteps * exponential sequence
    emb = timesteps.float()[:, None] * emb[None, :]

    # Concatenate sin and cos to get full embedding
    # embedding(t)[2i] = sin(t / 10000^(2i/d))
    # embedding(t)[2i+1] = cos(t / 10000^(2i/d))
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    # Handle odd embedding dimensions by zero-padding 
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb 




class Model(nn.Module):

    """ 
    U-Net architecture with timestep conditioning and optional attention machanisms 

    Args:
        ch: Base channel count 
        out_ch: Output channels 
        ch_mult: Channel multiplier for each resolution level 
        num_res_blocks: Number residual blocks per resolutions 
        attn_resolutions: Resolutions to apply attention at 
        dropout: Dropout probability 
        resamp_with_conv: Use convolution in down/up sampling 
        in_channels: Input Image channels 
        resolution: Input image resolution 
        use_timestep: Whether to use timestep conditioning 
        use_linear_attn: Use linear attention variant 
        attn_type: "vanilla" or "linear" attention 
    """

    def __init__(self, 
                 *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolution,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 use_timestep=True,
                 use_linear_attn=False,
                 attn_type="vanilla"):
        
        super().__init__()

        # Attention type handling 
        if use_linear_attn: attn_type = "linear"

        # Store configuration 
        self.ch = ch 
        self.temb_ch = self.ch*4 
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep

        # Timestep embedding MLP
        if self.use_timestep:
            self.temb = nn.Module()
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch, self.temb_ch),  
                torch.nn.Linear(self.temb_ch, self.temb_ch)
            ])



        # Initial conv
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # Downsampling path 
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            # Create blocks for current resolution
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                
                block_in = block_out 

                # Add attention if at specified resolution
                if curr_res in attn_resolution:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            # Create downsampling module 
            down = nn.ModuleList()
            down.block = block 
            down.attn = attn 

            # Add downsampler if not last level 
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2  # half resolution

            self.down.append(down)


        # middle blocks 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, 
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        


        # upsampling blocks 
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            # Create blocks for current resolution
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                # Last block uses skip connection from encoder 
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]

                block.append(ResnetBlock(in_channels=block_in+skip_in,  # skip connection
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                
                block_in = block_out
                if curr_res in attn_resolution:
                    attn.append(make_attn(block_in,
                                          attn_type=attn_type))
                    
            
            # Create upsampling module 
            up = nn.ModuleList()
            up.block = block
            up.attn = attn 

            # Add upsampler if not first level 
            if i_level != 0:
                up.upsample = Upsample(block_in, 
                                       resamp_with_conv)
                curr_res = curr_res * 2   # Double resolution 

            self.up.insert(0, up)  # Prepend to maintain order 




        # Final layers 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)





    def forward(self, x, t=None, context=None):

        # Handle context (for class-conditional generation)
        if context is not None:
            x = torch.cat((x, context), dim=1)  # concat alogn channels 

        # Timestep embedding 
        if self.use_timestep:
            assert t is not None, "Timestep required when use_timestep=True"
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)  # First linear layer
            temb = nonlinearity(temb)       # Activation (typically SiLU)
            temb = self.temb.dense[1](temb) # Second linear layer 

        else:
            temb = None



        # downsampling blocks 
        hs = [self.conv_in(x)]  # Store encoder features for skip connections 
        for i_level in range(self.num_resolutions):
            # Process all blocks at this resolutions
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)

                # Apply attention if exists
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)


            # Downsample if not last level 
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))



        # middle blocks 
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)


        # upsampling blocks 
        for i_level in reversed(range(self.num_resolutions)):
            # Process all blocks at this resolution
            for i_block in range(self.num_res_blocks + 1):
                # Concatenate with skip connection from encoder 
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                # Apply attention if exists
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)


            # Upsample if not first level 
            if i_level != 0:
                h = self.up[i_level].upsample(h)



        # Final output 
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h 
    




    def get_last_layer(self):

        """Return the weights of the final output convolution."""
        return self.conv_out.weight



if __name__ == "__main__":

    ddconfig = {
    "z_channels": 8,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1,2,2,4],
    "num_res_blocks": 2,
    "attn_resolution": [16],
    "dropout": 0.0,
    "double_z": False
    }

    encoder = Encoder(**ddconfig)
    print(encoder)
    x = torch.randn(1, 3, 256, 256)
    encoder = encoder(x)
    print(encoder.shape)
    # print(x)
    
