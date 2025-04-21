
from torch import nn 
import torch 
from abc import abstractmethod
from ldm.models.attention import SpatialTransformer, zero_module
from ldm.models.utils import conv_nd, normalization, avg_pool_nd, checkpoint, timestep_embedding
import math 
import numpy as np 

class TimestepBlock(nn.Module):

    """
    An abstract base class for neural network blocks that incorporate timestep embeddings.

    This class is typically used in diffusion models or other time-dependent architectures 
    where the behavior of the network needs to be conditioned on a timestep or 
    noise level. represented by embeddings.

    Subclassed must implement for forward method that processes both the input tensor 
    and corresponding timestep embeddings.

    Example:
        >>> class MyTimestepBlock(TimestepBlock):
        ...     def __init__(self, channels):
        ...         super().__init__()
        ...         self.dense = nn.Linear(128, channels*2)
        ...         self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        ...     
        ...     def forward(self, x, emb):
        ...         # x: [batch_size, channels, height, width]
        ...         # emb: [batch_size, embedding_dim]
        ...         emb_out = self.dense(emb)[:, :, None, None]     # [batch_size, channels*2, 1, 1]
        ...         scale, shift = torch.chunk(emb_out, 2, dim=1)
        ...         return self.conv(x * (1 + scale) + shift)
        >>>
        >>> block= MyTimestepBlock(64)
        >>> x = torch.randn(4, 64, 32, 32) # 4 samples, 64 channels, 32x32
        >>> emb = torch.randn(4, 128) # 4 samples, 128-dim embedding 
        >>> output = block(x, emb) # [4, 64, 32, 32]
    """

    @abstractmethod
    def forward(self, x, emb):
        """ 
        Forward pass of the module incorporating timestep embeddings.

        Args:
            x: Input tensor of any shape (typically [batch_size, ...])
            emb: Timestep embeddings tensor of shape [batch_size, embedding_dim]

        Returns:
            Output tensor of same shape as input (typically [batch_size, ...])

        Note:
            The exact shapes depend on the specific implementation in subclasses.
            The embedding is typically used to modulate the network's behaviour
            at different timestep (e.g., through scaling and shifting)
        """
        raise NotImplementedError("Subclasses must implement forward method.")



class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    """ 
    A sequential module that passes timestep embeddings to the children that 
    supported is as an extra input.
    """

    def forward(self, x, emb, context=None):
        print("X: ---->", x.shape)
        print("emb: --->", emb.shape)
        print("self: -----> ", self)


        for layer in self:
            


            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
                

            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)

            else:
                x = layer(x)

        return x 
    




class Upsample(nn.Module):

    """ 
    An upsampling layer with an optional conv.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a conv is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D if 3D, then 
                upsampling occures in the inner-two dims
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims 

        if use_conv:
            # Add padding options for size matching
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding)

        


    def forward(self, x, output_size=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = torch.nn.functional.interpolate(
                input=x,
                size = (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest"
                
            )

        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x 

        
 


class Downsample(nn.Module):

    """ 
    A downsampling layer that can use either convolution or average pooling to reduce spatial dimensions.

    This layer supports 1D, 2D and 3D inputs. For 3D inputs, downsampling is performed only in the 
    inner two dimension (typically height and wdith while preserving temporal dimension).

    Args:
        channels (int): Number of input channels 
        use_conv (bool): If True, uses a strided convolution for downsampling. If False, uses average pooling.
        dims (int, optional): Dimenisonally of the input (1, 2 or 3). Default to 2.
        out_channels (int, optional): Number of output channels. If None, same as input channels Default to None.
        padding (int, optional): Padding size for the convolution. Default to 1.


    Attribute:
        channels (int): Stores the input channels.
        out_channels (int): Store the output channels.
        use_conv (bool): Flag for using conv or pooling 
        dims (int): Dimensionality of the input.
        conv (nn.Module): Conv layer (if use_conv is True)
        pool (nn.Module): Average pooling layer (if use_conv is False)
    """


    def __init__(self, 
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims 
        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.op = conv_nd(
                self.dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )

        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
            


    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

        
        
        
# class Downsample(nn.Module):

#     """ 
#     A downsampling layer with optional conv.
#     :param channels: channels in inputs and outputs.
#     :params use_conv: a bool determining if conv is applied.
#     :param dims: determines if the signal is 1D, 2D or 3D if 3D then 
#                 downsampling occurs in the inner-two dims.
#     """

#     def __init__(self, 
#                  channels,
#                  use_conv,
#                  dims=2,
#                  out_channels=None,
#                  padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv
#         self.dims = dims 
#         stride = 2 if dims != 3 else (1, 2, 2)

#         if use_conv:
#             self.conv = conv_nd(
#                 self.dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
#             )

#         else:
#             # assert self.channels == self.out_channels
#             # self.pool = avg_pool_nd(dims, kernel_size=stride, stride=stride)
#             self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


#     def forward(self, x):
#         # assert x.shape[1] == self.channels
#         # return self.op(x)

#         # Handle odd dimension by adding padding 
#         if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
#             x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant")


#         if self.use_conv:
#             return self.conv(x)
        
#         else:
#             return self.pool(x)

        
    


class ResBlock(TimestepBlock):

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False
    ):
        
        super().__init__()
        self.channels = channels
        self.emb_channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )

        self.updown = up or down 
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)

        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)

        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels
            )
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()

        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )

        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x, emb):

        """ 
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )
    


    def _forward(self, x, emb):

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)


        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h 
    
def count_flop_attn(model, _x, y):
    """ 
    A counter for the `thop` package to count the operations in an 
    attention operation.

    Meant to be used like:
        maxs, params = thop.profile(
        model,
        inputs=(inputs, timesteps),
        custom_ops={QKVAttention: QKVAttention.count_flops})
    """

    b, c, *spatial = y[0].shape 
    num_spatial = int(np.prod(spatial))

    matmul_ops = 2 * b * (num_spatial ** 2) * c 
    model.total_ops += torch.DoubleTensor([matmul_ops])




class QKVAttention(nn.Module):

    """ A module which perform QKV attention and splits in a different order."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):

        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """

        bs, width, length = qkv.shape 
        assert width % (3 * self.n_heads) == 0 
        ch = width // (3 * self.n_heads) 

        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "b c t, b c s -> b t s",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length)
        )

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("b t s, b c s -> b c t", weight, v.reshape(bs * self.n_heads, ch, length))

        return a.reshape(bs, -1, length)
    

    @staticmethod
    def count_flops(model, _x, y):
        return count_flop_attn(model, _x, y)
    


class QKVAttentionLegacy(nn.Module):
    """A module which perform QKV attention. Matches legacy QKVAttention + input/output heads shaping"""

    def __init__(self,
                 n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """ 
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qc, Ks and Vs
        :return: an [N x (H * C) x T] tensor after attention
        """

        bs, width, length = qkv.shape 
        assert width % (3 * self.n_heads) == 0 
        ch = width // (3 * self.n_heads) 
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "b c t, b c s -> b t s", q * scale, k* scale
        )

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("b t s, b c s -> b c t", weight, v)
        return a.reshape(bs, -1, length)
    

    @staticmethod
    def count_flops(model, _x, y):
        return count_flop_attn(model, _x, y)
    



class AttentionBlock(nn.Module):

    def __init__(self,
                 channels,
                 num_heads=1,
                 num_head_channels=-1,
                 use_checkpoint=False,
                 use_new_attention_order=False):
        
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0 
            ), f"q, k, v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        # self.qkv.weight = nn.Parameter(self.qkv.weight.half())  # Weights in float16
        # if self.qkv.bias is not None:
        #     self.qkv.bias = nn.Parameter(self.qkv.bias.float())  # Biases in float32



        if use_new_attention_order:
            # split qkv before split heads 
            self.attention = QKVAttention(self.num_heads)

        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))


    def forward(self, x):
        # if x.dtype == torch.float16:
        #     # Convert input to float32 for attention operations
        #     with torch.cuda.amp.autocast(enabled=False):
        #         x_float = x.float()
        #         out = checkpoint(self._forward, (x_float,), self.parameters(), self.use_checkpoint)
        #         return out.half()
        # return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

        if self.training and torch.is_autocast_cache_enabled():
            # Convert weights to match input type during mixed precision 
            with torch.cuda.amp.autocast(enabled=False):
                x = x.float()
                self.qkv.weight.data = self.qkv.weight.data.float()
                if self.qkv.bias is not None:
                    self.qkv.bias.data = self.qkv.bias.data.float()


                h = checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
                return h.to(x.dtype)
            
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
    



    def _forward(self, x):
        b, c, *spatial = x.shape 
        x = x.reshape(b, c, -1)
        print("check the dtype :", x)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    






class UNetModel(nn.Module):

    """ 
    A U-Net achitecture with residual blocks, attention mechanism, and optional conditioning.

    The model is designed for diffusion process and supports:
    - The step embedding 
    - class conditioning 
    - Cross-attention with spatial transformers
    - Multiple resolution attention 
    - Gradient checkpointing 
    - Mixed precision training

    Args:
        image_size (int): Input image size (assumed square)
        in_channels (int): Number of input channels 
        model_channels (int): Base channel count 
        out_channels (int): Number of output channels 
        num_res_blocks (int): Number of residual blocks per resolution level 
        attention_resolution (int): List of resolutions (as fraction of original) to apply attention 
        dropout (float): Dropout probability (0 = no dropout)
        channel_mult (tuple): Channel multiplies for each resolution level 
        conv_resample (bool): Use conv down/upsampling if True 
        dims (int): Number of spatial dimensions (2 for images)
        num_classes (int, optional): Number of classes for conditioning 
        use_checkpoint (bool): Use gradient checkpointing to save memory 
        use_fp16 (bool): Use mixed precision (float16) training 
        use_heads (int): Number of attention heads (-1 = calculate from num_head_channels)
        num_head_channels  (int): Channels per attention head (-1 = calculate from num_heads)
        num_head_upsample (int): Number of heads in upsampling pass (-1 = same as num_heads)
        use_scale_shift_norm (bool): use scale/shift in group norm 
        resblock_updown (bool): Use residual blocks for down/upsampling 
        use_new_attention_order (bool): Use different attention computation order 
        use_spatial_transformer (bool): Use spatial transformer block instead of regular attention
        transformer_depth (int): Depth of transformer blocks 
        context_dim (int, optional): Dimension of cross-attention context 
        n_embed (int, optional): If specified, predicts codebook indices 
        legacy (bool): use legacy attention head calculation 

    Input Shapes:
        - x: (Batch, in_channels, height, width)
        - timesteps: (Batch,)
        - context: (batch, context_len, context_dim) [if use_spatial_transformer]
        - y: (batch,) [if num_classes specified]

    Output Shapes:
        - Main output: (Batch, out_channels, height, width)
        - If predict_codebook_ids: (batch, n_embed, height, width)

    """
    
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult = (1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True
    ):
        
        super().__init__()

        # Validate spatial transformer configuration 
        if use_spatial_transformer:
            assert context_dim is not None, 'Must specify context_dim for spatial transformer.'

        if context_dim is not None:
            assert use_spatial_transformer, "Must enable spatial transformer for context conditioning."
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)


        # Handle attention head configuration 
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads 

        if num_heads == -1:
            assert num_head_channels != -1, "Must be specified eith num_head or num_head_channels"

        if num_head_channels == -1:
            assert num_heads != -1, "Must be specify either num_heads or num_head_channels"

        # Convert model to half precision but keep certain layers in float32
        self.half()  # Convert all parameters to float16
        
        # Ensure normalization layers and certain operations stay in float32
        for module in self.modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                module.float()  # Normalization layers work better in float32
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.float())  # Keep biases in float32


        # Store model configuration 
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None 
        self.dims = dims 

        # Time embedding network
        time_embed_dim = model_channels * 4 
        self.time_embed = nn.Sequential(
            torch.nn.Linear(in_features=model_channels, out_features=time_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=time_embed_dim, out_features=time_embed_dim)
        )

        # Class conditioning embedding 
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        # Build input blocks (downsampling path)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(self.dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1  # current downsampling factor 

        # Construct downsampling path 
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels

                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    # Calculate attention head configuration 
                    if num_head_channels == -1:
                        dim_head = ch // num_heads

                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_checkpoint=use_checkpoint,
                            use_new_attention_order=use_new_attention_order
                        ) if not use_spatial_transformer else SpatialTransformer(
                            in_channels=ch,
                            n_heads=num_heads,
                            d_head=dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch 
                input_block_chans.append(ch)

            # Add downsampling layer (except last level)
            if level != len(channel_mult) -1:
                out_ch = ch 
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(
                            channels=ch,
                            use_conv=conv_resample,
                            dims=dims,
                            out_channels=out_ch
                        )
                    )
                )
                ch = out_ch 
                input_block_chans.append(ch)
                ds *= 2 
                self._feature_size += ch 

        # Middle block
        if num_head_channels == -1:
            dim_head = ch // num_heads

        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order
            ) if not use_spatial_transformer else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
        )
        self._feature_size += ch 

        # Build output blocks (upsampling path)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Get channels for corresponding input block
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ] 
                ch = model_channels * mult 

                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    # Calculate attention head configuration 
                    if num_head_channels == -1:
                        dim_head = ch // num_heads

                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,

                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )


                # Add upsampling layer (except first and last blocks)
                if level and i == num_res_blocks:
                    out_ch = ch 
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )

                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch 

        # Final output layers 
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1))
        )

        # Optional codebook prediction head 
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1)
            )



    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):

        """ 
        Forward pass of the U-Net model.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            timesteps: Diffusion timesteps of shape (batch,)
            context: Optional context for cross-attention (batch, context_len, context_dim)
            y: Optional class lebels of conditioning (batch,)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
            Or codebook predictions if predict_codebook_ids is True 
        """
        # validate the image size 
        assert self.image_size % 8 == 0, "Image Size must be divisible by 8"

        # Validate inputs 
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        

        # Time embedding 
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)


        # Class conditioning 
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)


        # Downsampling path 
        h = x.type(self.dtype)
        print("h:  ->", h.shape)
        for module in self.input_blocks:
            print("module:  ->", module)
            h = module(h, emb, context)
            hs.append(h)


        # Middle block
        h = self.middle_block(h, emb, context)


        # Upsampling path with skip connections
        for module in self.output_blocks:
            h_skip = hs.pop()
            print(f"Current h shape: {h.shape}, Skip connection shape: {h_skip.shape}")

            # Resize h to match h_skip spatial dim 
            if h.shape[2:] != h_skip.shape[2:]:
                h = nn.functional.interpolate(h, size=h_skip.shape[2:], mode="nearest")

            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        
        else:
            return self.out(h)
            


        
        

if __name__ == "__main__":

# -------------------------------------------------------------------------------------------------------------------------
# Example 1: Basic UNet for Diffusion Model 

    model = UNetModel(
        image_size=64,      # Imput image size (64x64)
        in_channels=3,      # RGB images 
        model_channels=224,  # Base channel count
        out_channels=3,     # Same as input for diffusion
        num_res_blocks=2,   # 3 residual blocks per level 
        attention_resolutions=[8, 4, 2],  # Apply attention at 8x8 and 16x16
        dropout=0.1,        # 10% dropout 
        channel_mult=(1, 2, 3, 4),   # channel multipliers at each level 
        num_heads=4,                    # 4 attention heads 
        use_checkpoint=False,       # Disable gradient checkpointing for this example 
        use_fp16=False,              # Disable mixed precision
        num_head_channels=32
    )


    # Example forward pass 
    batch_size = 4 
    x = torch.randn(batch_size, 3, 64, 64)
    timestep = torch.randint(0, 1000, (batch_size,))



    output = model(x, timestep)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

# --------------------------------------------------------------------------

    # downsample = Downsample(channels=3,
    #                         use_conv=True,
    #                         out_channels=3)
    
    # x = torch.randn(4, 3, 256, 256)
    # output = downsample(x)
    # print(output.shape)


    # upsample = Upsample(channels=3,
    #                     use_conv=False,
    #                     out_channels=3)
    # x = torch.randn(4, 3, 256, 256)
    # output = upsample(x)
    # print(output.shape)


# python -m ldm.models.openaimodel