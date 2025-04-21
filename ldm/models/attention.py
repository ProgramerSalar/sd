import torch 
from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat
from torch.nn import functional as F
from .utils import checkpoint

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, 
                        num_channels=in_channels,
                        eps=1e-6,
                        affine=True)


def exists(val):
    return val is not None 



def default(val, d):
    if exists(val):
        return val 
    
    return d() if isfunction(d) else d



class CrossAttention(nn.Module):

    """ 
    Cross-attention layer implementing mult-head attention with optional context and masking.

    This layer perform attention between a query sequence (x) and a key-value context sequence.
    with optional masking of certain positions. When no context is provided. it performs 
    self-attention on the input.

    Args:
        query_dim (int): Dimension of the query input 
        context_dim (int, optional): Dimension of the context input. Default to query_dim
        heads (int): Number of attention heads. Default: 8
        dim_head (int): Dimension of each attention head. Default: 64
        dropout (float): Droput probability for output. Default: 0.0

    Shapes:
        - x: (batch_size, sequence_length, query_dim)
        - context: (Batch_size, context_length, context_dim)
        - mask: (Batch_size, context_length) or broadcastable shape 
        - Output: (batch_size, sequence_length, query_dim)

    Example:
        >>> attn = CrossAttention(query_dim=512, heads=8)
        >>> x = torch.randn(1, 10, 512) # 1 batch, 10 tokens, 512 dim 
        >>> context = torch.randn(1, 15, 512) # different length context 
        >>> output = attn(x, context) # shape (1, 10, 512)
    """
    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5 # 1/sqrt(dim_head) for scaled dot-product attention
        self.heads = heads

        # Linear transformations for queries, key and values 
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        # Output projection with droput
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, context=None, mask=None):
        """
        Forward pass for cross-attention.

        Args:
            x: Query tensor [batch, seq_len, query_dim]
            context: Context tensor [batch, context_len, context_dim] (optional)
            mask: Boolean mask tensor [batch, context_len] (optionall)

        Returns:
            Output tensor [batch, seq_len, query_dim]
        """

        h = self.heads # Number of attention heads 

        # Project inputs to queries, keys, values
        q = self.to_q(x) # # [batch, context_len, inner_dim]
        context = default(context, x)   # Use x as context if not provided 
        k = self.to_k(context)  # [batch, context_len, inner_dim]
        v = self.to_v(context) # [batch, context_len, inner_dim]

        # Split into multiple heads 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Scaled dot-product attention
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # Apply mask if provided 
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)') # Flatten mask 
            max_neg_value = -torch.finfo(sim.dtype).max # Large negative value 
            mask = repeat(mask, 'b j -> (b h) () j', h=h)   # Repeat for all heads 
            sim.masked_fill_(~mask, max_neg_value)  # Apply mask 

        # Compute attention weights
        attn = sim.softmax(dim=-1) # [batch*heads, seq_len, context_len]

        # Apply attention to values 
        out = einsum('b i j, b j d -> b i d', attn, v)  # [batch*heads, seq_len, dim_head]

        # Combine heads and project back to original dimension
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h) # [batch, seq_len, inner_dim]
        return self.to_out(out) # [batch, seq_len, query_dim]

class GELU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    


class FeedForward(nn.Module):

    def __init__(self, 
                 dim,
                 dim_out=None,
                 mult=4,
                 glu=False,
                 dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GELU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    
    def forward(self, x):
        return self.net(x)
    



class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim,
                                    heads=n_heads,
                                    dim_head=d_head,
                                    dropout=dropout)
        self.ff = FeedForward(dim=dim,
                              dropout=dropout,
                              glu=gated_ff)
        
        self.att2 = CrossAttention(query_dim=dim,
                                   context_dim=context_dim,
                                   heads=n_heads,
                                   dim_head=d_head,
                                   dropout=dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None):
        return checkpoint(self._forward, 
                          (x, context),
                          self.parameters(),
                          self.checkpoint)
    

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x 
        x = self.attn1(self.norm2(x), context=context)
        x = self.ff(self.norm3(x)) + x 
        return x 
    


def zero_module(module):
    """ 
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()

    return module






class SpatialTransformer(nn.Module):

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None):
        super().__init__()

        self.in_channels = in_channels
        self.inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 self.inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(dim=self.inner_dim,
                                   n_heads=n_heads,
                                   d_head=d_head,
                                   dropout=dropout,
                                   context_dim=context_dim)
            for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(in_channels=self.inner_dim,
                                              out_channels=in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        
    def forward(self, x, context=None):

        b, c, h, w = x.shape 
        x_in = x 

        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x, context)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)

        return x + x_in
    

    