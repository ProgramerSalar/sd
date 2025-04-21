import torch 
import math 
from einops import repeat

class CheckpointFunction(torch.autograd.Function):

    """ 
    Gradient Checkpointing implementation that trades compute for memory.

    This class implements a custom autograd Function that perform checkpointing 
    It recomputes the forward pass during backward to save memory that would 
    otherwise be used to store intermediate activations.

    Usages:
        output: CheckpointFunction.apply(run_function, num_inputs, *inputs_and_params)

    Args:
        run_function: The function to checkpoint (typically a sequence of layers)
        length: Number of input tensors  (seperates inputs from parameters)
        args: All arguments (inputs followed by parameters)

    Note:
        - Input tensors must be at the begining of args 
        - Parameters must come after the first "length" arguments 
        - Returns the same outputs as run_function would normally
    """
    
    @staticmethod
    def forward(ctx, run_function, length, *args):

        """ 
        Forward pass with checkpointing - runs without saving activation.

        Args:
            ctx: Context object to save information for backward.
            run_function: Function to execute 
            length: Number of input tensors in args 
            args: Input tensors followed by parameters

        Returns:
            Output of run_function
        """

        # Save necessary information for backward pass 
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length]) # Seperate inputs
        ctx.input_params = list(args[length:])  # Seperate parameters 

        # Run forward pass without tracking gradients 
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    

    @staticmethod
    def backward(ctx, *output_grads):
        """ 
        Backward pass with recomputation of forward pass.

        Args:
            ctx: Context with saved information
            output_grads: Gradients of the outputs

        Returns:
            Tuple of gradients (None for run_function and length, then input grads)
        """

        print(f"check the data of dtype : {ctx}")

        # Prepare input tensors for gradient computation
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        # Recompute forward pass with gradient tracking 
        with torch.enable_grad():
            # Create shallow copies to avoid modifying original tensors
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            print(f"Shadown copies of data: {shallow_copies}")
            output_tensors = ctx.run_function(*shallow_copies)

        
        # Compute gradients for both inputs and params 
        input_grads = torch.autograd.grad(
            output_tensors,     # Output of recomputed forward
            ctx.input_tensors + ctx.input_params,   # All inputs we want gradients for 
            output_grads,       # Gradients for upper layers 
            allow_unused=True   # Some parameters might not recieve gradients
        )

        # Clean up to save memory
        del ctx.input_tensors 
        del ctx.input_params
        del output_tensors

        # Return gradients (None for run_function and length placeholders)
        return (None, None) + input_grads
    


    



def checkpoint(func, inputs, params, flag):

    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    
    else:
        return func(*inputs)
    


def conv_nd(dims, *args, **kwargs):

    """Create a 1D, 2D or 3D conv module."""

    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    
    elif dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    
    elif dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)
    
    raise ValueError(f"unsupported dimensions: {dims}")



class GroupNorm32(torch.nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    



def normalization(channels):

    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization
    """

    return GroupNorm32(32, channels)


def avg_pool_nd(dims, *args, **kwargs):

    """ 
    Create a 1D, 2D or 3D average pooling module.
    """

    if dims == 1:
        return torch.nn.AvgPool1d(*args, **kwargs)
    
    elif dims == 2:
        return torch.nn.AvgPool2d(*args, **kwargs)
    
    elif dims == 3:
        return torch.nn.AvgPool3d(*args, **kwargs)
    
    raise ValueError(f"unsupported dims: {dims}")


def timestep_embedding(timesteps, dim, max_period=1000, repeat_only=False):

    """ 
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                    There may be fractional.
    :param dim: the dimension of the output 
    :param max_period: controls the minimum frequency of the embeddings 
    :return: an [N x dim] Tensor of positional embeddings
    """

    if not repeat_only:
        half = dim // 2 
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)

    return embedding


