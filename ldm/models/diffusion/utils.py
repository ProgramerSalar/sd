
from inspect import isfunction
import torch 
import numpy as np 
from PIL import Image, ImageDraw, ImageFont


def count_params(model, verbose=False):

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")

    return total_params

def exists(x):
    return x is not None 


def default(val, d):
    if exists(val):
        return val 
    return d() if isfunction(d) else d 



def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn(shape, device=device)
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()



def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape 
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



def make_beta_schedule(schedule,
                       n_timestep,
                       linear_start=1e-4,
                       linear_end=2e-2,
                       cosine_s=8e-3):
    
    if schedule == "linear":
        betas = (
            torch.linspace(start=linear_start ** 0.5,
                           end=linear_end ** 0.5,
                           steps=n_timestep,
                           dtype=torch.float64) ** 2 
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2 
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]

        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)


    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)

    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5 

    else:
        raise ValueError(f"schedule '{schedule}' unknown. ")
    
    return betas.numpy()



def log_txt_as_img(wh, xc, size=10):

    # wh a tuple of (width, height)
    # xc a list of captions to plot 

    b = len(xc)
    txts = list()

    for bi in range(b):
        txt = Image.new("RGB", wh, color="white") 
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)

        except UnicodeEncodeError:
            print("cant encode string for logging. Skipping.")


        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0 
        txts.append(txt)

    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts 


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    
    return (len(x.shape) == 4) and (x.shape[1] > 3)



def make_ddim_timesteps(ddim_discr_method,
                        num_ddim_timesteps,
                        num_ddpm_timesteps,
                        verbose=True):
    
    if ddim_discr_method == "uniform":
        c = num_ddim_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

    elif ddim_discr_method == "quad":
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) * 2).astype(int)

    else:
        raise NotImplementedError(f"There is not ddim discretization method called {ddim_discr_method}")
    
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1 
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")

    return steps_out




def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):

    # select alphas for computing the variance schedule 
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.array([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according to the formula in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}")
        print(f"For the choosen value of eta, which is {eta}, "
              f"this results in the following sigma_t schedule for ddim sampler {sigmas}")
        
    return sigmas, alphas, alphas_prev


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,), * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise 


def extract_into_tensor(a, t, x_shape):

    b, *_ = t.shape 
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))




def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

