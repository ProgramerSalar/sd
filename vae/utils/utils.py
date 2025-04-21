import requests, os
from tqdm import tqdm
import hashlib
import yaml
import importlib

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def md5_has(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)

    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))

        with tqdm(total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data: 
                        f.write(data)
                        pbar.update(chunk_size)



def instantiate_from_config(config):

    """ 
    Instantiates an object from a configuration dictionary.

    The config dictinary must contain:
    - 'target': Full import path to the class (e.g., 'module.submodule.ClassName')
    - Optional 'params': Dictionary of constructor arguments

    Args:
        config (dict): Configuration dictionary with at least `target` key 

    Returns:
        object: Instance of the specified class initialized with given parameters.

    Raises:
        KeyError: If `target` key is missing and config is not special identifier
    """

    # Check if target key exists in config 
    if not "target" in config:
        # handle special case identifiers 
        if config == "__is_first_stage__":
            return None   # Spatial maker for first stage models 
        
        elif config == "__is_unconditional__":
            return None  # Special maker for unconditional models
        
        # If not a special case and no target raise error
        raise KeyError("Expected key `target` to instantiate.")
    
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    Dynamically imports and returns a class or function from a string reference.
    
    The string should be in format 'module.submodule.ClassName'
    
    Args:
        string (str): Full import path to the object
        reload (bool): Whether to reload the module before import
        
    Returns:
        object: The requested class or function
        
    Example:
        >>> get_obj_from_str('torch.optim.Adam')
        <class 'torch.optim.adam.Adam'>
    """

    # split the full path into module and class name 
    module, cls = string.rsplit('.', 1)

    # Reload the module if requested (useful for development)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    # Import the module and get the class/function
    return getattr(importlib.import_module(module, package=None), cls)




def get_ckpt_path(name, root, check=False):

    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])

    if not os.path.exists(path) or (check and not md5_has(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_has(path)
        
        assert md5 == MD5_MAP[name], md5 

    return path 
        

def load_config(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config



