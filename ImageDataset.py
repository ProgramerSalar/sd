from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image 
from vae.utils.utils import load_config


# class ImageDataset(Dataset):

#     def __init__(
#             self, 
#             config 
#     ):
        
#         self.config = config
#         self.image_folder = config["data"]["dataset"]['image_folder']

#         # get a list of all image file paths in the folder 
#         self.image_paths = [
#             os.path.join(self.image_folder, f) 
#             for f in os.listdir(self.image_folder)
#             if f.lower().endswith(tuple(config["data"]["dataset"]["extension"]))
#         ]

#         # Build transforms dynamically from config 
#         transform_list = [
#             transforms.Resize(config["data"]["dataset"]["image_size"]),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 config["data"]["transform"]["normalize_mean"],
#                 config["data"]["transform"]["normalize_std"]
#             )
#         ]

#         # Add random flip if configured 
#         if config["data"]["transform"].get("random_flip", False):
#             transform_list.insert(1, transforms.RandomHorizontalFlip())

#         self.transforms = transforms.Compose(transform_list)

#     def __len__(self):
#         return len(self.image_paths)


#     def __getitem__(self, index):
        
#         image_path = self.image_paths[index]
#         image = Image.open(image_path).convert("RGB")
#         image = self.transforms(image)
#         return image 
    




import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image 
import os 


class ImageDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split="train",
                 image_size = 256):
        
        self.roo_dir = os.path.join(root_dir, split)
        self.image_files = [f for f in os.listdir(self.roo_dir) if f.endswith(('.png', '.jpeg', 'jpg'))]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.image_files)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.roo_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {'image': image}
    
