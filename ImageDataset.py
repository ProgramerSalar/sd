from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image 
from vae.utils.utils import load_config





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
    
