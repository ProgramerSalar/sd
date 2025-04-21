import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image 
import os 


class VQVAEDataset(Dataset):
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
    


if __name__ == "__main__":

    train_dataset = VQVAEDataset(root_dir="E:\\YouTube\\stable-diffusion\\dataset\\cat_dog_images",
                                split="train",
                                image_size=256)
    

    val_dataset = VQVAEDataset(root_dir="E:\\YouTube\\stable-diffusion\\dataset\\cat_dog_images",
                                split="val",
                                image_size=256)
    
    train_datloader = DataLoader(dataset=train_dataset,
                               batch_size=1,
                               shuffle=True)
    
    val_datloader = DataLoader(dataset=val_dataset,
                               batch_size=1,
                               shuffle=True)
    

    for i in train_datloader:
        print(i['image'])
    
    