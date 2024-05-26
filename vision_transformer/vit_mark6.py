import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import OrderedDict

DATA_ROOT = Path.home()/"Desktop/research"
EXP_DIR = Path.home()/"Desktop/projects/deep_learning_essentials/vision_transformer/experiments"
NUM_WORKERS = 2
BATCH_SIZE = 32 

def to_rgb(image):
  """Converts a grayscale image to RGB format."""
  if len(image.getbands()) == 1:
    # Add two dummy channels to make it RGB
    return image.convert('RGB')
  else:
    return image

def caltech101_dataset(config):

    transform = transforms.Compose([
    transforms.Lambda(to_rgb),
        transforms.Resize((config["img_size"], config["img_size"])),  # Resize images 64x64 : caltech101 has pics of around 200x300 
        transforms.ToTensor(), 
        transforms.RandomRotation(7),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.5], [0.5]),# Convert images to tensors
    ])

    dataset = datasets.Caltech101(DATA_ROOT, transform=transform, download=True)
    # dataset = datasets.Caltech101(DATA_ROOT, transform=transform, download=True)
    print("Dataset size : ",len(dataset))
    indices = list(range(len(dataset)))

    split = int(0.8 * len(dataset))
    train_indices, test_indices = indices[:split], indices[split:]

    # Create training and test subsets using Subset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    classes = dataset.categories

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, test_loader, classes

class Patches(nn.Module):
    """https://mrinath.medium.com/vit-part-1-patchify-images-using-pytorch-unfold-716cd4fd4ef6"""
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.img_size = config["img_size"]
        self.num_patches = (self.img_size//self.patch_size )
        self.num_channels = config["num_channels"]
        self.unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        
    def display_patches(self, patches):
        fig = plt.figure(figsize=(8, 8))
        grid = ImageGrid(fig, 111, nrows_ncols=(self.num_patches, self.num_patches), axes_pad=0.1)

        for i, ax in enumerate(grid):
            patch = patches[i].permute(1, 2, 0).numpy() 
            ax.imshow(patch)
            ax.axis('off')

        plt.show()
        
            
    def forward(self, x):
        bs, c, h, w = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        patches = x.view(bs, c, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        return patches
        

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.num_patches = (self.img_size //self.patch_size)**2
        self.embed_dim = config["embed_dim"]
        self.device = config["device"]
        
        self.projection = nn.Linear(in_features=self.num_patches, out_features=self.embed_dim)  # Replace Dense with Linear
        self.position_embedding = nn.Embedding(num_embeddings=self.num_patches, embedding_dim=self.embed_dim)

    def forward(self, patches):
        # The input here is of shape ( Bs ,num_patches, channels, patch_size, patch_size )
        
        batch_size, num_patches, channels, patch_size, patch_size  = patches.shape
        patches = patches.reshape(batch_size, channels, patch_size, patch_size, num_patches)
        projected_patches = self.projection(patches)
        positions = torch.arange(num_patches, device=self.device).expand(batch_size, num_patches)
        
        print(projected_patches.shape)
        encoded = projected_patches + self.position_embedding(positions)
        
        return encoded

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config["embed_dim"])

class ViT(nn.Module):
    def __init__(self, config):
        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.num_layers = config["num_layers"]
        self.embed_dim = config["embed_dim"]
                
        self.patcher = Patches(config)
        self.patch_embeddings = PatchEmbeddings(config)
        
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        layers = OrderedDict()
        # self.transformer_layers =  
    def forward(self, x):
        # patches = self.patcher(x)
        # encoded_patches = self.patch_embeddings(patches)
        
        # for _ in range(self.num_layers):
            
        pass
        
    
    
if __name__=="__main__":
    config={
        "img_size":64,
        "patch_size":32,
        "embed_dim":4*3, # (img_size// patch_size)**2 * channels
        "num_channels":3,
        "num_classes":101,
        "num_layers":8,
        "num_heads":4,
        "device":"cuda" if torch.cuda.is_available() else "cpu"
        
    }
    train_loader, test_loader, classes = caltech101_dataset(config)
    patcher = Patches(config)
    patch_embed = PatchEmbeddings(config)

    for i , (imgs, labels) in enumerate(train_loader):
        all_patches = patcher(imgs)
        print(all_patches.shape)
        # patcher.display_patches(all_patches[0])
        embeds = patch_embed(all_patches) 
        print(embeds.shape)       
        break
    
        
    
    
    

    