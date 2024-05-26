"""https://github.com/chingisooinar/AI_self-driving-car/blob/main/model/SimpleTransformer.py"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import OrderedDict

from torchvision import models

"""
Special Points
    - ResNet18 used from Positional Encoding
"""


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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, classes




class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.seq_len = config["seq_len"] #seq_length = (self.img_size // self.patch_size) ** 2 Most probably
        self.position_encoder = models.resnet18(pretrained=True)
        self.d_model = 512
        self.position_embedder = nn.Linear(in_features =1000, out_features = self.d_model, bias=True)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2, norm=None) #nn.LayerNorm(512)
        
        self.reduce_combined = nn.Linear(in_features =self.d_model, out_features = 64, bias=True)
        
        self.steering_predictor = nn.Linear(in_features = 64, out_features = 1, bias=True)
        self.speed_predictor = nn.Linear(in_features = 64, out_features = 1, bias=True)
        
        
    def generate_square_subsequent_mask(self,sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        print("initial",x.shape)
        x = self.position_encoder(x)
        print("resnet shape : ",x.shape)
        x = nn.functional.relu(self.position_embedder(x) )
        print(x.shape)
        # x.shape = (torch.Size([32, 512])) = > occasionally 16, 512
        x = x.reshape(-1, self.seq_len, self.d_model).permute(1, 0 ,2) 
        # x.shape = 1,32,512 ==> 32,1,512
        attn_mask = self.generate_square_subsequent_mask(x.shape[0]).cuda()
        fused_embedding = F.relu(self.transformer_encoder(x, mask=attn_mask))
        
        fused_embedding = fused_embedding.permute(1, 0, 2)
        fused_embedding = fused_embedding.reshape(-1, self.d_model)
        reduced = F.relu(self.reduce_combined(fused_embedding))
        
        angle = self.steering_predictor(reduced)
        # speed = self.speed_predictor(reduced)
        
        return angle
    

if __name__ =="__main__":
    config={
    "img_size":32,
    "patch_size":4,
    "dropout":0.0,
    "seq_len":5
    
}
    train_loader, test_loader, classes = caltech101_dataset(config)
    model = VisionTransformer(config)
    
    for i, (imgs, labels ) in enumerate(train_loader):
        angle = model(imgs)
        break

