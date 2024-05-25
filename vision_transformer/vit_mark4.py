"""https://github.com/chingisooinar/AI_self-driving-car/blob/main/model/SimpleTransformer.py"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

"""
Special Points
    - ResNet18 used from Positional Encoding
"""

config={
    "img_size":32,
    "patch_size":4,
    "dropout":0.0,
    
}

class VisionTransformer(nn.Module):
    def __init__(self, seq_len, config):
        super(VisionTransformer, self).__init__()
        self.seq_len = seq_len #seq_length = (self.img_size // self.patch_size) ** 2 Most probably
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
        x = F.relu(self.position_embedder(self.position_encoder(x))) 
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
                
        