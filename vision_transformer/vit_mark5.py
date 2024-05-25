"""https://github.com/uygarkurt/ViT-PyTorch/blob/main/vit-implementation.ipynb"""

import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config["img_size"]
        self.embed_dim = config["embed_dim"]
        self.patch_size = config["patch_size"]
        self.num_patches = (self.img_size // self.patch_size) **2
        self.num_channels = config["num_channels"]
        
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(
            torch.randn(size=(1, self.num_channels,self.embed_dim)),
            requires_grad=True
            )
        self.position_embeddings = nn.Parameter(
            torch.randn(size=(1, self.num_patches+1, self.embed_dim)),
            requires_grad=True
            )
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token,x ], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        
        return x

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_channels = config["num_channels"]
        
        self.embeddings = PatchEmbeddings(config)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dropout=config["dropout"],
            activation = "gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers = config["num_layers"]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_channels)
        )
        
    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :]) # apply MLP on the CLS token only
        return x
    
if __name__ == "__main__":
    
    config ={
        "img_size":28,
        "embed_dim":16, # (PATCH_SIZE**2)*NUM_CHANNELS
        "patch_size":4,
        "num_patches":49 , # (IMG_SIZE// PATCH_SIZE)**2
        "dropout":0.01,
        "num_channels":1,
        "num_heads":4,
        "num_layers":8,
    }
    model = ViT(config)
    x = torch.randn(32, 1, 28,28)
    print(model(x).shape)

