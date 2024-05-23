"""MODEL"""
"""based on official PyTorch source code based on original paper : https://pytorch.org/vision/main/models/vision_transformer.html"""

import torch
import torch.nn as nn
from collections import OrderedDict
import math

config = {
    "img_size":32,
	"patch_size":6,
	"num_channels":3,
	"num_layers":7,
	"num_heads":8,
	"embed_dim":768,
	"mlp_hidden_dim":4*768,
	"dropout":0.0,
	"num_classes":10,
    "lr":0.01,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs":100,
    "exp_name":"vit_cifar10_mark2_100_epochs",
    "save_model_every":0
}

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.hidden_dim = config["mlp_hidden_dim"]
        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.dropout=nn.Dropout(config["dropout"])

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_heads"]
        self.embed_dim = config["embed_dim"]
            
        # Attention block
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.self_attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=config["dropout"], batch_first=True)
        self.dropout = nn.Dropout(config["dropout"])

        # MLP block
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)

    def forward(self, inp: torch.Tensor):
        x = self.layer_norm1(inp) # inp as in  input
        x, _ = self.self_attention(x,x,x,need_weights=False)
        x = self.dropout(x)
        x = x + inp

        y = self.layer_norm2(x)
        y = self.mlp(y)

        return x+y



class Encoder(nn.Module):
    def __init__(self, seq_length ,config):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, config["embed_dim"]).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(config["dropout"])
        self.embed_dim =  config["embed_dim"]
        self.layers = nn.ModuleList([
			EncoderBlock(config) for _ in range(config["num_layers"])]
			)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x):
        x = x + self.pos_embedding
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_proj = nn.Conv2d(
			config["num_channels"], 
		    config["embed_dim"],
			kernel_size=config["patch_size"],
			stride=config["patch_size"])
        self.patch_size = config["patch_size"]
        self.img_size = config["img_size"]
        seq_length = (config["img_size"] // config["patch_size"]) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, config["embed_dim"]))
        seq_length += 1

        self.encoder = Encoder(seq_length, config)
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        heads_layers["pre_logits"] = nn.Linear(config["embed_dim"], config["mlp_hidden_dim"])
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(config["mlp_hidden_dim"] ,config["num_classes"])

        self.heads = nn.Sequential(heads_layers)


        # Initializing weights

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)


    def forward(self,x ):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.img_size, f"Wrong image height! Expected {self.img_size} but got {h}!")
        torch._assert(w == self.img_size, f"Wrong image width! Expected {self.img_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


