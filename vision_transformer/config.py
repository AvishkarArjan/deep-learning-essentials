import torch
from pathlib import Path

DATA_ROOT = Path.home()/"Desktop/research"
CHECKPOINT_PATH = Path.home()/"Desktop/projects/deep_learning_essentials/vision_transformer/model_checkpoints"
RESUME=False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 300
BATCH_SIZE = 32
