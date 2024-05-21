"""
USEFULL STUFF
Pathlib - https://pythonandvba.com/wp-content/uploads/2021/11/Pathlib_Cheat_Sheet.pdf
Caltech256 Dataset : https://pytorch.org/vision/stable/generated/torchvision.datasets.Caltech256.html#torchvision.datasets.Caltech256

"""

import torch
from torchvision import datasets
from torchvision.transforms import transforms

from config import *

torch.manual_seed(42)

def to_rgb(image):
  """Converts a grayscale image to RGB format."""
  if len(image.getbands()) == 1:
    # Add two dummy channels to make it RGB
    return image.convert('RGB')
  else:
    return image

transform = transforms.Compose([
  transforms.Lambda(to_rgb),
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),           # Convert images to tensors
])

dataset = datasets.Caltech256(DATA_ROOT, transform=transform, download=True)
# dataset = datasets.Caltech101(DATA_ROOT, transform=transform, download=True)
print("Dataset size : ",len(dataset))
indices = list(range(len(dataset)))

split = int(0.8 * len(dataset))
train_indices, test_indices = indices[:split], indices[split:]

# Create training and test subsets using Subset
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
