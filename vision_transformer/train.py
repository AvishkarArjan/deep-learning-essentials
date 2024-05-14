
import torch.nn as nn
import torch.optim as optim
from config import *
from model import VisionTransformer
from dataset import train_loader, test_loader
import time
from utils import save_checkpoint, load_pretrained


if __name__ == "__main__":

  model = VisionTransformer()
  if CHECKPOINT_PATH.exists() and RESUME:
    models = []
    for file in CHECKPOINT_PATH.iterdir():
      models.append(file.stem)
    models.sort()
    ckpt = models[-1]
    epoch = ckpt[-1]
    print(epoch)

    model = load_pretrained(model, CHECKPOINT_PATH/f"{ckpt}{epoch}.pth")
  model = model.to(DEVICE)
  print("Num parameters of model : ", sum(p.numel() for p in model.parameters()))

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  print("_______________________\nStarting Training\n")

  for epoch in range(NUM_EPOCHS):
    print("Training for epoch : ", epoch+1)
    start = time.time()
    for i, (imgs, labels) in enumerate(train_loader):
      
      imgs = imgs.to(DEVICE)
      labels = labels.to(DEVICE)
      
      prediction = model(imgs)
      loss = criterion(prediction, labels)
      print(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i%50==0):
        print("Saving checkpoint : ", i)
        save_checkpoint(model.state_dict(), epoch+1, loss, CHECKPOINT_PATH)

    end = time.time()
    print(f"Epoch {epoch+1} Training time : {end-start}s")

