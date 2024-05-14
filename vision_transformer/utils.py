import torch
from pathlib import Path

def save_checkpoint(state_dict, epoch, loss, path):
    p = Path(path)
    if not p.exists():
        print("Creating folder")
        p.mkdir(parents=True, exist_ok=True)

    model_details = {
        "epoch":epoch,
        "state_dict": state_dict,
        "loss" : loss,
    }
    torch.save(model_details, f"{p}/vit{epoch}.pth")
    print(f"model saved at path : {p}/vit{epoch}.pth")


def load_pretrained(model, path, epoch):
    model.load_state_dict(torch.load(f"{path}/vit{epoch}.pth")["state_dict"])
    return model

