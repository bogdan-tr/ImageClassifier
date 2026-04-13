"""Generate saliency maps for the given model architecture. Used for analysis"""

import argparse

import baseline_cnn_lab4_frequent_metrics as basefm

# import baseline_cnn_lab4_frequent_metrics as basefm
import helper_loads.load_dataloader as ld
import matplotlib.pyplot as plt
import mesonet_imp as mesi
import mesonet_imp_v2 as mesi2
import mesonet_imp_v3 as mesi3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xception as xcep

MODEL_PATH = "model.pth"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=MODEL_PATH, help="Path to saved model .pth")
parser.add_argument("--out", default="saliency.png")
parser.add_argument("--arch", default="mesi")
args = parser.parse_args()
model_arch = args.arch
# get model architecture to pass to saliency map
match model_arch:
    case "mesi":
        model = mesi.CNN()
    case "mesi2":
        model = mesi2.CNN()
    case "mesi3":
        model = mesi3.CNN()
    case "xcep":
        model = xcep.CNN()
    case "basefm":
        model = basefm.CNN()
    case _:
        model = mesi.CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(args.model, map_location=device))
model = model.to(device)
model.eval()

loaders = ld.get_dataloaders_task4(batch_size=1)
val_loader = loaders["devloader"]
iter_loader = iter(val_loader)
for x in range(100):
    img_tensor, target = next(iter_loader)

    img = img_tensor.clone().to(device).requires_grad_(True)

    # # Forward pass
    output = model(img)
    # model.zero_grad()

    _, predicted = torch.max(output.data, 1)
    predicted_class = predicted.item()

    # Calculate the gradient of the score of the predicted class with respect to the input image
    output[:, predicted_class].backward()

    # Get the gradient
    saliency_map = img.grad.data
    saliency_map = saliency_map.abs()
    saliency_map, _ = torch.max(saliency_map, dim=1)
    saliency_map = saliency_map.squeeze()

    # Convert the saliency map to a numpy array for visualization
    saliency_map = saliency_map.cpu().numpy()
    orig_img = (
        img.cpu().detach().permute(3, 2, 1, 0).squeeze(-1).numpy()  # reorder dimensions
    )  # .squeeze(3).permute(1, 2, 0).numpy()
    orig_img = orig_img * 0.5 + 0.5  # unnormalize
    orig_img = np.rot90(orig_img, k=3)
    # print(orig_img.size())
    plt.axis("off")
    plt.title("Saliency Overlay for " + args.out)
    plt.imshow(orig_img, cmap="gray")
    plt.imshow(saliency_map, cmap="hot", alpha=0.6)
    plt.savefig(args.arch + "_" + str(x), dpi=150)
