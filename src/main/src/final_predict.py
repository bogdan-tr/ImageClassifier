"""This file generates the final predictions for task4 .load the selected model pkl file and make prediciton for final predictions for eval, returning a
a vector in R^N where N is the number of test images where the ith element is the probability model assigns
to the image being FAKE. Output type should be np.float32 named task4_predictions.npy"""

import sys

import numpy
import torch
import torch.nn.functional as f
from PIL import Image

import mesonet_imp_v3 as arch
from helper_loads import load_transformations as lt

# WARN: temp, for testing only

# CONT: save to .npy file


def make_prediction(model, test_file, test_dir, device, temp=9):
    transformations = lt.transformations()
    transformation = transformations.val_transform
    prob_fake_list = []
    for line in test_file:
        # img = Image.open(test_dir + line.strip()).convert("RGB")
        line = line.strip().replace("test/", "")
        img = Image.open(test_dir + line).convert("RGB")  # WARN: temp for testing
        img_tensor = transformation(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = f.softmax(output / temp, dim=1)
            prob_fake = probs[0][0].item()
            prob_real = probs[0][1].item()
            prob_fake = numpy.clip(prob_fake, 1e-6, 1 - 1e-6)
            print(
                f"real: {prob_real:.2f}, fake: {prob_fake:.2f}, total: {(prob_real+prob_fake):.2f}"
            )
            prob_fake_list.append(prob_fake)
    return prob_fake_list


def main():
    model_name = sys.argv[1]
    test_script_name = sys.argv[2]
    test_dir = "/local/202610_csci581_project/project_test_data/task4/"
    model = arch.CNN()
    test_file = open("test_script/" + test_script_name + ".txt", "r")
    weights = torch.load("../models/" + model_name + ".pth")
    model.load_state_dict(weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prob_fake_list = make_prediction(model, test_file, test_dir, device)
    preds = numpy.array(prob_fake_list, dtype=numpy.float32)
    print("test size:", len(prob_fake_list))
    print("output shape:", preds.shape)
    print("output dtype:", preds.dtype)
    numpy.save("task4_predictions.npy", preds)


if __name__ == "__main__":
    main()
