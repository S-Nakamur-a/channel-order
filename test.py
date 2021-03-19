import argparse
from pathlib import Path

import torch
from cv2 import cv2
from serde.yaml import from_yaml
import numpy as np
from efficientnet_pytorch import EfficientNet

from channel_order.config import Config
from channel_order.dataset import ChannelOrder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    return parser.parse_args()


def run():
    args = get_args()
    config = from_yaml(Config, Path(args.config).read_text())
    state_dict = torch.load(args.model)["state_dict"]
    state_dict = {
        key[6:]: val for key, val in state_dict.items() if key.startswith("model.")
    }
    model = EfficientNet.from_name(config.model, num_classes=1)
    print(model.load_state_dict(state_dict))
    model.eval()

    image = cv2.imread(args.target)[:, :, :3]

    images = image.transpose((2, 0, 1))[None, :, :, :].astype(np.float32) / 255.0
    bgr_image = torch.tensor(images)

    images = image.transpose((2, 0, 1))[None, ::-1, :, :].astype(np.float32) / 255.0
    rgb_image = torch.tensor(images)

    with torch.no_grad():
        output = model.forward(bgr_image)
        prediction = ChannelOrder.BGR if output < 0.5 else ChannelOrder.RGB
        print(f"BGR image prediction is {prediction}. Output is {output}")
        output = model.forward(rgb_image)
        prediction = ChannelOrder.BGR if output < 0.5 else ChannelOrder.RGB
        print(f"RGB image prediction is {prediction}. Output is {output}")

    cv2.imshow("BGR", image)
    cv2.imshow("RGB", image[:, :, ::-1])
    cv2.waitKey()


if __name__ == "__main__":
    run()
