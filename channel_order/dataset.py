import enum
from pathlib import Path
import random
from typing import Tuple

from cv2 import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch import ToTensorV2


class ChannelOrder(enum.Enum):
    BGR = 0
    RGB = 1


class ChannelOrderDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        glob_search_word: str = "*.png",
        crop_size: Tuple[int, int] = (224, 224),
    ):
        assert root_dir.is_dir(), f"{root_dir} is not a directory"
        self.paths = list(map(str, root_dir.glob(glob_search_word)))
        assert len(self.paths) > 0, f"Not found: {root_dir}/{glob_search_word}"
        self.labels = list(ChannelOrder)
        self.transforms = alb.Compose(
            [
                alb.Resize(height=crop_size[0] * 2, width=crop_size[1] * 2),
                alb.RandomCrop(height=crop_size[0], width=crop_size[1]),
                alb.VerticalFlip(),
                alb.RandomRotate90(),
                alb.ToFloat(),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        image: np.ndarray = cv2.imread(image_path)
        # RGBA to BGR
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        this_label: ChannelOrder = random.choice(self.labels)

        if this_label is ChannelOrder.BGR:
            pass
        elif this_label is ChannelOrder.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"{this_label} is not supported")
        image = self.transforms(image=image)["image"]
        return image, this_label.value
