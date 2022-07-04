import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Transform:
    def __init__(self, load_size, crop_size, mean, std, mode):
        self.mode = mode

        self.train_transforms = A.Compose(
            [
                A.Resize(width=load_size, height=load_size, always_apply=True),
                A.RandomCrop(width=crop_size, height=crop_size, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
                ToTensorV2(always_apply=True),
            ]
        )

        self.test_transforms = A.Compose(
            [
                A.Resize(width=crop_size, height=crop_size, always_apply=True),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
                ToTensorV2(always_apply=True),
            ]
        )

    def apply(self, input):
        input = np.array(input)
        if self.mode == 'train':
            tensor = self.train_transforms(image=input)
        else:
            tensor = self.test_transforms(image=input)
        input = tensor["image"]
        return input
