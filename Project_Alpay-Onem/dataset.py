from __future__ import print_function, division
import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import numpy as np


class DatasetHDR(Dataset):
    def __init__(self, data_path, validation_path, split="train", transform=None):
        self.root_dir = data_path
        self.validation_path = validation_path
        self.type = split
        self.transform = transform

        if split == "train":
            # list of training scenes dirs (full path)
            self.trainScenesDirs = [
                os.path.join(data_path, d)
                for d in sorted(os.listdir(data_path))
                if os.path.isdir(os.path.join(data_path, d))
            ]
            print("Training dataset scenes: ", len(self.trainScenesDirs))
        else:
            validScenesPath1 = os.path.join(validation_path, "EXTRA")
            validScenesPath2 = os.path.join(validation_path, "PAPER")

            self.validScenesDirs = [
                os.path.join(validScenesPath1, d)
                for d in sorted(os.listdir(validScenesPath1))
                if os.path.isdir(os.path.join(validScenesPath1, d))
            ] + [
                os.path.join(validScenesPath2, d)
                for d in sorted(os.listdir(validScenesPath2))
                if os.path.isdir(os.path.join(validScenesPath2, d))
            ]
            print("Validation dataset scenes: ", len(self.validScenesDirs))

    def __len__(self):
        if self.type == "train":
            return len(self.trainScenesDirs)
        else:
            return len(self.validScenesDirs)

    def __getitem__(self, idx):
        # read and preprocess the images of the idx'th scene
        sceneDir = (
            self.trainScenesDirs[idx]
            if self.type == "train"
            else self.validScenesDirs[idx]
        )
        ldrTifFiles = [
            os.path.join(sceneDir, f)
            for f in sorted(os.listdir(sceneDir))
            if os.path.splitext(os.path.join(sceneDir, f))[-1].lower() == ".tif"
        ]
        exposuresFile = os.path.join(sceneDir, "exposure.txt")
        gtHdrFile = os.path.join(sceneDir, "HDRImg.hdr")

        # read exposures into a list
        with open(exposuresFile) as file:
            lines = file.readlines()
            exposures = [2 ** float(line.rstrip()) for line in lines]

        # read gt hdr img as np array
        gtHdr = cv2.imread(gtHdrFile, -1)[
            :, :, ::-1
        ]  # dont return BGR. so use ::-1. second param "-1" is for reading as float32 (unchanged)

        gtHdr = np.clip(
            gtHdr, 0, 1
        )  # clip hdr to 0-1 (just in case. no training example has this normally)

        # read ldr .tif images as float32 in 0-1 range. clip just in case.
        ldrTifs = [
            np.clip(cv2.imread(f)[:, :, ::-1].astype(np.float32) / 255.0, 0, 1)
            for f in ldrTifFiles
        ]

        # get Hi hdrs from ldrs
        hdrsFromLdrs = []
        for idx, ldr in enumerate(ldrTifs):
            hdrsFromLdrs.append((ldr**2.2) / exposures[idx])

        # get concatenated input training dataset
        Xs = np.zeros((3, ldrTifs[0].shape[0], ldrTifs[0].shape[1], 6)).astype(
            np.float32
        )
        for i in range(3):
            Xs[i] = np.concatenate((ldrTifs[i], hdrsFromLdrs[i]), axis=2)

        input = np.concatenate((Xs[0], Xs[1], Xs[2]), axis=2)  # shape is 1000x1500x18
        gt = gtHdr  # 1000x1500x3

        input = input.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))

        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)

        allData = torch.cat((input, gt), 0)

        if self.transform:
            try:
                allData = self.transform(allData)
            except:
                print("Oops!", sys.exc_info()[0], " occurred.")

        input, gt = allData[0:18, :, :], allData[18:21, :, :]

        return input, gt


def getTransforms():
    trns = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
            ]
        ),
    }
    # no transformation is applied on the validation set
    return trns


def getDataLoaders(transforms, batchSize, dataPath, validation_path):

    trainset = DatasetHDR(
        dataPath, validation_path, split="train", transform=transforms["train"]
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchSize, shuffle=True, num_workers=16, pin_memory=True
    )

    validset = DatasetHDR(dataPath, validation_path, split="valid", transform=None)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
    )

    return {"train": trainloader, "valid": validloader}
