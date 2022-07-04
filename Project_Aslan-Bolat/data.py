import os
import copy
import numpy as np
import torch
from torchvision.io import read_image

class MINDatasetSampler(object):
    """Create Dataset.
    Dataset size is small. Thus, directly loaded to the memory.
    Unlike, most of the time, the data path is loaded first, then, image is opened at the runtime.
    Images from: https://lyy.mpi-inf.mpg.de/mtl/download/
    """
    def __init__(self, img_dirs=None, transform=None, read_all=True, device="cpu"):
        self.data_dict = {}
        for img_dir in img_dirs:
            for root, dirs, _ in os.walk(img_dir):
                for class_ind, class_num in enumerate(dirs):
                    class_dir = os.path.join(root, class_num)
                    img_arr = []
                    for _, _, files in os.walk(class_dir):
                        for ind, file in enumerate(files):
                            img_path = os.path.join(class_dir, file)
                            img = read_image(img_path)/255.0
                            if transform: img = transform(img)
                            img_arr.append(img.to(device))
                            if not read_all and ind+1 == 4:
                                break
                    self.data_dict[class_ind] = torch.stack(img_arr, dim=0)
        assert self.data_dict != {}, "Data Dict is empty!"
        self.num_classes = len(self.data_dict)
        self.num_data_in_class = len(self.data_dict[0])

    def random_sample_classes(self, nc, ns, nq):
        support_set_list = []
        query_set_list = []
        random_class_indices = np.random.choice(self.num_classes, nc)
        for random_class_index in random_class_indices:
            random_class_data = self.data_dict[random_class_index]
            random_sample = np.random.choice(self.num_data_in_class, ns+nq)
            support_set = random_class_data[random_sample][:ns]
            query_set = random_class_data[random_sample][ns:]
            support_set_list.append(support_set)
            query_set_list.append(query_set)
        return support_set_list, query_set_list

if __name__=="__main__":
    nc = 5
    ns = 5
    nq = 16
    mindatasampler = MINDatasetSampler("images/train", read_all=True, device="cuda")
    for _ in range(1000):
        support_set_list, query_set_list = mindatasampler.random_sample_classes(nc, ns, nq)
        print(support_set_list[0].size())
        break
