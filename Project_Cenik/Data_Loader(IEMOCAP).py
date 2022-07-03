import torch
from torch.utils.data import Dataset, DataLoader

LABEL_DICT1 = {
    '01': 'neutral',
    '02': 'frustration',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    '07': 'excitement'
    # '08': 'surprised'
}



class DataSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        x = torch.from_numpy(x)
        x = x.float()
        y = self.Y[index]
        y = dict[y]
        y = y.long()
        return x, y

    def __len__(self):
        return len(self.X)