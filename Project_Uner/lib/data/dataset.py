import random
import warnings
from PIL import Image
from pathlib import Path

from lib.base import BaseDataset
from lib.data.helper import Transform

warnings.filterwarnings('ignore', message=r'.*Corrupt EXIF data\.  Expecting to read .+ bytes but only got .+\.')
warnings.filterwarnings('ignore', message=r'.*Truncated File Read.*')


class UnpairedDataset(BaseDataset):
    def __init__(self, root, A_B_dirs, load_size, crop_size, mean, std, mode, **kwargs):
        super().__init__(mean, std, mode)
        self.root = Path(root).resolve()
        A_dir, B_dir = A_B_dirs
        self.A_paths = list(sorted((self.root / A_dir).glob("*.*")))
        self.B_paths = list(sorted((self.root / B_dir).glob("*.*")))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = Transform(load_size, crop_size, mean, std, mode)

    def _load_data_(self, index):
        index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert('RGB')
        return A_img, B_img

    def __getitem__(self, index):
        A_img, B_img = self._load_data_(index)
        A = self.transform.apply(A_img)
        B = self.transform.apply(B_img)
        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)