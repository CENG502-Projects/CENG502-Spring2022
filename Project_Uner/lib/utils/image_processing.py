import torch
import cv2
import os
import numpy as np
from tqdm import tqdm

_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()


def gram(input):
    """
    Calculate Gram Matrix

    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image, device='cuda'):
    '''
    https://en.wikipedia.org/wiki/YUV

    output: Image of shape (H, W, C) (channel last)
    '''
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(
        image,
        _rgb_to_yuv_kernel.to(device),
        dims=([image.ndim - 3], [0]))

    return yuv_img


def divisible(dim):
    '''
    Make width and height divisible by 32
    '''
    width, height = dim
    return width - (width % 32), height - (height % 32)


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width and height:
        return cv2.resize(image, divisible((width, height)),  interpolation=inter)

    if width is None and height is None:
        return cv2.resize(image, divisible((w, h)),  interpolation=inter)

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, divisible(dim), interpolation=inter)


def normalize_input(images):
    '''
    [0, 255] -> [-1, 1]
    '''
    return images / 127.5 - 1.0


def denormalize_input(images, dtype=None):
    '''
    [-1, 1] -> [0, 255]
    '''
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):
            images = images.type(dtype)
        else:
            # numpy.ndarray
            images = images.astype(dtype)

    return images


def compute_data_mean(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    print(f"Compute mean (R, G, B) from {len(image_files)} images")

    for img_file in tqdm(image_files):
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[...,::-1]  # Convert to BGR for training


def rgb_to_gray(rgb_tensor):
    assert len(rgb_tensor.shape) == 4, "rgb tensor shape must be 4d"
    r = rgb_tensor[:, 0, ...]
    g = rgb_tensor[:, 1, ...]
    b = rgb_tensor[:, 2, ...]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.unsqueeze(1)
    gray = gray.repeat(1, 3, 1, 1)
    return gray


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image

    img_path = '/Users/onuruner/Downloads/IMG_7019.JPG'
    img = Image.open(img_path).convert('RGB')
    img = to_tensor(img)[None, ...]
    imgs = torch.cat([img, img], dim=0)
    imgs = rgb_to_gray(imgs)
    img = to_pil_image(imgs[1])
    img.show()