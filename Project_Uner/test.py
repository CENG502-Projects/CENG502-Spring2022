import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from easydict import EasyDict as edict
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

import lib.models as models
from lib.utils import load_checkpoint


class ModelWrapper(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        checkpoint = load_checkpoint(config.deployment.best_model)

        self.model = getattr(models, config.generator.type)(**config.generator.args)
        self.model.to(device)
        self.model.load_pretrained_weights(checkpoint['generator_state_dict'])
        self.model.eval()

    @torch.no_grad()
    def forward(self, image):
        return self.model(image)


class Transform:
    def __init__(self, size, mean, std):
        self.to_tensor = A.Compose(
            [
                A.Resize(width=size["width"], height=size["height"]),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def apply(self, input):
        input = np.array(input)
        input = self.to_tensor(image=input)["image"]
        return input


parser = argparse.ArgumentParser(description='Video Inference')
parser.add_argument('-c', '--config', default='./configs/CUT.json',
                    type=str, help='Path to the config file')
parser.add_argument('-i', '--image', default="X.png", type=str,
                    help='Path to the input image file or directory')
parser.add_argument('-o', '--out-dir', default="../../saved/", type=str,
                    help='Path to the output directory')


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config))
    config = edict(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = ModelWrapper(config, device=device)
    input_size = {"width": config.datamanager.width, "height": config.datamanager.height}
    transform = Transform(input_size, config.datamanager.norm_mean, config.datamanager.norm_std)

    test_time = datetime.now().strftime('%d_%m_%Y_%H:%M')
    out_dir = Path(args.out_dir) / Path(args.config).stem / test_time / 'images'
    out_dir.mkdir(exist_ok=True, parents=True)
    print(out_dir)

    image_paths = Path(args.image)
    if image_paths.is_file():
        image_paths = [image_paths]
    elif image_paths.is_dir():
        image_paths = image_paths.glob('*.*')

    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert("RGB")
        inp = transform.apply(image).to(device)
        out = generator(inp.unsqueeze(0))
        out = out[0].detach().cpu().numpy().transpose(1, 2, 0)
        #out = out * 0.5 + 0.5
        out = out * 127.5 + 127.5
        out = out.astype(np.uint8)
        out = to_pil_image(out)
        out.save(str(out_dir / image_path.name))



