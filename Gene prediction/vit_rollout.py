import timm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd
import openslide
from openslide.deepzoom import DeepZoomGenerator
from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainnorm import get_normalizer
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f"Using device: {device}")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(device)}")

root = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
tile_encoder = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True, pretrained_cfg_overlay=dict(
    file=os.path.join(root, 'pretrained/pytorch_model.bin')))

class Encoder(nn.Module):
    def __init__(self, tile_encoder, output):
        super(Encoder, self).__init__()
        self.tile_encoder = tile_encoder
        self.head_drop = nn.Dropout(p=0.5)
        self.new_head = nn.Linear(1536, output)

    def forward(self, x):
        x = self.tile_encoder(x)
        x = self.head_drop(x)
        x = self.new_head(x)
        return x

    def compute_loss(self, regression_output, target):
        return F.mse_loss(regression_output.view(-1), target.view(-1))

# Initialize stain normalizer
target_image = stain_norm_target()
stain_normalizer = get_normalizer("vahadane")
stain_normalizer.fit(target_image)

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def stain_norm_func(img):
    if img.var() > 200:
        try:
            img = stain_normalizer.transform(img)
        except:
            pass

    # for vit
    image = Image.fromarray(img)
    return transform(image)

def process_tile(tile, patch_size=(256, 256)):
    """ Preprocess and normalize the tile. """
    tile_resized = cv2.resize(tile, patch_size)
    return stain_norm_func(tile_resized)

def process_batch(batch):
    """ Process a batch of images. """
    batch = torch.stack(batch, dim=0).to(device)
    return batch

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    model = Encoder(tile_encoder, output=138).to(device)
    model_path = r'../save_model/best_model_epoch.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    svs_root = "../YALE-HER2/wsi/"

    for svs_name in os.listdir(svs_root):
        sample_name = svs_name.split(".")[0]
        feature_matrix = pd.DataFrame([])
        new_index = []

        slide = openslide.OpenSlide(os.path.join(svs_root, svs_name))
        data_gen = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)

        batch_images = []
        tile_positions = []

        # Process in batches
        for j in range(data_gen.level_tiles[-1][1] - 1):  # 0:w, 1:h
            for i in range(data_gen.level_tiles[-1][0] - 1):
                patch = data_gen.get_tile(len(data_gen.level_tiles) - 1, (i, j))
                np_patch = np.array(patch)

                if np_patch.var() > 300:
                    image = process_tile(np_patch)
                    batch_images.append(image)
                    tile_positions.append((i, j))

        # Process images in batch
        if batch_images:
            # Process batch of images
            img_batch = process_batch(batch_images)

            # Compute Grad Rollout for the entire batch
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.9)
            grad_masks = grad_rollout(img_batch, category_index=0)

            for idx, (i, j) in enumerate(tile_positions):
                mask = grad_masks[idx]
                if idx == 0:
                    temp_result = np.array(batch_images[idx])
                    temp_mask = mask
                elif i == data_gen.level_tiles[-1][0] - 2:
                    if j == 0:
                        result = temp_result
                        mask_combined = temp_mask
                    else:
                        result = np.vstack([result, temp_result])
                        mask_combined = np.vstack([mask_combined, temp_mask])
                else:
                    temp_result = np.hstack([temp_result, np.array(batch_images[idx])])
                    temp_mask = np.hstack([temp_mask, mask])

        # Post-processing and visualization
        result = np.float32(result) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_combined), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        vit_atten = heatmap + np.float32(result)
        vit_atten = vit_atten / np.max(vit_atten)

        # Save results
        output_path = os.path.join('../vit_rollout', f'{sample_name}.png')
        cv2.imwrite(output_path, vit_atten)
        print(f"Processed {sample_name}")
