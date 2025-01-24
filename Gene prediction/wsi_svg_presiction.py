import timm
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import openslide
from openslide.deepzoom import DeepZoomGenerator
from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainnorm import get_normalizer
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

root = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory

# Initialize the tile encoder model
tile_encoder = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True, pretrained_cfg_overlay=dict(
    file=os.path.join(root, 'save_model/best_model_epoch.bin')))
tile_encoder = tile_encoder.cuda()

# Initialize stain normalizer
target_image = stain_norm_target()
stain_normalizer = get_normalizer("vahadane")
stain_normalizer.fit(target_image)

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def stain_norm_func(img):
    """ Normalize the stain of the image and preprocess for the model input """
    if img.var() > 500:
        try:
            img = stain_normalizer.transform(img)
        except:
            pass
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).cuda()  # Move to GPU
    return img


def process_svs(svs_name):
    """ Process a single SVS file and extract features """
    sample_name = svs_name.split(".")[0]
    feature_matrix = pd.DataFrame([])
    new_index = []

    # Skip if the output already exists
    if os.path.exists(os.path.join(root, 'inferred_svg', sample_name, ".csv")):
        return

    try:
        slide = openslide.OpenSlide(os.path.join(svs_root, svs_name))
        data_gen = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)  ### 256 for 20x; 512 for 40x

        for j in range(data_gen.level_tiles[-1][1] - 1):  # Iterate through height
            for i in range(data_gen.level_tiles[-1][0] - 1):  # Iterate through width
                patch = data_gen.get_tile(len(data_gen.level_tiles) - 1, (i, j))
                np_patch = np.array(patch)

                new_index.append(f"{i}_{j}")
                # Perform stain normalization and preprocessing
                sample_input = stain_norm_func(np_patch)
                with torch.no_grad():
                    output = tile_encoder(sample_input).squeeze()
                    data = pd.DataFrame(output.detach().cpu().numpy())  # 1024-D
                    feature_matrix = pd.concat([feature_matrix, data.T])

        feature_matrix.index = new_index
        feature_matrix.to_csv(os.path.join(root, 'inferred_svg', sample_name, ".csv"), sep=',')
    except Exception as e:
        print(f"Error processing {svs_name}: {e}")


# Path to SVS files
svs_root = os.path.join(root, 'svs')

# Using ProcessPoolExecutor to process multiple SVS files in parallel (optimized for I/O bound tasks)
cpu_count = multiprocessing.cpu_count()
batch_size = max(cpu_count // 2, 1)  # Set batch size based on number of available CPUs

with ProcessPoolExecutor(max_workers=batch_size) as executor:
    svs_files = [f for f in os.listdir(svs_root) if f.endswith('.svs')]  # List of SVS files
    executor.map(process_svs, svs_files)


