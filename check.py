import torch
import os
import numpy as np

from PIL import Image
from pdb import set_trace

root_dataset = 'watermark_data/vietnam/annotations/training'

img_path = [
    'chotot_1.png',
    'batdongsan_1.png'
]

for img in img_path:
    print(f'Image {img}\n')
    mask_path = os.path.join(root_dataset, img)
    mask_img = Image.open(mask_path).convert('L')

    mask_arr = np.array(mask_img)
    print(f"(Mask) Min: {np.min(mask_arr)}, Max: {np.max(mask_arr)}, Unique: {np.unique(mask_arr)}\n")

    new_mask_arr = np.where(mask_arr>=1, 151, mask_arr)
    print(f"(New Mask) Min: {np.min(new_mask_arr)}, Max: {np.max(new_mask_arr)}, Unique: {np.unique(new_mask_arr)}\n")

    tensor_mask = torch.from_numpy(new_mask_arr).long() - 1
    print(f"(Annot) Min: {torch.min(tensor_mask)}, Max: {torch.max(tensor_mask)}, Unique: {torch.unique(tensor_mask)}")

    new_img = Image.fromarray(new_mask_arr)
    new_img.save(f'watermark_data/new_{img}')

# threshold = 2
# new_img = new_mask_img.convert('L').point(lambda p: p > threshold and 1)