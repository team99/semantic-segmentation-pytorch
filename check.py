import torch
import os
import numpy as np

from PIL import Image
from pdb import set_trace

root_dataset = 'watermark_data/vietnam/annotations/training'

segm_path = os.path.join(root_dataset, "ADE_train_00000001.png")
mask_path = os.path.join(root_dataset, "chotot_1_gs/label_L.png")
new_mask_path = os.path.join(root_dataset, "chotot_1_gs/label_P.png")
# annot_path = os.path.join(root_dataset, "chotot_1_gs/label_gs.png")

segm_img = Image.open(segm_path)
mask_img = Image.open(mask_path)
new_mask_img = Image.open(new_mask_path)
# annot_img = Image.open(annot_path)
# annot_img_L = Image.open(annot_path).convert('L')

# threshold = 2
# new_img = new_mask_img.convert('L').point(lambda p: p > threshold and 1)

segm_arr = np.array(segm_img)
mask_arr = np.array(mask_img)
new_mask_arr = np.array(new_mask_img)
# annot_arr = np.array(annot_img)

# mask_img_gs = mask_img.convert('L')
__import__('pdb').set_trace()

# new_img_arr = np.array(new_img)

print(f"(Segm) Min: {np.min(segm_arr)}, Max: {np.max(segm_arr)}, Unique: {np.unique(segm_arr)}")
print(f"(Mask) Min: {np.min(mask_arr)}, Max: {np.max(mask_arr)}, Unique: {np.unique(mask_arr)}")
# print(f"(New Mask) Min: {np.min(new_mask_arr)}, Max: {np.max(new_mask_arr)}, Unique: {np.unique(new_mask_arr)}")
print(f"(Annot) Min: {np.min(annot_arr)}, Max: {np.max(annot_arr)}, Unique: {np.unique(annot_arr)}")


# new_annot_img = Image.fromarray(new_img_arr)
# new_annot_img.save('watermark_data/vietnam/annotations/training/new_mask_1.png')