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

def expand_to_rectangle_box(segm_arr):
    print("Expand watermark into rectangle box")
    new_segm = segm_arr.copy()

    x_amax_value = np.amax(new_segm, axis=0)
    y_amax_value = np.amax(new_segm, axis=1)

    x_index_neighbour_diff = np.where(x_amax_value[:-1] != x_amax_value[1:])[0]
    y_index_neighbour_diff = np.where(y_amax_value[:-1] != y_amax_value[1:])[0]

    x_index_coords = x_index_neighbour_diff.reshape((-1, 2))
    y_index_coords = y_index_neighbour_diff.reshape((-1, 2))

    for i in range(len(x_index_coords)):
        min_x, max_x = x_index_coords[i]
        min_y, max_y = y_index_coords[i]

        new_segm[min_y+1:max_y+1, min_x+1:max_x+1] = 255

    return new_segm

def segm_transform(segm):
    segm = np.array(segm)
    print(f"(Segm) Min: {np.min(segm)}, Max: {np.max(segm)}, Unique: {np.unique(segm)}\n")

    threshold = 1
    new_segm = np.where(segm<threshold, 0, segm)
    new_segm = np.where(new_segm>=threshold, 255, new_segm)
    print(f"(New Segm) Min: {np.min(new_segm)}, Max: {np.max(new_segm)}, Unique: {np.unique(new_segm)}\n")

    return new_segm

for img in img_path:
    print(f'Image {img}\n')
    mask_path = os.path.join(root_dataset, img)
    mask_img = Image.open(mask_path).convert('L')

    new_mask = segm_transform(mask_img)

    new_img = Image.fromarray(new_mask)
    new_img.save(f'watermark_data/new_{img}')

    new_rectangle_segm = expand_to_rectangle_box(new_mask)

    new_rectangle_img = Image.fromarray(new_rectangle_segm)
    new_rectangle_img.save(f'watermark_data/rectangle_{img}')