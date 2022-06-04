import torch
import os
import numpy as np

from PIL import Image
from pdb import set_trace

root_dataset = 'watermark_data/vietnam/annotations/training'
save_directory = 'watermark_data/vietnam/check'

img_path = [
    'chotot_1.png',
    'batdongsan_1.png',
    'batdongsan_2.png',
    'batdongsan_3.png',
    'batdongsan_4.png',
    'batdongsan_5.png',
    'batdongsan_6.png',
]

def expand_to_rectangle_box(segm_arr):
    print("Expand watermark into rectangle box")
    new_segm = segm_arr.copy()

    # Scan through x-axis for sudden pixel value change from 0-255 / 255-0
    x_amax_value = np.amax(new_segm, axis=0)
    x_index_neighbour_diff = np.where(x_amax_value[:-1] != x_amax_value[1:])[0]
    x_index_coords = x_index_neighbour_diff.reshape((-1, 2)) # Pair into 2-index tuple

    # Iterate through each pair of x-indices
    for (min_x, max_x) in x_index_coords:
        # Scan through y-axis for sudden pixel value change from 0-255 / 255-0
        y_amax_value = np.amax(new_segm[:, min_x+1:max_x+1], axis=1)
        y_index_neighbour_diff = np.where(y_amax_value[:-1] != y_amax_value[1:])[0]
        y_index_coords = y_index_neighbour_diff.reshape((-1, 2))

        # Iterate through each pair of y-indices
        for (min_y, max_y) in y_index_coords:
            # Update value of the area within x-y coordinates into 255
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
    new_img.save(f'{save_directory}/new_{img}')

    new_rectangle_segm = expand_to_rectangle_box(new_mask)

    new_rectangle_img = Image.fromarray(new_rectangle_segm)
    new_rectangle_img.save(f'{save_directory}/rectangle_{img}')