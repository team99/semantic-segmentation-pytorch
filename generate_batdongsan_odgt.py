import os
import cv2
import json

def odgt(img_path):
    seg_path = img_path.replace('images','masks')
    seg_path = seg_path.replace('.jpg','.png')

    if os.path.exists(seg_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        odgt_dic = {}
        odgt_dic["fpath_img"] = img_path
        odgt_dic["fpath_segm"] = seg_path
        odgt_dic["width"] = h
        odgt_dic["height"] = w
        return odgt_dic
    else:
        print('the corresponded annotation does not exist')
        print(img_path)
        return None

def save_to_odgt(file_path, img_list):
    with open(file_path, mode='wt', encoding='utf-8') as myodgt:
        for i, img in enumerate(img_list):
            a_odgt = odgt(img)
            if a_odgt is not None:
                myodgt.write(f'{json.dumps(a_odgt)}\n')


if __name__ == "__main__":
    modes = ['training','validation']
    custom_name = "batdongsan_"

    dir_path = "watermark_data/batdongsan/images"
    img_list = os.listdir(dir_path)
    img_list.sort()
    img_list = [os.path.join(dir_path, img) for img in img_list]
    training_split = int(0.8 * len(img_list))
    training_list = img_list[:training_split]
    validation_list = img_list[training_split:]

    save_to_odgt('watermark_data/batdongsan_training.odgt', training_list)
    save_to_odgt('watermark_data/batdongsan_validation.odgt', validation_list)