import os
import random

label_origin = "/root/SMOKE/datasets/kitti/ImageSets/trainval.txt"
back_ground = "/root/SMOKE/datasets/kitti/ImageSets/backgrounds_in_48666.txt"
label_enhanced = "/root/SMOKE/datasets/kitti/ImageSets/train_9472.txt"

if __name__ == "__main__":
    kitti_root = "/root/Dataset/kitti"
    with open(label_origin, 'r') as f:
        lines = f.readlines()
    train_img_ids = [int(line) for line in lines]

    with open(back_ground, 'r') as f:
        lines = f.readlines()
    bk_img_ids = [int(line) for line in lines]

    selected = random.sample(bk_img_ids, 1991)
    print(len(selected))

    img_ids = train_img_ids + selected
    print("+++++++++>: ", len(img_ids))
    with open(label_enhanced,'w') as f:
        for idx in img_ids:
            frame_name = "{:06d}".format(idx)
            f.write(frame_name)
            f.write("\n")