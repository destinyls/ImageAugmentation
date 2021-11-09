import pickle
import os 
import sys
import numpy as np
from shutil import copyfile

kitti_root = "/root/Dataset/kitti"
imagetxt = "/root/SMOKE/datasets/kitti/ImageSets/backgrounds_in_48666.txt"


def copy_file(file_src, file_dest):
    if not os.path.exists(file_dest):
        print(file_dest)
        try:
            copyfile(file_src, file_dest)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

if __name__ == "__main__":
    with open(imagetxt, 'r') as f:
        lines = f.readlines()
    total_img_ids = [int(line) for line in lines]

    for img_id in total_img_ids:
        img_id_key = "{:06d}".format(img_id)
        print(img_id_key)

        dest_image_2_file = os.path.join(kitti_root, "training/image_3/" + img_id_key + ".png")
        dest_calib_file = os.path.join(kitti_root, "training/calib/" + img_id_key + ".txt")
        dest_label_2_file = os.path.join(kitti_root, "training/label_2/" + img_id_key + ".txt")

        src_image_2_file = os.path.join(kitti_root, "testing/image_3/" + img_id_key + ".png")
        src_calib_file = os.path.join(kitti_root, "testing/calib/" + img_id_key + ".txt")
        src_label_2_file = os.path.join(kitti_root, "testing/label_2/" + img_id_key + ".txt")

        #copy_file(src_calib_file, dest_calib_file)
        copy_file(src_image_2_file, dest_image_2_file)
        #copy_file(src_label_2_file, dest_label_2_file)


        

