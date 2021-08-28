import os
import cv2
from shutil import copyfile

split_path = "/root/Dataset/kitti/kitti_splits"
merge_path = "/root/Dataset/kitti/kitti_raw_dataset"

split_names = ["split_1", "split_2", "split_3", "split_4", "split_5", "split_6", "split_7"]

def copy_file(file_src, file_dest):
    if not os.path.exists(file_dest):
        try:
            copyfile(file_src, file_dest)
        except IOError as e:
            print()
            print("Unable to copy file", file_dest)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

if __name__ == "__main__":
    if not os.path.exists(os.path.join(merge_path, "image_2")):
            os.makedirs(os.path.join(merge_path, "image_2"))
    if not os.path.exists(os.path.join(merge_path, "image_3")):
        os.makedirs(os.path.join(merge_path, "image_3"))
    if not os.path.exists(os.path.join(merge_path, "calib")):
        os.makedirs(os.path.join(merge_path, "calib"))
    if not os.path.exists(os.path.join(merge_path, "velodyne")):
        os.makedirs(os.path.join(merge_path, "velodyne"))
    if not os.path.exists(os.path.join(merge_path, "label_2")):
        os.makedirs(os.path.join(merge_path, "label_2"))


    global_idx = 0
    for split_n in split_names:
        print(split_n, "processing ...")
        split_sub_path = os.path.join(split_path, split_n)
        split_image_2_path = os.path.join(split_sub_path, "image_2")
        split_image_3_path = os.path.join(split_sub_path, "image_3")
        split_calib_path = os.path.join(split_sub_path, "calib")
        split_label_2_path = os.path.join(split_sub_path, "label_2")
        split_velodyne_path = os.path.join(split_sub_path, "velodyne")

        for idx in range(7518):
            print("frame id: ", idx)
            source_frame_name = "{:06d}".format(idx)
            source_image_2 = os.path.join(split_image_2_path, source_frame_name + ".png")
            source_image_3 = os.path.join(split_image_3_path, source_frame_name + ".png")
            source_calib = os.path.join(split_calib_path, source_frame_name + ".txt")
            source_label_2 = os.path.join(split_label_2_path, source_frame_name + ".txt")
            source_velodyne = os.path.join(split_velodyne_path, source_frame_name + ".bin")

            dest_frame_name = "{:06d}".format(global_idx)
            dest_image_2 = os.path.join(merge_path, "image_2", dest_frame_name + ".png")
            dest_image_3 = os.path.join(merge_path, "image_3", dest_frame_name + ".png")
            dest_calib = os.path.join(merge_path, "calib", dest_frame_name + ".txt")
            dest_label_2 = os.path.join(merge_path, "label_2", dest_frame_name + ".txt")
            dest_velodyne = os.path.join(merge_path, "velodyne", dest_frame_name + ".bin")
            global_idx = global_idx + 1

            copy_file(source_image_2, dest_image_2)
            copy_file(source_image_3, dest_image_3)
            copy_file(source_calib, dest_calib)
            copy_file(source_velodyne, dest_velodyne)
            copy_file(source_label_2, dest_label_2)
















