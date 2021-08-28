import os
import sys
from shutil import copyfile

source_path = "/home/yanglei/Dataset/kitti/kitti_raw_dataset"
dest_path = "/home/yanglei/Dataset/kitti/kitti_splits"
split_path = ["split_1", "split_2", "split_3", "split_4", "split_5", "split_6", "split_7"]

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
    for data_id in range(47490):
        src_frame_id = data_id
        split_id = data_id / 7518
        dest_frame_id = data_id % 7518
        src_frame_name = "{:06d}".format(src_frame_id)
        dest_frame_name = "{:06d}".format(dest_frame_id)

        split_name = split_path[split_id]
        split_folder_name = os.path.join(dest_path, split_name)
        if not os.path.exists(os.path.join(split_folder_name, "image_2")):
            os.makedirs(os.path.join(split_folder_name, "image_2"))
        if not os.path.exists(os.path.join(split_folder_name, "image_3")):
            os.makedirs(os.path.join(split_folder_name, "image_3"))
        if not os.path.exists(os.path.join(split_folder_name, "calib")):
            os.makedirs(os.path.join(split_folder_name, "calib"))
        if not os.path.exists(os.path.join(split_folder_name, "velodyne")):
            os.makedirs(os.path.join(split_folder_name, "velodyne"))

        src_image_02_name = os.path.join(source_path, "image_2", src_frame_name + ".png")
        src_image_03_name = os.path.join(source_path, "image_3", src_frame_name + ".png")
        src_velodyne_name = os.path.join(source_path, "velodyne", src_frame_name + ".bin")
        src_calib_name = os.path.join(source_path, "calib", src_frame_name + ".txt")
       
        dest_image_02_name = os.path.join(split_folder_name, "image_2", dest_frame_name + ".png")
        dest_image_03_name = os.path.join(split_folder_name, "image_3", dest_frame_name +".png")
        dest_velodyne_name = os.path.join(split_folder_name, "velodyne", dest_frame_name + ".bin")
        dest_calib_name = os.path.join(split_folder_name, "calib", dest_frame_name + ".txt")

        copy_file(src_image_02_name, dest_image_02_name)
        copy_file(src_image_03_name, dest_image_03_name)
        copy_file(src_velodyne_name, dest_velodyne_name)
        copy_file(src_calib_name, dest_calib_name)

        

        




