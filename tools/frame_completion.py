import os
import sys

from shutil import copyfile

source_path = "/home/yanglei/Dataset/kitti/kitti_splits/split_1"
dest_path = "/home/yanglei/Dataset/kitti/kitti_splits/split_7"


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

for image_name in os.listdir(os.path.join(source_path, "image_2")):
    frame_id = image_name.split(".")[0]

    src_image_02_name = os.path.join(source_path, "image_2", frame_id + ".png")
    src_image_03_name = os.path.join(source_path, "image_3", frame_id + ".png")
    src_velodyne_name = os.path.join(source_path, "velodyne", frame_id + ".bin")
    src_calib_name = os.path.join(source_path, "calib", frame_id + ".txt")
    
    dest_image_02_name = os.path.join(dest_path, "image_2", frame_id + ".png")
    dest_image_03_name = os.path.join(dest_path, "image_3", frame_id +".png")
    dest_velodyne_name = os.path.join(dest_path, "velodyne", frame_id + ".bin")
    dest_calib_name = os.path.join(dest_path, "calib", frame_id + ".txt")

    copy_file(src_image_02_name, dest_image_02_name)
    copy_file(src_image_03_name, dest_image_03_name)
    copy_file(src_velodyne_name, dest_velodyne_name)
    copy_file(src_calib_name, dest_calib_name)