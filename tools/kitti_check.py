import argparse
import os
import time
import csv
import cv2
import numpy as np
from PIL import Image

from kitti_utils import *
from utils import (
  load_intrinsic, compute_box_3d_camera, project_to_image, 
  draw_box_3d, draw_3d_box_on_image, draw_box_on_bev_image)

import warnings
warnings.filterwarnings("ignore")



def kitti_format_check(kitti_root, pred_path, save_path):
  if not os.path.exists(kitti_root):
    raise ValueError("kitti_root Not Found")

  if not os.path.exists(save_path):
    os.makedirs(save_path)

  image_path = os.path.join(kitti_root, "image_02/data")
  velodyne_path = os.path.join(kitti_root, "velodyne_points/data")
  calib_path = os.path.join(kitti_root, "../")
  label_path = os.path.join(kitti_root, "training/label_2")
  image_ids = []
  for image_file in os.listdir(image_path):
    image_ids.append(image_file.split(".")[0])
  for i in range(len(image_ids)):

    print("processing ...", image_ids[i])
    image_2_file = os.path.join(image_path, str(image_ids[i]) + ".png")
    velodyne_file = os.path.join(velodyne_path, str(image_ids[i]) + ".bin")
    cam_calib_file = os.path.join(calib_path, "calib_cam_to_cam.txt")
    lidar_calib_file = os.path.join(calib_path, "calib_velo_to_cam.txt")

    label_2_file = os.path.join(pred_path, str(image_ids[i]) + ".txt")

    # Image Checking
    image = cv2.imread(image_2_file)
    K, P2 = load_intrinsic(cam_calib_file)
    _, cam_to_vel, R0_rect, Tr_velo_to_cam = KittiCalibration.get_transform_matrix(lidar_calib_file, cam_calib_file)

    image = draw_3d_box_on_image(image, label_2_file, P2, (255, 0, 0))
    width = image.shape[1]
    height = image.shape[0]
    side_distance = height * 0.04
    fwd_distance = width * 0.08
    range_list = [(-side_distance, side_distance), (-fwd_distance, fwd_distance), (-5., 30.), 0.08]

    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])

    if not os.path.exists(velodyne_file):
    	continue

    bev_image = points_filter.get_bev_image(velodyne_file, R0_rect, Tr_velo_to_cam, P2, (height, width))
    bev_image = 255 - bev_image

    gt_bev_image = draw_box_on_bev_image(bev_image, points_filter, label_2_file, cam_to_vel, (255, 0, 0))

    origin_bev_image = gt_bev_image.copy()
    origin_bev_image = np.rot90(origin_bev_image)
    '''
    origin_bev_image = np.rot90(origin_bev_image)
    origin_bev_image = np.rot90(origin_bev_image)
    '''
    origin_image = image[:, :int(width), :]
    origin_bev_image = origin_bev_image[:, :int(width), :]

    print(origin_image.shape)
    print(origin_bev_image.shape)
    
    origin_total_image = np.hstack([origin_image, origin_bev_image[:-1,...]])
    cv2.imwrite(os.path.join(save_path, str(image_ids[i]) + ".jpg"), origin_total_image)

def main():
  parser = argparse.ArgumentParser(description="Dataset in KITTI format Checking ...")
  parser.add_argument("--kitti_root", type=str,
                      default="",
                      help="Path to KITTI Dataset root")

  parser.add_argument("--pred_path", type=str,
                      default="",
                      help="Path to model predictions")

  parser.add_argument("--save_path", type=str,
                      default="",
                      help="Path to save iamges")

  args = parser.parse_args()
  kitti_format_check(args.kitti_root, args.pred_path, args.save_path)


if __name__ == "__main__":
  main()
