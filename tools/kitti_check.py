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

  image_path = os.path.join(kitti_root, "testing/image_2")
  velodyne_path = os.path.join(kitti_root, "testing/velodyne")
  calib_path = os.path.join(kitti_root, "testing/calib")
  label_path = os.path.join(kitti_root, "testing/label_2")
  pred_path_baseline = os.path.join(pred_path, "baseline")
  pred_path_ssl = os.path.join(pred_path, "mix_teaching_21.29")
  image_ids = []
  for image_file in os.listdir(pred_path_baseline):
    image_ids.append(image_file.split(".")[0])
  for i in range(len(image_ids)):
    image_2_file = os.path.join(image_path, str(image_ids[i]) + ".png")
    velodyne_file = os.path.join(velodyne_path, str(image_ids[i]) + ".bin")
    calib_file = os.path.join(calib_path, str(image_ids[i]) + ".txt")
    label_2_file = os.path.join(label_path, str(image_ids[i]) + ".txt")
    pred_file_baseline = os.path.join(pred_path_baseline, str(image_ids[i]) + ".txt")
    pred_file_ssl = os.path.join(pred_path_ssl, str(image_ids[i]) + ".txt")

    # Image Checking
    image_baseline = cv2.imread(image_2_file)
    image_ssl = image_baseline.copy()

    K, P2 = load_intrinsic(calib_file)
    _, cam_to_vel = KittiCalibration.get_transform_matrix_origin(calib_file)
    
    image_baseline = draw_3d_box_on_image(image_baseline, pred_file_baseline, P2, (255, 0, 0))
    image_ssl = draw_3d_box_on_image(image_ssl, pred_file_ssl, P2, (0, 255, 0))
    width = image_baseline.shape[1]
    height = image_baseline.shape[0]
    side_distance = height * 0.04
    fwd_distance = width * 0.08
    range_list = [(-side_distance, side_distance), (-fwd_distance, fwd_distance), (-5., 30.), 0.08]

    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    if not os.path.exists(velodyne_file):
    	continue
    bev_image = points_filter.get_bev_image(velodyne_file)
    bev_image = 255 - bev_image
    gt_bev_image = draw_box_on_bev_image(bev_image, points_filter, label_2_file, cam_to_vel, (0, 0, 255))
    bev_image_baseline = gt_bev_image.copy()
    bev_image_ssl = gt_bev_image.copy()

    bev_image_baseline = draw_box_on_bev_image(bev_image_baseline, points_filter, pred_file_baseline, cam_to_vel, (255, 0, 0))
    bev_image_ssl = draw_box_on_bev_image(bev_image_ssl, points_filter, pred_file_ssl, cam_to_vel, (0, 255, 0))

    bev_image_baseline = np.rot90(bev_image_baseline)
    bev_image_ssl = np.rot90(bev_image_ssl)
    
    image_baseline = image_baseline[:, :int(width), :]
    image_ssl = image_ssl[:, :int(width), :]

    bev_image_baseline = bev_image_baseline[:, :int(width), :]
    bev_image_ssl = bev_image_ssl[:, :int(width), :]

    print(image_baseline.shape)
    print(bev_image_ssl.shape)
    
    total_image_baseline = np.vstack([image_baseline, bev_image_baseline[:-1,...]])
    total_image_ssl = np.vstack([image_ssl, bev_image_ssl[:-1,...]])

    print(total_image_baseline.shape, total_image_ssl.shape)
    total_image = np.hstack([total_image_baseline, total_image_ssl])

    cv2.imwrite(os.path.join(save_path, str(image_ids[i]) + ".jpg"), total_image)

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
