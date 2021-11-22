import argparse
import os
import cv2
import numpy as np

from visualizer.kitti_utils import *
from visualizer.utils import (
  load_intrinsic, compute_box_3d_camera, project_to_image, 
  draw_box_3d, draw_3d_box_on_image, draw_box_on_bev_image_v2)

import warnings
warnings.filterwarnings("ignore")

def kitti_visual_tool_api(image_file, calib_file, label_file, pred_path, velodyne_file=None, overall_new_boxes=None, gt_boxes=None):
  image = cv2.imread(image_file)
  K, P2 = load_intrinsic(calib_file)
  image = draw_3d_box_on_image(image, pred_path, P2)
  
  range_list = [(-60, 60), (-100, 100), (-2., -2.), 0.1]
  points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
  bev_image = points_filter.get_meshgrid()
  bev_image = cv2.merge([bev_image, bev_image, bev_image])
  if velodyne_file is not None:
    bev_image = points_filter.get_bev_image(velodyne_file)

  bev_image = draw_box_on_bev_image_v2(bev_image, points_filter, label_file, pred_path, calib_file, overall_new_boxes, gt_boxes)
  rows, cols = bev_image.shape[0], bev_image.shape[1]
  bev_image = bev_image[int(0.30*rows) : int(0.5*rows), int(0.20*cols): int(0.80*cols)]
  width = int(bev_image.shape[1])
  height = int(image.shape[0] * bev_image.shape[1] / image.shape[1])
  image = cv2.resize(image, (width, height))
  image = np.vstack([image, bev_image])
  return image


def kitti_visual_tool(kitti_root):
  if not os.path.exists(kitti_root):
    raise ValueError("kitti_root Not Found")

  image_path = os.path.join(kitti_root, "training/image_2")
  velodyne_path = os.path.join(kitti_root, "training/velodyne")
  calib_path = os.path.join(kitti_root, "training/calib")
  label_path = os.path.join(kitti_root, "training/label_2")
  image_ids = []
  for image_file in os.listdir(image_path):
    image_ids.append(image_file.split(".")[0])
  for i in range(len(image_ids)):
    image_2_file = os.path.join(image_path, str(image_ids[i]) + ".png")
    velodyne_file = os.path.join(velodyne_path, str(image_ids[i]) + ".bin")
    calib_file = os.path.join(calib_path, str(image_ids[i]) + ".txt")
    label_2_file = os.path.join(label_path, str(image_ids[i]) + ".txt")

    image = kitti_visual_tool_api(image_2_file, calib_file, label_2_file, velodyne_file)
    cv2.imshow("Image", image)
    cv2.waitKey(300)

def main():
  parser = argparse.ArgumentParser(description="Dataset in KITTI format Checking ...")
  parser.add_argument("--kitti_root", type=str,
                      default="",
                      help="Path to KITTI Dataset root")
  args = parser.parse_args()
  kitti_visual_tool(args.kitti_root)


if __name__ == "__main__":
  main()
