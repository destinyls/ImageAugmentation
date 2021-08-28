import argparse
import os
import cv2
import json
import math

from tqdm import tqdm
import numpy as np

from extract_conf import parse_calib_conf, get_camera_params, get_trans_matrix, trans_to_matrix
from kitti_utils import *
from utils import (
  load_intrinsic, compute_box_3d_camera, project_to_image, 
  draw_box_3d, draw_3d_box_on_image, draw_box_on_bev_image,
  load_pcd_data, compute_box_3d_lidar)

from protos.sensing_header_pb2 import FrameId
from protos.sensor_calibration_param_pb2 import CalibrationParam
from google.protobuf import text_format

def setup_kitti_dirs(kitti_root):
  if not os.path.exists(kitti_root):
    os.makedirs(kitti_root)
  if not os.path.exists(os.path.join(kitti_root, "training/image_2")):
    os.makedirs(os.path.join(kitti_root, "training/image_2"))
  if not os.path.exists(os.path.join(kitti_root, "training/label_2")):
    os.makedirs(os.path.join(kitti_root, "training/label_2"))
  if not os.path.exists(os.path.join(kitti_root, "training/calib")):
    os.makedirs(os.path.join(kitti_root, "training/calib"))
  if not os.path.exists(os.path.join(kitti_root, "training/velodyne")):
    os.makedirs(os.path.join(kitti_root, "training/velodyne"))

def takeFirst(elem):
    return elem[0]

def to_kitti_string(class_name, xyz, whl, alpha, rot_y, bbox_2d = (-1.0, -1.0, -1.0, -1.0), truncation = -1.0, occlusion = -1):
  name = class_name + " "
  trunc = "{:.2f} ".format(truncation)
  occ = "{:d} ".format(occlusion)
  a = "{:.2f} ".format(alpha)
  bb = "{:.2f} {:.2f} {:.2f} {:.2f} ".format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
  hwl = "{:.2} {:.2f} {:.2f} ".format(whl[1], whl[0], whl[2])  # height, width, length.
  loc = "{:.2f} {:.2f} {:.2f} ".format(xyz[0], xyz[1], xyz[2])  # x, y, z.
  y = "{:.2f}".format(rot_y)  # Yaw angle.
  output = name + trunc + occ + a + bb + hwl + loc + y
  return output

def convert_to_kitti(sensor_calib_param_path, image_path, pcd_path, annos_3d_path, kitti_root, vis):
  setup_kitti_dirs(kitti_root)
  image_ids = []
  for image_file in os.listdir(image_path):
    image_ids.append((int(image_file[:19]), image_file[19:]))
  image_ids.sort(key=takeFirst)

  for i in tqdm(range(len(image_ids))):
    # Jdx file path
    image_file = str(image_ids[i][0]) + image_ids[i][1]
    pcd_file = os.path.join(pcd_path, image_file.split("_front")[0] + ".pcd")
    annos_file = os.path.join(annos_3d_path, image_file.split("_front")[0] + "_pcd.json")
    image_file = os.path.join(image_path, image_file)
 
    # KITTI file path
    image_2_file = os.path.join(kitti_root, "training/image_2/" + str(image_ids[i][0]) + ".png")
    velodyne_file = os.path.join(kitti_root, "training/velodyne/" + str(image_ids[i][0]) + ".bin")
    calib_file = os.path.join(kitti_root, "training/calib/" + str(image_ids[i][0]) + ".txt")
    label_2_file = os.path.join(kitti_root, "training/label_2/" + str(image_ids[i][0]) + ".txt")

    # image_2
    img = cv2.imread(image_file)
    cv2.imwrite(image_2_file, img)
    # velodyne
    points = load_pcd_data(pcd_file)
    points = points.astype(np.float32)
    with open(velodyne_file, 'w') as f:
      points.tofile(f)
    # calib
    content = parse_calib_conf(sensor_calib_param_path)
    camera_matrix, dist_coeffs = get_camera_params("FRAME_CAMERA360_FRONT", content)
    P2 = np.zeros((3, 4))
    P2[:3, :3] = camera_matrix
    Tr = get_trans_matrix("FRAME_LIDAR_TOP", "FRAME_CAMERA360_FRONT", content)
    extrinsic_matrix = Tr[list(Tr.keys())[0]]
    kitti_transforms = dict()
    kitti_transforms["P0"] = np.zeros((3, 4))
    kitti_transforms["P1"] = np.zeros((3, 4))
    kitti_transforms["P2"] = P2 
    kitti_transforms["P3"] = np.zeros((3, 4))
    kitti_transforms["R0_rect"] = np.identity(3)
    kitti_transforms["Tr_velo_to_cam"] = extrinsic_matrix[:3, :]
    kitti_transforms["Tr_imu_to_velo"] = np.zeros((3, 4))
    with open(calib_file, "w") as f:
        for (key, val) in kitti_transforms.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            f.write("%s: %s\n" % (key, val_str))
    # label_2
    with open(annos_file, "r") as f:
      annos = json.load(f)
    boxes_3d = annos["elem"]

    with open(label_2_file, "w") as label_file:
      for box3d in boxes_3d:
        class_name = box3d["class"]
        whl = np.array([box3d["size"]["width"], box3d["size"]["height"], box3d["size"]["depth"]])
        xyz = np.array([box3d["position"]["x"], box3d["position"]["y"], box3d["position"]["z"]-0.5*whl[1]])  
        yaw = box3d["yaw"]
        # valid box
        if xyz[0] < 0.0:
          continue
        box_3d = compute_box_3d_lidar(whl, xyz, -yaw)
        centre = box_3d.sum(axis=0) / 8
        centre = np.array([centre[0], centre[1], centre[2], 1.0])
        centre_camera = np.dot(centre.reshape(1, 4), extrinsic_matrix.T)[:, :3]
        centre_image = project_to_image(centre_camera, P2)[0]
        if centre_image[0] < 0 or centre_image[0] > img.shape[1] or centre_image[1] < 0 or centre_image[1] > img.shape[0]:
          continue
        
        box_3d = np.hstack((box_3d, np.array([1., 1., 1., 1., 1., 1., 1., 1.]).reshape(8, 1)))
        box_3d = np.dot(box_3d, extrinsic_matrix.T)[:, :3]
        box_2d = project_to_image(box_3d, P2)
        bbox_2d = np.zeros(4)
        bbox_2d[:2] = np.min(box_2d, axis=0)
        bbox_2d[2:] = np.max(box_2d, axis=0)
        bbox_2d = np.maximum(bbox_2d, 0.)
        bbox_2d[0] = np.minimum(bbox_2d[0], img.shape[1])
        bbox_2d[1] = np.minimum(bbox_2d[1], img.shape[0])
        bbox_2d[2] = np.minimum(bbox_2d[2], img.shape[1])
        bbox_2d[3] = np.minimum(bbox_2d[3], img.shape[0])

        rot_y = -0.5 * np.pi - yaw
        if rot_y > np.pi:
          rot_y -= np.pi
        elif rot_y < -np.pi:
          rot_y += np.pi
        alpha = -np.arctan2(-xyz[1], xyz[0]) + rot_y
        xyz = np.hstack((xyz.reshape(1, 3), np.array([1]).reshape(1, 1)))
        xyz = np.dot(xyz, extrinsic_matrix.T)[:, :3].squeeze()
        output = to_kitti_string(class_name, xyz, whl, alpha, rot_y, bbox_2d)
        label_file.write(output + "\n")
    
    if vis:
      img = draw_3d_box_on_image(img, label_2_file, P2, (0, 0, 255))
      range_list = [(-60, 60), (-100, 100), (-2., -2.), 0.1]
      points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
      bev_image = points_filter.get_bev_image(velodyne_file)
      bev_image = draw_box_on_bev_image(bev_image, points_filter, label_2_file, calib_file)
      rows, cols = bev_image.shape[0], bev_image.shape[1]
      bev_image = bev_image[int(0.30*rows) : int(0.5*rows), int(0.20*cols): int(0.80*cols)]
      image_total = np.vstack([img, bev_image])
      cv2.imwrite("demo/" + str(image_ids[i][0]) + ".jpg", image_total)
      cv2.imshow("Image", image_total)
      cv2.waitKey(1)

def main():
  parser = argparse.ArgumentParser(description="Convert from Jdx to KITTI format ...")
  parser.add_argument("--sensor_calib_param_path", type=str,
                      default="../FusionV_131/sensor_calib_param.conf",
                      help="Path to sensor_calib_param.conf")
  parser.add_argument("--image_path", type=str,
                      default="",
                      help="Path to image in jpg format.")
  parser.add_argument("--pcd_path", type=str,
                      default="",
                      help="Path to pointcloud in pcd format.")
  parser.add_argument("--annos_3d_path", type=str,
                      default="",
                      help="Path to 3d annotations in json format.")
  parser.add_argument("--kitti_root", type=str,
                      default="",
                      help="Path to the dataset root in KITTI format.")
  parser.add_argument("--vis", type=bool,
                      default=False,
                      help="Path to the dataset root in KITTI format.")
  args = parser.parse_args()

  convert_to_kitti(args.sensor_calib_param_path,
                   args.image_path,
                   args.pcd_path,
                   args.annos_3d_path,
                   args.kitti_root,
                   args.vis)

if __name__ == "__main__":
  main()
