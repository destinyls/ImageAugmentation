import sys
import cv2
import csv
import numpy as np
from kitti_utils import *

range = 1000
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]


def draw_3d_box_on_image(image, label_2_file, P2, c=(255, 0, 0)):
    with open(label_2_file) as f:
      for line in f.readlines():
          line_list = line.split('\n')[0].split(' ')
          object_type = line_list[0]

          if object_type != "Car":
              continue
          occluded = line_list[2]
          dim = np.array(line_list[8:11]).astype(float)
          location = np.array(line_list[11:14]).astype(float)
          rotation_y = float(line_list[14])
          box_3d = compute_box_3d_camera(dim, location, rotation_y)
          box_2d = project_to_image(box_3d, P2)
          color = color_list[int(occluded)]
          image = draw_box_3d(image, box_2d, c=c)
    return image

def draw_box_on_bev_image_v2(bev_image, points_filter, label_2_file, cam_to_vel, c=(0, 255, 0)):
    with open(label_2_file) as f:
      for line in f.readlines():
          line_list = line.split('\n')[0].split(' ')
          object_type = line_list[0]
          occluded = line_list[2]
          if object_type != "Car":
              continue

          dimensions = np.array(line_list[8:11]).astype(float)
          location = np.array(line_list[11:14]).astype(float)
          rotation_y = float(line_list[14])
          corner_points = get_object_corners_in_lidar(cam_to_vel, dimensions, location, rotation_y)

          x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
          cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[1], y_img[1]), c, 2)
          cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[2], y_img[2]), c, 2)
          cv2.line(bev_image, (x_img[1], y_img[1]), (x_img[3], y_img[3]), c, 2)
          cv2.line(bev_image, (x_img[2], y_img[2]), (x_img[3], y_img[3]), c, 2)

          center_point = (int(np.mean(x_img)), int(np.mean(y_img)))
          front_point = (int(0.5 * (x_img[0] + x_img[1])), int(0.5 * (y_img[0] + y_img[1])))
          cv2.line(bev_image, center_point, front_point, c, 2)
          
    return bev_image

def load_intrinsic(calib_file):
    with open(os.path.join(calib_file), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P_rect_02:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
            if row[0] == 'P_rect_03:':
                P3 = row[1:]
                P3 = [float(i) for i in P3]
                P3 = np.array(P3, dtype=np.float32).reshape(3, 4)
                break
    return P3, P2

def load_pcd_data(pcd_file):
    pts = []
    with open(pcd_file, 'r') as f:
      data = f.readlines()
    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        if "nan" in xyzi:
          continue
        x, y, z = [eval(i) for i in xyzi[:3]]
        intensity = float(xyzi[-1])
        pts.append([x, y, z, intensity])
    point_cloud = np.array(pts)
    return point_cloud

def compute_box_3d_lidar(dim, location, yaw):
  c, s = np.cos(yaw), np.sin(yaw)
  R = np.array([[ c,  s, 0],
                [-s,  c, 0],
                [ 0,  0, 1]], dtype=np.float32)
  w, h, l = dim[0], dim[1], dim[2]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  z_corners = [0,0,0,0,h,h,h,h]
  y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def compute_box_3d_camera(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def draw_box_3d(image, corners, c=(0, 255, 0)):
  face_idx = [[0,1,5,4],
              [1,2,6,5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in [3, 2, 1, 0]:
    f = face_idx[ind_f]
    for j in [0, 1, 2, 3]:
      cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
               (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
               (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
               (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
  return image

# return whether one lidar point valid
def is_point_valid(point, valid_area):
    nan = [np.isnan(x) for x in point]
    if any(nan):
        return False
    elif point[0] > valid_area[0] and point[0] < valid_area[1] \
            and point[1] > valid_area[2] and  point[1] < valid_area[3]:
        return True
    else:
        return False

# project lidar points to image points
def lidar_to_image(points_cloud, extrinsic_matrix, camera_matrix):
    points_cloud = points_cloud.copy()
    points_cloud[:, 3] = 1.0
    points_cloud = points_cloud.T
    image_points = np.dot(np.dot(camera_matrix, extrinsic_matrix), points_cloud).T.reshape([-1, 3])
    image_points[:, 0] = image_points[:, 0] / image_points[:, 2]
    image_points[:, 1] = image_points[:, 1] / image_points[:, 2]
    image_points = image_points[:, :2].reshape([-1, 1, 2])
    return image_points

class Projector():
    def __init__(self, camera_view, camera_matrix):
        color_map = np.arange(256, dtype=np.uint8) 
        self.color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
        self.camera_matrix = camera_matrix
        self.set_camera_view(camera_view)

    def set_camera_view(self, camera_view):
        self.camera_view = camera_view
        if camera_view == 'right':
            self.valid_area = [-range, range, -range, 0]
        elif camera_view == 'rear':
            self.valid_area = [-range, 0, -range,  range]
        elif camera_view == 'left':
            self.valid_area = [-range, range, 0, range]
        elif camera_view in ['front', 'long1', 'long2']:
            self.valid_area = [0, range, -range,  range]
        else:
            print("warning: cannot decide valid_area!")
            sys.exit()

    def project(self, points_cloud, image, extrinsic_matrix):
        points_cloud = points_cloud[points_cloud[:, 0] > self.valid_area[0]]
        points_cloud = points_cloud[points_cloud[:, 0] < self.valid_area[1]]
        points_cloud = points_cloud[points_cloud[:, 1] > self.valid_area[2]]
        points_cloud = points_cloud[points_cloud[:, 1] < self.valid_area[3]]

        # obtain the distance, which will be used to draw the points
        distance = np.linalg.norm(points_cloud[:, :3], axis=1)
        image_points = lidar_to_image(points_cloud, extrinsic_matrix, self.camera_matrix).squeeze()
        # draw the image points
        h, w, _ = image.shape
        draw_points = []
        draw_distances = []
        for ip, dis in zip(image_points, distance):
            if ip[0] >= 0 and ip[0] < 1200 and ip[1] >= 0 and ip[1] < 800:
                draw_points.append(ip)
                draw_distances.append(dis)

        max_distance = max(draw_distances)
        min_distance = min(draw_distances)
        if max_distance - min_distance > 5:
            max_distance -= 5
        for ip, dis in zip(draw_points, draw_distances):
                dis = (dis - min_distance) / max_distance * 256
                dis = min(int(dis), 255)
                color = tuple(self.color_map[dis, 0].astype(np.int))
                color = (int(color[0]), int(color[1]), int(color[2]))
                image = cv2.circle(image, (int(ip[0]), int(ip[1])), 1, color, -1)
        return image
