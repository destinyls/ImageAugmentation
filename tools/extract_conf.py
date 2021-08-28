import cv2
import math
import numpy as np
import os

from protos.sensing_header_pb2 import FrameId
from protos.sensor_calibration_param_pb2 import CalibrationParam
from google.protobuf import text_format

def undistort_image(image, camera_matrix, dist_coeffs):
    image = cv2.undistort(image, camera_matrix, dist_coeffs)
    return image

def get_camera_params(frame, calib_content):
    camera_params = calib_content.camera_calibration_intrinsic_map.camera_param_map[frame]
    camera_matrix = np.array([float(i) for i in camera_params.intrisic_param]).reshape((3, 3))
    dist_coeffs = np.array([float(i) for i in camera_params.distortion_param])
    return camera_matrix, dist_coeffs

def trans_to_matrix(trans):
    extrinsic_param = [float(i) for i in trans]
    rmat = eulerAnglesToRotationMatrix(extrinsic_param[3:])
    tvec = np.array([extrinsic_param[:3]]).astype(np.float64)
    extrinsic_matrix = np.concatenate([rmat, tvec.reshape(3, 1)], axis=1)
    trans_matrix = np.vstack((extrinsic_matrix, np.array([0.,0.,0.,1.]).reshape(1,4)))
    return trans_matrix

def get_trans_matrix(source_frame, target_frame, calib_content):
    matrix_key = ""
    trans_matrix = []
    trans = calib_content.connected_frame_trans_container.connected_frame_trans
    for tr in trans:
        tr_target = FrameId.Name(tr.target_frame)
        tr_source = FrameId.Name(tr.source_frame)
        if tr_target == target_frame and tr_source == source_frame:
            matrix_key = "Tr_" + source_frame + "_to_" + target_frame 
            trans_matrix = trans_to_matrix(tr.trans)
            break
        elif tr_target == source_frame and tr_source == target_frame:
            matrix_key = "Tr_" + target_frame + "_to_" + source_frame 
            trans_matrix = trans_to_matrix(tr.trans)
            break
    return {matrix_key: trans_matrix}

def parse_calib_conf(calib_file):
    pb = CalibrationParam()
    with open(calib_file, "rb") as f:
        content = text_format.Parse(f.read(), pb)
    return content

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0],
                    [0,         math.cos(theta[0]), -math.sin(theta[0])],
                    [0,         math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                    [0,                     1,      0],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
