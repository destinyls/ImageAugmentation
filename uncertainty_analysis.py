import os
import cv2

from kitti_common import iou 
import numpy as np
import torch
import random
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from evaluation.kitti_object_eval_python.eval import bev_box_overlap, d3_box_overlap
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D

from tools.visualizer.kitti_utils import *

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Van': 3,
    'Person_sitting': 4,
    'Truck': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1,
}

def prepare_boxes(annos, used_classes = ["Car", "Pedestrian", "Cyclist"]):
    class_boxes_3d = dict()
    for class_key in used_classes:
        class_boxes_3d[class_key] = None
    names = annos["name"] 
    dimensions = annos["dimensions"]
    locations = annos["location"]
    rotys = annos["rotation_y"]
    num_obj = np.sum(annos["index"] >= 0)
    if num_obj == 0:
        return class_boxes_3d
    
    # 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry'
    boxes_3d = np.hstack((locations, dimensions, rotys[:, np.newaxis]))
    class_indices = dict()
    for class_key in used_classes:
        class_indices[class_key] = []
    for i in range(num_obj):
        for class_key in used_classes:
            if names[i] == class_key:
                class_indices[class_key].append(i)
    
    for class_key in used_classes:
        class_boxes_3d[class_key] = boxes_3d[class_indices[class_key], :]
    return class_boxes_3d

def plot_scatter_diagram(total_frame_samples):
    plt.xlabel('geo_conf/score')
    plt.ylabel('iou_3d')
    plt.xlim(xmax=1.0, xmin=0)
    plt.ylim(ymax=1.0, ymin=0)
    colors_1 = '#DC143C'
    colors_2 = '#BDEE3C'
    area = np.pi * 1**2
    count = 0
    for frame_samples in tqdm(total_frame_samples):
        for sample in frame_samples:
            plt.scatter(sample["score"], sample["iou_3d"], s=area, marker='+', c=colors_1, alpha=0.4, label='score')
            plt.scatter(sample["geo_conf"], sample["iou_3d"], s=area, marker='o', c=colors_2, alpha=0.4, label='geo_conf')
            # plt.text(sample["score"], sample["iou_3d"], count, color='r', ha='center', va='bottom')
            # plt.text(sample["geo_conf"], sample["iou_3d"], count, color='b', ha='center', va='bottom')
            count += 1
    print("sample count: ", count)
    plt.savefig("scatter_diagram_geo_conf_0.25.png", dpi=1500)

def visualization(image_idx, frame_samples):
    kitti_root = "/root/Dataset/kitti"
    calib_file = os.path.join(kitti_root, "training", "calib", "{:06d}".format(image_idx) + ".txt")
    velodyne_file = os.path.join(kitti_root, "training", "velodyne", "{:06d}".format(image_idx) + ".bin")
    _, cam_to_vel = KittiCalibration.get_transform_matrix(calib_file)
    range_list = [(-60, 60), (-100, 100), (-2., -2.), 0.1]
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    bev_image = points_filter.get_bev_image(velodyne_file)

    print(len(frame_samples))
    for sample in frame_samples:
        # gt_box3d
        gt_box3d = sample["match_gt_box3d"]
        loc = gt_box3d[:3]
        dim = gt_box3d[3:6]
        roty = gt_box3d[6]

        dim = np.array([dim[1], dim[2], dim[0]])
        corner_points = get_object_corners_in_lidar(cam_to_vel, dim, loc, roty)
        x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
        c = (0, 0, 255)
        for i in np.arange(4):
            cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[1], y_img[1]), c, 3)
            cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[2], y_img[2]), c, 3)
            cv2.line(bev_image, (x_img[1], y_img[1]), (x_img[3], y_img[3]), c, 3)
            cv2.line(bev_image, (x_img[2], y_img[2]), (x_img[3], y_img[3]), c, 3)

        # pred_box3d
        loc = sample["loc"]
        dim = sample["dim"]
        roty = sample["roty"]
        dim = np.array([dim[1], dim[2], dim[0]])

        corner_points = get_object_corners_in_lidar(cam_to_vel, dim, loc, roty)
        x_img, y_img = points_filter.pcl2xy_plane(corner_points[:, 0], corner_points[:, 1])
        c = (0, 255, 255)
        for i in np.arange(4):
            cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[1], y_img[1]), c, 3)
            cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[2], y_img[2]), c, 3)
            cv2.line(bev_image, (x_img[1], y_img[1]), (x_img[3], y_img[3]), c, 3)
            cv2.line(bev_image, (x_img[2], y_img[2]), (x_img[3], y_img[3]), c, 3)
        bev_image = cv2.putText(bev_image, str(round(sample["iou_3d"], 2)), (x_img[0], y_img[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imwrite(os.path.join("debug", "{:06d}".format(image_idx) + ".jpg"), bev_image[:bev_image.shape[0]//2, :, :])
    return bev_image

if __name__ == "__main__":
    gt_info_path = "kitti_infos/kitti_infos_val.pkl"
    pred_info_path = "kitti_infos/kitti_infos_val_uncertainty.pkl"
    iou_calculator = BboxOverlaps3D(coordinate="camera")

    with open(gt_info_path, 'rb') as f:
        kitti_gt_infos = pickle.load(f)
    with open(pred_info_path, 'rb') as f:
        kitti_pred_infos = pickle.load(f)
    
    total_frame_samples = []
    count = 0
    total_ious_3d = 0.0
    for idx in tqdm(range(len(kitti_gt_infos))):
        gt_info, pred_info = kitti_gt_infos[idx], kitti_pred_infos[idx] 
        image_idx = gt_info["image_idx"]
        if gt_info["image_idx"] != image_idx:
            print("image_idx is inconsist ...")
        gt_annos = gt_info["annos"]
        pred_annos = pred_info["annos"]
        used_classes = ["Car", "Pedestrian", "Cyclist"]
        class_boxes_3d = prepare_boxes(gt_annos, used_classes)
        names = pred_annos["name"]
        difficulty = pred_annos["difficulty"] 
        scores = pred_annos["score"]
        geo_confs = pred_annos["geo_conf"]
        frame_samples = []

        for i in range(len(names)):
            name = names[i]
            loc = pred_annos["location"][i]
            dim = pred_annos["dimensions"][i]
            roty = pred_annos["rotation_y"][i]
            box3d = np.array([loc[0], loc[1], loc[2], dim[0], dim[1], dim[2], roty])
            box3d = box3d[np.newaxis, ...]
            gt_bboxes = class_boxes_3d[name]
            gt_bboxes_tensor = torch.tensor(gt_bboxes)
            box3d_tensor = torch.tensor(box3d)
            ious_3d = iou_calculator(gt_bboxes_tensor, box3d_tensor, "iou").numpy() 
            iou_3d = 0.0
            if len(ious_3d) > 0:
                iou_3d = np.max(ious_3d)
            '''
            box_3d = np.array([loc[0], loc[1], loc[2], dim[0], dim[1], dim[2], roty])
            box_3d = box_3d[np.newaxis, ...]
            gt_bboxes_3d = class_boxes_3d[name]
            overlap_part = d3_box_overlap(gt_bboxes_3d, box_3d).astype(np.float64)
            iou_3d_1 = 0.0
            if len(overlap_part) > 0:
                iou_3d_1 = np.max(overlap_part)
            '''
            sample_info = {
                "name": names[i],
                "label": TYPE_ID_CONVERSION[names[i]],
                "roty": roty,
                "dim": dim,
                "loc": loc,
                "P2": pred_info['calib/P2'],
                "image_idx": image_idx,
                "difficulty": pred_annos["difficulty"][i],
                "score": pred_annos["score"][i],
                "geo_conf": pred_annos["geo_conf"][i],
                "iou_3d": iou_3d,
                "match_gt_box3d": 0
            }
            frame_samples.append(sample_info)
            total_ious_3d += iou_3d
            count += 1
        # visualization(image_idx, frame_samples)
        total_frame_samples.append(frame_samples)

    print("total num: ", count, total_ious_3d / count)
    #plot_scatter_diagram(total_frame_samples)
    score_list, geo_conf_list, fusion_list = [], [], []
    for frame_samples in tqdm(total_frame_samples):
        for sample in frame_samples:
            score_list.append([sample["score"], sample["iou_3d"]])
            geo_conf_list.append([sample["geo_conf"], sample["iou_3d"]])
            fusion_list.append([sample["geo_conf"]**2 *  sample["score"], sample["iou_3d"]])

    score_list = np.array(score_list)
    geo_conf_list = np.array(geo_conf_list)
    fusion_list = np.array(fusion_list)

    score_list = score_list[np.argsort(score_list[:, 0])]
    geo_conf_list = geo_conf_list[np.argsort(geo_conf_list[:, 0])]
    fusion_list = fusion_list[np.argsort(fusion_list[:, 0])]

    total_num = score_list.shape[0]
    sub_num = int(total_num / 7)

    score_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    geo_conf_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fusion_means = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    score_variances = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    geo_conf_variances = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fusion_variances = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(7):
        start_idx = i * sub_num
        sub_score_list = score_list[start_idx: start_idx + sub_num, 1]
        sub_geo_conf_list = geo_conf_list[start_idx: start_idx + sub_num, 1]
        sub_fusion_list = fusion_list[start_idx: start_idx + sub_num, 1]

        score_means[i] = sum(sub_score_list.tolist()) / sub_num
        geo_conf_means[i] = sum(sub_geo_conf_list.tolist()) / sub_num
        fusion_means[i] = sum(sub_fusion_list.tolist()) / sub_num

        score_variances[i] = np.var(sub_score_list)
        geo_conf_variances[i] = np.var(sub_geo_conf_list)
        fusion_variances[i] = np.var(sub_fusion_list)
    
    fig = plt.figure()
    intervals = np.array(range(len(score_means)))
    total_width, n = 0.9, 3
    width = total_width / n
    x1 = intervals - width
    x2 = intervals
    x3 = intervals + width
    print(score_means, sum(score_means[-3:])/3)
    print(geo_conf_means, sum(geo_conf_means[-3:])/3)
    print(fusion_means, sum(fusion_means[-3:])/3)

    print(score_variances, sum(score_variances[-3:])/3)
    print(geo_conf_variances, sum(geo_conf_variances[-3:])/3)
    print(fusion_variances, sum(fusion_variances[-3:])/3)

    plt.bar(x1, score_means, 0.3, color="green")
    plt.bar(x2, geo_conf_means, 0.3, color="blue")
    plt.bar(x3, fusion_means, 0.3, color="red")

    '''
    for iv1, iv2, score_iou, geo_conf_iou in zip(x1, x2, score_list, geo_conf_list):
        plt.text(iv1, score_iou + 0.05, score_iou, ha='center',va='bottom')
        plt.text(iv2, geo_conf_iou + 0.05, geo_conf_iou, ha='center',va='bottom')
    '''
    plt.xlabel("score")
    plt.ylabel("iou_3d")
    plt.savefig("bar_diagram_bev_3.0.jpg", dpi=500)






            
        
