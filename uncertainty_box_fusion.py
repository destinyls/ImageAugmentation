import os
import csv
import pickle
import warnings
import cv2
import torch
import numpy as np
import kitti_common_fusion as kitti

from numba import jit
from tqdm import tqdm

from create_date import _read_imageset_file
from evaluation import evaluate_python
from visualizer.kitti_visual_tool import kitti_visual_tool_api

from evaluation.kitti_object_eval_python.eval import bev_box_overlap, d3_box_overlap
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}
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

def decode_location(P2, proj_p, depth):
    loc_img = np.array([proj_p[0] * depth, proj_p[1] * depth, depth, 1.0]).reshape((4, 1))
    P2_inv = np.linalg.inv(P2)
    loc = np.matmul(P2_inv, loc_img)
    return loc[:3, 0]

def encode_location(P, ry, locs, h):
    x, y, z = locs[0], locs[1], locs[2]
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    loc_center = np.array([x, y - h / 2, z, 1.0])
    proj_point = np.matmul(P, loc_center)
    depth = proj_point[2]
    proj_point = proj_point[:2] / depth
    return proj_point[0], proj_point[1], depth

@jit(nopython=True)
def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def generate_kitti_3d_detection(prediction, predict_txt, P2):
    prediction[:, 8:14] = prediction[:, [11,12,13,8,9,10]]
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                # ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry', 'score', 'geo_conf']
                prediction
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[3:18].tolist()
                
                w.writerow(row)
    check_last_line_break(predict_txt)

def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()

class UncertaintyBoxFusion():
    def __init__(self, root, pred_dir, split, model_list=["0025000",]):
        super(UncertaintyBoxFusion, self).__init__()
        self.root = root
        self.pred_dir = pred_dir
        self.split = split
        self.split_path = "training" if self.split in ["train", "val", "trainval"] else "testing"
        self.model_list = model_list
        self.iou_calculator = BboxOverlaps3D(coordinate="camera")
        self.has_gts = False if self.split == "test" else True
        
        if self.split == "val":
            imageset_txt = os.path.join(root, "ImageSets", "val.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "ImageSets", "test.txt")
        else:
            raise ValueError("Invalid split!")

        label_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            label_name = base_name + ".txt"
            if not os.path.exists(os.path.join(self.root, self.split_path, "image_2", base_name + ".png")):
                continue
            label_files.append(label_name)
        self.label_files = label_files
        self.num_samples = len(self.label_files)

    def __len__(self):
        return self.num_samples

    def visualize_labels(self, fusion_dir, predict_txt, overall_new_boxes):
        image_path = os.path.join(self.root, self.split_path, "image_2", predict_txt.replace("txt", "png"))
        velodyne_path = os.path.join(self.root, self.split_path, "velodyne", predict_txt.replace("txt", "bin"))
        calib_path = os.path.join(self.root, self.split_path, "calib", predict_txt)
        label_path = os.path.join(self.root, self.split_path, "label_2", predict_txt)
        pred_path = os.path.join(fusion_dir, predict_txt)

        if not os.path.exists(velodyne_path):
            velodyne_path = None
        
        image = kitti_visual_tool_api(image_path, calib_path, label_path, pred_path, velodyne_path, overall_new_boxes)
        return image

    def load_annotations(self, model_name, idx):
        file_name = self.label_files[idx]
        image_id = int(file_name.split(".")[0])
        if self.split in ["train", "val", "trainval"]:
            calib_path = os.path.join(self.root, "training/calib", file_name)
        elif self.split == "test":
            calib_path = os.path.join(self.root, "testing/calib", file_name)

        with open(calib_path, 'r') as f:
            lines = f.readlines()
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
            P2 = np.concatenate([P2, np.array([[0., 0., 0., 1.]])], axis=0)

        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry', 'score']
        frame_boxes, frame_scores, frame_labels = [], [], []
        label_file = os.path.join(self.pred_dir, model_name, file_name)
        if os.path.exists(label_file):
            with open(label_file, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
                for line, row in enumerate(reader):
                    box3d = np.zeros(16, dtype=np.float32)
                    box3d[0] = TYPE_ID_CONVERSION[row["type"]]
                    box3d[1] = float(row["truncated"])
                    box3d[2] = float(row["occluded"])
                    box3d[3] = float(row["alpha"])
                    box3d[4] = (float(row["xmin"]) / 1280.0)
                    box3d[5] = (float(row["ymin"]) / 384.0)
                    box3d[6] = (float(row["xmax"]) / 1280.0)
                    box3d[7] = (float(row["ymax"]) / 384.0)
                    '''
                    proj_x, proj_y, depth = \
                        encode_location(P2, float(row["ry"]), [float(row["lx"]), float(row["ly"]), float(row["lz"])], float(row["dh"]))
                    '''
                    box3d[8] = float(row["lx"])
                    box3d[9] = float(row["ly"])
                    box3d[10] = float(row["lz"])
                    box3d[11] = float(row["dh"])
                    box3d[12] = float(row["dw"])
                    box3d[13] = float(row["dl"])
                    box3d[14] = float(row["ry"])
                    box3d[15] = float(row["score"])
                    frame_boxes.append(box3d)
                    frame_scores.append(float(row["score"]))
                    frame_labels.append(TYPE_ID_CONVERSION[row["type"]])
        return frame_boxes, frame_scores, frame_labels, P2, image_id

    def prepare_boxes(self, idx):
        total_boxes_list, total_scores_list, total_labels_list = [], [], []
        for model_name in self.model_list:
            frame_boxes, frame_scores, frame_labels, P2, image_idx = self.load_annotations(model_name, idx)
            total_boxes_list.append(frame_boxes)
            total_scores_list.append(frame_scores)
            total_labels_list.append(frame_labels)
            print(model_name, len(frame_boxes))
        return total_boxes_list, total_scores_list, total_labels_list, P2, image_idx

    def find_matching_box(self, boxes_list, new_box, match_iou):
        best_iou = match_iou
        best_index = -1
        for i in range(len(boxes_list)):
            box = boxes_list[i]
            if box[0] != new_box[0]:
                continue
            iou = bb_intersection_over_union(box[4:8], new_box[4:8])
            if iou > best_iou:
                best_index = i
                best_iou = iou
        return best_index, best_iou

    def prefilter_boxes(self, boxes, scores, labels, weights, thr):
        # Create dict with boxes stored by its label
        new_boxes = dict()
        for t in range(len(boxes)):
            if len(boxes[t]) != len(scores[t]):
                print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
                exit()
            if len(boxes[t]) != len(labels[t]):
                print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
                exit()
            for j in range(len(boxes[t])):
                score = scores[t][j]
                if score < thr:
                    continue
                label = int(labels[t][j])
                box_part = boxes[t][j]
                x1 = float(box_part[4])
                y1 = float(box_part[5])
                x2 = float(box_part[6])
                y2 = float(box_part[7])

                # Box data checks
                if x2 < x1:
                    warnings.warn('X2 < X1 value in box. Swap them.')
                    x1, x2 = x2, x1
                if y2 < y1:
                    warnings.warn('Y2 < Y1 value in box. Swap them.')
                    y1, y2 = y2, y1
                if x1 < 0:
                    warnings.warn('X1 < 0 in box. Set it to 0.')
                    x1 = 0
                if x1 > 1:
                    warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                    x1 = 1
                if x2 < 0:
                    warnings.warn('X2 < 0 in box. Set it to 0.')
                    x2 = 0
                if x2 > 1:
                    warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                    x2 = 1
                if y1 < 0:
                    warnings.warn('Y1 < 0 in box. Set it to 0.')
                    y1 = 0
                if y1 > 1:
                    warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                    y1 = 1
                if y2 < 0:
                    warnings.warn('Y2 < 0 in box. Set it to 0.')
                    y2 = 0
                if y2 > 1:
                    warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                    y2 = 1
                if (x2 - x1) * (y2 - y1) == 0.0:
                    warnings.warn("Zero area box skipped: {}.".format(box_part))
                    continue

                # [label, score, weight, model index, x1, y1, x2, y2]

                #['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry']
                # + ['score', weights, 'model index']
                b = box_part[:-1].tolist() + [box_part[-1] * weights[t], weights[t], t]
                if label not in new_boxes:
                    new_boxes[label] = []
                new_boxes[label].append(b)

        # Sort each list in dict by score and transform it to numpy array
        for k in new_boxes:
            current_boxes = np.array(new_boxes[k])
            new_boxes[k] = current_boxes[current_boxes[:, -3].argsort()[::-1]]

        return new_boxes

    def get_weighted_box(self, boxes, conf_type='avg'):
        """
        Create weighted box for set of boxes
        :param boxes: set of boxes to fuse
        :param conf_type: type of confidence one of 'avg' or 'max'
        :return: weighted box (label, score, weight, x1, y1, x2, y2)
        """
        box = np.zeros(18, dtype=np.float32)
        conf = 0
        conf_list = []
        w = 0

        box3d = boxes.copy()
        box3d = np.array(box3d)
        box3d = box3d[:, [8,9,10,13,11,12,14]]
        box3d = torch.tensor(box3d)
        bbox_ious_3d = self.iou_calculator(box3d, box3d, "iou")
        bbox_ious_3d = bbox_ious_3d.numpy()

        matrix_w = np.ones((len(self.model_list), len(self.model_list)))
        tempalte_ious_3d = np.zeros((len(self.model_list), len(self.model_list)))
        if tempalte_ious_3d.shape[0] >= bbox_ious_3d.shape[0]:
            tempalte_ious_3d[:bbox_ious_3d.shape[0], :bbox_ious_3d.shape[1]] = bbox_ious_3d
        else:
            tempalte_ious_3d = bbox_ious_3d[:tempalte_ious_3d.shape[0], :tempalte_ious_3d.shape[1]]
        matrix_w[range(matrix_w.shape[0]), range(matrix_w.shape[1])] = 1.0
        weighted_ious_3d = tempalte_ious_3d * matrix_w
        geo_conf = np.sum(weighted_ious_3d) / (np.sum(matrix_w) + 10e-6)

        '''
        matrix_w = np.ones_like(bbox_ious_3d)
        matrix_w[range(matrix_w.shape[0]), range(matrix_w.shape[1])] = 0.0
        weighted_ious_3d = bbox_ious_3d * matrix_w
        geo_conf = np.sum(weighted_ious_3d) / (np.sum(matrix_w) + 10e-6)
        # geo_conf = max(geo_conf_1, geo_conf_2)
        '''
        for b in boxes:
            # [label, score, weight, model_index, x1, y1, x2, y2]
            # ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry']
            # + ['score', 'weights', 'model index']
            conf += b[-3]
            conf_list.append(b[-3])
            w += b[-2]

        if conf_type == 'avg':
            box[-3] = conf / len(boxes)
        elif conf_type == 'max':
            box[-3] = np.array(conf_list).max()
        elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
            box[-3] = conf / len(boxes)
        
        box[-2] = w
        box[-1] = -1 # model index field is retained for consistensy but is not used.
        box[:-2] = boxes[0][:-2]
        if box[10] < 4.2:
            box[-2] = max(0.85, box[-2])
        else:
            box[-2] = geo_conf
        box[-1] = len(boxes)
        # ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl', 'ry']
        # + ['score', 'geo_conf', 'model index']
        return box

    def boxes_cluster(self, boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
        '''
        :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
        It has 3 dimensions (models_number, model_preds, 15)
        Order of boxes: 'type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl' 'ry', 'score'. 
        We expect float normalized coordinates [0; 1]
        :param scores_list: list of scores for each model
        :param labels_list: list of labels for each model
        :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
        :param iou_thr: IoU value for boxes to be a match
        :param skip_box_thr: exclude boxes with score lower than this variable
        :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
        :param allows_overflow: false if we want confidence score not exceed 1.0
        :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
        :return: scores: confidence scores
        :return: labels: boxes labels
        '''

        if weights is None:
            weights = np.ones(len(boxes_list))
        if len(weights) != len(boxes_list):
            print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
            weights = np.ones(len(boxes_list))
        weights = np.array(weights)

        if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
            print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
            exit()

        filtered_boxes = self.prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
        if len(filtered_boxes) == 0:
            return np.zeros((0, 18)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

        overall_boxes = []
        overall_new_boxes = {}
        for label in filtered_boxes:
            boxes = filtered_boxes[label]
            new_boxes = []
            weighted_boxes = []
            # Clusterize boxes
            for j in range(0, len(boxes)):
                index, best_iou = self.find_matching_box(weighted_boxes, boxes[j], iou_thr)
                if index != -1:
                    new_boxes[index].append(boxes[j])
                    weighted_boxes[index] = self.get_weighted_box(new_boxes[index], conf_type)
                else:
                    new_boxes.append([boxes[j].copy()])
                    candidate = boxes[j].copy()
                    if candidate[10] < 4.2:
                        candidate[-2] = max(0.85, candidate[-2])
                    else:
                        candidate[-2] = 1 / len(self.model_list)
                    weighted_boxes.append(candidate)
            overall_boxes.append(np.array(weighted_boxes))
            overall_new_boxes[label] = new_boxes
        
        overall_boxes = np.concatenate(overall_boxes, axis=0)
        overall_boxes = overall_boxes[overall_boxes[:, -3].argsort()[::-1]]
        boxes = overall_boxes[:, :]
        scores = overall_boxes[:, -3]
        labels = overall_boxes[:, 0]

        #   ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'lx', 'ly', 'lz','dh', 'dw', 'dl' 'ry']
        # + ['score', 'weights', 'model index']
        return boxes, scores, labels, overall_new_boxes
    
    def load_gt_boxes(self, image_idx, used_classes = ["Car", "Pedestrian", "Cyclist"]):  
        class_boxes_3d = dict()
        for class_key in used_classes:
            class_boxes_3d[class_key] = None
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry', 'score']
        kitti_root = "/root/Dataset/kitti"
        label_file = os.path.join(kitti_root, "training", "label_2", "{:06d}".format(image_idx) + ".txt")
        dims, locs, rotys, names = [], [], [], []
        with open(label_file, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                if row["type"] not in used_classes:
                    continue
                dim = [float(row["dh"]), float(row["dw"]), float(row["dl"])]
                loc = [float(row["lx"]), float(row["ly"]), float(row["lz"])]
                roty = float(row["ry"])
                dims.append(dim)
                locs.append(loc)
                rotys.append(roty)
                names.append(row["type"])
        dimensions = np.array(dims)
        locations = np.array(locs)
        rotys = np.array(rotys)
        num_obj = len(names)
        if num_obj == 0:
            return class_boxes_3d
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

    def __getitem__(self, idx):
        predict_txt = self.label_files[idx]
        boxes_list, scores_list, labels_list, P2, image_idx = self.prepare_boxes(idx)
        boxes, scores, labels, overall_new_boxes = self.boxes_cluster(boxes_list, scores_list, labels_list, iou_thr=0.70, skip_box_thr=0.0)
        
        if self.has_gts:
            class_boxes_3d = self.load_gt_boxes(image_idx)
            for i in range(boxes.shape[0]):
                name = ID_TYPE_CONVERSION[boxes[i, 0]]
                loc = boxes[i, 8:11]
                dim = boxes[i, 11:14]
                roty = boxes[i, 14]
                
                box_3d = np.array([loc[0], loc[1], loc[2], dim[2], dim[0], dim[1], roty])
                box_3d = box_3d[np.newaxis, ...]
                gt_bboxes_3d = class_boxes_3d[name][:, [0,1,2,5,3,4,6]]
                overlap_part = d3_box_overlap(gt_bboxes_3d, box_3d).astype(np.float64)
                iou_3d = 0.0
                if len(overlap_part) > 0:
                    iou_3d = np.max(overlap_part)
                boxes[i, -1] = iou_3d
        
        return boxes, scores, labels, predict_txt, P2, overall_new_boxes

if __name__ == "__main__":
    root = "/root/Dataset/kitti"
    pred_dir = "/root"
    fusion_dir = "./fusion_results"

    # 22.71, 22.12, 22.10, 21.98, 23.35, 22.42, 21.51, 21.49\
    '''
    model_list =["MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_002nd/kitti_train/inference_40832/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_DIFFICULTY_1_003nd/kitti_train/inference_43152/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_003nd/kitti_train/inference_41760/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_FLIP_006nd/kitti_train/inference_39440/data/",
                 "MonoFlex/output/DLA-34_Baseline_MergeHead_SCH2_4_GPUS_001nd/kitti_train/inference_46400/data/", \
                 "MonoFlex_Backup/output/DLA-34_Baseline_001/kitti_train/inference_37584/data/", \
                 "MonoFlex_Backup/output/DLA-34_Baseline_005/kitti_train/inference_38976/data/"]

    model_list =["MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_002nd/kitti_train/inference_40832/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_002nd/kitti_train/inference_43616/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_DIFFICULTY_1_003nd/kitti_train/inference_43152/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_DIFFICULTY_1_006nd/kitti_train/inference_40368/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_003nd/kitti_train/inference_41760/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_004nd/kitti_train/inference_42688/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_FLIP_001nd/kitti_train/inference_38512/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_FLIP_006nd/kitti_train/inference_39440/data/",]
    
    model_list =["MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_004nd/kitti_train/inference_77184/data/",\
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_003nd/kitti_train/inference_80400/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_DB_TRAIN_002nd/kitti_train/inference_82008/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_DB_TRAIN_003nd/kitti_train/inference_77194/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_DB_TRAIN_004nd/kitti_train/inference_67536_/data/"]
    
    model_list =["MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_001nd/kitti_train/inference_40832/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_002nd/kitti_train/inference_45936/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_003nd/kitti_train/inference_43152/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_004nd/kitti_train/inference_39904/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_005nd/kitti_train/inference_43152/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_006nd/kitti_train/inference_41296/data/"]
    
    model_list =["MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_005nd/kitti_train/inference_25984/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_005nd/kitti_train/inference_30160/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_005nd/kitti_train/inference_35728/data/", \
                 "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_005nd/kitti_train/inference_43152/data/"]

    model_list = ["TRAIN_NOISE_002nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_006nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_DIFFICULTY_1_003nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_DIFFICULTY_1_006nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_NO_RIGHT_003nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_NO_RIGHT_004nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_NO_RIGHT_FLIP_001nd/inference/kitti_test/data", \
                  "TRAIN_NOISE_NO_RIGHT_FLIP_006nd/inference/kitti_test/data"]
                  
    # state 2:
    model_list = ["MonoFlex/checkpoints/STAGE_59953_002/MORE_BACKGROUND_003nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_59953_002/MORE_BACKGROUND_004nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_59953_002/MORE_BACKGROUND_DB_TRAIN_002nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_59953_002/MORE_BACKGROUND_DB_TRAIN_003nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_59953_002/MORE_BACKGROUND_DB_TRAIN_004nd/inference/kitti_test/data"]

    # stage 3 | test
    model_list = ["MonoFlex/checkpoints/STAGE_003/MORE_BACKGROUND_STAGE_002_003nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_003/MORE_BACKGROUND_STAGE_002_005nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_003/MORE_BACKGROUND_STAGE_002_011nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_003/MORE_BACKGROUND_STAGE_002_012nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/STAGE_003/MORE_BACKGROUND_STAGE_002_017nd/inference/kitti_test/data"
                ]
    
    # stage 3 | val
    model_list = ["MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_STAGE_002_003nd/kitti_train/inference_69144/data", \
                  "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_STAGE_002_005nd/kitti_train/inference_67536/data",\
                  "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_STAGE_002_011nd/kitti_train/inference_77184/data",\
                  "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_STAGE_002_012nd/kitti_train/inference_77988/data",\
                  "MonoFlex/checkpoints/DLA-34_BCP_TRAIN_MULTI_CLASSES_PSEUDO_LABELS_FROM_TRAIN_NOISE_NO_RIGHT_UNCERTAINTY_FLIP_MORE_BACKGROUND_STAGE_002_017nd/kitti_train/inference_73164/data"
                  ]
    '''
    # GUP NET TEST
    model_list = ["MonoFlex/checkpoints/MONOFLEX_TEST_17.91_STAGE_002/Trainval_labels_12000+_On_Trainval_001nd/inference/kitti_test/data", \
                  "MonoFlex/checkpoints/MONOFLEX_TEST_17.91_STAGE_002/TEST_UNION_001nd/inference/kitti_test/data"
                 ]

    model_list = ["/root/SMOKE/checkpoints/RESNET-34_WEIGHTED_LOSS_RIGHT_IMAGE_ROI_BCP_AUG_RAW_DATA_R40_001nd/inference/kitti_train/data"]

    if not os.path.exists(fusion_dir):
        os.makedirs(fusion_dir)

    box_fusion = UncertaintyBoxFusion(root, pred_dir, split='val', model_list=model_list)

    for idx in tqdm(range(len(box_fusion))):
        boxes, scores, labels, predict_txt, P2, overall_new_boxes = box_fusion[idx]
        boxes[:,[4,6]] *= 1280.0
        boxes[:,[5,7]] *= 384.0

        generate_kitti_3d_detection(boxes, os.path.join(fusion_dir, predict_txt), P2)
        '''
        image = box_fusion.visualize_labels(fusion_dir, predict_txt, overall_new_boxes)
        cv2.imwrite(os.path.join("debug", predict_txt.replace("txt", "jpg")), image)
        '''
    val_img_ids = _read_imageset_file(root, "ImageSets/val.txt")
    kitti_infos_val = kitti.get_kitti_image_info(
        root,
        training=True,
        label_info=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=True)
    filename = os.path.join("kitti_infos", 'kitti_infos_val_uncertainty.pkl')
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)

    gt_label_path = "datasets/kitti/training/label_2/"
    imageset_txt = "datasets/kitti/ImageSets/val.txt"
    result, ret_dict = evaluate_python(label_path=gt_label_path, 
                                        result_path=fusion_dir,
                                        label_split_file=imageset_txt,
                                        current_class=["Car",],
                                        metric='R11')
    print(result)
    