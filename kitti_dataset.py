import cv2
import os
import pickle

import numpy as np
import kitti_common as kitti

from random import sample

class AUGDataset():
    def __init__(self, kitti_root, is_train=True, split="train", classes=["Car", "Pedestrian", "Cyclist"]):
        super(AUGDataset, self).__init__()
        self.kitti_root = kitti_root
        self.split = split
        self.is_train = is_train
        self.classes = classes
        self.max_objs = 30
        self.use_right = False

        if self.split == "train":
            info_path = os.path.join(kitti_root, "kitti_infos_train.pkl")
            db_info_path = os.path.join(kitti_root, "kitti_dbinfos_train.pkl")
        elif self.split == "val":
            info_path = os.path.join(kitti_root, "kitti_infos_val.pkl")
        elif self.split == "trainval":
            info_path = os.path.join(kitti_root, "kitti_infos_trainval.pkl")
            db_info_path = os.path.join(kitti_root, "kitti_dbinfos_trainval.pkl")
        elif self.split == "test":
            info_path = os.path.join(kitti_root, "kitti_infos_test.pkl")
        else:
            raise ValueError("Invalid split!")

        with open(info_path, 'rb') as f:
            self.kitti_infos = pickle.load(f)
        with open(db_info_path, 'rb') as f:
            db_infos = pickle.load(f)
        self.car_db_infos = db_infos["Car"]
        self.ped_db_infos = db_infos["Pedestrian"]
        self.cyc_db_infos = db_infos["Cyclist"]

        self.num_samples = len(self.kitti_infos)

    def set_use_right(self, use_right):
        self.use_right = use_right

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        info = self.kitti_infos[idx]
        img_path = os.path.join(self.kitti_root, info["img_path"])
        if self.use_right:
            img_path = img_path.replace("image_2", "image_3")
        img = cv2.imread(img_path)
        image_idx = info["image_idx"]
        P2 = info["calib/P2"]
        P3 = info["calib/P3"]

        if not self.is_train:
            return img, P2, P3, image_idx

        annos = info["annos"]
        names = annos["name"] 
        bboxes = annos["bbox"]
        alphas = annos["alpha"]
        dimensions = annos["dimensions"]
        locations = annos["location"]
        rotys = annos["rotation_y"]
        difficulty = annos["difficulty"]
        truncated = annos["truncated"]
        occluded = annos["occluded"]
        scores = annos["score"]
        class_to_label = kitti.get_class_to_label_map()
        embedding_annos = []
        init_bboxes = []
        for i in range(len(names)):
            if names[i] not in self.classes:
                continue
            init_bboxes.append(bboxes[i])
            ins_anno = {
                    "name": names[i],
                    "lable": class_to_label[names[i]],
                    "bbox": bboxes[i],
                    "alpha": alphas[i],
                    "dim": dimensions[i],
                    "loc": locations[i],
                    "roty": rotys[i],
                    "P2": info['calib/P2'],
                    "P3": info['calib/P3'],
                    "difficulty": difficulty[i],
                    "truncated": truncated[i],
                    "occluded": occluded[i],
                    "score": scores[i]
                }
            embedding_annos.append(ins_anno)
        origin_num = len(embedding_annos)

        if self.use_right:
            aug_num = len(embedding_annos) - origin_num
            return img, embedding_annos, aug_num, image_idx

        init_bboxes = np.array(init_bboxes)
        img_shape_key = f"{img.shape[0]}_{img.shape[1]}"
        if img_shape_key in self.car_db_infos.keys():
            car_db_infos_t = self.car_db_infos[img_shape_key]
            ins_ids = sample(range(len(car_db_infos_t)), min(0, self.max_objs - len(annos)))
            for i in ins_ids:
                ins = car_db_infos_t[i]
                box2d = ins["bbox"]
                if ins['difficulty'] > 0:
                    continue
                if len(init_bboxes.shape) > 1:
                    ious = kitti.iou(init_bboxes, box2d[np.newaxis, ...])
                    if np.max(ious) > 0.0:
                        continue
                    init_bboxes = np.vstack((init_bboxes, box2d[np.newaxis, ...]))
                else:
                    init_bboxes = bbox[np.newaxis, ...].copy()            
                patch_img_path = os.path.join(self.kitti_root, ins["path"])
                patch_img = cv2.imread(patch_img_path)
                img[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = patch_img
                
                ins_anno = {
                    "name": ins["name"],
                    "lable": class_to_label[ins["name"]],
                    "bbox": box2d,
                    "alpha": ins["alpha"],
                    "dim": ins["dim"],
                    "loc": ins["loc"],
                    "roty": ins["roty"],
                    "P2": ins["P2"],
                    "P3": ins["P3"],
                    "difficulty": ins["difficulty"],
                    "truncated": ins["truncated"],
                    "occluded": ins["occluded"],
                    "score": ins["score"]
                }
                embedding_annos.append(ins_anno)   
        aug_num = len(embedding_annos) - origin_num
        return img, embedding_annos, aug_num, image_idx

    def visualization(self, img, annos, save_path):
        image = img.copy()
        for anno in annos:
            dim = anno["dim"]
            loc = anno["loc"]
            roty = anno["roty"]
            P2 = anno["P2"]
            box3d = kitti.compute_box_3d_image(P2, roty, dim, loc)
            image = kitti.draw_box_3d(image, box3d)
        cv2.imwrite(os.path.join(save_path, image_index + ".jpg"), image)

if __name__ == "__main__":
    
    dataset = KITTIDataset(kitti_root="datasets/kitti")
    for i in range(100):
        idx = i
        image_index = kitti.get_image_index_str(idx)
        img, embedding_annos = dataset[idx]    
