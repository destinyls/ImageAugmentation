import os
import cv2
import kitti_common as kitti

import copy
import pathlib
import pickle

import numpy as np
from tqdm import tqdm


def encode_bbox(P, ry, dims, locs, img_shape):
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]
    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])
    corners_3d_extend = corners_3d.transpose(1, 0)
    corners_3d_extend = np.concatenate(
        [corners_3d_extend, np.ones((corners_3d_extend.shape[0], 1), dtype=np.float32)], axis=1)    
    corners_2d = np.matmul(P, corners_3d_extend.transpose(1, 0))
    corners_2d = corners_2d[:2] / corners_2d[2]
    bbox = np.array([min(corners_2d[0]), min(corners_2d[1]),
                     max(corners_2d[0]), max(corners_2d[1])])

    bbox[[0,2]] = np.clip(bbox[[0,2]], 0, img_shape[1])
    bbox[[1,3]] = np.clip(bbox[[1,3]], 0, img_shape[0])
    return bbox

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def create_kitti_info_file(kitti_root,
                           info_path=None,
                           create_trainval=True,
                           relative_path=True):
    train_img_ids = _read_imageset_file("datasets/kitti/ImageSets/train.txt")
    val_img_ids = _read_imageset_file("datasets/kitti/ImageSets/val.txt")
    trainval_img_ids = _read_imageset_file("datasets/kitti/ImageSets/trainval.txt")
    test_img_ids = _read_imageset_file("datasets/kitti/ImageSets/test.txt")
    print("Generate info. this may take several minutes.")

    if info_path is None:
        info_path = pathlib.Path(kitti_root)
    else:
        info_path = pathlib.Path(info_path)
    info_path.mkdir(parents=True, exist_ok=True)

    kitti_infos_train = kitti.get_kitti_image_info(
        kitti_root,
        training=True,
        label_info=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    filename = info_path / 'kitti_infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)

    kitti_infos_val = kitti.get_kitti_image_info(
        kitti_root,
        training=True,
        label_info=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    filename = info_path / 'kitti_infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)

    if create_trainval:
        kitti_infos_trainval = kitti.get_kitti_image_info(
            kitti_root,
            training=True,
            label_info=True,
            calib=True,
            image_ids=trainval_img_ids,
            relative_path=relative_path)
        filename = info_path / 'kitti_infos_trainval.pkl'
        print(f"Kitti info trainval file is saved to {filename}")
        with open(filename, 'wb') as f:
            pickle.dump(kitti_infos_trainval, f)

    kitti_infos_test = kitti.get_kitti_image_info(
        kitti_root,
        training=False,
        label_info=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = info_path / 'kitti_infos_test.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)

def create_groundtruth_database(kitti_root,
                                info_path=None,
                                used_classes=None,
                                database_save_path=None,
                                relative_path=True):
    root_path = pathlib.Path(kitti_root)
    if info_path is None:
        info_save_path = root_path / 'kitti_infos_test.pkl'
    else:
        path_info = pathlib.Path(info_path)
        path_info.mkdir(parents=True, exist_ok=True)
        info_save_path = path_info / 'kitti_infos_test.pkl'
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = pathlib.Path(database_save_path)
    database_save_path.mkdir(parents=True, exist_ok=True)

    if info_path is None:
        db_info_save_path = root_path / "kitti_dbinfos_test.pkl"
    else:
        db_info_save_path = path_info / "kitti_dbinfos_test.pkl"
    with open(info_save_path, 'rb') as f:
        kitti_infos = pickle.load(f)

    all_db_infos = {}
    if used_classes is None:
        used_classes = list(kitti.get_classes())
        used_classes.pop(used_classes.index('DontCare'))
    for name in used_classes:
        all_db_infos[name] = {}
    for info in tqdm(kitti_infos):
        image_idx = info["image_idx"]
        annos = info["annos"]

        names = annos["name"]
        alphas = annos["alpha"]
        rotys = annos["rotation_y"]
        locs = annos["location"]
        dims = annos["dimensions"]
        difficulty = annos["difficulty"]
        truncated = annos["truncated"]
        occluded = annos["occluded"]
        scores = annos["score"]
        gt_idxes = annos["index"]
        num_obj = np.sum(annos["index"] >= 0)

        img_path = info["img_path"]
        img_path_l = os.path.join(root_path, img_path)
        img_path_r = img_path_l.replace("image_2", "image_3")

        if not os.path.exists(img_path_l):
            continue
        img_l = cv2.imread(img_path_l)
        img_r = cv2.imread(img_path_r)

        if np.all(img_l == img_r):
            print("shape not consistence ...")        
        img_shape = img_l.shape
        img_shape_key = f"{img_l.shape[0]}_{img_l.shape[1]}"

        for i in range(num_obj):
            if difficulty[i] == -1:
                continue
            # print(scores[i])
            if scores[i] < 0.65:
                continue
            if img_shape_key not in all_db_infos[names[i]].keys():
                all_db_infos[names[i]][img_shape_key] = []

            filename = "image_2/" + f"{image_idx}_{names[i]}_{gt_idxes[i]}_{difficulty[i]}.png"
            filepath = database_save_path / filename
            if names[i] in used_classes:
                if relative_path:
                    relative_filepath = os.path.join(database_save_path.stem, filename)

                box2d_l = encode_bbox(info['calib/P2'], rotys[i], dims[i], locs[i], img_shape)
                box2d_r = encode_bbox(info['calib/P3'], rotys[i], dims[i], locs[i], img_shape)

                cropImg_l = img_l[int(box2d_l[1]):int(box2d_l[3]), int(box2d_l[0]):int(box2d_l[2]), :]
                cropImg_r = img_r[int(box2d_r[1]):int(box2d_r[3]), int(box2d_r[0]):int(box2d_r[2]), :]
                filepath_l = str(filepath)
                filepath_r = filepath_l.replace("image_2", "image_3")

                if np.min(cropImg_l.shape) == 0 or np.min(cropImg_r.shape) == 0:
                    continue
                if np.max(cropImg_l.shape) > 1000 or np.max(cropImg_r.shape) > 1000:
                    continue

                cv2.imwrite(filepath_l, cropImg_l)
                cv2.imwrite(filepath_r, cropImg_r)

                class_to_label = kitti.get_class_to_label_map()
                db_info = {
                    "name": names[i],
                    "label": class_to_label[names[i]],
                    "path": relative_filepath,
                    "bbox_l": box2d_l,
                    "bbox_r": box2d_r,
                    "alpha": alphas[i],
                    "roty": rotys[i],
                    "dim": dims[i],
                    "loc": locs[i],
                    "P2": info['calib/P2'],
                    "P3": info['calib/P3'],
                    "img_shape": img_shape,
                    "image_idx": image_idx,
                    "gt_idx": gt_idxes[i],
                    "difficulty": difficulty[i],
                    "truncated": truncated[i],
                    "occluded": occluded[i],
                }
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                all_db_infos[names[i]][img_shape_key].append(db_info)
    for class_key, class_db_infos in all_db_infos.items():
        for k, v in class_db_infos.items():
            print(f"load {len(v)} {k}_{class_key} database infos")
    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

if __name__ == "__main__":
    kitti_root = "datasets/kitti"
    # create_kitti_info_file(kitti_root, info_path="kitti_infos") 
    create_groundtruth_database(kitti_root, info_path="kitti_infos", database_save_path="gt_database")

    # create_kitti_info_file(kitti_root) 
    # create_groundtruth_database(kitti_root)

