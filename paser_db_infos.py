import pickle
import os 
import cv2
import random
import numpy as np

from shutil import copyfile

def copy_file(file_src, file_dest):
    if not os.path.exists(file_dest):
        print(file_dest)
        try:
            copyfile(file_src, file_dest)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

def _read_imageset_file(kitti_root, path):
    imagetxt = os.path.join(kitti_root, path)
    with open(imagetxt, 'r') as f:
        lines = f.readlines()
    total_img_ids = [int(line) for line in lines]
    img_ids = []
    for img_id in total_img_ids:
        if "test" in path:
            img_path = os.path.join(kitti_root, "testing/image_2", "{:06d}".format(img_id) + ".png")
        else:
            img_path = os.path.join(kitti_root, "training/image_2", "{:06d}".format(img_id) + ".png")
        if os.path.exists(img_path):
            img_ids.append(img_id)
    return img_ids

if __name__ == "__main__":
    db_infos_path = "datasets/kitti/kitti_dbinfos_test_uncertainty_59953_stage_003.pkl"
    # db_infos = "kitti_infos/kitti_dbinfos_test_31552_006nd.pkl"
    infos_path = "datasets/kitti/kitti_infos_train_enhanced.pkl"
    save_db_infos_path = "datasets/kitti/kitti_dbinfos_test_uncertainty_59953_stage_003_half_samples_dense.pkl"
    with open(db_infos_path, 'rb') as f:
        kitti_db_infos = pickle.load(f)
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
        print("total: ", len(infos))
    
    img_shape_dict = {}
    for idx in range(len(infos)):
        info = infos[idx]
        img_shape = info["img_shape"]
        img_shape_key = f"{img_shape[0]}_{img_shape[1]}"
        if img_shape_key not in img_shape_dict:
            img_shape_dict[img_shape_key] = 1
        else:
            img_shape_dict[img_shape_key] += 1

    for img_shape_key, number in img_shape_dict.items():
        print(img_shape_key, number)

    print(kitti_db_infos["Car"].keys())
    sub_db_infos = dict()
    total_valid_ins = 0
    total_ins = 0
    print(kitti_db_infos.keys())
    for class_name, class_db_infos in kitti_db_infos.items():
        if class_name not in sub_db_infos:
            sub_db_infos[class_name] = {}
        for img_shape, shape_dbinfos in class_db_infos.items():
            if img_shape not in sub_db_infos[class_name]:
                sub_db_infos[class_name][img_shape] = []
            geo_count = 0
            for ins_idx in range(len(shape_dbinfos)):
                ins = shape_dbinfos[ins_idx]                    
                if ins["difficulty"] == 0:
                    if ins["score"] > 0.55:
                        if ins["geo_conf"] > 0.60:
                            if random.random() < 1.5:
                                sub_db_infos[class_name][img_shape].append(ins)
                                geo_count = geo_count + 1
            total_valid_ins += geo_count
            total_ins += len(shape_dbinfos)
            print(class_name, img_shape, "geo_count: ", geo_count, "total", len(shape_dbinfos))
    print("toyal_ins num: ", total_valid_ins, total_ins)
'''
    with open(save_db_infos_path, 'wb') as f:
        pickle.dump(sub_db_infos, f)

if __name__ == "__main__":

    kitti_root = "/root/Dataset/kitti/"
    src_root = os.path.join(kitti_root, "testing")
    dest_root = os.path.join(kitti_root, "training")

    # kitti_infos = "kitti_infos/kitti_infos_test_monoflex_baseine_trainval.pkl"
    kitti_infos = "kitti_infos/kitti_infos_train_enhanced.pkl"

    with open(kitti_infos, 'rb') as f:
        kitti_infos = pickle.load(f)

    num = 0
    copyed_num = 0
    for info in kitti_infos:
        num += 1
        annos = info["annos"]
        num_obj = np.sum(annos["index"] >= 0)
        image_idx = info["image_idx"]
        if image_idx < 8000 or num_obj == 0:
            continue
        copyed_num += 1
        image_id_key = "{:06d}".format(image_idx)
        src_image_2 = os.path.join(src_root, "image_2", image_id_key + ".png")
        src_image_3 = os.path.join(src_root, "image_3", image_id_key + ".png")
        src_label_2 = os.path.join(src_root, "label_2", image_id_key + ".txt")
        src_calib = os.path.join(src_root, "calib", image_id_key + ".txt")

        dest_image_2 = os.path.join(dest_root, "image_2", image_id_key + ".png")
        dest_image_3 = os.path.join(dest_root, "image_3", image_id_key + ".png")
        dest_label_2 = os.path.join(dest_root, "label_2", image_id_key + ".txt")
        dest_calib = os.path.join(dest_root, "calib", image_id_key + ".txt")

        copy_file(src_image_2, dest_image_2)
        copy_file(src_image_3, dest_image_3)
        copy_file(src_calib, dest_calib)
        copy_file(src_label_2, dest_label_2)
        
        if image_idx > 8000:
            image = cv2.imread(dest_image_2)
            cv2.imwrite(os.path.join("debug", image_id_key + ".jpg"), image)
        
        # print(dest_image_2)
    print(num, copyed_num)

if __name__ == "__main__":
    kitti_root = "/root/Dataset/kitti/"
    dest_root = os.path.join(kitti_root, "testing")
    src_root = os.path.join(kitti_root, "training")

    image_txt = "ImageSets/val.txt"
    img_val_ids = _read_imageset_file(kitti_root, image_txt)
    print(len(img_val_ids))

    new_img_id = 53000
    for img_id in img_val_ids:
        image_id_key = "{:06d}".format(img_id)
        src_image_2 = os.path.join(src_root, "image_2", image_id_key + ".png")
        src_image_3 = os.path.join(src_root, "image_3", image_id_key + ".png")
        src_calib = os.path.join(src_root, "calib", image_id_key + ".txt")

        new_image_id_key = "{:06d}".format(new_img_id)
        new_img_id += 1
        dest_image_2 = os.path.join(dest_root, "image_2", new_image_id_key + ".png")
        dest_image_3 = os.path.join(dest_root, "image_3", new_image_id_key + ".png")
        dest_calib = os.path.join(dest_root, "calib", new_image_id_key + ".txt")
    
        copy_file(src_image_2, dest_image_2)
        copy_file(src_image_3, dest_image_3)
        copy_file(src_calib, dest_calib)

'''



