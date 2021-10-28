import pickle
import os 
import cv2
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

if __name__ == "__main__":
    '''
    db_infos = "kitti_infos/kitti_dbinfos_test_uncertainty_flipped.pkl"
    # db_infos = "kitti_infos/kitti_dbinfos_test_31552_006nd.pkl"
    with open(db_infos, 'rb') as f:
        kitti_db_infos = pickle.load(f)
    
    score_count = 0
    geo_count = 0
    print(kitti_db_infos["origin"]["Car"].keys())
    for ins in kitti_db_infos["origin"]["Car"]["375_1242"]:
        if ins["score"] > 0.50 and ins["difficulty"] == 0:
            score_count = score_count + 1
            if ins["geo_conf"] > 0.75:
                geo_count = geo_count + 1
            
    print("score_count: ", score_count)
    print("geo_count: ", geo_count)
    '''

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

        '''
        copy_file(src_image_2, dest_image_2)
        copy_file(src_image_3, dest_image_3)
        copy_file(src_calib, dest_calib)
        copy_file(src_label_2, dest_label_2)
        
        if image_idx > 8000:
            image = cv2.imread(dest_image_2)
            cv2.imwrite(os.path.join("debug", image_id_key + ".jpg"), image)
        '''
        # print(dest_image_2)
    print(num, copyed_num)
