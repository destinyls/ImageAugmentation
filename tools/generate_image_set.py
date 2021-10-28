import os

label_origin = "/root/SMOKE/datasets/kitti/ImageSets/train.txt"
label_enhanced = "/root/SMOKE/datasets/kitti/ImageSets/train_enhanced.txt"

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
    kitti_root = "/root/Dataset/kitti"
    img_train_ids = _read_imageset_file(kitti_root, "ImageSets/train.txt")

    for file_name in os.listdir(os.path.join(kitti_root, "training", "image_2")):
        image_idx = int(file_name.split('.')[0])
        if image_idx < 7481:
            continue
        img_train_ids.append(image_idx)

    print(len(img_train_ids))

    with open(label_enhanced,'w') as f:
        for idx in img_train_ids:
            frame_name = "{:06d}".format(idx)
            f.write(frame_name)
            f.write("\n")