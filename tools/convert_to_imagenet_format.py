import os
import cv2

kitti_root = "/root/Desktop/scenes/kitti_raw_dataset/image_2"

imagenet_root = "/root/Desktop/scenes/imagenet"
imagenet_train_path = os.path.join(imagenet_root, "train")
imagenet_meta_path = os.path.join(imagenet_root, "meta")

image_list = []
if __name__ == "__main__":
    for file_name in os.listdir(kitti_root):
        image_id = file_name.split('.')[0]
        file_name = os.path.join(kitti_root, file_name)
        print(file_name)

        image = cv2.imread(file_name)
        image_list.append(image_id + ".jpg")

        dest_file_name = os.path.join(imagenet_train_path, image_id + ".jpg")
        cv2.imwrite(dest_file_name, image)

    print("Hello World ...")
    
    meta_train_file_name = os.path.join(imagenet_meta_path, "train.txt")
    with open(meta_train_file_name, "w") as calib_file:
        for file_name in image_list:
            calib_file.write(file_name + '\n')

